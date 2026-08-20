import os
import logging
import duckdb
from typing import Optional, Dict, Any, List, Tuple
import uuid

from open_webui.config import DUCKDB_PATH
from open_webui.env import SRC_LOG_LEVELS

from open_webui.models.files import Files

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS.get("RAG", logging.INFO))


# --- Local tabular storage via file-service --------------------------------
# When storage is local, a tabular file's parquet is written to a local temp
# file, uploaded to file-service (the shared HTTP blob store) instead of S3, and
# the parquet record stores file-service's HTTP download URL. DuckDB reads the
# parquet over HTTP (httpfs) from that URL — so BOTH this service and other
# services (e.g. nextbestaction) can query it. Mirrors how data-board stores
# files. Toggle with TABULAR_STORAGE_PROVIDER (falls back to STORAGE_PROVIDER).
TABULAR_USE_LOCAL = (
    os.environ.get("TABULAR_STORAGE_PROVIDER", os.environ.get("STORAGE_PROVIDER", ""))
    .strip()
    .lower()
    == "local"
)
# file-service endpoints. Upload is internal; the download base is embedded in
# stored records and must be reachable by every service that queries tabular
# data — override to an app-gateway/public URL for cross-host deployments.
TABULAR_FS_UPLOAD_URL = (
    os.environ.get("TABULAR_FS_UPLOAD_URL")
    or "http://file-service:8866/api/files/upload"
)
TABULAR_FS_DOWNLOAD_BASE = (
    os.environ.get("TABULAR_FS_DOWNLOAD_BASE")
    or "http://file-service:8866/api/files/download"
).rstrip("/")


def _tabular_basename(file_id_or_path: str) -> str:
    """Storage key for a tabular parquet: ``<id>.parquet`` (slash-free)."""
    base = str(file_id_or_path).rstrip("/").rsplit("/", 1)[-1]
    return base if base.endswith(".parquet") else f"{base}.parquet"


def tabular_uri(file_id: str, bucket: str = "") -> str:
    """Canonical readable location for a file's parquet.

    Local mode → file-service HTTP download URL. Otherwise → ``s3://`` URI.
    DuckDB reads both via read_parquet (httpfs handles http/s3).
    """
    if TABULAR_USE_LOCAL:
        return f"{TABULAR_FS_DOWNLOAD_BASE}/{_tabular_basename(file_id)}"
    return f"s3://{bucket}/tabular/{file_id}.parquet"


def resolve_tabular_path(path: str, bucket: str = "") -> str:
    """Map a stored/relative tabular path to a readable one.

    In local mode any path (a legacy ``s3://.../<id>.parquet``, a bare local
    path, or a basename) is resolved to the file-service download URL so older
    records keep working. Already-HTTP URLs pass through unchanged.
    """
    if not path:
        return path
    if TABULAR_USE_LOCAL:
        if path.startswith("http://") or path.startswith("https://"):
            return path
        return f"{TABULAR_FS_DOWNLOAD_BASE}/{_tabular_basename(path)}"
    if not path.startswith("s3://"):
        return f"s3://{bucket}/{path}"
    return path


class DuckDBClient:
    def __init__(
        self, aws_config: Optional[Dict[str, str]] = None, in_memory: bool = False
    ):
        """
        Initialize the DuckDB client with extensions for S3 access.

        Args:
            aws_config: Optional dictionary with AWS configuration.
                        If not provided, will try to get credentials from environment variables.
            in_memory: If True, use an in-memory database instead of a file-based one.
        """
        # Set up database path (used only if in_memory is False)
        self.db_path = DUCKDB_PATH or os.path.join(os.getcwd(), "tabular_store.db")
        self.table_prefix = "open_webui_tabular"
        self.in_memory = in_memory

        # If aws_config not provided, try to get from environment variables

        self.aws_config = {
            "access_key_id": os.environ.get("S3_ACCESS_KEY_ID"),
            "secret_access_key": os.environ.get("S3_SECRET_ACCESS_KEY"),
            "region": os.environ.get("S3_REGION_NAME"),
            "bucket": os.environ.get("S3_BUCKET_NAME"),
        }

        # Connect to the database (in-memory or file-based)
        if in_memory:
            self.conn = duckdb.connect()
        else:
            self.conn = duckdb.connect(self.db_path)

        # Load extensions and configure AWS
        self._load_extensions()

        # Initialize database schema (only needed for persisted databases)
        if not in_memory:
            self._initialize_db()

    def _load_extensions(self):
        """Load DuckDB extensions and configure AWS access if credentials are provided."""
        try:
            # Load the httpfs extension to enable HTTP/HTTPS support
            self.conn.execute("INSTALL httpfs; LOAD httpfs;")

            # Install and load AWS extension
            self.conn.execute("INSTALL aws; LOAD aws;")

            # Check if we have AWS credentials (either from param or env vars)
            access_key_id = self.aws_config.get("access_key_id")
            secret_access_key = self.aws_config.get("secret_access_key")
            region = self.aws_config.get("region")
            bucket = self.aws_config.get("bucket")

            # Configure AWS S3 access if credentials are provided
            if all([access_key_id, secret_access_key, region]):
                # Create S3 secret with proper credentials
                endpoint = os.environ.get(
                    "S3_ENDPOINT_URL", f"s3.{region}.amazonaws.com"
                )
                scope = f"s3://{bucket}" if bucket else ""

                self.conn.execute(
                    f"""
                    CREATE SECRET encrypted (
                        TYPE s3,
                        KEY_ID '{access_key_id}',
                        SECRET '{secret_access_key}',
                        REGION '{region}',
                        ENDPOINT '{endpoint}',
                        SCOPE '{scope}'
                    );
                """
                )
            else:
                missing = []
                if not access_key_id:
                    missing.append("AWS_ACCESS_KEY_ID")
                if not secret_access_key:
                    missing.append("AWS_SECRET_ACCESS_KEY")
                if not region:
                    missing.append("AWS_REGION")
        except Exception as e:
            log.exception(f"Error loading extensions: {e}")
            # Continue with local database functionality even if extension loading fails
            log.warning(
                "Continuing with limited functionality (without cloud storage access)"
            )

    def _initialize_db(self):
        """Initialize the database schema with required tables."""
        try:
            # Create metadata table for tabular data
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tabular_metadata (
                    id VARCHAR PRIMARY KEY,
                    name VARCHAR NOT NULL,
                    file_type VARCHAR NOT NULL,
                    row_count INTEGER,
                    column_count INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

        except Exception as e:
            log.exception(f"Error initializing database: {e}")
            raise

    def _get_table_name(self, file_id: str) -> str:
        """Generate the internal table name for a file."""
        # Replace hyphens with underscores for SQL compatibility
        safe_name = file_id.replace("-", "_").replace(".", "_")
        return f"{self.table_prefix}_{safe_name}"

    def tabular_uri(self, file_id: str) -> str:
        """Storage location (local path or s3:// URI) for a file's parquet."""
        return tabular_uri(file_id, self.aws_config.get("bucket", ""))

    def read_file_from_s3(self, s3_path: str, file_type: str):
        """
        Read a file directly from S3 into a DataFrame.

        Args:
            s3_path: S3 path to the file (s3://bucket/path/to/file)
            file_type: Type of file ('csv', 'excel', 'parquet')

        Returns:
            A pandas DataFrame
        """
        s3_path = resolve_tabular_path(s3_path, self.aws_config.get("bucket", ""))

        try:
            if file_type.lower() == "csv":
                return self.conn.execute(f"SELECT * FROM read_csv('{s3_path}')").df()
            elif file_type.lower() == "parquet":
                return self.conn.execute(
                    f"SELECT * FROM read_parquet('{s3_path}')"
                ).df()
            elif file_type.lower() in ("excel", "xlsx"):
                # For Excel files, DuckDB doesn't have direct support, need to use httpfs
                # This may require additional handling
                return self.conn.execute(
                    f"SELECT * FROM read_csv_auto('{s3_path}')"
                ).df()
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
        except Exception as e:
            log.exception(f"Error reading file from S3: {e}")
            raise

    def write_file_to_s3(
        self,
        file_id: str,
        df,
        s3_path: str,
        file_type: str,
        file_name: Optional[str] = None,
    ):
        """
        Write a DataFrame to S3.

        Args:
            file_id: File ID for the tabular data
            df: Pandas DataFrame to write
            s3_path: S3 path to write to (s3://bucket/path/to/file)
            file_type: Type of file to write ('csv', 'parquet')
            file_name: Optional file name to use for table naming (if not provided, uses file_id)

        Returns:
            True if successful, False otherwise
        """

        s3_path = f"s3://{self.aws_config.get('bucket', '')}/{s3_path}"

        try:
            # Determine table name - use provided file_name or fallback to file_id
            if file_name:
                base_name = file_name
            else:
                # Try to get from Files table, fallback to file_id
                file = Files.get_file_by_id(file_id)
                if file is not None and file.filename:
                    base_name, _ = os.path.splitext(file.filename)
                else:
                    base_name = f"tabular_{file_id[:8]}"  # Use first 8 chars of file_id

            table_name = f"{base_name.replace(' ', '_').replace('/', '_').replace('-', '_')}_{file_id.replace('-', '_')}"
            s3_path = self.tabular_uri(file_id)
            self.conn.register(table_name, df)

            # Only update file metadata if the file exists in the Files table
            file = Files.get_file_by_id(file_id)
            if file is not None:
                Files.update_file_metadata_by_id(
                    file_id,
                    {
                        "table_name": table_name,
                    },
                )

            if file_type.lower() == "csv":
                fmt = "CSV, HEADER"
            elif file_type.lower() == "parquet":
                fmt = "PARQUET"
            else:
                raise ValueError(
                    f"Unsupported file type for writing: {file_type}"
                )

            if TABULAR_USE_LOCAL:
                # Local mode: COPY to a temp file, then upload to file-service
                # (shared HTTP store). `s3_path` here is the file-service
                # download URL — we don't COPY to it (httpfs http is read-only).
                import tempfile

                with tempfile.TemporaryDirectory() as _td:
                    local_file = os.path.join(_td, _tabular_basename(file_id))
                    self.conn.execute(
                        f"COPY \"{table_name}\" TO '{local_file}' (FORMAT {fmt})"
                    )
                    self._upload_to_fileservice(local_file)
            else:
                self.conn.execute(
                    f"COPY \"{table_name}\" TO '{s3_path}' (FORMAT {fmt})"
                )

            return True
        except Exception as e:
            log.exception(f"Error writing tabular file: {e}")
            return False

    def _upload_to_fileservice(self, local_path: str) -> None:
        """Upload a local parquet file to file-service (shared HTTP blob store).

        file-service stores by basename, so the object is reachable at
        ``TABULAR_FS_DOWNLOAD_BASE/<basename>`` — i.e. ``tabular_uri(file_id)``.
        """
        import requests

        filename = os.path.basename(local_path)
        with open(local_path, "rb") as fh:
            resp = requests.post(
                TABULAR_FS_UPLOAD_URL,
                files={"file": (filename, fh, "application/octet-stream")},
                timeout=120,
            )
        resp.raise_for_status()
        log.info(f"Uploaded tabular parquet to file-service: {filename}")

    def register_table(self, file_id: str, df, file_name: str, file_type: str):
        """Register a pandas DataFrame as a table in DuckDB."""
        table_name = self._get_table_name(file_id)

        try:
            # Register the DataFrame
            self.conn.register(table_name, df)

            # Save metadata (only for persistent databases)
            if not self.in_memory:
                self.conn.sql(
                    "CREATE TABLE IF NOT EXISTS as SELECT * FROM " + table_name
                )
                self.conn.sql(
                    "COPY "
                    + table_name
                    + " TO 's3://"
                    + self.aws_config.get("bucket", "")
                    + "/tabular/"
                    + file_id
                    + ".parquet' (FORMAT PARQUET)"
                )
            return True
        except Exception as e:
            log.exception(f"Error registering table for file '{file_id}': {e}")
            return False

    def get_file_from_s3(self, file_id: str) -> bytes:
        """Get a file from S3 and return it as bytes."""
        try:
            s3_path = self.tabular_uri(file_id)
            print(f"Getting tabular file from: {s3_path}")
            df = self.conn.execute(f"SELECT * FROM read_parquet('{s3_path}')")
            return df
        except Exception as e:
            log.exception(f"Error getting file from S3: {e}")
            raise

    def execute(self, file_id: str, sql_query: str) -> bytes:
        """Execute a SQL query and return the result as bytes."""
        try:
            file = Files.get_file_by_id(file_id)
            if file is not None and file.filename:
                file_name, _ = os.path.splitext(file.filename)
            table_name = f"{file_name.replace(' ', '_').replace('/', '_')}_{file_id.replace('-', '_')}"
            s3_path = self.tabular_uri(file_id)
            self.conn.execute(
                f"CREATE TABLE \"{table_name}\" as SELECT * FROM read_parquet('{s3_path}')"
            )
            return self.conn.execute(sql_query).fetchall()
        except Exception as e:
            log.exception(f"Error executing query for file '{file_id}': {e}")
            raise

    def execute_multi(self, file_ids: List[str], sql_query: str) -> bytes:
        """Execute a SQL query on multiple files, supporting joins between them."""
        try:
            table_names = []
            for file_id in file_ids:
                file = Files.get_file_by_id(file_id)
                if file is not None and file.filename:
                    file_name, _ = os.path.splitext(file.filename)
                table_name = f"{file_name.replace(' ', '_').replace('/', '_')}_{file_id.replace('-', '_')}"
                s3_path = self.tabular_uri(file_id)
                print(f"Creating table {table_name} from {s3_path}")
                self.conn.execute(
                    f"CREATE TABLE \"{table_name}\" as SELECT * FROM read_parquet('{s3_path}')"
                )
                table_names.append(table_name)

            log.info(f"Created tables: {', '.join(table_names)}")
            log.info(f"Executing query: {sql_query}")
            return self.conn.execute(sql_query).fetchall()
        except Exception as e:
            log.exception(f"Error executing multi-file query: {e}")
            raise
