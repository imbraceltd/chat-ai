import pandas as pd
import io
from ..interface.tabular_file import TabularFile
from ..duckdb import DuckDBClient
from open_webui.models.files import Files

class TabularFileProcessor(TabularFile):
    """Processor for CSV files stored as Parquet in S3."""

    def __init__(self):
        pass

    def validateFile(self) -> Exception | None:
        """Validate that the DataFrame is not empty."""
        try:
            assert not self.df.empty
            return None
        except Exception as e:
            return e

    def describe(self, file_id: str) -> list[tuple[str, str]]:
        """Return column names and types for the file."""
        try:
            df = self._load_df_from_s3(file_id)
            return [(col, str(dtype)) for col, dtype in df.dtypes.items()]
        except Exception as e:
            print(f"Error describing file {file_id}: {e}")
            raise

    def summarize(self, file_id: str) -> dict:
        """Return a summary (describe) of the file as a dictionary."""
        try:
            df = self._load_df_from_s3(file_id)
            return df.describe(include='all').fillna("").to_dict()
        except Exception as e:
            print(f"Error summarizing file {file_id}: {e}")
            raise

    def top(self, file_id: str) -> list[dict]:
        """Return the top 10 rows of the DataFrame as a list of dicts."""
        try:
            df = self._load_df_from_s3(file_id)
            return df.head(10).to_dict(orient="records")
        except Exception as e:
            print(f"Error getting top rows from file {file_id}: {e}")
            raise

    def to_parquet(self, file_id: str) -> bytes:
        """Convert the file to Parquet format and return as bytes."""
        try:
            buffer = io.BytesIO()
            df = self._load_df_from_s3(file_id)
            df.to_parquet(buffer, engine='pyarrow', index=False)
            buffer.seek(0)
            return buffer.getvalue()
        except Exception as e:
            print(f"Error converting file {file_id} to parquet: {e}")
            raise

    def query(self, file_id: str | list[str], sql_query: str) -> pd.DataFrame:
        """Run an SQL query on the DataFrame(s) loaded from the Parquet file(s) using DuckDBClient."""
        duckdb_client = DuckDBClient(in_memory=True)
        try:
            if isinstance(file_id, list):
                # Use the execute_multi method to load all files and enable joins between them
                result = duckdb_client.execute_multi(file_id, sql_query)
                return pd.DataFrame(result, columns=[col[0] for col in duckdb_client.conn.description])
            else:
                # For a single file, just return the results directly
                result = duckdb_client.execute(file_id, sql_query)
                return pd.DataFrame(result, columns=[col[0] for col in duckdb_client.conn.description])
        finally:
            print("Closing DuckDB connection")
            duckdb_client.conn.close()

    def _load_df_from_s3(self, file_id: str) -> pd.DataFrame:
        """Load a DataFrame from S3 using DuckDBClient."""
        duckdb_client = DuckDBClient(in_memory=True)
        try:
            s3_path = f"tabular/{file_id}.parquet"
            return duckdb_client.read_file_from_s3(s3_path, file_type='parquet')
        finally:
            if hasattr(duckdb_client, 'conn'):
                print("Closing DuckDB connection")
                duckdb_client.conn.close()
