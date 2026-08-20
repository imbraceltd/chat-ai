from abc import ABC, abstractmethod
import pandas as pd

class TabularFile(ABC):
    @abstractmethod
    def validateFile(self) -> Exception:
        """Validate the file contents."""
        pass

    @abstractmethod
    def describe(self) -> list[tuple[str, str]]:
        """Return list of (column name, data type)."""
        pass

    @abstractmethod
    def summarize(self) -> dict:
        """Return column statistics."""
        pass

    @abstractmethod
    def top(self, n: int) -> list[dict]:
        """Return the top n rows as a list of dictionaries."""
        pass

    @abstractmethod
    def to_parquet(self) -> bytes:
        """Export valid data to parquet format (as bytes)."""
        pass

    @abstractmethod
    def query(self, sql: str) -> pd.DataFrame:
        """Execute SQL on the data and return a pandas DataFrame."""
        pass
