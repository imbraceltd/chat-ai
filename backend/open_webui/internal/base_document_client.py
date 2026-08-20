from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseDocumentClient(ABC):
    """Abstract base class defining the document client contract.

    Both MongoDocumentClient and PgDocumentClient must implement this interface.
    All logic/repository layers import only from internal/mongo_db.py which
    re-exports the active implementation chosen via DB_TYPE env var.
    """

    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def reconnect(self) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...

    @abstractmethod
    async def query(
        self,
        database_name: str = "",
        collection_name: str = "",
        query: Dict[str, Any] = None,
    ) -> List[Dict[str, Any]]: ...

    @abstractmethod
    async def query_one(
        self,
        database_name: str = "",
        collection_name: str = "",
        query: Dict[str, Any] = None,
    ) -> Optional[Dict[str, Any]]: ...

    @abstractmethod
    async def query_one_with_projection(
        self,
        database_name: str = "",
        collection_name: str = "",
        query: Dict[str, Any] = None,
        projection: Dict[str, Any] = None,
    ) -> Optional[Dict[str, Any]]: ...

    @abstractmethod
    async def query_with_projection(
        self,
        database_name: str = "",
        collection_name: str = "",
        query: Dict[str, Any] = None,
        projection: Dict[str, Any] = None,
    ) -> List[Dict[str, Any]]: ...

    @abstractmethod
    async def query_with_sort_and_pagination(
        self,
        database_name: str = "",
        collection_name: str = "",
        query: Dict[str, Any] = None,
        sort_field: str = "updated_at",
        sort_order: int = -1,
        limit: int = 10,
        skip: int = 0,
    ) -> Dict[str, Any]: ...

    @abstractmethod
    async def query_one_by_pipeline(
        self,
        database_name: str = "",
        collection_name: str = "",
        pipeline: List[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]: ...

    @abstractmethod
    async def query_by_pipeline(
        self,
        database_name: str = "",
        collection_name: str = "",
        pipeline: List[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]: ...

    @abstractmethod
    async def insert(
        self,
        database_name: str = "",
        collection_name: str = "",
        data: Dict[str, Any] = None,
    ) -> Optional[Dict[str, Any]]: ...

    @abstractmethod
    async def insert_many(
        self,
        database_name: str = "",
        collection_name: str = "",
        data_array: List[Dict[str, Any]] = None,
    ) -> Dict[str, Any]: ...

    @abstractmethod
    async def update(
        self,
        database_name: str = "",
        collection_name: str = "",
        query: Dict[str, Any] = None,
        data: Dict[str, Any] = None,
    ) -> Optional[Dict[str, Any]]: ...

    @abstractmethod
    async def update_many(
        self,
        database_name: str = "",
        collection_name: str = "",
        query: Dict[str, Any] = None,
        data: Dict[str, Any] = None,
    ) -> Dict[str, Any]: ...

    @abstractmethod
    async def delete_one(
        self,
        database_name: str = "",
        collection_name: str = "",
        query: Dict[str, Any] = None,
    ) -> Dict[str, Any]: ...

    @abstractmethod
    async def delete_many(
        self,
        database_name: str = "",
        collection_name: str = "",
        query: Dict[str, Any] = None,
    ) -> Dict[str, Any]: ...
