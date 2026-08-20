from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any
from typing import Optional, List, Dict, Annotated, Union, Any
from pydantic import BaseModel, Field, constr, field_validator
from bson import ObjectId
from datetime import datetime
from open_webui.utils.misc import convert_date_to_timestamp
from .mongo import get_assistants_collection


class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)


class Tool(BaseModel):
    type: str


class AssistantBase(BaseModel):
    name: Annotated[str, Field(max_length=256)]
    description: Optional[Annotated[str, Field(max_length=512)]] = None
    instructions: Optional[Annotated[str, Field(max_length=32768)]] = None
    model: Optional[str] = None
    tools: Optional[List[Tool]] = None
    top_p: Optional[float] = 1.0
    temperature: Optional[float] = 1.0
    file_ids: Optional[List[str]] = None
    folder_ids: Optional[List[str]] = None
    board_ids: Optional[List[str]] = None
    metadata: Optional[Dict] = None
    response_format: Optional[str] = "auto"
    mode: Optional[str] = "standard"
    channel: Optional[str] = ""
    category: Optional[List[str]] = []
    personality_role: Optional[str] = ""
    core_task: Optional[str] = ""
    tone_and_style: Optional[str] = ""
    response_length: Optional[str] = ""
    banned_words: Optional[str] = ""
    workflow_function_call: Optional[List] = []
    knowledge_hubs: Optional[str] = ""
    model_id: Optional[str] = ""
    show_thinking_process: Optional[bool] = False
    preload_information: Optional[str] = ""
    guardrail_id: Optional[str] = ""
    default_folder_id: Optional[str] = None
    document_ai: Optional[Dict] = None
    vibe_coding: Optional[bool] = False

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class AssistantDB(AssistantBase):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    object: str = "assistant"
    created_at: Optional[Union[int, str]] = None
    updated_at: Optional[str] = None

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
            "example": {
                "_id": "64f12eb7f4c46669e5a6c285",
                "object": "assistant",
                "name": "Data Analysis",
                "description": "Expert in data analysis",
                "instructions": "Help with data analysis tasks",
                "model": "gpt-4o-mini",
                "tools": [{"type": "file_search"}],
                "top_p": 1.0,
                "temperature": 1.0,
                "file_ids": ["file-qZleuycUPtV8TS6ghk8SyHPn"],
                "metadata": {"key_1": "value_1"},
                "response_format": "auto",
                "mode": "standard",
                "channel": "",
                "category": [],
                "personality_role": "",
                "core_task": "",
                "tone_and_style": "",
                "response_length": "",
                "banned_words": "",
                "workflow_function_call": [],
                "knowledge_hubs": "",
                "model_id": "",
                "show_thinking_process": False,
                "preload_information": "",
                "guardrail_id": "",
                "created_at": 1723629862,
                "updated_at": "2023-09-01T00:00:00Z",
            }
        }


class AssistantCreate(AssistantBase):
    pass


class AssistantResponse(BaseModel):
    # --- All Fields Defined Here ---
    id: str
    object: str = "assistant"
    created_at: int
    name: str
    description: str
    model: str
    instructions: str
    tools: List[Dict] = []
    top_p: float = 1.0
    temperature: float = 1.0
    file_ids: List[str] = []
    folder_ids: List[str] = []
    board_ids: List[str] = []
    metadata: Dict = {}
    response_format: str = "auto"
    mode: str = "standard"
    channel: str = ""
    category: List[str] = []
    personality_role: str = ""
    core_task: str = ""
    tone_and_style: str = ""
    response_length: str = ""
    banned_words: str = ""
    workflow_function_call: List = []
    knowledge_hubs: str = ""
    model_id: str = ""
    show_thinking_process: bool = False
    preload_information: str = ""
    guardrail_id: str = ""
    default_folder_id: Optional[str] = None
    document_ai: Optional[Dict] = None
    vibe_coding: bool = False

    # --- Validators for Data Cleaning ---

    @field_validator("id", mode="before")
    @classmethod
    def convert_objectid_to_str(cls, v: Any) -> str:
        """Converts a BSON ObjectId to a string."""
        if isinstance(v, ObjectId):
            return str(v)
        # Handles both 'assistant_id' and '_id' if one is already a string
        return v

    @field_validator("created_at", mode="before")
    @classmethod
    def format_created_at(cls, v: Any) -> int:
        return convert_date_to_timestamp(v)

    @field_validator("workflow_function_call", mode="before")
    @classmethod
    def handle_none_as_list(cls, v: Any) -> list:
        return v or []

    @field_validator("knowledge_hubs", mode="before")
    @classmethod
    def handle_list_as_string(cls, v: Any) -> str:
        return v if isinstance(v, str) else ""

    @field_validator("model_id", mode="before")
    @classmethod
    def handle_model_id_none(cls, v: Any) -> str:
        """Ensures model_id is a string, defaulting None to empty."""
        return v or ""

    class Config:
        populate_by_name = True


class AssistantUpdate(BaseModel):
    name: Optional[Annotated[str, Field(max_length=256)]] = None
    description: Optional[Annotated[str, Field(max_length=512)]] = None
    instructions: Optional[Annotated[str, Field(max_length=32768)]] = None
    file_ids: Optional[List[str]] = None
    folder_ids: Optional[List[str]] = None
    board_ids: Optional[List[str]] = None
    metadata: Optional[Dict] = None
    mode: Optional[str] = None
    channel: Optional[str] = None
    category: Optional[List[str]] = None
    personality_role: Optional[str] = None
    core_task: Optional[str] = None
    tone_and_style: Optional[str] = None
    response_length: Optional[str] = None
    banned_words: Optional[str] = None
    default_folder_id: Optional[str] = None
    document_ai: Optional[Dict] = None
    vibe_coding: Optional[bool] = None

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class BulkDeleteAssistants(BaseModel):
    ids: List[str]

    @property
    def object_ids(self) -> List[ObjectId]:
        return [ObjectId(id_) for id_ in self.ids]

    class Config:
        schema_extra = {
            "example": {"ids": ["64f12eb7f4c46669e5a6c285", "64f12eb7f4c46669e5a6c286"]}
        }


# MongoDB CRUD operations


async def create_assistant(assistant: AssistantCreate) -> AssistantDB:
    collection = get_assistants_collection()

    assistant_dict = assistant.dict()
    assistant_dict["created_at"] = datetime.utcnow().isoformat()
    assistant_dict["updated_at"] = assistant_dict["created_at"]

    result = await collection.insert_one(assistant_dict)

    created_assistant = await collection.find_one({"_id": result.inserted_id})
    return AssistantDB(**created_assistant)


async def get_assistant(assistant_id: str) -> Optional[AssistantDB]:
    collection = get_assistants_collection()
    assistant = await collection.find_one({"_id": ObjectId(assistant_id)})
    if assistant:
        return AssistantDB(**assistant)
    return None


async def get_all_assistants() -> List[AssistantDB]:
    collection = get_assistants_collection()
    assistants = []
    async for assistant in collection.find():
        assistants.append(AssistantDB(**assistant))
    return assistants


async def update_assistant(
    assistant_id: str, assistant_update: AssistantUpdate
) -> Optional[AssistantDB]:
    collection = get_assistants_collection()

    update_data = assistant_update.dict(exclude_unset=True)
    update_data["updated_at"] = datetime.utcnow().isoformat()

    if update_data:
        result = await collection.update_one(
            {"_id": ObjectId(assistant_id)}, {"$set": update_data}
        )
        if result.modified_count:
            return await get_assistant(assistant_id)
    return None


async def delete_assistant(assistant_id: str) -> bool:
    collection = get_assistants_collection()
    result = await collection.delete_one({"_id": ObjectId(assistant_id)})
    return result.deleted_count > 0


async def bulk_delete_assistants(bulk_delete: BulkDeleteAssistants) -> int:
    collection = get_assistants_collection()
    result = await collection.delete_many({"_id": {"$in": bulk_delete.object_ids}})
    return result.deleted_count
