from enum import Enum


class MESSAGES(str, Enum):
    DEFAULT = lambda msg="": f"{msg if msg else ''}"
    MODEL_ADDED = lambda model="": f"The model '{model}' has been added successfully."
    MODEL_DELETED = (
        lambda model="": f"The model '{model}' has been deleted successfully."
    )


class WEBHOOK_MESSAGES(str, Enum):
    DEFAULT = lambda msg="": f"{msg if msg else ''}"
    USER_SIGNUP = lambda username="": (
        f"New user signed up: {username}" if username else "New user signed up"
    )


class ERROR_MESSAGES(str, Enum):
    def __str__(self) -> str:
        return super().__str__()

    DEFAULT = (
        lambda err="": f'{"Something went wrong :/" if err == "" else "[ERROR: " + str(err) + "]"}'
    )
    ENV_VAR_NOT_FOUND = "Required environment variable not found. Terminating now."
    CREATE_USER_ERROR = "Oops! Something went wrong while creating your account. Please try again later. If the issue persists, contact support for assistance."
    DELETE_USER_ERROR = "Oops! Something went wrong. We encountered an issue while trying to delete the user. Please give it another shot."
    EMAIL_MISMATCH = "Uh-oh! This email does not match the email your provider is registered with. Please check your email and try again."
    EMAIL_TAKEN = "Uh-oh! This email is already registered. Sign in with your existing account or choose another email to start anew."
    USERNAME_TAKEN = (
        "Uh-oh! This username is already registered. Please choose another username."
    )
    PASSWORD_TOO_LONG = "Uh-oh! The password you entered is too long. Please make sure your password is less than 72 bytes long."
    COMMAND_TAKEN = "Uh-oh! This command is already registered. Please choose another command string."
    FILE_EXISTS = "Uh-oh! This file is already registered. Please choose another file."

    ID_TAKEN = "Uh-oh! This id is already registered. Please choose another id string."
    MODEL_ID_TAKEN = "Uh-oh! This model id is already registered. Please choose another model id string."
    NAME_TAG_TAKEN = "Uh-oh! This name tag is already registered. Please choose another name tag string."

    INVALID_TOKEN = (
        "Your session has expired or the token is invalid. Please sign in again."
    )
    INVALID_CRED = "The email or password provided is incorrect. Please check for typos and try logging in again."
    INVALID_EMAIL_FORMAT = "The email format you entered is invalid. Please double-check and make sure you're using a valid email address (e.g., yourname@example.com)."
    INVALID_PASSWORD = (
        "The password provided is incorrect. Please check for typos and try again."
    )
    INVALID_TRUSTED_HEADER = "Your provider has not provided a trusted header. Please contact your administrator for assistance."

    EXISTING_USERS = "You can't turn off authentication because there are existing users. If you want to disable WEBUI_AUTH, make sure your web interface doesn't have any existing users and is a fresh installation."

    UNAUTHORIZED = "401 Unauthorized"
    ACCESS_PROHIBITED = "You do not have permission to access this resource. Please contact your administrator for assistance."
    ACTION_PROHIBITED = (
        "The requested action has been restricted as a security measure."
    )

    FILE_NOT_SENT = "FILE_NOT_SENT"
    FILE_NOT_SUPPORTED = "Oops! It seems like the file format you're trying to upload is not supported. Please upload a file with a supported format and try again."

    NOT_FOUND = "We could not find what you're looking for :/"
    USER_NOT_FOUND = "We could not find what you're looking for :/"
    API_KEY_NOT_FOUND = "Oops! It looks like there's a hiccup. The API key is missing. Please make sure to provide a valid API key to access this feature."
    API_KEY_NOT_ALLOWED = "Use of API key is not enabled in the environment."

    MALICIOUS = "Unusual activities detected, please try again in a few minutes."

    PANDOC_NOT_INSTALLED = "Pandoc is not installed on the server. Please contact your administrator for assistance."
    INCORRECT_FORMAT = (
        lambda err="": f"Invalid format. Please use the correct format{err}"
    )
    RATE_LIMIT_EXCEEDED = "API rate limit exceeded"

    MODEL_NOT_FOUND = lambda name="": f"Model '{name}' was not found"
    OPENAI_NOT_FOUND = lambda name="": "OpenAI API was not found"
    OLLAMA_NOT_FOUND = "WebUI could not connect to Ollama"
    CREATE_API_KEY_ERROR = "Oops! Something went wrong while creating your API key. Please try again later. If the issue persists, contact support for assistance."
    API_KEY_CREATION_NOT_ALLOWED = "API key creation is not allowed in the environment."

    EMPTY_CONTENT = "The content provided is empty. Please ensure that there is text or data present before proceeding."

    DB_NOT_SQLITE = "This feature is only available when running with SQLite databases."

    INVALID_URL = (
        "Oops! The URL you provided is invalid. Please double-check and try again."
    )

    WEB_SEARCH_ERROR = (
        lambda err="": f"{err if err else 'Oops! Something went wrong while searching the web.'}"
    )

    OLLAMA_API_DISABLED = (
        "The Ollama API is disabled. Please enable it to use this feature."
    )

    FILE_TOO_LARGE = (
        lambda size="": f"Oops! The file you're trying to upload is too large. Please upload a file that is less than {size}."
    )

    DUPLICATE_CONTENT = (
        "Duplicate content detected. Please provide unique content to proceed."
    )
    FILE_NOT_PROCESSED = "Extracted content is not available for this file. Please ensure that the file is processed before proceeding."


class TASKS(str, Enum):
    def __str__(self) -> str:
        return super().__str__()

    DEFAULT = lambda task="": f"{task if task else 'generation'}"
    TITLE_GENERATION = "title_generation"
    TAGS_GENERATION = "tags_generation"
    EMOJI_GENERATION = "emoji_generation"
    QUERY_GENERATION = "query_generation"
    IMAGE_PROMPT_GENERATION = "image_prompt_generation"
    AUTOCOMPLETE_GENERATION = "autocomplete_generation"
    FUNCTION_CALLING = "function_calling"
    MOA_RESPONSE_GENERATION = "moa_response_generation"

class TOOLS(str, Enum):
    def __str__(self) -> str:
        return super().__str__()

    DEFAULT = lambda tool="": f"{tool if tool else 'default'}"
    VISUALIZE_DATA = "visualize_data"

# In Python, module-level constants are typically written in UPPER_SNAKE_CASE.

# --- Model Definitions ---

OPENAI_MODELS = [
    {"name": "gpt-4o", "is_toolCall_available": True, "is_vision_available": True, "provider": "openai"},
    {"name": "gpt-4o-mini", "is_toolCall_available": True, "is_vision_available": True, "provider": "openai"},
    {"name": "gpt-4.1", "is_toolCall_available": True, "is_vision_available": True, "provider": "openai"},
    {"name": "gpt-5-nano", "is_toolCall_available": True, "is_vision_available": True, "provider": "openai"},
    {"name": "gpt-5.1", "is_toolCall_available": True, "is_vision_available": False, "provider": "openai"},
    {"name": "gpt-5-mini", "is_toolCall_available": True, "is_vision_available": False, "provider": "openai"},
]

OLLAMA_MODELS = [
    {"name": "llama3.2", "is_toolCall_available": True, "is_vision_available": False, "provider": "ollama"},
    {
        "name": "llama3.3:70b-instruct-q2_K",
        "is_toolCall_available": True,
        "is_vision_available": False,
        "provider": "ollama",
    },
    {"name": "llama3.2-vision", "is_toolCall_available": True, "is_vision_available": True, "provider": "ollama"},
]

GROQ_MODELS = [
    {
        "name": "deepseek-r1-distill-llama-70b",
        "is_toolCall_available": True,
        "is_vision_available": False,
    },
    {
        "name": "deepseek-r1-distill-qwen-32b",
        "is_toolCall_available": True,
        "is_vision_available": False,
    },
    {
        "name": "meta-llama/llama-4-scout-17b-16e-instruct",
        "is_toolCall_available": True,
        "is_vision_available": True,
    },
]

NOKIA_MODELS = [
    {"name": "llama3.2", "is_toolCall_available": True, "is_vision_available": False},
    {
        "name": "MFDoom/deepseek-r1-tool-calling:32b",
        "is_toolCall_available": True,
        "is_vision_available": False,
    },
    {"name": "deepseek-r1:14B", "is_toolCall_available": False, "is_vision_available": False},
    {
        "name": "MFDoom/deepseek-r1-tool-calling:70b",
        "is_toolCall_available": True,
        "is_vision_available": False,
    },
]

GOOGLE_MODELS = [
    {"name": "gemini-2.0-flash", "is_toolCall_available": True, "is_vision_available": False, "provider": "google"},
    {"name": "gemini-2.0-flash-lite", "is_toolCall_available": True, "is_vision_available": False, "provider": "google"},
    {"name": "gemini-pro", "is_toolCall_available": True, "is_vision_available": False, "provider": "google"},
]

LMDEPLOY_MODELS = [
    {
        "name": "OpenGVLab/InternVL2_5-8B-MPO-AWQ",
        "is_toolCall_available": False,
        "is_vision_available": True,
    },
]

BEDROCK_MODELS = [
    {"name": "openai.gpt-oss-120b-1:0", "is_toolCall_available": True, "is_vision_available": False, "provider": "bedrock"},
    {"name": "openai.gpt-oss-20b-1:0", "is_toolCall_available": True, "is_vision_available": False, "provider": "bedrock"},
    {"name": "qwen.qwen3-235b-a22b-2507-v1:0", "is_toolCall_available": True, "is_vision_available": False, "provider": "bedrock"},
]

# A dictionary that groups all the model lists together, like the JS export
MODEL_LIST = {
    "OpenAIModels": OPENAI_MODELS,
    "OllamaModels": OLLAMA_MODELS,
    "GroqModels": GROQ_MODELS,
    "NokiaModels": NOKIA_MODELS,
    "GoogleModels": GOOGLE_MODELS,
    "LmdeployModels": LMDEPLOY_MODELS,
    "BedrockModels": BEDROCK_MODELS,
}

def is_tool_call_available(model_name: str) -> bool:
    """
    Checks if a given model name has tool calling available.

    This function iterates through all model lists to find a match.
    The nested `any()` calls are the Pythonic equivalent of nested `.some()`.
    """
    return any(
        any(
            model["name"] == model_name and model["is_toolCall_available"]
            for model in model_list
        )
        for model_list in MODEL_LIST.values()
    )