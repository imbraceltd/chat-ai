import tempfile
import uuid
import os
import aiofiles  # Use aiofiles for async file operations
from open_webui.utils.tools import download_file, convert_to_16khz
import whisper

async def transcribe(
    audio_file: object, # Assuming this is a file-like object from a web framework (e.g., FastAPI's UploadFile)
    audio_uri: str,
    language_code: str,
    api_key: str = None  # api_key is often not needed if GOOGLE_APPLICATION_CREDENTIALS is set
):
    """
    Transcribes an audio file using Google Cloud Speech-to-Text.

    The function handles two cases:
    1. A remote audio file specified by `audio_uri`.
    2. An uploaded audio file passed as `audio_file`.

    It converts the audio to a 16kHz mono WAV file before transcription.

    Args:
        audio_file (object): An object with a `read()` method containing the audio data.
        audio_uri (str): The public URI of an audio file to download.
        language_code (str): The language code for transcription (e.g., "en-US").
        api_key (str, optional): Google Cloud API key. Defaults to None.
                                 Note: Authentication is typically handled by Application Default Credentials.

    Returns:
        str: The transcribed text.
        
    Raises:
        ValueError: If neither audio_file nor audio_uri is provided.
        Exception: For errors during download, conversion, or transcription.
    """
    # Use two distinct paths: one for the initial download/write, and one for the converted file.
    # Using the same path for input and output of ffmpeg is a common source of errors.
    temp_dir = tempfile.gettempdir()
    
    # Fallback to current directory if temp directory is not accessible
    if not os.access(temp_dir, os.W_OK):
        print(f"Warning: Cannot write to temp directory {temp_dir}, using current directory")
        temp_dir = os.getcwd()
    
    unique_id = uuid.uuid4().hex
    input_path = os.path.join(temp_dir, f"{unique_id}_input.wav")
    converted_path = os.path.join(temp_dir, f"{unique_id}_16khz_mono.wav")
    
    try:
        if audio_uri:
            await download_file(audio_uri, input_path)
        elif audio_file:
            async with aiofiles.open(input_path, "wb") as f:
                buffer = await audio_file.read()
                await f.write(buffer)
        else:
            raise ValueError("No audio file or audio URI provided")

        convert_to_16khz(input_path, converted_path)

        model = whisper.load_model("turbo")
        result = model.transcribe(converted_path)
        
        return result["text"]

    except Exception as e:
        raise e
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(converted_path):
            os.remove(converted_path)