from ..repository.assistant import AssistantRepository
# Assuming you have this utility
from ..utils.misc import convert_date_to_timestamp, generate_ai_assistant_instruction
from typing import Dict, List
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
from open_webui.config import OPENAI_API_KEY, GOOGLE_SPEECH_API_KEY
import logging
from open_webui.env import SRC_LOG_LEVELS
from open_webui.speech import google as google_stt
from open_webui.speech import whisper as whisper_stt
from open_webui.speech import openai as openai_tts
from open_webui.speech import fano as fano_tts

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["MODELS"])

async def transcribe(audio_file, audio_uri, language_code: str, vendor: str):
    """
    Transcribes audio using the specified vendor's service.,
    """
    if vendor == "google":
        transcription = await google_stt.transcribe(audio_file, audio_uri, language_code, GOOGLE_SPEECH_API_KEY)
    else:
        transcription = await whisper_stt.transcribe(audio_file, audio_uri, language_code)
    
    return {
        "transcription": transcription,
    }

async def speechify(text, language_code, vendor):
    try:
        if vendor == "openai":
            result = await openai_tts.tts(text, OPENAI_API_KEY)
        else:
            result = await fano_tts.tts(text, language_code)
        return result
    except Exception as e:
        log.exception(e)
        return None

