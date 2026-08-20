from open_webui.config import OPENAI_API_KEY
from openai import OpenAI

async def tts(text, api_key):
    try:
        client = OpenAI(api_key=api_key)
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text,
            response_format="wav"
        )
        audio_bytes = response.read()
        return audio_bytes
    except Exception as e:
        print(f"Error in tts: {e}")
        raise e




