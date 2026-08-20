import os
import requests
import base64

from open_webui.config import FANO_URL, FANO_API_KEY
async def tts(language_code: str = "yue", message: str = "") -> bytes:
    """
    Synthesizes speech using the FANO TTS API and returns the audio as bytes.

    Args:
        language_code (str, optional): The language code for the voice. Defaults to "yue".
        message (str, optional): The text message to synthesize.

    Returns:
        bytes: The raw MP3 audio data.
        
    Raises:
        requests.exceptions.HTTPError: If the API returns an error status code.
        Exception: For other issues like network problems.
    """
    if not message:
        raise ValueError("Message cannot be empty.")

    # If language_code is not provided or is an empty string, default to "yue"
    if not language_code:
        language_code = "yue"

    api_url = f"{FANO_URL}/speech/synthesize-speech"

    # The request payload (body) as a Python dictionary
    payload = {
        "input": {
            "text": message,
        },
        "voice": {
            "languageCode": language_code,
            "gender": "MALE",
        },
        "audioConfig": {
            "encoding": "MP3",
            "sampleRateHertz": 22050,
            "speakingRate": 1,
        },
        "maxexpand": 100000,
    }

    headers = {
        "Authorization": f"Bearer {FANO_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload)

        # Raise an exception for bad status codes (4xx or 5xx)
        response.raise_for_status()

        # Extract the 'audioContent' from the JSON response
        response_data = response.json()
        audio_content_base64 = response_data.get("audioContent")

        if not audio_content_base64:
            raise ValueError("API response did not contain 'audioContent'.")

        # Decode the Base64 string into bytes
        # This is the Python equivalent of Buffer.from(audioContent, "base64")
        return base64.b64decode(audio_content_base64)

    except requests.exceptions.RequestException as e:
        # Handle connection errors, timeouts, etc.
        print(f"An HTTP request error occurred: {e}")
        raise e
    except Exception as e:
        # Handle other errors like JSON decoding or ValueErrors
        print(f"An error occurred: {e}")
        raise e