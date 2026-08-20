import sys
from fastapi import File, HTTPException, Request, status, UploadFile
import logging
import requests

from open_webui.env import GLOBAL_LOG_LEVEL, IMBRACE_API_URL, SRC_LOG_LEVELS
from open_webui.constants import ERROR_MESSAGES

logging.basicConfig(stream=sys.stdout, level=GLOBAL_LOG_LEVEL)
log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["MAIN"])

async def upload_assistant_file(request: Request, assistant_id: str, file: UploadFile = File(...), is_rag: bool = True):
    token = request.headers.get("X-Access-Token")
    auth_token = request.headers.get("Authorization")
    
    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.INVALID_TOKEN,
        )
    
    try:
        # Read file content
        file_content = await file.read()
        filename = file.filename
        
        # Prepare headers for the API request
        headers = {
            'x-access-token': token,
        }
        
        # Add Authorization header if present
        if auth_token:
            headers['Authorization'] = auth_token
        
        # Prepare form data for file upload
        files = {
            'file': (filename, file_content, file.content_type or 'application/octet-stream')
        }
        
        data = {
            'assistant_id': assistant_id,
            'is_rag': str(is_rag).lower(),
            'enable_pdf_table_extraction': 'false'  
        }
        
        # Make request to Imbrace API file upload endpoint
        response = requests.post(
            f"{IMBRACE_API_URL}/api/v2/ai/rag/files",
            headers=headers,
            files=files,
            data=data,
        )
        
        # Handle response based on status code
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 401:
            log.error(f"Imbrace API authentication failed: {response.text}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid access token"
            )
        elif response.status_code == 404:
            log.error(f"Assistant not found: {assistant_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Assistant with ID {assistant_id} not found"
            )
        else:
            log.error(f"Imbrace API file upload failed: Status {response.status_code}, Response: {response.text}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to upload file to Imbrace API: {response.text}"
            )
    except requests.RequestException as e:
        log.exception(f"Request to Imbrace API failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Connection to Imbrace API failed: {str(e)}"
        )
    except Exception as e:
        log.exception(f"Unexpected error in upload_agent_file: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )
    
