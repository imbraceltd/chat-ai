import requests
import json
import os
from typing import Union, Optional


def upload_file(org_id: str, file_content: Union[str, dict], file_name: str, backend_url: str) -> str:
    """
    Upload a file to the backend server.
    
    Args:
        org_id (str): Organization ID
        file_content (Union[str, dict]): File content as string or dict
        file_name (str): Name of the file
        backend_url (str): Backend URL
    
    Returns:
        str: URL of the uploaded file
    
    Raises:
        Exception: If upload fails or unsupported file extension
    """
    try:
        url = f"{backend_url}/v1/board/_fileupload/{org_id}"
        
        # Detect file type from extension
        ext = file_name.split(".")[-1].lower()
        
        if ext == "json":
            if isinstance(file_content, str):
                content = file_content
            else:
                content = json.dumps(file_content, indent=2)
            content_bytes = content.encode('utf-8')
            content_type = "application/json; charset=utf-8"
        elif ext in ["csv", "txt"]:
            content_bytes = file_content.encode('utf-8')
            content_type = "text/csv; charset=utf-8"
        elif ext == "md":
            content_bytes = file_content.encode('utf-8')
            content_type = "text/markdown; charset=utf-8"
        else:
            raise Exception(f"Unsupported file extension: .{ext}")
        
        # Create files dict for multipart form data
        files = {
            'text': (file_name, content_bytes, content_type)
        }
        
        response = requests.post(url, files=files)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        data = response.json()
        return data['url']
        
    except Exception as error:
        print(f"Error uploading file: {error}")
        raise error