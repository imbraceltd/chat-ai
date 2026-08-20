import logging
import uuid
import jwt
import base64
import hmac
import hashlib
import requests
import os


from datetime import datetime, timedelta
from pytz import UTC
from typing import Optional, Union

from open_webui.models.users import Users

from open_webui.constants import ERROR_MESSAGES
from open_webui.env import (
    WEBUI_SECRET_KEY,
    TRUSTED_SIGNATURE_KEY,
    STATIC_DIR,
    SRC_LOG_LEVELS,
    IMBRACE_API_URL
)

from fastapi import BackgroundTasks, Depends, HTTPException, Request, Response, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from passlib.context import CryptContext


logging.getLogger("passlib").setLevel(logging.ERROR)

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["OAUTH"])

SESSION_SECRET = WEBUI_SECRET_KEY
ALGORITHM = "HS256"

##############
# Auth Utils
##############


def verify_signature(payload: str, signature: str) -> bool:
    """
    Verifies the HMAC signature of the received payload.
    """
    try:
        expected_signature = base64.b64encode(
            hmac.new(TRUSTED_SIGNATURE_KEY, payload.encode(), hashlib.sha256).digest()
        ).decode()

        # Compare securely to prevent timing attacks
        return hmac.compare_digest(expected_signature, signature)

    except Exception:
        return False


def override_static(path: str, content: str):
    # Ensure path is safe
    if "/" in path or ".." in path:
        log.error(f"Invalid path: {path}")
        return

    file_path = os.path.join(STATIC_DIR, path)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "wb") as f:
        f.write(base64.b64decode(content))  # Convert Base64 back to raw binary


def get_license_data(app, key):
    if key:
        try:
            res = requests.post(
                "https://api.openwebui.com/api/v1/license/",
                json={"key": key, "version": "1"},
                timeout=5,
            )

            if getattr(res, "ok", False):
                payload = getattr(res, "json", lambda: {})()
                for k, v in payload.items():
                    if k == "resources":
                        for p, c in v.items():
                            globals().get("override_static", lambda a, b: None)(p, c)
                    elif k == "count":
                        setattr(app.state, "USER_COUNT", v)
                    elif k == "name":
                        setattr(app.state, "WEBUI_NAME", v)
                    elif k == "metadata":
                        setattr(app.state, "LICENSE_METADATA", v)
                return True
            else:
                log.error(
                    f"License: retrieval issue: {getattr(res, 'text', 'unknown error')}"
                )
        except Exception as ex:
            log.exception(f"License: Uncaught Exception: {ex}")
    return False


bearer_security = HTTPBearer(auto_error=False)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password, hashed_password):
    return (
        pwd_context.verify(plain_password, hashed_password) if hashed_password else None
    )


def get_password_hash(password):
    return pwd_context.hash(password)


def create_token(data: dict, expires_delta: Union[timedelta, None] = None) -> str:
    payload = data.copy()

    if expires_delta:
        expire = datetime.now(UTC) + expires_delta
        payload.update({"exp": expire})

    encoded_jwt = jwt.encode(payload, SESSION_SECRET, algorithm=ALGORITHM)
    return encoded_jwt


def decode_token(token: str) -> Optional[dict]:
    try:
        decoded = jwt.decode(token, SESSION_SECRET, algorithms=[ALGORITHM])
        return decoded
    except Exception:
        return None


def extract_token_from_auth_header(auth_header: str):
    return auth_header[len("Bearer ") :]


def create_api_key():
    key = str(uuid.uuid4()).replace("-", "")
    return f"sk-{key}"


def get_http_authorization_cred(auth_header: Optional[str]):
    if not auth_header:
        return None
    try:
        scheme, credentials = auth_header.split(" ")
        return HTTPAuthorizationCredentials(scheme=scheme, credentials=credentials)
    except Exception:
        return None


def get_current_user(
    request: Request,
    background_tasks: BackgroundTasks,
    auth_token: HTTPAuthorizationCredentials = Depends(bearer_security),
):
    token = None
    
    # log.info(f"Token details: {auth_token}")
    # log.info(f"imbrace details {IMBRACE_API_URL} {IMBRACE_DEFAULT_PASSWORD}")
    
    # Finally try to get token from query parameters
    # log.info(f"Raw URL: {str(request.url)}")
    is_imbrace = False
    is_existing_user = True
    if token is None and "token" in request.query_params:
        imbrace_token = request.query_params.get("token")
        # log.info(f"Token from query params: {imbrace_token}")
        imbrace_user = get_imbrace_user(imbrace_token)
        is_imbrace = True
        # log.info(f"imbrace user {imbrace_user}")
        current_user = Users.get_user_by_email(imbrace_user["email"])
        if current_user is None:
           is_existing_user = False
           log.info(f"User doesn't exist {imbrace_user['email']}")
           current_user = imbrace_user
          
        else:
            log.info(f"User already exists {imbrace_user['email']}")
            
    if is_imbrace is False and auth_token is not None:
        token = auth_token.credentials
    if is_imbrace is False and token is None and "token" in request.cookies:
        token = request.cookies.get("token")   
        
    if is_imbrace:
       if is_existing_user: 
          background_tasks.add_task(Users.update_user_last_active_by_id, current_user.id)
       return current_user 

    if token is None:
        raise HTTPException(status_code=403, detail="Not authenticated")
    


    # auth by api key
    if token.startswith("sk-"):
        if not request.state.enable_api_key:
            raise HTTPException(
                status.HTTP_403_FORBIDDEN, detail=ERROR_MESSAGES.API_KEY_NOT_ALLOWED
            )

        if request.app.state.config.ENABLE_API_KEY_ENDPOINT_RESTRICTIONS:
            allowed_paths = [
                path.strip()
                for path in str(
                    request.app.state.config.API_KEY_ALLOWED_ENDPOINTS
                ).split(",")
            ]

            # Check if the request path matches any allowed endpoint.
            if not any(
                request.url.path == allowed
                or request.url.path.startswith(allowed + "/")
                for allowed in allowed_paths
            ):
                raise HTTPException(
                    status.HTTP_403_FORBIDDEN, detail=ERROR_MESSAGES.API_KEY_NOT_ALLOWED
                )

        return get_current_user_by_api_key(token)

    # auth by jwt token
    try:
        data = decode_token(token)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )

    if data is not None and "id" in data:
        user = Users.get_user_by_id(data["id"])
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.INVALID_TOKEN,
            )
        else:
            # Refresh the user's last active timestamp asynchronously
            # to prevent blocking the request
            if background_tasks:
                background_tasks.add_task(Users.update_user_last_active_by_id, user.id)
        return user
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.UNAUTHORIZED,
        )


def get_current_user_by_api_key(api_key: str):
    user = Users.get_user_by_api_key(api_key)

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.INVALID_TOKEN,
        )
    else:
        Users.update_user_last_active_by_id(user.id)

    return user


def get_verified_user(user=Depends(get_current_user)):
    if user.role not in {"user", "admin"}:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )
    return user


def get_admin_user(user=Depends(get_current_user)):
    if user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )
    return user

def get_imbrace_user(token: str) -> dict:
    """
    Fetches user information from Imbrace API using the provided access token.
    
    Args:
        token: The Imbrace access token in format "acc_uuid"
        
    Returns:
        dict: User information if successful
        
    Raises:
        HTTPException: If the token is invalid or the request fails
    """
    try:
        log.info(f"Fetching user info from Imbrace API with token: {token}")
        headers = {
            'X-Access-Token': token
        }
        
        url = f"{IMBRACE_API_URL}/v1/account"
        log.info(f"Imbrace API URL: {url}")
        
        response = requests.get(
            url,
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            user_data = response.json()
            return user_data
        elif response.status_code == 401:
            log.error(f"Imbrace API authentication failed: {response.text}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid Imbrace access token"
            )
        else:
            log.error(f"Imbrace API request failed: Status {response.status_code}, Response: {response.text}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to fetch user information from Imbrace"
            )
    except requests.RequestException as e:
        log.exception(f"Request to Imbrace API failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Connection to Imbrace API failed"
        )
    except Exception as e:
        log.exception(f"Unexpected error in get_imbrace_user: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )


def auth_imbrace(request: Request):
    organization_id = request.headers.get("x-organization-id")
    api_key = request.headers.get("x-api-key")
    if not organization_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="x-organization-id is required",
        )
    userContext = {"organization_id": organization_id, "api_key": api_key}
    return userContext
