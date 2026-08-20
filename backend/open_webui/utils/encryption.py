"""
Encryption utilities using Fernet symmetric encryption.

This module provides functions to encrypt and decrypt data using the Fernet
encryption scheme from the cryptography library. It derives a proper Fernet
key from ENCRYPTION_SECRET_KEY using PBKDF2.
"""

import base64
import json
import logging
from typing import Any, Dict

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from open_webui import env as webui_env

log = logging.getLogger(__name__)

# Salt for key derivation - this should be consistent across deployments
# Using a fixed salt derived from the purpose of encryption
ENCRYPTION_SALT = b"provider_config_encryption_v1"


def _get_fernet_key() -> bytes:
    """
    Derive a valid Fernet key from ENCRYPTION_SECRET_KEY using PBKDF2.
    
    Fernet requires a 32-byte URL-safe base64-encoded key.
    PBKDF2 with SHA256 will derive a proper 32-byte key.
    
    Returns:
        bytes: A valid Fernet key
    """
    # Prefer dedicated ENCRYPTION_SECRET_KEY; fall back to empty string if missing.
    raw_secret = getattr(webui_env, "ENCRYPTION_SECRET_KEY", "") or ""

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=ENCRYPTION_SALT,
        iterations=100_000,
    )
    key = kdf.derive(raw_secret.encode("utf-8"))
    return base64.urlsafe_b64encode(key)


def _get_fernet() -> Fernet:
    """
    Get a Fernet instance with the derived key.
    
    Returns:
        Fernet: A Fernet encryption instance
    """
    return Fernet(_get_fernet_key())


def encrypt_dict(data: Dict[str, Any]) -> str:
    """
    Encrypt a dictionary to an encrypted string.
    
    Args:
        data: Dictionary to encrypt
        
    Returns:
        str: Base64-encoded encrypted string
    """
    if not data:
        return ""
    
    try:
        fernet = _get_fernet()
        json_str = json.dumps(data, ensure_ascii=False, default=str)
        encrypted = fernet.encrypt(json_str.encode("utf-8"))
        return encrypted.decode("utf-8")
    except Exception as e:
        log.error(f"Failed to encrypt data: {e}")
        raise


def decrypt_dict(encrypted_data: str) -> Dict[str, Any]:
    """
    Decrypt an encrypted string back to a dictionary.
    
    Args:
        encrypted_data: Base64-encoded encrypted string
        
    Returns:
        Dict[str, Any]: Decrypted dictionary
    """
    if not encrypted_data:
        return {}
    
    try:
        fernet = _get_fernet()
        decrypted = fernet.decrypt(encrypted_data.encode("utf-8"))
        return json.loads(decrypted.decode("utf-8"))
    except InvalidToken:
        log.error("Failed to decrypt data: invalid token or corrupted data")
        raise ValueError("Failed to decrypt: invalid token or data corrupted")
    except json.JSONDecodeError as e:
        log.error(f"Failed to parse decrypted JSON: {e}")
        raise ValueError(f"Failed to parse decrypted data as JSON: {e}")
    except Exception as e:
        log.error(f"Failed to decrypt data: {e}")
        raise


def encrypt_string(data: str) -> str:
    """
    Encrypt a string.
    
    Args:
        data: String to encrypt
        
    Returns:
        str: Base64-encoded encrypted string
    """
    if not data:
        return ""
    
    try:
        fernet = _get_fernet()
        encrypted = fernet.encrypt(data.encode("utf-8"))
        return encrypted.decode("utf-8")
    except Exception as e:
        log.error(f"Failed to encrypt string: {e}")
        raise


def decrypt_string(encrypted_data: str) -> str:
    """
    Decrypt an encrypted string.
    
    Args:
        encrypted_data: Base64-encoded encrypted string
        
    Returns:
        str: Decrypted string
    """
    if not encrypted_data:
        return ""
    
    try:
        fernet = _get_fernet()
        decrypted = fernet.decrypt(encrypted_data.encode("utf-8"))
        return decrypted.decode("utf-8")
    except InvalidToken:
        log.error("Failed to decrypt string: invalid token or corrupted data")
        raise ValueError("Failed to decrypt: invalid token or data corrupted")
    except Exception as e:
        log.error(f"Failed to decrypt string: {e}")
        raise


def is_encrypted(data: str) -> bool:
    """
    Check if a string appears to be Fernet-encrypted.
    
    Fernet tokens start with 'gAAAAA' when base64-encoded.
    
    Args:
        data: String to check
        
    Returns:
        bool: True if the string appears to be encrypted
    """
    if not data or not isinstance(data, str):
        return False
    
    # Fernet tokens are base64-encoded and have a specific prefix
    # They start with 'gAAAAA' after base64 encoding
    return data.startswith("gAAAAA")

