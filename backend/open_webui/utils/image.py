# open_webui/utils/image.py

from typing import Dict, Optional, List
import logging
import uuid
import base64
import requests
from open_webui.config import OPENAI_API_KEY, GOOGLE_VISION_API_KEY
from open_webui.env import SRC_LOG_LEVELS
from ..repository.image import ImageRepository
from ..repository.api_key import ApiKeyRepository

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["MODELS"])

image_repo = ImageRepository()
api_key_repo = ApiKeyRepository()

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    log.warning("Langchain packages not available. Image summarization will be disabled.")


async def get_image_by_id(organization_id: str, image_id: str) -> Optional[Dict]:
    """
    Get an image by its ID and organization ID.
    """
    try:
        image = await image_repo.get_by_image_id(
            organization_id, image_id
        )
        return image

    except Exception as error:
        log.error(f"Error getting image by id: {error}")
        raise error


async def summarize_image(image: Dict, api_key: str) -> Dict:
    """
    Generate a summary for an image using OpenAI's vision model.
    
    Args:
        image: Dictionary containing image data (url, buffer, mimetype, etc.)
        api_key: OpenAI API key
        
    Returns:
        Dictionary with success status, image_id, and summary
    """
    try:
        if not LANGCHAIN_AVAILABLE:
            log.warning("Langchain not available, returning empty summary")
            return {
                "success": False,
                "image_id": image.get("image_id"),
                "summary": "Summarization not available",
            }

        image_url = image.get("url")
        
        # If no URL provided, create data URL from buffer
        if not image_url and image.get("buffer"):
            buffer = image.get("buffer")
            mimetype = image.get("mimetype", "image/png")
            
            # Convert bytes to base64 if needed
            if isinstance(buffer, bytes):
                b64_data = base64.b64encode(buffer).decode("utf-8")
            else:
                b64_data = buffer
                
            image_url = f"data:{mimetype};base64,{b64_data}"
        
        if not image_url:
            raise ValueError("No image URL or buffer provided")

        # Build the vision LLM per LLM_DEFAULT_PROVIDER (override with
        # IMAGE_SUMMARY_PROVIDER / IMAGE_SUMMARY_MODEL). Point it at a
        # vision-capable model on the target provider.
        from open_webui.llm.utils.task_llm import build_task_llm

        model = build_task_llm(
            "IMAGE_SUMMARY",
            default_openai_model="gpt-4o-mini",
            temperature=0.2,
            api_key=api_key,
            max_retries=5,
            timeout=30,
        )

        # Create messages
        messages = [
            SystemMessage(
                content="You are a helpful assistant that can analyze images and provide detailed summaries. Focus on the main elements, context, and any notable details in the image."
            ),
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": "Please provide a detailed summary of this image with no more than 50 words.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                ]
            ),
        ]

        # Call the model
        response = await model.ainvoke(messages)

        return {
            "success": True,
            "image_id": image.get("image_id"),
            "summary": response.content,
        }

    except Exception as error:
        log.error(f"Error summarizing image {image.get('image_id')}: {error}")
        return {
            "success": False,
            "image_id": image.get("image_id"),
            "summary": "",
        }


async def get_image_labels(images: List[Dict]) -> List[Dict]:
    """
    Get classification labels for images using Google Vision API.
    
    Matches JavaScript implementation:
    - Uses Google Vision API LABEL_DETECTION
    - Supports both URL and buffer-based images
    - Returns labels with description and confidence score
    - Handles errors per image
    
    Args:
        images: List of image dictionaries with url/buffer, filename, image_id
        
    Returns:
        List of label dictionaries with filename, image_id, success, labels, error
    """
    try:
        api_key = GOOGLE_VISION_API_KEY
        
        if not api_key:
            log.warning("GOOGLE_VISION_API_KEY not set, returning mock labels")
            # Return mock labels for testing
            return [
                {
                    "filename": image.get("filename"),
                    "image_id": image.get("image_id"),
                    "success": True,
                    "labels": [
                        {"description": "object", "score": 85.0},
                        {"description": "scene", "score": 75.0},
                    ],
                }
                for image in images
            ]
        
        # Prepare the requests for each image (matches JS implementation)
        requests_data = []
        for image in images:
            if image.get("url"):
                # URL-based image
                request_item = {
                    "image": {"source": {"imageUri": image.get("url")}},
                    "features": [{"maxResults": 5, "type": "LABEL_DETECTION"}],
                }
            else:
                # Buffer-based image - convert to base64
                buffer = image.get("buffer")
                if isinstance(buffer, bytes):
                    b64_data = base64.b64encode(buffer).decode("utf-8")
                else:
                    b64_data = buffer
                    
                request_item = {
                    "image": {"content": b64_data},
                    "features": [{"maxResults": 5, "type": "LABEL_DETECTION"}],
                }
            
            requests_data.append(request_item)
        
        # Call Google Vision API
        response = requests.post(
            f"https://vision.googleapis.com/v1/images:annotate?key={api_key}",
            json={"requests": requests_data},
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        
        response.raise_for_status()
        response_data = response.json()
        
        # Process results (matches JS implementation)
        results = []
        for index, result in enumerate(response_data.get("responses", [])):
            image = images[index]
            
            # Check for errors
            if result.get("error"):
                results.append({
                    "filename": image.get("filename"),
                    "image_id": image.get("image_id"),
                    "success": False,
                    "error": result["error"].get("message", "Unknown error"),
                })
                continue
            
            # Check for labels
            if not result.get("labelAnnotations"):
                results.append({
                    "filename": image.get("filename"),
                    "image_id": image.get("image_id"),
                    "success": False,
                    "error": "No labels found",
                    "labels": [],
                })
                continue
            
            # Extract labels and scores (round to 2 decimals like JS)
            labels = [
                {
                    "description": label.get("description"),
                    "score": round(label.get("score", 0) * 100, 2),
                }
                for label in result.get("labelAnnotations", [])
            ]
            
            results.append({
                "filename": image.get("filename"),
                "image_id": image.get("image_id"),
                "success": True,
                "labels": labels,
            })
        
        return results
        
    except requests.exceptions.RequestException as error:
        log.error(f"Google Vision API error: {error}", exc_info=True)
        # Return error for all images
        return [
            {
                "filename": image.get("filename"),
                "image_id": image.get("image_id"),
                "success": False,
                "error": str(error),
                "labels": [],
            }
            for image in images
        ]
    except Exception as error:
        log.error(f"Error getting image labels: {error}", exc_info=True)
        raise


async def summary_images(images: List[Dict], api_key: str = None) -> List[Dict]:
    """
    Generate summaries for images using OpenAI vision model.
    
    Matches JavaScript implementation:
    - Processes each image individually to get separate summaries
    - Uses Promise.all pattern (parallel processing)
    - Returns array of summaries
    
    Args:
        images: List of image dictionaries with image_id, buffer, mimetype, filename
        api_key: OpenAI API key (optional, falls back to OPENAI_API_KEY)
        
    Returns:
        List of summary dictionaries with success, image_id, and summary
    """
    try:
        if not api_key:
            api_key = OPENAI_API_KEY
            
        if not api_key:
            log.error("No OPENAI_API_KEY provided for image summarization")
            raise ValueError("OpenAI API key is required for image summarization")

        # Process each image individually to get separate summaries (parallel)
        import asyncio
        image_summaries = await asyncio.gather(
            *[summarize_image(image, api_key) for image in images],
            return_exceptions=False
        )

        return image_summaries

    except Exception as error:
        log.error(f"Error summarizing images: {error}", exc_info=True)
        raise


async def label_images(images: List[Dict], organization_id: str = None) -> List[Dict]:
    """
    Label and classify images using AI models.
    
    Matches JavaScript implementation:
    1. Gets classification labels for images
    2. Generates summaries using OpenAI vision model (parallel processing)
    3. Combines labels and summaries by index
    4. Saves to database if organization_id is provided
    
    Args:
        images: List of image dictionaries with image_id, url/buffer, mimetype, filename
        organization_id: Organization ID for caching (optional, None disables caching)
        
    Returns:
        List of labeled images with summaries
    """
    try:
        # Step 1: Get image classification labels
        image_labels = await get_image_labels(images)
        
        if not image_labels or len(image_labels) == 0:
            raise ValueError("Error labelling images")

        # Step 2: Get summaries for each image using OpenAI (parallel processing)
        api_key = OPENAI_API_KEY
        if not api_key:
            log.warning("No OPENAI_API_KEY provided, summaries will be empty")
            image_summaries = [
                {"image_id": img.get("image_id"), "summary": "", "success": False}
                for img in images
            ]
        else:
            # Process all images in parallel like JS Promise.all
            import asyncio
            image_summaries = await asyncio.gather(
                *[summarize_image(img, api_key) for img in images],
                return_exceptions=False
            )

        # Step 3: Combine labels and summaries by index (matches JS implementation)
        labeled_images = [
            {
                **label,
                "summary": image_summaries[index].get("summary", ""),
            }
            for index, label in enumerate(image_labels)
        ]

        # Step 4: Save to database if organization_id is provided
        if organization_id:
            # Save all images in parallel like JS Promise.all
            import asyncio
            await asyncio.gather(
                *[
                    image_repo.create(
                        organization_id,
                        {
                            "image_id": image.get("image_id"),
                            "url": image.get("url"),
                            "filename": image.get("filename"),
                            "mimetype": image.get("mimetype"),
                        },
                    )
                    for image in images
                    if image.get("url") or image.get("image_id")
                ],
                return_exceptions=True  # Don't fail if some saves fail
            )

        return labeled_images

    except Exception as error:
        log.error(f"Error labelling images: {error}", exc_info=True)
        raise


async def create_image(
    organization_id: str = "",
    api_key: str = "",
    image_details: Dict = None
) -> Optional[Dict]:
    """
    Create a new image record.

    Args:
        organization_id: The organization ID
        api_key: The API key for authentication  
        image_details: Dictionary containing image details

    Returns:
        Dict containing created image data

    Raises:
        Exception: If there's an error creating the image
    """
    try:
        if not api_key:
            api_key = OPENAI_API_KEY

        # Validate API key
        api_key = await api_key_repo.check_key_is_paid(organization_id, api_key)

        # Create image
        created_image = await image_repo.create(
            organization_id=organization_id,
            image_details=image_details
        )

        if created_image:
            # Convert ObjectId to string if present
            if "_id" in created_image:
                created_image["_id"] = str(created_image["_id"])

            log.info(
                f"Successfully created image {created_image.get('image_id')}")

        return created_image

    except Exception as error:
        log.error(f"Error creating image: {error}", exc_info=True)
        raise error


async def update_image(
    organization_id: str = "",
    api_key: str = "",
    image_id: str = "",
    image_details: Dict = None
) -> Optional[Dict]:
    """
    Update an existing image record.

    Args:
        organization_id: The organization ID
        api_key: The API key for authentication
        image_id: The image ID to update
        image_details: Dictionary containing updated image details

    Returns:
        Dict containing updated image data

    Raises:
        Exception: If there's an error updating the image
    """
    try:
        if not api_key:
            api_key = OPENAI_API_KEY

        # Validate API key
        api_key = await api_key_repo.check_key_is_paid(organization_id, api_key)

        # Update image
        updated_image = await image_repo.update(
            image_id=image_id,
            image_details=image_details
        )

        if updated_image:
            # Convert ObjectId to string if present
            if "_id" in updated_image:
                updated_image["_id"] = str(updated_image["_id"])

            log.info(f"Successfully updated image {image_id}")

        return updated_image

    except Exception as error:
        log.error(f"Error updating image {image_id}: {error}", exc_info=True)
        raise error


async def remove_image(
    organization_id: str = "",
    api_key: str = "",
    image_id: str = ""
) -> bool:
    """
    Remove an image record (soft delete).

    Args:
        organization_id: The organization ID
        api_key: The API key for authentication
        image_id: The image ID to remove

    Returns:
        bool indicating if removal was successful

    Raises:
        Exception: If there's an error removing the image
    """
    try:
        if not api_key:
            api_key = OPENAI_API_KEY

        # Validate API key
        api_key = await api_key_repo.check_key_is_paid(organization_id, api_key)

        # Remove image
        removed = await image_repo.remove(image_id=image_id)

        if removed:
            log.info(f"Successfully removed image {image_id}")

        return removed

    except Exception as error:
        log.error(f"Error removing image {image_id}: {error}", exc_info=True)
        raise error
