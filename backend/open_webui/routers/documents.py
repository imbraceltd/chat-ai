import logging
import json
import os

from typing import Optional, Dict, Any
from open_webui.utils import document
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    status,
)
from pydantic import BaseModel

from open_webui.constants import ERROR_MESSAGES
from open_webui.env import SRC_LOG_LEVELS
from open_webui.utils.auth import auth_imbrace
from pydantic import Field
from typing import List
from open_webui.utils import databoard, fillers, misc
from langchain_openai import ChatOpenAI
from open_webui.config import OPENAI_API_KEY


log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS.get("DOCUMENT", logging.INFO))

router = APIRouter()

class DocumentAIRequest(BaseModel):
    modelName: str = Field(..., alias="modelName")
    url: str
    organizationId: str = Field(..., alias="organizationId")
    boardId: Optional[str] = Field(None, alias="boardId")
    language: Optional[str] = None
    additionalInstructions: Optional[str] = Field(None, alias="additionalInstructions")
    additionalDocumentInstructions: Optional[str] = Field(None, alias="additionalDocumentInstructions")
    processModelName: Optional[str] = Field(None, alias="processModelName")
    fileUrlToFill: Optional[str] = Field(None, alias="fileUrlToFill")
    tools: Optional[List[dict]] = None
    utc: Optional[int] = None
    # Custom provider support
    visionProviderId: Optional[str] = Field("system", alias="visionProviderId")
    processProviderId: Optional[str] = Field("system", alias="processProviderId")
    # Enhanced PDF processing options
    chunkSize: Optional[int] = Field(None, alias="chunkSize")
    maxConcurrent: Optional[int] = Field(None, alias="maxConcurrent")
    maxRetries: Optional[int] = Field(1, alias="maxRetries")
    maxBatchSize: Optional[int] = Field(None, alias="maxBatchSize")
    useEnhancedProcessing: Optional[bool] = Field(True, alias="useEnhancedProcessing")
    # vLLM-specific options for vision model (step 1)
    visionEnableThinking: Optional[bool] = Field(False, alias="visionEnableThinking")
    visionMaxTokens: Optional[int] = Field(None, alias="visionMaxTokens")
    # vLLM-specific options for process model (step 2)
    processEnableThinking: Optional[bool] = Field(False, alias="processEnableThinking")
    processMaxTokens: Optional[int] = Field(None, alias="processMaxTokens")
    # When True, extract raw data including borders/boxes from original files
    extractRawData: Optional[bool] = Field(False, alias="extractRawData")
    # Real-OCR confidence grounding: run an OCR engine alongside the vision LLM and
    # attach a grounded {value, box, confidence} to each extracted field.
    enableOcrConfidence: Optional[bool] = Field(False, alias="enableOcrConfidence")
    # Per-request OCR engine override; falls back to DOCUMENT_AI_OCR_ENGINE when omitted.
    ocrEngine: Optional[str] = Field(None, alias="ocrEngine")
    # Drop grounded fields whose confidence is below this threshold (0..1).
    confidenceThreshold: Optional[float] = Field(0.0, alias="confidenceThreshold")


def openai_llm(model_name: str) -> ChatOpenAI:
    return ChatOpenAI(
        api_key=OPENAI_API_KEY,  # API key will be set in the environment
        model=model_name,
        temperature=0,
        max_tokens=3000,
        request_timeout=60,
        
    )

async def resolve_model_from_provider(
    organization_id: str,
    provider_id: str,
    model_name: str,
):
    """
    Resolve a LangChain chat model from a provider.

    If provider_id is "system", uses the global CONFIG (same as agent.py).
    Otherwise, loads a custom provider from DB.

    Returns:
        tuple of (langchain_chat_model, detected_provider_type)
    """
    from open_webui.utils import provider as provider_service
    from open_webui.llm.agent import (
        CONFIG,
        PROVIDER,
        get_llm_provider,
        get_all_available_models,
        get_all_available_models_from_custom_provider,
    )
    from open_webui.llm.utils.models import detect_provider

    if model_name == "Default":
        from open_webui.config import LLM_CONFIG
        model_name = LLM_CONFIG.get("defaultModel", "gpt-4o") or "gpt-4o"
        log.info(f"Resolved 'Default' model to: {model_name}")

    if provider_id == "system":
        provider_config = CONFIG
        all_models = await get_all_available_models(config=CONFIG)
        default_provider = PROVIDER
    else:
        log.info("Resolved provider id")
        custom_provider = await provider_service.get_by_id(
            organization_id=organization_id,
            provider_id=provider_id,
        )
        if not custom_provider:
            raise ValueError(f"Provider '{provider_id}' not found")

        provider_config = custom_provider.get("config") or {}
        default_provider = custom_provider.get("type", "openai").lower()
        all_models = get_all_available_models_from_custom_provider(custom_provider)

    detected_provider = detect_provider(
        model_id=model_name,
        models=all_models,
        default_provider=default_provider,
    )

    llm_provider = get_llm_provider(detected_provider, provider_config)
    model = llm_provider.create_llm(
        model=model_name,
        available_models=all_models,
        streaming=False,
        temperature=0,
    )

    return model, detected_provider


@router.post("")
async def document_ai_assistant(request: DocumentAIRequest, userContext=Depends(auth_imbrace)):
    try:
        from open_webui.utils.model_providers import (
            ModelService,
            openai_llm,
            ollama_llm,
            google_llm,
            lmstudio_llm,
            kimi_vision_model,
            olm_vision_model,
            google_vision_model,
        )
        from open_webui.utils.document_assistant import (
            vision_assistant_v2,
            vision_assistant_v3,
            kimi_assistant_v2,
            kimi_assistant_v3,
            gemini_assistant_v3,
        )
        from open_webui.config import OPENAI_API_KEY
        
        # Validate required fields
        if not request.modelName or not request.url or not request.organizationId:
            raise HTTPException(
                status_code=400,
                detail={"success": False, "message": "Missing necessary info."}
            )
        
        # Get output schema if boardId is provided
        output_schema = None
        if request.boardId:
            board_info = await databoard.get_board_schema(request.organizationId, request.boardId)
            if not board_info:
                raise HTTPException(
                    status_code=404,
                    detail={"success": False, "message": "Cannot get boardInfo."}
                )
            log.info(f"Board Info: {board_info}")
            
            output_schema = await databoard.get_board_as_function_call_schema(board_info=board_info, organization_id=request.organizationId)
            if not output_schema:
                raise HTTPException(
                    status_code=404,
                    detail={"success": False, "message": "Cannot get outputSchema."}
                )
            print(f"Output Schema: {json.dumps(output_schema)}")
        
        # Determine model providers
        process_data_model = None
        vision_model = None
        is_kimi = False
        is_olm = False
        is_gemini = False
        vision_provider = "openai"  # Default
        process_provider = "openai"  # Default, tracked separately for structured output

        # Load processing model
        if request.processModelName:
            if request.processProviderId:
                try:
                    process_data_model, detected_type = await resolve_model_from_provider(
                        organization_id=request.organizationId,
                        provider_id=request.processProviderId,
                        model_name=request.processModelName,
                    )
                    process_provider = detected_type
                    if detected_type == "google":
                        is_gemini = True
                        # Use ChatGoogleGenerativeAI for document AI instead of GoogleRESTChatModel
                        provider_api_key = getattr(process_data_model, 'api_key', None)
                        process_data_model = google_llm(request.processModelName, api_key=provider_api_key)
                    log.info(f"Resolved process model from custom provider '{request.processProviderId}': {request.processModelName} (type: {detected_type})")
                except Exception as e:
                    log.error(f"Failed to resolve custom process provider '{request.processProviderId}': {e}")
                    raise HTTPException(
                        status_code=400,
                        detail={"success": False, "message": f"Failed to resolve custom process provider: {e}"}
                    )
            else:
                model_list = await ModelService.get_models()
                model = next((m for m in model_list if m.get("name") == request.processModelName), None)
                print(f"Model found: {model}")

                if model:
                    provider = model.get("provider", "openai")
                    process_provider = provider
                    print(f"Provider: {provider}")

                    if provider == "openai":
                        process_data_model = openai_llm(model["name"])
                    elif provider == "ollama":
                        process_data_model = ollama_llm(model["name"])
                    elif provider == "google":
                        process_data_model = google_llm(model["name"])
                        is_gemini = True
                    elif provider == "lmstudio":
                        process_data_model = lmstudio_llm(model["name"])

        # Determine vision model based on modelName
        if request.visionProviderId:
            # Resolve vision model from custom provider (returns LangChain BaseChatModel)
            try:
                vision_model, vision_provider_type = await resolve_model_from_provider(
                    organization_id=request.organizationId,
                    provider_id=request.visionProviderId,
                    model_name=request.modelName,
                )
                vision_provider = vision_provider_type
                if vision_provider_type == "google":
                    is_gemini = True
                    # Use ChatGoogleGenerativeAI for document AI instead of GoogleRESTChatModel
                    # GoogleRESTChatModel uses REST v1 which doesn't support systemInstruction
                    provider_api_key = getattr(vision_model, 'api_key', None)
                    vision_model = google_vision_model(request.modelName, api_key=provider_api_key)
                log.info(f"Resolved vision model from custom provider '{request.visionProviderId}': {request.modelName} (type: {vision_provider_type})")
            except Exception as e:
                log.error(f"Failed to resolve custom vision provider '{request.visionProviderId}': {e}")
                raise HTTPException(
                    status_code=400,
                    detail={"success": False, "message": f"Failed to resolve custom vision provider: {e}"}
                )
        elif request.modelName == "Kimi-VL-A3B-Thinking-2506":
            print("Using Kimi-VL-A3B-Thinking-2506 model")
            is_kimi = True
            vision_provider = "kimi"
            vision_model = kimi_vision_model()
        elif request.modelName in ["olmOCR-2-7B-1025-FP8", "llama3.2-vision"] or "llama" in request.modelName.lower():
            vision_provider = "kimi"  # Use kimi provider for OLM/Ollama compatibility
            is_olm = True
            print(f"Using OLM/Ollama Vision Model: {request.modelName}")
            vision_model = olm_vision_model()
        elif "gemini" in request.modelName.lower():
            # Use Google Gemini vision
            is_gemini = True
            vision_provider = "google"
            vision_model = google_vision_model(request.modelName)
        else:
            # Default to Google vision for other models
            vision_model = google_vision_model(request.modelName)
            if "gemini" in request.modelName.lower():
                is_gemini = True
                vision_provider = "google"

        # Set default process model if not specified
        if not process_data_model:
            process_data_model = openai_llm("gpt-4o-mini")
            process_provider = "openai"

        log.info(f"Vision provider: {vision_provider}, Process provider: {process_provider}")
        
        # Configure processing options
        customer_chunk_size = 3
        customer_max_concurrent = 3
        
        if is_olm:
            customer_chunk_size = 1
            customer_max_concurrent = 1
        
        # Force useEnhancedProcessing when extractRawData is enabled
        use_enhanced = request.useEnhancedProcessing or (request.extractRawData or False)

        processing_options = {
            "chunkSize": request.chunkSize or (customer_chunk_size if (is_kimi or is_olm) else 1),
            "maxConcurrent": request.maxConcurrent or (customer_max_concurrent if (is_kimi or is_olm) else 2),
            "maxRetries": request.maxRetries or 1,
            "modelProvider": vision_provider,
            "processProvider": process_provider,
            "useEnhancedProcessing": use_enhanced,
            "visionEnableThinking": request.visionEnableThinking or False,
            "visionMaxTokens": request.visionMaxTokens,
            "processEnableThinking": request.processEnableThinking or False,
            "processMaxTokens": request.processMaxTokens,
            "extractRawData": request.extractRawData or False,
            "maxBatchSize": request.maxBatchSize or 5,
            "enableOcrConfidence": request.enableOcrConfidence or False,
            "ocrEngine": request.ocrEngine,
            "confidenceThreshold": request.confidenceThreshold or 0.0,
        }

        print(f"Processing options: {processing_options}")

        # Execute document processing
        data = None
        try:
            if use_enhanced:
                # Use V3 enhanced processing
                if is_kimi or is_olm:
                    data = await kimi_assistant_v3(
                        modelName=request.modelName,
                        model=process_data_model,
                        visionModel=vision_model,
                        url=request.url,
                        language=request.language,
                        additional_instructions=request.additionalInstructions,
                        additional_document_instructions=request.additionalDocumentInstructions,
                        output_schema=output_schema,
                        options=processing_options,
                    )
                elif is_gemini:
                    data = await gemini_assistant_v3(
                        model=process_data_model,
                        visionModel=vision_model,
                        url=request.url,
                        language=request.language,
                        additional_instructions=request.additionalInstructions,
                        additional_document_instructions=request.additionalDocumentInstructions,
                        output_schema=output_schema,
                        options=processing_options,
                    )
                else:
                    data = await vision_assistant_v3(
                        model=process_data_model,
                        visionModel=vision_model,
                        url=request.url,
                        language=request.language,
                        additional_instructions=request.additionalInstructions,
                        additional_document_instructions=request.additionalDocumentInstructions,
                        output_schema=output_schema,
                        options=processing_options,
                    )
            else:
                # Use V2 original processing for backward compatibility
                if is_kimi or is_olm:
                    data = await kimi_assistant_v2(
                        modelName=request.modelName,
                        model=process_data_model,
                        visionModel=vision_model,
                        url=request.url,
                        language=request.language,
                        additional_instructions=request.additionalInstructions,
                        additional_document_instructions=request.additionalDocumentInstructions,
                        output_schema=output_schema,
                        model_provider=process_provider,
                        process_enable_thinking=request.processEnableThinking or False,
                        process_max_tokens=request.processMaxTokens,
                        extract_raw_data=request.extractRawData or False,
                        enable_ocr_confidence=request.enableOcrConfidence or False,
                        ocr_engine=request.ocrEngine,
                        confidence_threshold=request.confidenceThreshold or 0.0,
                    )
                else:
                    data = await vision_assistant_v2(
                        model=process_data_model,
                        visionModel=vision_model,
                        url=request.url,
                        language=request.language,
                        additional_instructions=request.additionalInstructions,
                        additional_document_instructions=request.additionalDocumentInstructions,
                        output_schema=output_schema,
                        model_provider=process_provider,
                        process_enable_thinking=request.processEnableThinking or False,
                        process_max_tokens=request.processMaxTokens,
                        extract_raw_data=request.extractRawData or False,
                        vision_provider=vision_provider,
                        enable_ocr_confidence=request.enableOcrConfidence or False,
                        ocr_engine=request.ocrEngine,
                        confidence_threshold=request.confidenceThreshold or 0.0,
                    )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail={"success": False, "message": f"Error from document-ai assistant: {e}"}
            )
        
        if not data:
            raise HTTPException(
                status_code=500,
                detail={"success": False, "message": "No response from LLM"}
            )
        
        # Fix timezone if needed
        data = misc.fix_time_zone(data=data, utc=request.utc)
        
        # Optional action: fill data to file
        if request.fileUrlToFill:
            file_extension = os.path.splitext(request.fileUrlToFill)[1].lower().strip('.')
            filler_map = {
                "pdf": fillers.pdf_filler,
                "xlsx": fillers.excel_filler,
                "xls": fillers.excel_filler,
            }
            try:
                filler_function = filler_map.get(file_extension)
                if filler_function:
                    data["filledPdfUrl"] = await filler_function(
                        data=data,
                        fileUrlToFill=request.fileUrlToFill,
                        organizationId=request.organizationId,
                    )
            except Exception as filler_error:
                print(f"Unsupported file type for filling or error: {file_extension}, {filler_error}")

        # When OCR confidence grounding ran, surface a document-level aggregate
        # alongside the (in-place annotated) data without altering its shape.
        response = {"success": True, "data": data}
        if request.enableOcrConfidence or request.extractRawData:
            try:
                from open_webui.utils.ocr import aggregate_confidence

                response["confidence"] = aggregate_confidence(data)
            except Exception as agg_error:
                log.warning(f"Failed to compute document-level confidence: {agg_error}")

        return response
    
    except HTTPException as e:
        # Re-raise HTTPExceptions to let FastAPI handle the response
        raise e
    except Exception as e:
        # Catch any other unexpected errors and return a generic 500 error
        print(f"An unexpected server error occurred: {e}")
        logging.exception("Unexpected error in document_ai_assistant")
        raise HTTPException(
            status_code=500,
            detail={"success": False, "message": f"Something went wrong: {str(e)}"}
        )



@router.get("/models")
async def get_models():
    try:
        from open_webui.utils.model_providers import ModelService
        models = await ModelService.get_models()
        return models
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(str(e))
        )
