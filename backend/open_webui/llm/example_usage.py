"""
Example usage of the LLM Provider Pattern
Demonstrates how to use the factory pattern to create providers
"""

import asyncio
from typing import Dict, Any

from .agent import create_llm_agent
from .main import AnswerQuestionInputs


async def example_usage():
    """Example of how to use the LLM provider pattern"""
    
    # Example configuration (would typically come from settings/environment)
    config = {
        "mongoDB": {
            "openai_db_name": "rag_db"
        },
        "vectorConfig": {
            "collectionName": "rag_collection",
            "newCollectionName": "new_rag_collection"
        },
        "openApi": {
            "api_key": "your-openai-api-key",
            "api_type": "gpt-4"
        },
        "ollama": {
            "host": "http://localhost:11434",
            "model": "llama2",
            "retriever_model": "llama2",
            "embedding_model": "nomic-embed-text",
            "search_k_value": "10"
        },
        "lmstudio": {
            "host": "http://localhost:1234",
            "api_type": "gpt-3.5-turbo",
            "embedding_model": "text-embedding-nomic-embed-text-v1.5"
        },
        "embedding": {
            "chunkSize": "1000",
            "chunkOverlap": "200"
        },
        "environment": "development"
    }
    
    # Example inputs - streaming enabled
    streaming_inputs = AnswerQuestionInputs(
        question="What is artificial intelligence?",
        assistant_id="assistant-123",
        file_ids=["file-1", "file-2"],
        instruction="You are a helpful AI assistant specialized in technology.",
        thread_id="thread-456",
        created_at="2024-01-15T10:00:00Z",
        streaming=True
    )
    
    # Example inputs - streaming disabled
    non_streaming_inputs = AnswerQuestionInputs(
        question="What is machine learning?",
        assistant_id="assistant-123",
        file_ids=["file-1", "file-2"],
        instruction="You are a helpful AI assistant specialized in technology.",
        thread_id="thread-789",
        created_at="2024-01-15T10:00:00Z",
        streaming=False
    )
    
    print("=== OpenAI Provider Example (Streaming) ===")
    try:
        # Create OpenAI provider
        openai_agent = create_llm_agent("openai", config)
        
        # Use the agent with streaming
        async for chunk in openai_agent.answer_question(
            organization_id="org-123",
            model="gpt-4",
            inputs=streaming_inputs
        ):
            print(f"OpenAI Stream: {chunk.message}")
            if not chunk.is_partial:
                break
                
    except Exception as e:
        print(f"OpenAI Streaming Error: {e}")
    
    print("\n=== OpenAI Provider Example (Non-Streaming) ===")
    try:
        # Use the agent without streaming
        async for chunk in openai_agent.answer_question(
            organization_id="org-123",
            model="gpt-4",
            inputs=non_streaming_inputs
        ):
            print(f"OpenAI Complete: {chunk.message}")
            break  # Only one response for non-streaming
                
    except Exception as e:
        print(f"OpenAI Non-Streaming Error: {e}")
    
    print("\n=== Ollama Provider Example ===")
    try:
        # Create Ollama provider
        ollama_agent = create_llm_agent("ollama", config)
        
        # Use the agent
        async for chunk in ollama_agent.answer_question(
            organization_id="org-123",
            model="llama2",
            inputs=streaming_inputs
        ):
            print(f"Ollama: {chunk.message}")
            if not chunk.is_partial:
                break
                
    except Exception as e:
        print(f"Ollama Error: {e}")
    
    print("\n=== LMStudio Provider Example ===")
    try:
        # Create LMStudio provider
        lmstudio_agent = create_llm_agent("lmstudio", config)
        
        # Use the agent
        async for chunk in lmstudio_agent.answer_question(
            organization_id="org-123",
            model="gpt-3.5-turbo",
            inputs=streaming_inputs
        ):
            print(f"LMStudio: {chunk.message}")
            if not chunk.is_partial:
                break
                
    except Exception as e:
        print(f"LMStudio Error: {e}")


def demonstrate_provider_pattern():
    """
    Demonstrates the key benefits of the provider pattern:
    1. No if/else branching in main logic
    2. Easy to add new providers
    3. Clean separation of concerns
    4. Reusable shared logic
    """
    
    print("Provider Pattern Benefits:")
    print("1. ✅ Clean separation: Each provider only handles its specific logic")
    print("2. ✅ No if/else: Factory pattern handles provider selection")
    print("3. ✅ Extensible: Add new providers by implementing LLMProvider interface")
    print("4. ✅ DRY: Shared RAG logic in base class, no duplication")
    print("5. ✅ Maintainable: Provider-specific changes don't affect others")
    print("6. ✅ Configurable: Streaming can be enabled/disabled per request")
    print("\nRunning example usage...")


if __name__ == "__main__":
    demonstrate_provider_pattern()
    # Uncomment to run actual example (requires proper configuration)
    # asyncio.run(example_usage())
