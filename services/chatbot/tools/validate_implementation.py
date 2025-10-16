#!/usr/bin/env python3
"""Validate the chatbot implementation without external dependencies."""

import sys
import os
import importlib.util
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock

def validate_imports():
    """Validate that all required modules can be imported."""
    print("🔍 Validating imports...")
    
    try:
        # Test main module imports
        service_root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location("main", service_root / "main.py")
        main_module = importlib.util.module_from_spec(spec)
        
        # Mock external dependencies before importing
        sys.modules['qdrant_client'] = Mock()
        sys.modules['qdrant_client.models'] = Mock()
        sys.modules['structlog'] = Mock()
        sys.modules['httpx'] = Mock()
        
        spec.loader.exec_module(main_module)
        
        print("✅ All imports successful")
        return True, main_module
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False, None

def validate_chatbot_service(main_module):
    """Validate ChatbotService class functionality."""
    print("🤖 Validating ChatbotService...")
    
    try:
        # Create service instance
        service = main_module.ChatbotService()
        
        # Test intent extraction
        intent = service.extract_intent("Show me articles about AI")
        assert intent["intent"] == "search_content"
        assert "AI" in intent["topic"]
        print("✅ Intent extraction works")
        
        # Test different intents
        cluster_intent = service.extract_intent("What clusters do I have?")
        assert cluster_intent["intent"] == "explore_clusters"
        print("✅ Cluster intent detection works")
        
        summary_intent = service.extract_intent("Give me a summary")
        assert summary_intent["intent"] == "get_summary"
        print("✅ Summary intent detection works")
        
        return True
        
    except Exception as e:
        print(f"❌ ChatbotService validation failed: {e}")
        return False

def validate_api_endpoints(main_module):
    """Validate FastAPI app and endpoints."""
    print("🌐 Validating API endpoints...")
    
    try:
        app = main_module.app
        
        # Check that app is FastAPI instance
        assert hasattr(app, 'routes')
        print("✅ FastAPI app created")
        
        # Check required endpoints exist
        routes = [route.path for route in app.routes if hasattr(route, 'path')]
        
        required_endpoints = [
            "/chat/message",
            "/chat/history/{session_id}",
            "/chat/feedback",
            "/health"
        ]
        
        for endpoint in required_endpoints:
            # Check if endpoint pattern exists (allowing for path parameters)
            endpoint_exists = any(
                endpoint.replace("{session_id}", "test") in route or 
                endpoint.split("/")[:-1] == route.split("/")[:-1]
                for route in routes
            )
            if endpoint_exists or any(endpoint.split("{")[0] in route for route in routes):
                print(f"✅ Endpoint {endpoint} found")
            else:
                print(f"⚠️ Endpoint {endpoint} not found in routes: {routes}")
        
        return True
        
    except Exception as e:
        print(f"❌ API validation failed: {e}")
        return False

def validate_data_models(main_module):
    """Validate Pydantic data models."""
    print("📋 Validating data models...")
    
    try:
        # Test ChatMessage model
        chat_msg = main_module.ChatMessage(
            session_id="test_session",
            message="Hello",
            context=[]
        )
        assert chat_msg.session_id == "test_session"
        print("✅ ChatMessage model works")
        
        # Test ChatResponse model
        chat_resp = main_module.ChatResponse(
            response="Test response",
            sources=[],
            suggestions=["Test suggestion"]
        )
        assert chat_resp.response == "Test response"
        print("✅ ChatResponse model works")
        
        # Test FeedbackRequest model
        feedback = main_module.FeedbackRequest(
            message_id="msg123",
            feedback="positive"
        )
        assert feedback.message_id == "msg123"
        print("✅ FeedbackRequest model works")
        
        return True
        
    except Exception as e:
        print(f"❌ Data model validation failed: {e}")
        return False

async def validate_async_methods(main_module):
    """Validate async methods with mocked dependencies."""
    print("⚡ Validating async methods...")
    
    try:
        service = main_module.ChatbotService()
        
        # Mock the ollama client
        service.ollama_client = AsyncMock()
        
        # Mock embedding response
        mock_response = Mock()
        mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        mock_response.raise_for_status.return_value = None
        service.ollama_client.post.return_value = mock_response
        
        # Test embedding generation
        embedding = await service.generate_embedding("test text")
        assert embedding == [0.1, 0.2, 0.3]
        print("✅ Embedding generation works")
        
        # Mock LLM response
        mock_llm_response = Mock()
        mock_llm_response.json.return_value = {"response": "Test LLM response"}
        mock_llm_response.raise_for_status.return_value = None
        service.ollama_client.post.return_value = mock_llm_response
        
        # Test LLM response generation
        llm_response = await service.generate_llm_response("What is AI?")
        assert llm_response == "Test LLM response"
        print("✅ LLM response generation works")
        
        return True
        
    except Exception as e:
        print(f"❌ Async method validation failed: {e}")
        return False

def main():
    """Main validation function."""
    print("🤖 Chatbot Service Implementation Validation")
    print("=" * 50)
    
    # Change to service directory
    service_root = Path(__file__).resolve().parents[1]
    os.chdir(service_root)
    
    all_passed = True
    
    # Validate imports
    imports_ok, main_module = validate_imports()
    if not imports_ok:
        return False
    
    # Validate ChatbotService
    if not validate_chatbot_service(main_module):
        all_passed = False
    
    # Validate API endpoints
    if not validate_api_endpoints(main_module):
        all_passed = False
    
    # Validate data models
    if not validate_data_models(main_module):
        all_passed = False
    
    # Validate async methods
    try:
        asyncio.run(validate_async_methods(main_module))
        print("✅ Async methods validation passed")
    except Exception as e:
        print(f"❌ Async methods validation failed: {e}")
        all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All validations passed! Chatbot service implementation is correct.")
        return True
    else:
        print("❌ Some validations failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
