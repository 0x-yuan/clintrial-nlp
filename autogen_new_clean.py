#!/usr/bin/env python3
"""
Clean version of AutoGen framework for testing with new API.
"""

import os
import time
import asyncio

try:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.messages import TextMessage
    from autogen_agentchat.ui import Console
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    AUTOGEN_AVAILABLE = True
    print("✅ New AutoGen API imported successfully")
except ImportError as e:
    print(f"❌ New AutoGen API import failed: {e}")
    AUTOGEN_AVAILABLE = False

async def test_new_autogen():
    """Test new AutoGen API"""
    if not AUTOGEN_AVAILABLE:
        return "New AutoGen API not available"
    
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return "No API key found"
    
    try:
        # Note: The new API might not directly support Gemini yet
        # Let's try with OpenAI API for now to test the structure
        print("⚠️ Testing with OpenAI API structure (Gemini support may vary)")
        
        # This is a placeholder - actual Gemini integration might be different
        model_client = OpenAIChatCompletionClient(
            model="gpt-3.5-turbo",  # This would need to be Gemini
            api_key="placeholder"   # This would need to be your Gemini key
        )
        
        # Create assistant
        assistant = AssistantAgent(
            name="test_assistant",
            model_client=model_client,
            system_message="You are a helpful assistant."
        )
        
        # Test conversation
        start_time = time.time()
        
        # Send a message
        result = await assistant.on_messages(
            [TextMessage(content="Hello, respond with 'New AutoGen test successful'", source="user")],
            cancellation_token=None
        )
        
        duration = time.time() - start_time
        
        return f"New AutoGen test completed in {duration:.2f}s"
        
    except Exception as e:
        return f"New AutoGen test failed: {e}"

def test_autogen_fallback():
    """Test if we can access old autogen through ag2"""
    try:
        import ag2
        print("✅ ag2 available")
        from ag2 import AssistantAgent, UserProxyAgent
        print("✅ ag2 agents imported")
        
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return "No API key found"
        
        # Try the old configuration style
        config_list = [
            {
                "model": "gemini-2.5-flash",
                "api_key": api_key,
                "api_type": "google"
            }
        ]
        
        llm_config = {
            "config_list": config_list,
            "temperature": 0.1,
            "timeout": 60,
        }
        
        assistant = AssistantAgent(
            name="test_assistant",
            system_message="You are a helpful assistant.",
            llm_config=llm_config,
        )
        
        user_proxy = UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1,
            code_execution_config=False,
        )
        
        start_time = time.time()
        user_proxy.initiate_chat(
            assistant, 
            message="Hello, respond with 'AG2 test successful'"
        )
        duration = time.time() - start_time
        
        return f"AG2 test completed in {duration:.2f}s"
        
    except ImportError:
        return "ag2 not available"
    except Exception as e:
        return f"AG2 test failed: {e}"

if __name__ == "__main__":
    print("Testing AutoGen with new API...")
    
    print("\n1. New AutoGen API test:")
    if AUTOGEN_AVAILABLE:
        result = asyncio.run(test_new_autogen())
    else:
        result = "New API not available"
    print(result)
    
    print("\n2. AG2 fallback test:")
    result2 = test_autogen_fallback()
    print(result2)
    
    print("\n✅ AutoGen tests completed")