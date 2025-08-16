#!/usr/bin/env python3
"""
Clean version of AutoGen framework for testing.
"""

import os
import time
try:
    import autogen
    from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
    AUTOGEN_AVAILABLE = True
except ImportError:
    print("⚠️ AutoGen not available, using fallback")
    AUTOGEN_AVAILABLE = False

def test_autogen_simple():
    """Test simple AutoGen setup"""
    if not AUTOGEN_AVAILABLE:
        return "AutoGen not available"
    
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return "No API key found"
    
    # Configuration for Gemini
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
    
    # Create assistant
    assistant = AssistantAgent(
        name="test_assistant",
        system_message="You are a helpful assistant. Please respond concisely.",
        llm_config=llm_config,
    )
    
    # Create user proxy
    user_proxy = UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1,
        code_execution_config=False,
    )
    
    # Test conversation
    start_time = time.time()
    try:
        user_proxy.initiate_chat(
            assistant, 
            message="Hello, please respond with 'AutoGen test successful'"
        )
        duration = time.time() - start_time
        
        return f"AutoGen test completed in {duration:.2f}s"
    except Exception as e:
        return f"AutoGen test failed: {e}"

def test_autogen_multi_agent():
    """Test multi-agent AutoGen setup"""
    if not AUTOGEN_AVAILABLE:
        return "AutoGen not available"
    
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return "No API key found"
    
    # Configuration for Gemini
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
        "timeout": 120,
    }
    
    # Create multiple agents
    supervisor = AssistantAgent(
        name="supervisor",
        system_message="You are a supervisor coordinating analysis. Keep responses brief.",
        llm_config=llm_config,
    )
    
    medical_expert = AssistantAgent(
        name="medical_expert",
        system_message="You are a medical expert. Analyze from medical perspective. Keep responses brief.",
        llm_config=llm_config,
    )
    
    aggregator = AssistantAgent(
        name="aggregator",
        system_message="You are an aggregator. Make final decisions. End with 'FINAL_DECISION: Entailment' or 'FINAL_DECISION: Contradiction'",
        llm_config=llm_config,
    )
    
    user_proxy = UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=8,
        is_termination_msg=lambda x: "FINAL_DECISION:" in x.get("content", ""),
        code_execution_config=False,
    )
    
    # Create group chat
    groupchat = GroupChat(
        agents=[supervisor, medical_expert, aggregator, user_proxy],
        messages=[],
        max_round=8,
        speaker_selection_method="round_robin",
    )
    
    manager = GroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config
    )
    
    # Test multi-agent conversation
    context = """
    Analyze this statement: "This trial has 100 participants"
    Trial data: {"enrollment": 100, "status": "completed"}
    
    Supervisor: coordinate analysis. Medical expert: analyze medically. Aggregator: make final decision.
    """
    
    start_time = time.time()
    try:
        user_proxy.initiate_chat(manager, message=context)
        duration = time.time() - start_time
        
        messages_count = len(groupchat.messages)
        return f"Multi-agent test completed in {duration:.2f}s with {messages_count} messages"
        
    except Exception as e:
        return f"Multi-agent test failed: {e}"

if __name__ == "__main__":
    print("Testing AutoGen...")
    
    print("\n1. Simple AutoGen test:")
    result1 = test_autogen_simple()
    print(result1)
    
    print("\n2. Multi-agent AutoGen test:")
    result2 = test_autogen_multi_agent()
    print(result2)
    
    print("\n✅ AutoGen tests completed")