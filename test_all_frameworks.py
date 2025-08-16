#!/usr/bin/env python3
"""
Test script to verify all AI agent frameworks are working correctly.
This script tests API connectivity, execution time, and basic functionality.
"""

import os
import sys
import time
import json
import traceback
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_api_key():
    """Test if API key is available"""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ No API key found. Please set GEMINI_API_KEY or GOOGLE_API_KEY")
        return False
    print(f"✅ API key found: {api_key[:8]}...{api_key[-4:]}")
    return True

def test_api_connection():
    """Test direct connection to Gemini API"""
    try:
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=api_key)
        
        model = genai.GenerativeModel("gemini-2.5-flash")
        start_time = time.time()
        response = model.generate_content("Hello, respond with 'API test successful'")
        duration = time.time() - start_time
        
        print(f"✅ Direct API test successful in {duration:.2f}s: {response.text[:50]}...")
        return True
    except Exception as e:
        print(f"❌ Direct API test failed: {e}")
        return False

def test_atomic_agents():
    """Test Atomic Agents framework"""
    print("\n🧪 Testing Atomic Agents...")
    try:
        start_time = time.time()
        
        # Import atomic agents components
        from atomic_agents_baseline import analyze_with_gemini
        
        # Test with simple prompt
        test_prompt = "Analyze this statement: 'This trial has 100 participants'"
        test_context = "Trial data: {'enrollment': 100, 'status': 'completed'}"
        
        result = analyze_with_gemini(test_prompt, test_context)
        duration = time.time() - start_time
        
        if duration < 0.5:
            print(f"⚠️ Atomic Agents completed too quickly ({duration:.2f}s) - might not be calling LLM")
            return False
        
        print(f"✅ Atomic Agents test successful in {duration:.2f}s")
        print(f"   Result length: {len(result)} characters")
        return True
        
    except Exception as e:
        print(f"❌ Atomic Agents test failed: {e}")
        traceback.print_exc()
        return False

def test_langchain():
    """Test LangChain framework"""
    print("\n🧪 Testing LangChain...")
    try:
        start_time = time.time()
        
        # Test LangChain setup
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.1,
            google_api_key=api_key
        )
        
        # Test with simple prompt
        result = llm.invoke("Hello, respond with 'LangChain test successful'")
        duration = time.time() - start_time
        
        if duration < 0.5:
            print(f"⚠️ LangChain completed too quickly ({duration:.2f}s) - might not be calling LLM")
            return False
        
        print(f"✅ LangChain test successful in {duration:.2f}s")
        print(f"   Result: {result.content[:100]}...")
        return True
        
    except Exception as e:
        print(f"❌ LangChain test failed: {e}")
        traceback.print_exc()
        return False

def test_agno():
    """Test Agno/Phidata framework"""
    print("\n🧪 Testing Agno (Phidata)...")
    try:
        start_time = time.time()
        
        # Test Agno setup
        from phi.agent import Agent
        from phi.model.google import Gemini
        
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        model = Gemini(
            id="gemini-2.5-flash",
            api_key=api_key,
            temperature=0.1
        )
        
        # Create simple agent
        agent = Agent(
            model=model,
            instructions="You are a helpful assistant. Respond clearly and concisely."
        )
        
        # Test with simple prompt
        result = agent.run("Hello, respond with 'Agno test successful'")
        duration = time.time() - start_time
        
        if duration < 0.5:
            print(f"⚠️ Agno completed too quickly ({duration:.2f}s) - might not be calling LLM")
            return False
        
        print(f"✅ Agno test successful in {duration:.2f}s")
        print(f"   Result: {str(result)[:100]}...")
        return True
        
    except Exception as e:
        print(f"❌ Agno test failed: {e}")
        traceback.print_exc()
        return False

def test_autogen():
    """Test AutoGen framework"""
    print("\n🧪 Testing AutoGen...")
    try:
        start_time = time.time()
        
        # Test AutoGen setup
        import autogen
        
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
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
        
        # Create a simple assistant
        assistant = autogen.AssistantAgent(
            name="test_assistant",
            system_message="You are a helpful assistant. Respond with 'AutoGen test successful'.",
            llm_config=llm_config,
        )
        
        # Create user proxy
        user_proxy = autogen.UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1,
            is_termination_msg=lambda x: "AutoGen test successful" in x.get("content", ""),
            code_execution_config=False,
        )
        
        # Test conversation
        user_proxy.initiate_chat(assistant, message="Hello, please respond.")
        duration = time.time() - start_time
        
        if duration < 1.0:
            print(f"⚠️ AutoGen completed too quickly ({duration:.2f}s) - might not be calling LLM")
            return False
        
        print(f"✅ AutoGen test successful in {duration:.2f}s")
        return True
        
    except Exception as e:
        print(f"❌ AutoGen test failed: {e}")
        traceback.print_exc()
        return False

def test_data_loading():
    """Test if training data can be loaded"""
    print("\n🧪 Testing data loading...")
    try:
        # Test if training data exists
        if not os.path.exists("training_data"):
            print("❌ training_data directory not found")
            return False
        
        # Test loading train.json
        train_path = "training_data/train.json"
        if not os.path.exists(train_path):
            print("❌ train.json not found")
            return False
        
        with open(train_path, "r", encoding="utf-8") as f:
            train_data = json.load(f)
        
        print(f"✅ Training data loaded: {len(train_data)} examples")
        
        # Test loading a clinical trial
        ct_path = "training_data/CT json"
        if not os.path.exists(ct_path):
            print("❌ CT json directory not found")
            return False
        
        ct_files = [f for f in os.listdir(ct_path) if f.endswith('.json')]
        print(f"✅ Found {len(ct_files)} clinical trial files")
        
        # Test loading one file
        if ct_files:
            with open(os.path.join(ct_path, ct_files[0]), "r", encoding="utf-8") as f:
                ct_data = json.load(f)
            print(f"✅ Sample clinical trial loaded: {ct_data.get('Clinical Trial ID', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting comprehensive framework tests...")
    print("="*80)
    
    results = {}
    
    # Test 1: API Key
    results['api_key'] = test_api_key()
    
    # Test 2: Direct API Connection
    results['api_connection'] = test_api_connection()
    
    # Test 3: Data Loading
    results['data_loading'] = test_data_loading()
    
    # Test 4: Individual Frameworks
    if results['api_key'] and results['api_connection']:
        results['atomic_agents'] = test_atomic_agents()
        results['langchain'] = test_langchain()
        results['agno'] = test_agno()
        results['autogen'] = test_autogen()
    else:
        print("⚠️ Skipping framework tests due to API issues")
        results.update({
            'atomic_agents': False,
            'langchain': False,
            'agno': False,
            'autogen': False
        })
    
    # Summary
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:15} {status}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! All frameworks are working correctly.")
    elif passed_tests > 0:
        print("⚠️ Some tests failed. Check the output above for details.")
    else:
        print("❌ All tests failed. Please check your configuration.")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)