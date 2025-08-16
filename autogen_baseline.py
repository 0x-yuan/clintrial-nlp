#!/usr/bin/env python
# coding: utf-8

# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/0x-yuan/clintrial-nlp/blob/main/autogen_baseline.ipynb)
# 
# # AutoGen 框架基線 - 臨床試驗 NLP
# 
# ## 概述
# 
# 本notebook展示如何使用Microsoft AutoGen建構一個多代理系統，用於臨床試驗自然語言推理(NLI)。AutoGen在透過對話模式進行多代理協作方面表現卓越。
# 
# ## 📚 學習目標
# 完成本教學後，您將學會：
# - 理解 AutoGen 的多代理對話框架
# - 建立專業化的角色代理
# - 實作群組聊天協作模式
# - 管理代理間的任務分配和協調
# 
# ### 為什麼選擇 AutoGen？
# - **多代理對話**: 代理間的自然對話
# - **角色專業化**: 每個代理都有特定的專業知識
# - **彈性協調**: 監督者管理任務分配
# - **強健推理**: 多重視角提升準確性
# 
# ### 🎭 代理架構
# 根據教學材料，我們實作：
# 1. **監督者**: 協調任務並管理工作流程
# 2. **醫療專家**: 理解醫學術語和概念
# 3. **數值分析員**: 處理量化資料和統計
# 4. **邏輯檢查員**: 驗證邏輯關係
# 5. **聚合員**: 基於所有輸入做出最終決策
# 
# > 💬 **核心概念**: AutoGen的獨特之處在於代理透過對話進行協作，就像人類團隊討論問題一樣。這種自然的互動模式使得複雜推理過程更加透明和可解釋。

# In[ ]:


# 🔧 Colab 環境設置 - 一鍵安裝 AutoGen 相關套件
# 這個cell會靜默安裝Microsoft AutoGen多代理協作框架所需的套件
get_ipython().system('pip install -q pyautogen[gemini] python-dotenv pandas tqdm')
get_ipython().system('pip install -q google-generativeai gdown')

# 確保 AutoGen 相關依賴完整安裝
get_ipython().system('pip install -q openai>=1.0.0  # AutoGen 需要 OpenAI 客戶端庫（即使不使用 OpenAI）')
get_ipython().system('pip install -q docker  # 如果需要代碼執行功能')

print("✅ AutoGen 多代理協作框架安裝完成！可以開始建構對話式代理團隊了")


# In[ ]:


# 📥 從 Google Drive 下載訓練資料
# 這個cell會自動下載並解壓縮 clinicaltrial-nlp.zip，確保在Colab中可以直接運行
import os
import gdown
import zipfile
import shutil

# Google Drive zip 檔案 ID
file_id = "15GA5XI39DDxQ5QkIZXsFbApx1yEvCpcR"
zip_url = f"https://drive.google.com/uc?id={file_id}"
zip_filename = "clinicaltrial-nlp.zip"

# 檢查是否已有訓練資料
if not os.path.exists("training_data"):
    print("📥 從 Google Drive 下載 clinicaltrial-nlp.zip...")
    print("⚠️ 如果下載失敗，請確認:")
    print("1. Google Drive 連結的權限設定為 '知道連結的使用者'")
    print("2. 網路連線正常")
    print(f"3. 檔案連結: {zip_url}")
    
    try:
        # 下載 zip 檔案
        print("📥 正在下載 zip 檔案...")
        gdown.download(zip_url, zip_filename, quiet=False)
        
        # 解壓縮檔案
        print("📦 正在解壓縮檔案...")
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(".")
        
        # 移動 training_data 到正確位置（如果在子資料夾中）
        if os.path.exists("clintrial-nlp/training_data") and not os.path.exists("training_data"):
            print("📁 移動 training_data 到正確位置...")
            shutil.move("clintrial-nlp/training_data", "training_data")
            # 清理解壓縮的資料夾
            if os.path.exists("clintrial-nlp"):
                shutil.rmtree("clintrial-nlp")
            if os.path.exists("__MACOSX"):  # 清理 macOS 產生的隱藏檔案
                shutil.rmtree("__MACOSX")
        
        # 清理 zip 檔案
        os.remove(zip_filename)
        print("✅ 訓練資料下載並解壓縮完成！")
        
    except Exception as e:
        print(f"❌ 下載失敗: {e}")
        print("\n🔧 手動解決方案:")
        print("1. 點擊此連結下載 zip 檔案:")
        print("   https://drive.google.com/file/d/15GA5XI39DDxQ5QkIZXsFbApx1yEvCpcR/view?usp=sharing")
        print("2. 上傳 zip 檔案到 Colab")
        print("3. 解壓縮後重新執行後續的 cells")
        
        # 創建一個提示檔案
        os.makedirs("training_data", exist_ok=True)
        with open("training_data/DOWNLOAD_INSTRUCTIONS.txt", "w", encoding="utf-8") as f:
            f.write("請手動下載並解壓縮 clinicaltrial-nlp.zip:\n")
            f.write("https://drive.google.com/file/d/15GA5XI39DDxQ5QkIZXsFbApx1yEvCpcR/view?usp=sharing\n")
        
        print("\n📝 已創建下載指示檔案於 training_data/DOWNLOAD_INSTRUCTIONS.txt")
else:
    print("✅ 訓練資料已存在，跳過下載")

# 檢查下載的資料結構
if os.path.exists("training_data"):
    contents = os.listdir("training_data")
    print(f"📂 資料夾內容: {contents}")
    if os.path.exists("training_data/CT json"):
        ct_files = len([f for f in os.listdir("training_data/CT json") if f.endswith('.json')])
        print(f"📄 找到 {ct_files} 個臨床試驗JSON檔案")
    else:
        print("⚠️ 找不到 'CT json' 子資料夾，請檢查下載是否完整")


# In[ ]:


# 🧪 準備測試資料集
import json

def create_test_data_if_needed():
    if not os.path.exists("test.json"):
        try:
            with open("training_data/train.json", "r", encoding="utf-8") as f:
                train_data = json.load(f)
            test_data = dict(list(train_data.items())[:100])
            with open("test.json", "w", encoding="utf-8") as f:
                json.dump(test_data, f, indent=2, ensure_ascii=False)
            print(f"✅ 已創建測試資料集，包含 {len(test_data)} 個樣本")
        except Exception as e:
            print(f"❌ 創建測試資料失敗: {e}")
    else:
        print("✅ test.json 已存在")

create_test_data_if_needed()


# ## 環境設置和安裝
# 
# 首先，讓我們安裝必要的套件並設置環境：
# 
# > 📝 **說明**: AutoGen框架支援多種LLM後端，這裡我們使用Google Gemini模型進行示範。

# In[ ]:


# Load environment variables
from dotenv import load_dotenv
import os

load_dotenv()
print("✅ Environment loaded")


# In[ ]:


# Import required libraries
import json
import pandas as pd
from tqdm import tqdm
import autogen
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

print("✅ Libraries imported successfully")


# ## Data Loading and Utilities
# 
# Let's create utility functions to load clinical trial data:

# In[ ]:


def load_clinical_trial(trial_id: str) -> Dict[str, Any]:
    """Load clinical trial data from JSON file.
    
    Args:
        trial_id: The NCT identifier for the clinical trial
        
    Returns:
        Dictionary containing trial data or error information
    """
    try:
        file_path = os.path.join("training_data", "CT json", f"{trial_id}.json")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        return {"error": f"Clinical trial {trial_id} not found"}
    except Exception as e:
        return {"error": f"Error loading {trial_id}: {str(e)}"}

def load_dataset(filepath: str) -> Dict[str, Any]:
    """Load training or test dataset.
    
    Args:
        filepath: Path to the JSON dataset file
        
    Returns:
        Dictionary containing the dataset
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return {}

# Test data loading
sample_trial = load_clinical_trial("NCT00066573")
print(f"✅ Data loading functions ready. Sample trial loaded: {sample_trial.get('Clinical Trial ID', 'Error')}")


# ## AutoGen 代理配置
# 
# 現在讓我們使用Google Gemini配置多代理系統。每個代理都有專門的角色和特定的指令：
# 
# > 🤖 **技術說明**: AutoGen的AssistantAgent類別允許我們創建具有特定專業知識和行為模式的AI代理，這些代理可以透過群組聊天進行協作。

# In[ ]:


# 🤖 Setup Google Gemini 2.5 Flash model for AutoGen
# AutoGen will use this configuration to initialize all agent LLM backends
import google.generativeai as genai
import os

print("🔑 Checking API key configuration...")

# Support both GEMINI_API_KEY and GOOGLE_API_KEY environment variables
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("⚠️ Please set GEMINI_API_KEY or GOOGLE_API_KEY environment variable")
    print("You can set it in Colab's 'Secrets' panel or use:")
    print("import os")
    print("os.environ['GEMINI_API_KEY'] = 'your-api-key'")
    print("or")
    print("os.environ['GOOGLE_API_KEY'] = 'your-api-key'")
    # Create fake config to avoid errors
    config_list = [{"model": "gpt-3.5-turbo", "api_key": "fake"}]
    api_key = "fake"
else:
    print(f"✅ Found API key: {api_key[:8]}...{api_key[-4:]}")
    genai.configure(api_key=api_key)
    print("✅ Google Gemini API configured")
    
    # Test API connection
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content("Hello, respond with 'API test successful'")
        print(f"✅ API connection test successful: {response.text[:50]}...")
    except Exception as e:
        print(f"⚠️ API connection test failed: {e}")

# AutoGen's LLM configuration - Use Google Gemini 2.5 Flash
# Based on official AutoGen documentation for Gemini integration
config_list = [
    {
        "model": "gemini-2.5-flash",
        "api_key": api_key,
        "api_type": "google"  # This is crucial for AutoGen to recognize it as Gemini
    }
]

# Test configuration
print(f"📋 LLM configuration model: {config_list[0]['model']}")
print(f"📋 API type: {config_list[0]['api_type']}")

llm_config = {
    "config_list": config_list,
    "temperature": 0.1,  # Low temperature for consistent results
    "timeout": 120,
    "seed": 42,  # Ensure reproducibility
}

print("✅ AutoGen configuration ready, will use Google Gemini 2.5 Flash model")
print(f"🔧 Configuration details: temperature={llm_config['temperature']}, timeout={llm_config['timeout']}s")


# ## Agent Definitions
# 
# Let's define each agent with their specific roles and capabilities:

# In[ ]:


# 1. Supervisor Agent - Coordinates the overall workflow
supervisor = autogen.AssistantAgent(
    name="supervisor",
    system_message="""You are the Supervisor Agent responsible for coordinating clinical trial analysis.
    
    Your responsibilities:
    1. Break down complex clinical trial questions into manageable tasks
    2. Assign tasks to appropriate specialist agents
    3. Coordinate information flow between agents
    4. Ensure all aspects of the statement are thoroughly analyzed
    5. Guide the final decision-making process
    
    When given a statement and clinical trial data, you should:
    - First understand what the statement claims
    - Identify which sections of the trial are relevant
    - Delegate specific analysis tasks to specialist agents
    - Collect and synthesize their findings
    
    Always maintain a systematic approach and ensure thorough analysis.
    """,
    llm_config=llm_config,
)

# 2. Medical Expert Agent - Handles medical terminology and concepts
medical_expert = autogen.AssistantAgent(
    name="medical_expert",
    system_message="""You are the Medical Expert Agent specializing in clinical trial analysis.
    
    Your expertise includes:
    1. Medical terminology and clinical concepts
    2. Understanding of trial phases, interventions, and outcomes
    3. Clinical significance of findings
    4. Medical relationships and causations
    5. Safety profiles and adverse events
    
    When analyzing statements:
    - Focus on the medical accuracy and clinical relevance
    - Identify key medical terms and their implications
    - Assess whether medical claims align with trial evidence
    - Consider clinical context and medical plausibility
    
    Provide clear medical reasoning for your analysis.
    """,
    llm_config=llm_config,
)

# 3. Numerical Analyzer Agent - Processes quantitative data
numerical_analyzer = autogen.AssistantAgent(
    name="numerical_analyzer",
    system_message="""You are the Numerical Analyzer Agent specializing in quantitative analysis.
    
    Your expertise includes:
    1. Statistical analysis and interpretation
    2. Numerical comparisons and calculations
    3. Percentage, ratios, and statistical significance
    4. Data ranges, confidence intervals, and error margins
    5. Trend analysis and numerical patterns
    
    When analyzing statements:
    - Extract and verify all numerical claims
    - Perform calculations to validate numerical relationships
    - Compare stated numbers with trial data
    - Assess statistical significance and clinical relevance
    - Identify any numerical inconsistencies or errors
    
    Be precise and show your calculations when relevant.
    """,
    llm_config=llm_config,
)

# 4. Logic Checker Agent - Validates logical relationships
logic_checker = autogen.AssistantAgent(
    name="logic_checker",
    system_message="""You are the Logic Checker Agent responsible for validating logical reasoning.
    
    Your expertise includes:
    1. Logical consistency and coherence
    2. Cause-and-effect relationships
    3. Conditional logic and implications
    4. Contradiction detection
    5. Multi-step reasoning validation
    
    When analyzing statements:
    - Evaluate the logical structure of claims
    - Check for internal consistency
    - Identify logical fallacies or contradictions
    - Assess the validity of inferences
    - Ensure conclusions follow from premises
    
    Focus on the logical soundness rather than medical or numerical details.
    """,
    llm_config=llm_config,
)

# 5. Aggregator Agent - Makes final decisions
aggregator = autogen.AssistantAgent(
    name="aggregator",
    system_message="""You are the Aggregator Agent responsible for making final entailment decisions.
    
    Your responsibilities:
    1. Synthesize findings from all specialist agents
    2. Weigh different types of evidence appropriately
    3. Resolve conflicts between agent analyses
    4. Make the final Entailment vs Contradiction decision
    5. Provide confidence assessment for the decision
    
    Decision criteria:
    - ENTAILMENT: Statement is directly supported by trial data
    - CONTRADICTION: Statement is refuted by trial data
    
    When making decisions:
    - Consider input from Medical Expert, Numerical Analyzer, and Logic Checker
    - Prioritize evidence that directly addresses the statement
    - Be conservative: if evidence is unclear, consider context carefully
    - Always provide reasoning for your final decision
    
    Output format: "FINAL_DECISION: [Entailment/Contradiction]"
    """,
    llm_config=llm_config,
)

# User proxy for managing the conversation
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").find("FINAL_DECISION:") >= 0,
    code_execution_config=False,
)

print("✅ All agents created successfully")


# ## 群組聊天設置
# 
# 我們將設置一個群組聊天，讓代理們能夠協作分析每個陳述：
# 
# > 🗣️ **協作說明**: GroupChat允許多個代理在同一個對話中互動，每個代理輪流發言，分享他們的分析見解，就像團隊會議一樣。

# In[ ]:


# Create group chat with all agents
groupchat = autogen.GroupChat(
    agents=[supervisor, medical_expert, numerical_analyzer, logic_checker, aggregator, user_proxy],
    messages=[],
    max_round=15,  # Allow sufficient rounds for thorough analysis
    speaker_selection_method="round_robin",  # Ensure all agents participate
)

# Create group chat manager
manager = autogen.GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config
)

print("✅ Group chat configured")


# ## Clinical Trial Analysis Pipeline
# 
# Now let's create our main analysis pipeline that coordinates the multi-agent analysis:

# In[ ]:


def analyze_statement_with_agents(statement: str, primary_id: str, secondary_id: Optional[str] = None, 
                                section_id: Optional[str] = None) -> str:
    """
    Analyze a statement using the multi-agent system.
    
    Args:
        statement: The natural language statement to analyze
        primary_id: Primary clinical trial ID
        secondary_id: Secondary trial ID for comparison statements
        section_id: Relevant section of the trial (Eligibility, Intervention, Results, Adverse Events)
        
    Returns:
        Final decision: 'Entailment' or 'Contradiction'
    """
    
    print(f"🔍 開始分析: {statement[:50]}...")
    
    # Load clinical trial data
    primary_data = load_clinical_trial(primary_id)
    secondary_data = None
    if secondary_id:
        secondary_data = load_clinical_trial(secondary_id)
    
    # 檢查資料是否載入成功
    if "error" in primary_data:
        print(f"❌ 主要試驗資料載入失敗: {primary_data['error']}")
        return "Contradiction"
    
    # Prepare secondary data section
    secondary_section = ""
    if secondary_data:
        secondary_section = f"SECONDARY TRIAL DATA:\n{json.dumps(secondary_data, indent=2)}"
    
    # Prepare the analysis context (縮短以避免 token 限制)
    context = f"""
CLINICAL TRIAL ANALYSIS TASK

Statement: "{statement}"
Primary Trial: {primary_id}
Secondary Trial: {secondary_id or "N/A"}
Section: {section_id or "All sections"}

PRIMARY TRIAL DATA (summarized):
Title: {primary_data.get('Official Title', 'N/A')}
Phase: {primary_data.get('Phase', 'N/A')}
Status: {primary_data.get('Overall Status', 'N/A')}
Conditions: {primary_data.get('Conditions', 'N/A')}
Interventions: {primary_data.get('Interventions', 'N/A')}

{secondary_section}

TASK: Analyze whether this statement is ENTAILMENT (supported) or CONTRADICTION (refuted).
Supervisor: Coordinate analysis with medical_expert, numerical_analyzer, logic_checker, then aggregator decides.
"""
    
    print("🤖 啟動多代理分析...")
    
    # Reset group chat for new analysis
    groupchat.messages = []
    
    # Start the multi-agent analysis
    try:
        print("📞 開始代理對話...")
        
        # 使用更詳細的錯誤處理
        user_proxy.initiate_chat(
            manager,
            message=context,
            clear_history=True
        )
        
        print(f"💬 對話完成，共 {len(groupchat.messages)} 條訊息")
        
        # Debug: 顯示最後幾條訊息
        if len(groupchat.messages) > 0:
            print("📋 最後的訊息:")
            for i, msg in enumerate(groupchat.messages[-3:]):  # 顯示最後3條訊息
                sender = msg.get("name", "unknown")
                content = msg.get("content", "")[:100] + "..."
                print(f"  {i+1}. {sender}: {content}")
        
        # Extract final decision from the last message
        final_message = groupchat.messages[-1]["content"] if groupchat.messages else ""
        
        # Parse the final decision
        if "FINAL_DECISION: Entailment" in final_message:
            decision = "Entailment"
        elif "FINAL_DECISION: Contradiction" in final_message:
            decision = "Contradiction"
        else:
            # Fallback: look for decision keywords in the final message
            if "entailment" in final_message.lower():
                decision = "Entailment"
            else:
                decision = "Contradiction"
        
        print(f"✅ 分析完成: {decision}")
        return decision
                
    except Exception as e:
        print(f"❌ 代理分析錯誤: {e}")
        print(f"錯誤類型: {type(e).__name__}")
        import traceback
        print(f"詳細錯誤: {traceback.format_exc()}")
        return "Contradiction"  # Conservative fallback

print("✅ Analysis pipeline ready")


# # 🧪 測試多代理系統是否正常工作
# def test_agent_system():
#     """測試多代理系統是否能正常協作"""
#     print("🧪 測試多代理系統...")
#     
#     # 簡單的測試案例
#     test_context = """
#     請各位代理協作分析：
#     陳述: "這個試驗有100名參與者"
#     試驗資料: {"enrollment": 100, "participants": "100 patients enrolled"}
#     
#     請 supervisor 協調，medical_expert 分析醫學相關性，numerical_analyzer 檢查數字，
#     logic_checker 驗證邏輯，最後 aggregator 做出決定。
#     
#     請確保每個代理都參與討論，最後輸出 "FINAL_DECISION: Entailment" 或 "FINAL_DECISION: Contradiction"
#     """
#     
#     # 重置群組聊天
#     groupchat.messages = []
#     
#     try:
#         print("📞 啟動測試對話...")
#         import time
#         start_time = time.time()
#         
#         user_proxy.initiate_chat(
#             manager,
#             message=test_context,
#             clear_history=True
#         )
#         
#         end_time = time.time()
#         duration = end_time - start_time
#         
#         print(f"⏱️ 對話耗時: {duration:.2f} 秒")
#         print(f"💬 總訊息數: {len(groupchat.messages)}")
#         
#         if len(groupchat.messages) == 0:
#             print("❌ 沒有生成任何對話訊息！")
#             return False
#         elif duration < 1.0:
#             print("⚠️ 對話時間太短，可能沒有真正執行 LLM 調用")
#             return False
#         else:
#             print("✅ 多代理系統正常工作")
#             
#             # 顯示代理參與情況
#             agents_participated = set()
#             for msg in groupchat.messages:
#                 if "name" in msg:
#                     agents_participated.add(msg["name"])
#             
#             print(f"🤖 參與的代理: {list(agents_participated)}")
#             return True
#             
#     except Exception as e:
#         print(f"❌ 測試失敗: {e}")
#         return False
# 
# # 運行測試
# test_result = test_agent_system()
# print(f"\n🎯 系統測試結果: {'通過' if test_result else '失敗'}")

# In[ ]:


# Test with a sample statement
test_statement = "there is a 13.2% difference between the results from the two the primary trial cohorts"
test_primary_id = "NCT00066573"

print(f"Testing with statement: '{test_statement}'")
print(f"Primary trial: {test_primary_id}")
print("\n" + "="*80 + "\n")

# Run the analysis
result = analyze_statement_with_agents(
    statement=test_statement,
    primary_id=test_primary_id,
    section_id="Results"
)

print(f"\n" + "="*80)
print(f"FINAL RESULT: {result}")
print("="*80)


# ## Evaluation on Training Data
# 
# Let's evaluate our AutoGen system on a subset of training data:

# In[ ]:


# Load training data
train_data = load_dataset("training_data/train.json")
print(f"Loaded {len(train_data)} training examples")

# Evaluate on first 20 examples (adjust as needed for testing)
sample_size = 20
examples = list(train_data.items())[:sample_size]

print(f"\nEvaluating on {len(examples)} examples...")

results = []
correct = 0

for i, (uuid, example) in enumerate(tqdm(examples, desc="Processing")):
    try:
        statement = example.get("Statement")
        primary_id = example.get("Primary_id")
        secondary_id = example.get("Secondary_id")
        section_id = example.get("Section_id")
        expected = example.get("Label")
        
        if not statement or not primary_id:
            results.append({
                "uuid": uuid,
                "expected": expected,
                "predicted": "SKIPPED",
                "correct": False
            })
            continue
        
        # Get prediction from multi-agent system
        predicted = analyze_statement_with_agents(
            statement=statement,
            primary_id=primary_id,
            secondary_id=secondary_id,
            section_id=section_id
        )
        
        # Check if correct
        is_correct = (predicted.strip() == expected.strip())
        if is_correct:
            correct += 1
            
        results.append({
            "uuid": uuid,
            "statement": statement[:100] + "..." if len(statement) > 100 else statement,
            "expected": expected,
            "predicted": predicted,
            "correct": is_correct
        })
        
        print(f"Example {i+1}: {expected} -> {predicted} {'✅' if is_correct else '❌'}")
        
    except Exception as e:
        print(f"Error processing example {i+1}: {e}")
        results.append({
            "uuid": uuid,
            "expected": expected,
            "predicted": "ERROR",
            "correct": False
        })

# Calculate accuracy
accuracy = correct / len(examples) if examples else 0
print(f"\n📊 AutoGen Results:")
print(f"Accuracy: {accuracy:.2%} ({correct}/{len(examples)})")


# ## Error Analysis
# 
# Let's examine some examples where our system made mistakes:

# In[ ]:


# Show incorrect predictions for analysis
incorrect_results = [r for r in results if not r["correct"] and r["predicted"] not in ["SKIPPED", "ERROR"]]

print(f"\n🔍 Error Analysis ({len(incorrect_results)} incorrect predictions):")
print("=" * 80)

for i, result in enumerate(incorrect_results[:5]):  # Show first 5 errors
    print(f"\nError #{i+1}:")
    print(f"Statement: {result['statement']}")
    print(f"Expected: {result['expected']}")
    print(f"Predicted: {result['predicted']}")
    print("-" * 40)


# ## Generate Submission File
# 
# Let's create a submission file for the test data:

# In[ ]:


def generate_autogen_submission(test_file="test.json", output_file="autogen_submission.json", sample_size=None):
    """
    Generate submission file using AutoGen multi-agent system.
    
    Args:
        test_file: Path to test data
        output_file: Output submission file
        sample_size: Number of examples to process (None for all)
    """
    
    # Load test data
    test_data = load_dataset(test_file)
    if not test_data:
        print(f"❌ Could not load test data from {test_file}")
        return
    
    examples = list(test_data.items())
    if sample_size:
        examples = examples[:sample_size]
        
    print(f"🚀 Generating AutoGen predictions for {len(examples)} examples...")
    
    submission = {}
    
    for i, (uuid, example) in enumerate(tqdm(examples, desc="AutoGen Processing")):
        try:
            statement = example.get("Statement")
            primary_id = example.get("Primary_id")
            secondary_id = example.get("Secondary_id")
            section_id = example.get("Section_id")
            
            if not statement or not primary_id:
                submission[uuid] = {"Prediction": "Contradiction"}  # Default fallback
                continue
                
            # Get prediction from multi-agent system
            prediction = analyze_statement_with_agents(
                statement=statement,
                primary_id=primary_id,
                secondary_id=secondary_id,
                section_id=section_id
            )
            
            submission[uuid] = {"Prediction": prediction}
            
        except Exception as e:
            print(f"Error processing {uuid}: {e}")
            submission[uuid] = {"Prediction": "Contradiction"}  # Conservative fallback
    
    # Save submission file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(submission, f, indent=2)
    
    print(f"✅ AutoGen submission saved to {output_file}")
    return submission

# Generate submission for a small sample (change sample_size as needed)
autogen_submission = generate_autogen_submission(
    test_file="test.json", 
    output_file="autogen_submission.json",
    sample_size=10  # Process only 10 examples for testing
)

print(f"Generated predictions for {len(autogen_submission)} examples")


# ## 結論與洞察
# 
# ### AutoGen 框架優勢：
# 1. **多視角分析**: 每個代理帶來專門的專業知識
# 2. **結構化協作**: 明確的角色和責任
# 3. **彈性協調**: 監督者管理複雜的工作流程
# 4. **強健推理**: 多重觀點提升決策品質
# 5. **可解釋性**: 代理對話提供推理過程的洞察
# 
# ### 關鍵功能展示：
# - **角色專業化**: 醫療、數值和邏輯分析
# - **群組聊天協調**: 代理透過結構化對話協作
# - **系統化方法**: 監督者確保全面分析
# - **錯誤處理**: 邊緣案例的優雅回退
# - **可擴展架構**: 易於添加新的專業代理
# 
# ### 優化機會：
# 1. **提示工程**: 微調代理指令以提升效能
# 2. **代理協調**: 改善對話流程和決策制定
# 3. **錯誤分析**: 利用失敗案例改進代理推理
# 4. **效能調整**: 最佳化模型參數和對話模式
# 5. **領域知識**: 納入更多醫學領域專業知識
# 
# ### 何時使用 AutoGen：
# - 需要多重視角的複雜推理任務
# - 從角色專業化中受益的問題
# - 可解釋性很重要的場景
# - 需要強健協作決策的應用
# 
# ## 🎓 學習重點總結
# - **對話式協作**: 代理透過自然對話進行協作
# - **角色分工**: 每個代理專注於特定分析領域
# - **監督協調**: 中央協調確保系統化分析
# - **透明推理**: 對話過程提供決策透明度
# 
# AutoGen在結構化多代理協作方面表現出色，並為推理過程提供優秀的透明度，使其成為複雜臨床分析任務的理想選擇。
