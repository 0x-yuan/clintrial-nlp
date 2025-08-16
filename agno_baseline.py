#!/usr/bin/env python
# coding: utf-8

# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/0x-yuan/clintrial-nlp/blob/main/agno_baseline.ipynb)
# 
# # Agno (Phidata) 框架基線 - 臨床試驗 NLP
# 
# ## 概述
# 
# 本notebook展示如何使用Agno（前身為Phidata）建構一個高效能、全堆疊的代理系統，用於臨床試驗自然語言推理(NLI)。Agno提供內建記憶體、知識管理和推理能力。
# 
# ## 📚 學習目標
# 完成本教學後，您將學會：
# - 理解 Agno 的全堆疊代理開發環境
# - 使用內建記憶體和知識管理功能
# - 實作RAG（檢索增強生成）系統
# - 建構持久化的代理對話系統
# 
# ### 為什麼選擇 Agno？
# - **全堆疊平台**: 完整的代理開發環境
# - **內建記憶體**: 持久的上下文和對話記憶體
# - **知識整合**: RAG功能和知識庫管理
# - **高效能**: 針對生產工作負載最佳化
# - **豐富工具**: 廣泛的內建工具和整合
# - **企業就緒**: 專為大規模應用設計
# 
# ### 🧠 代理架構
# 我們將實作一個協調的代理系統，包含：
# 1. **臨床研究助手**: 具備醫學知識的主要協調員
# 2. **資料分析員代理**: 專精於數值和統計分析
# 3. **邏輯驗證員代理**: 確保邏輯一致性
# 4. **決策制定員代理**: 最終蘊含分類
# 
# > 💡 **核心特色**: Agno的獨特之處在於其內建的記憶體系統和知識庫整合，讓代理能夠記住過往對話並利用領域知識進行更準確的推理。

# In[ ]:


# 🔧 Colab 環境設置 - 一鍵安裝 Agno (Phidata) 相關套件
# 這個cell會靜默安裝Agno全堆疊代理開發所需的套件
get_ipython().system('pip install -q phidata python-dotenv pandas tqdm')
get_ipython().system('pip install -q chromadb gdown  # 向量資料庫用於知識管理')
get_ipython().system('pip install -q google-generativeai')

print("✅ Agno (Phidata) 全堆疊代理環境安裝完成！可以開始建構具備記憶體和知識整合的代理系統了")


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


# ## Setup and Installation
# 
# First, let's set up our environment and import the necessary libraries:

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
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Phidata/Agno imports
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from phi.storage.agent.sqlite import SqlAgentStorage
from phi.knowledge.text import TextKnowledgeBase
from phi.vectordb.chroma import ChromaDb

print("✅ All libraries imported successfully")


# ## Data Loading and Utilities
# 
# Let's create utility functions for loading and processing clinical trial data:

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

def format_trial_data(trial_data: Dict[str, Any], focus_section: Optional[str] = None) -> str:
    """Format trial data for agent consumption.
    
    Args:
        trial_data: Clinical trial data dictionary
        focus_section: Optional section to focus on
        
    Returns:
        Formatted string containing trial information
    """
    if "error" in trial_data:
        return f"Error: {trial_data['error']}"
    
    # Extract key sections
    sections = {
        "Trial ID": trial_data.get("Clinical Trial ID", "Unknown"),
        "Eligibility": trial_data.get("Eligibility", []),
        "Intervention": trial_data.get("Intervention", []),
        "Results": trial_data.get("Results", []),
        "Adverse Events": trial_data.get("Adverse_Events", [])
    }
    
    # Format output
    formatted = [f"Clinical Trial: {sections['Trial ID']}"]
    
    # Focus on specific section if requested
    if focus_section and focus_section in ["Eligibility", "Intervention", "Results", "Adverse Events"]:
        section_data = sections[focus_section]
        formatted.append(f"\n{focus_section}:")
        if isinstance(section_data, list):
            for item in section_data:
                formatted.append(f"  - {item}")
        else:
            formatted.append(f"  {section_data}")
    else:
        # Include all sections
        for section_name, section_data in list(sections.items())[1:]:  # Skip Trial ID
            if section_data:
                formatted.append(f"\n{section_name}:")
                if isinstance(section_data, list):
                    for item in section_data[:5]:  # Limit to first 5 items for readability
                        formatted.append(f"  - {item}")
                    if len(section_data) > 5:
                        formatted.append(f"  ... ({len(section_data)-5} more items)")
                else:
                    formatted.append(f"  {section_data}")
    
    return "\n".join(formatted)

# Test utilities
sample_trial = load_clinical_trial("NCT00066573")
print(f"✅ Data utilities ready. Sample trial: {sample_trial.get('Clinical Trial ID', 'Error')}")


# ## Model and Storage Configuration
# 
# Set up the Gemini model and storage for agent memory:

# In[ ]:


# Model configuration with flexible API key support
import os

# Support both GEMINI_API_KEY and GOOGLE_API_KEY environment variables
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("⚠️ Please set GEMINI_API_KEY or GOOGLE_API_KEY environment variable")
    print("You can set it in Colab's 'Secrets' panel or use:")
    print("import os")
    print("os.environ['GEMINI_API_KEY'] = 'your-api-key'")
    print("or")
    print("os.environ['GOOGLE_API_KEY'] = 'your-api-key'")
    raise ValueError("Missing API key")
else:
    print(f"✅ Found API key: {api_key[:8]}...{api_key[-4:]}")

# Test API connection
try:
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    test_model = genai.GenerativeModel("gemini-2.5-flash")
    test_response = test_model.generate_content("Hello, respond with 'API test successful'")
    print(f"✅ API connection test successful: {test_response.text[:50]}...")
except Exception as e:
    print(f"❌ API connection test failed: {e}")
    raise

model = Gemini(
    id="gemini-2.5-flash",
    api_key=api_key,
    temperature=0.1  # Low temperature for consistent results
)

# Storage for agent memory (optional)
storage = SqlAgentStorage(
    table_name="clinical_trial_agents",
    db_file="agno_agents.db"
)

print("✅ Model and storage configured")


# ## 知識庫設置
# 
# 為RAG功能創建包含臨床試驗資訊的知識庫：
# 
# > 📚 **RAG說明**: 檢索增強生成(RAG)讓代理能夠從外部知識庫中檢索相關資訊，大幅提升回答的準確性和領域專業性。

# In[ ]:


# 創建包含臨床試驗概念的知識庫
clinical_knowledge = TextKnowledgeBase(
    sources=[
        # 添加臨床試驗領域知識
        """
        臨床試驗術語：
        
        蘊含(Entailment): 當陳述被試驗資料直接支持時，該陳述被資料蘊含。
        矛盾(Contradiction): 當陳述被試驗資料反駁時，該陳述與資料矛盾。
        
        試驗區段：
        - 適用條件(Eligibility): 參與者的納入和排除標準
        - 介入措施(Intervention): 使用的治療方法和程序
        - 結果(Results): 結果測量和統計發現
        - 不良事件(Adverse Events): 安全資料和報告的副作用
        
        統計術語：
        - 百分比(Percentage): 以百分之幾表示的比例
        - 信賴區間(Confidence Interval): 可能包含真實值的數值範圍
        - P值(P-value): 偶然觀察到結果的機率
        - 風險比(Hazard Ratio): 隨時間變化的相對風險測量
        
        醫學概念：
        - 療效(Efficacy): 治療在理想條件下的效果
        - 安全性(Safety): 治療無害作用
        - 主要終點(Primary Endpoint): 試驗的主要結果測量
        - 次要終點(Secondary Endpoint): 額外的結果測量
        """
    ],
    vector_db=ChromaDb(
        collection="clinical_knowledge",
        path="./agno_knowledge_db"
    )
)

# 載入知識庫
clinical_knowledge.load(recreate=False)  # 設為True以重新創建

print("✅ 知識庫配置完成")

# 🔍 知識庫說明：ChromaDB用於向量儲存，讓代理能夠快速檢索相關的醫學知識


# ## Agent Definitions
# 
# Now let's define our specialized agents using Agno's capabilities:

# In[ ]:


# 1. Clinical Research Assistant - Main coordinator with medical expertise
clinical_assistant = Agent(
    name="Clinical Research Assistant",
    model=model,
    storage=storage,
    knowledge=clinical_knowledge,
    description="Expert clinical researcher specializing in trial analysis and medical reasoning",
    instructions=[
        "You are a Clinical Research Assistant with deep expertise in clinical trials.",
        "Your primary role is to analyze clinical trial data and statements from a medical perspective.",
        "Focus on medical accuracy, clinical significance, and evidence-based reasoning.",
        "Consider the clinical context and medical plausibility of statements.",
        "Identify key medical concepts, terminology, and their implications.",
        "Always ground your analysis in the actual trial data provided.",
        "Provide clear medical reasoning for your assessments."
    ],
    show_tool_calls=False,
    markdown=True
)

print("✅ Clinical Research Assistant created")


# In[ ]:


# 2. Data Analyst Agent - Specialized in numerical and statistical analysis
data_analyst = Agent(
    name="Data Analyst",
    model=model,
    storage=storage,
    description="Statistical analyst specializing in clinical trial data analysis",
    instructions=[
        "You are a Data Analyst expert in statistical analysis of clinical trials.",
        "Your role is to analyze numerical claims, statistics, and quantitative relationships.",
        "Extract and verify all numerical values, percentages, and statistical measures.",
        "Perform calculations to validate numerical relationships and claims.",
        "Assess statistical significance and clinical meaningfulness of findings.",
        "Identify discrepancies between stated numbers and actual trial data.",
        "Be precise in your calculations and clearly show your work.",
        "Consider confidence intervals, error margins, and statistical uncertainty."
    ],
    show_tool_calls=False,
    markdown=True
)

print("✅ Data Analyst Agent created")


# In[ ]:


# 3. Logic Validator Agent - Ensures logical consistency
logic_validator = Agent(
    name="Logic Validator",
    model=model,
    storage=storage,
    description="Logic expert specializing in reasoning validation and consistency checking",
    instructions=[
        "You are a Logic Validator expert in logical reasoning and consistency.",
        "Your role is to validate the logical structure and coherence of claims.",
        "Analyze cause-and-effect relationships and logical implications.",
        "Check for internal consistency and logical contradictions.",
        "Evaluate the validity of inferences and conclusions.",
        "Identify logical fallacies or reasoning errors.",
        "Ensure that conclusions follow logically from premises.",
        "Focus on the logical soundness rather than medical or numerical details."
    ],
    show_tool_calls=False,
    markdown=True
)

print("✅ Logic Validator Agent created")


# In[ ]:


# 4. Decision Maker Agent - Final entailment classification
decision_maker = Agent(
    name="Decision Maker",
    model=model,
    storage=storage,
    description="Final decision authority for entailment classification",
    instructions=[
        "You are the Decision Maker responsible for final entailment classification.",
        "Your role is to synthesize analyses from all other agents and make the final decision.",
        "Consider medical, statistical, and logical evidence equally.",
        "Determine if a statement is ENTAILMENT (supported) or CONTRADICTION (refuted).",
        "ENTAILMENT: Statement is directly supported by the trial evidence.",
        "CONTRADICTION: Statement is refuted or contradicted by the trial evidence.",
        "Weigh evidence carefully and be conservative in your decisions.",
        "Provide clear reasoning for your final classification.",
        "Always end with: FINAL_DECISION: [Entailment/Contradiction]"
    ],
    show_tool_calls=False,
    markdown=True
)

print("✅ Decision Maker Agent created")


# ## Multi-Agent Analysis Pipeline
# 
# Create a coordinated pipeline that leverages all agents:

# In[ ]:


def agno_analysis_pipeline(statement: str, primary_id: str, secondary_id: Optional[str] = None, 
                          section_id: Optional[str] = None, verbose: bool = False) -> str:
    """
    Run the complete Agno multi-agent analysis pipeline.
    
    Args:
        statement: The natural language statement to analyze
        primary_id: Primary clinical trial ID
        secondary_id: Secondary trial ID for comparison statements
        section_id: Relevant section of the trial
        verbose: Whether to print intermediate results
        
    Returns:
        Final decision: 'Entailment' or 'Contradiction'
    """
    
    try:
        # Step 1: Load and format clinical trial data
        primary_data = load_clinical_trial(primary_id)
        secondary_data = None
        if secondary_id:
            secondary_data = load_clinical_trial(secondary_id)
        
        # Format trial data for analysis
        primary_formatted = format_trial_data(primary_data, section_id)
        secondary_formatted = None
        if secondary_data:
            secondary_formatted = format_trial_data(secondary_data, section_id)
        
        # Step 2: Create analysis prompt
        analysis_prompt = f"""
Analyze the following statement against the clinical trial evidence:

STATEMENT: "{statement}"

PRIMARY TRIAL EVIDENCE:
{primary_formatted}

{f'SECONDARY TRIAL EVIDENCE:\n{secondary_formatted}' if secondary_formatted else ''}

Please provide your expert analysis from your domain perspective.
        """.strip()
        
        if verbose:
            print(f"📄 Analyzing: {statement[:100]}...")
            print(f"🏥 Primary Trial: {primary_id}")
            if secondary_id:
                print(f"🏥 Secondary Trial: {secondary_id}")
        
        # Step 3: Clinical Research Assistant Analysis
        clinical_response = clinical_assistant.run(analysis_prompt)
        clinical_analysis = clinical_response.content
        
        if verbose:
            print(f"🩺 Clinical Analysis: Complete")
        
        # Step 4: Data Analyst Analysis
        data_response = data_analyst.run(analysis_prompt)
        data_analysis = data_response.content
        
        if verbose:
            print(f"📊 Data Analysis: Complete")
        
        # Step 5: Logic Validator Analysis
        logic_response = logic_validator.run(analysis_prompt)
        logic_analysis = logic_response.content
        
        if verbose:
            print(f"🧠 Logic Validation: Complete")
        
        # Step 6: Decision Making
        decision_prompt = f"""
Based on the following expert analyses, make the final entailment decision:

ORIGINAL STATEMENT: "{statement}"

CLINICAL RESEARCH ASSISTANT ANALYSIS:
{clinical_analysis}

DATA ANALYST ANALYSIS:
{data_analysis}

LOGIC VALIDATOR ANALYSIS:
{logic_analysis}

Synthesize these analyses and provide your final decision: Entailment or Contradiction?
        """.strip()
        
        decision_response = decision_maker.run(decision_prompt)
        final_analysis = decision_response.content
        
        # Step 7: Extract final decision
        if "FINAL_DECISION: Entailment" in final_analysis:
            decision = "Entailment"
        elif "FINAL_DECISION: Contradiction" in final_analysis:
            decision = "Contradiction"
        else:
            # Fallback parsing
            if "entailment" in final_analysis.lower() and "contradiction" not in final_analysis.lower():
                decision = "Entailment"
            else:
                decision = "Contradiction"
        
        if verbose:
            print(f"⚖️ Final Decision: {decision}")
            print("-" * 50)
        
        return decision
        
    except Exception as e:
        if verbose:
            print(f"❌ Error in Agno pipeline: {e}")
        return "Contradiction"  # Conservative fallback

print("✅ Agno analysis pipeline ready")


# ## Test Example
# 
# Let's test our Agno system with a sample case:

# In[ ]:


# Test with a sample statement
test_statement = "there is a 13.2% difference between the results from the two the primary trial cohorts"
test_primary_id = "NCT00066573"

print(f"Testing Agno system with statement:")
print(f"'{test_statement}'")
print(f"Primary trial: {test_primary_id}")
print("\n" + "="*80)

# Run the analysis with verbose output
result = agno_analysis_pipeline(
    statement=test_statement,
    primary_id=test_primary_id,
    section_id="Results",
    verbose=True
)

print(f"\n🎯 AGNO RESULT: {result}")
print("="*80)


# ## Evaluation on Training Data
# 
# Let's evaluate our Agno system on training data:

# In[ ]:


# Load training data
train_data = load_dataset("training_data/train.json")
print(f"Loaded {len(train_data)} training examples")

# Evaluate on a sample (adjust sample_size as needed)
sample_size = 25
examples = list(train_data.items())[:sample_size]

print(f"\nEvaluating Agno system on {len(examples)} examples...")

results = []
correct = 0

for i, (uuid, example) in enumerate(tqdm(examples, desc="Agno Processing")):
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
        
        # Get prediction from Agno system
        predicted = agno_analysis_pipeline(
            statement=statement,
            primary_id=primary_id,
            secondary_id=secondary_id,
            section_id=section_id,
            verbose=False
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
        
        status = "✅" if is_correct else "❌"
        print(f"Example {i+1:2d}: {expected:12} -> {predicted:12} {status}")
        
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
print(f"\n📊 Agno Results:")
print(f"Accuracy: {accuracy:.2%} ({correct}/{len(examples)})")

# Store results for comparison
agno_results = results.copy()


# ## Memory and Knowledge Analysis
# 
# Let's analyze how Agno's memory and knowledge features work:

# In[ ]:


# Demonstrate memory capabilities
print("🧠 Testing Agno Memory Capabilities:")
print("=" * 50)

# Test conversation memory
memory_test_1 = clinical_assistant.run("Remember that NCT00066573 is about breast cancer treatment comparing exemestane and anastrozole.")
print(f"Memory setup: {memory_test_1.content[:100]}...")

memory_test_2 = clinical_assistant.run("What trial were we just discussing?")
print(f"Memory recall: {memory_test_2.content[:200]}...")

# Test knowledge base usage
print("\n📚 Testing Knowledge Base Integration:")
print("=" * 50)

knowledge_test = clinical_assistant.run("What is the difference between entailment and contradiction in clinical trial analysis?")
print(f"Knowledge response: {knowledge_test.content[:300]}...")

print("\n✅ Memory and knowledge features demonstrated")


# ## Generate Submission File
# 
# Let's generate a submission file using our Agno system:

# In[ ]:


def generate_agno_submission(test_file="test.json", output_file="agno_submission.json", sample_size=None):
    """
    Generate submission file using Agno system.
    
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
        
    print(f"🚀 Generating Agno predictions for {len(examples)} examples...")
    
    submission = {}
    
    for i, (uuid, example) in enumerate(tqdm(examples, desc="Agno Processing")):
        try:
            statement = example.get("Statement")
            primary_id = example.get("Primary_id")
            secondary_id = example.get("Secondary_id")
            section_id = example.get("Section_id")
            
            if not statement or not primary_id:
                submission[uuid] = {"Prediction": "Contradiction"}  # Default fallback
                continue
                
            # Get prediction from Agno system
            prediction = agno_analysis_pipeline(
                statement=statement,
                primary_id=primary_id,
                secondary_id=secondary_id,
                section_id=section_id,
                verbose=False
            )
            
            submission[uuid] = {"Prediction": prediction}
            
        except Exception as e:
            print(f"Error processing {uuid}: {e}")
            submission[uuid] = {"Prediction": "Contradiction"}  # Conservative fallback
    
    # Save submission file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(submission, f, indent=2)
    
    print(f"✅ Agno submission saved to {output_file}")
    return submission

# Generate submission for a small sample
agno_submission = generate_agno_submission(
    test_file="test.json", 
    output_file="agno_submission.json",
    sample_size=10  # Adjust as needed
)

print(f"Generated predictions for {len(agno_submission)} examples")


# ## Error Analysis
# 
# Let's analyze errors to understand system performance:

# In[ ]:


# Analyze incorrect predictions
incorrect_results = [r for r in agno_results if not r["correct"] and r["predicted"] not in ["SKIPPED", "ERROR"]]

print(f"\n🔍 Agno Error Analysis ({len(incorrect_results)} incorrect predictions):")
print("=" * 80)

# Group errors by type
entailment_to_contradiction = [r for r in incorrect_results if r["expected"] == "Entailment" and r["predicted"] == "Contradiction"]
contradiction_to_entailment = [r for r in incorrect_results if r["expected"] == "Contradiction" and r["predicted"] == "Entailment"]

print(f"Entailment -> Contradiction errors: {len(entailment_to_contradiction)}")
print(f"Contradiction -> Entailment errors: {len(contradiction_to_entailment)}")

# Show some examples
print("\nSample errors:")
for i, result in enumerate(incorrect_results[:3]):
    print(f"\nError #{i+1}:")
    print(f"Statement: {result['statement']}")
    print(f"Expected: {result['expected']} | Predicted: {result['predicted']}")
    print("-" * 40)


# ## 結論與洞察
# 
# ### Agno 框架優勢：
# 1. **全堆疊能力**: 完整的代理開發環境
# 2. **內建記憶體**: 持久的對話上下文和學習
# 3. **知識整合**: 具備向量儲存的RAG功能
# 4. **高效能**: 針對生產工作負載最佳化
# 5. **豐富工具**: 廣泛的內建工具和整合
# 6. **企業就緒**: 專為大規模應用設計
# 
# ### 關鍵功能展示：
# - **多代理協調**: 臨床助手、資料分析員、邏輯驗證員、決策制定員
# - **知識庫**: 用於RAG的領域特定臨床試驗知識
# - **記憶體持久性**: 對話歷史和上下文保留
# - **儲存整合**: 用於代理狀態管理的SQLite儲存
# - **結構化分析**: 複雜推理任務的系統化方法
# 
# ### 架構優勢：
# - **可擴展設計**: 易於添加新代理和功能
# - **知識管理**: 內建RAG用於領域專業知識
# - **記憶體持續性**: 跨對話的持久上下文
# - **生產就緒**: 企業級功能和效能
# - **完整工具**: 代理開發的豐富生態系統
# 
# ### 優化機會：
# 1. **知識庫擴展**: 添加更多臨床試驗領域知識
# 2. **記憶體最佳化**: 微調記憶體保留和檢索
# 3. **代理專業化**: 增強每個代理的領域專業知識
# 4. **工具整合**: 利用額外的Agno工具和功能
# 5. **效能調整**: 速度與準確性權衡的最佳化
# 
# ### 何時使用 Agno：
# - 需要持久記憶體和學習的應用
# - 需要知識庫整合(RAG)的系統
# - 具有複雜代理工作流程的企業應用
# - 需要完整工具和基礎設施的專案
# - 代理狀態持久性很重要的使用案例
# 
# ### Agno vs 其他框架：
# - **vs AutoGen**: 記憶體/知識更佳，AutoGen對話更佳
# - **vs Atomic Agents**: 功能更多但較重，Atomic純速度更佳
# - **vs LangChain**: 更專注於代理，LangChain生態系統更廣
# 
# ## 🎓 學習重點總結
# - **記憶體系統**: 代理能夠記住並學習過往對話
# - **RAG整合**: 利用外部知識提升回答品質
# - **多代理協作**: 不同專業代理的協調工作
# - **持久化設計**: 狀態和記憶體的長期保存
# 
# Agno在需要持久記憶體、知識整合和企業級代理管理的場景中表現出色，使其成為需要上下文保留和領域專業知識的複雜臨床分析應用的理想選擇。
