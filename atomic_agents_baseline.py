#!/usr/bin/env python
# coding: utf-8

# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/0x-yuan/clintrial-nlp/blob/main/atomic_agents_baseline.ipynb)
# 
# # Atomic Agents 框架基線 - 臨床試驗 NLP
# 
# ## 概述
# 
# 本notebook展示如何使用Atomic Agents框架建構一個輕量級、高效能的多代理系統，用於臨床試驗自然語言推理(NLI)。Atomic Agents專為生產環境設計，具有極快的啟動時間(~3μs)和模組化架構。
# 
# ## 📚 學習目標
# 完成本教學後，您將學會：
# - 理解 Atomic Agents 框架的核心概念
# - 建立專門的醫療、數值和邏輯分析代理
# - 實作多代理協作pipeline
# - 評估和改進系統效能
# 
# ### 為什麼選擇 Atomic Agents？
# - **超輕量級**: 最小化開銷和快速執行
# - **高度模組化**: 易於組合和修改代理
# - **生產就緒**: 專為實際部署而設計
# - **簡單API**: 直觀的代理創建和協調
# - **記憶體管理**: 內建的上下文和記憶體處理
# 
# ### 🏗️ 代理架構
# 遵循教學圖表，我們實作一個結構化的pipeline：
# 1. **醫療專家代理**: 分析醫學術語和概念
# 2. **數值分析代理**: 處理量化數據
# 3. **邏輯檢查代理**: 驗證邏輯關係
# 4. **聚合代理**: 結合見解做出最終決策
# 5. **監督協調**: 管理整體工作流程
# 
# > 💡 **重要概念**: Atomic Agents 的"atomic"指的是每個代理都是一個獨立、最小的功能單元，可以輕鬆組合成複雜系統。

# In[ ]:


# 🔧 Colab 環境設置 - 一鍵安裝所需套件
# 這個cell會靜默安裝所有必要的Python套件，讓您可以在Colab中直接運行此notebook
get_ipython().system('pip install -q atomic-agents python-dotenv pandas tqdm')
get_ipython().system('pip install -q google-generativeai gdown')

print("✅ 所有套件安裝完成！可以開始使用 Atomic Agents 框架了")


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
# 如果沒有 test.json，從訓練資料創建一個測試集
import json

def create_test_data_if_needed():
    """如果不存在 test.json，從 train.json 創建一個小的測試集"""
    if not os.path.exists("test.json"):
        try:
            # 載入訓練資料
            with open("training_data/train.json", "r", encoding="utf-8") as f:
                train_data = json.load(f)
            
            # 取前100個樣本作為測試資料
            test_data = dict(list(train_data.items())[:100])
            
            # 儲存測試資料
            with open("test.json", "w", encoding="utf-8") as f:
                json.dump(test_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ 已創建測試資料集，包含 {len(test_data)} 個樣本")
        except Exception as e:
            print(f"❌ 創建測試資料失敗: {e}")
    else:
        print("✅ test.json 已存在")

# 執行測試資料準備
create_test_data_if_needed()


# ## 環境設置和安裝
# 
# 首先，讓我們設置環境並匯入必要的函式庫：
# 
# > 📝 **說明**: 在這個步驟中，我們會載入環境變數並確認所有必要的套件都已正確安裝。

# In[ ]:


# 載入環境變數
# 這個步驟會從 .env 檔案載入 API 金鑰等敏感資訊
from dotenv import load_dotenv
import os

load_dotenv()
print("✅ 環境變數載入完成")


# In[ ]:


# 匯入必要的函式庫
# 這些函式庫提供了資料處理、AI模型和Atomic Agents框架的功能
import json
import pandas as pd
from tqdm import tqdm  # 進度條顯示
import google.generativeai as genai  # Google Gemini API
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')  # 隱藏警告訊息

# Atomic Agents 核心組件
from atomic_agents.lib.components.agent_memory import AgentMemory  # 代理記憶體管理
from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig, BaseAgentInputSchema  # 基礎代理類別
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator  # 系統提示生成器

print("✅ 所有函式庫匯入成功")


# ## 資料載入和工具函式
# 
# 讓我們建立用於載入和處理臨床試驗資料的工具函式：
# 
# > 🔧 **功能說明**: 這些函式負責從JSON檔案中載入臨床試驗資料，並將其轉換為適合AI代理分析的格式。

# In[ ]:


def load_clinical_trial(trial_id: str) -> Dict[str, Any]:
    """載入臨床試驗資料從JSON檔案。
    
    Args:
        trial_id: 臨床試驗的NCT識別碼
        
    Returns:
        包含試驗資料的字典或錯誤資訊
    """
    try:
        # 確保使用下載的資料夾路徑
        file_path = os.path.join("training_data", "CT json", f"{trial_id}.json")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        return {"error": f"找不到臨床試驗 {trial_id}"}
    except Exception as e:
        return {"error": f"載入 {trial_id} 時發生錯誤: {str(e)}"}

def load_dataset(filepath: str) -> Dict[str, Any]:
    """載入訓練或測試資料集。
    
    Args:
        filepath: JSON資料集檔案的路徑（會自動檢查是否需要添加 training_data/ 前綴）
        
    Returns:
        包含資料集的字典
    """
    try:
        # 如果路徑不是絕對路徑且不包含 training_data，自動添加前綴
        if not os.path.isabs(filepath) and not filepath.startswith("training_data/"):
            # 對於標準檔案名，使用 training_data 前綴
            if filepath in ["train.json", "dev.json"]:
                filepath = os.path.join("training_data", filepath)
        
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"載入資料集時發生錯誤: {e}")
        return {}

def extract_relevant_sections(trial_data: Dict[str, Any], section_id: str) -> str:
    """根據section_id從試驗資料中提取相關部分。
    
    Args:
        trial_data: 臨床試驗資料字典
        section_id: 目標區段 (Eligibility, Intervention, Results, Adverse Events)
        
    Returns:
        包含相關區段資料的格式化字串
    """
    if "error" in trial_data:
        return f"錯誤: {trial_data['error']}"
    
    sections = {
        "Eligibility": trial_data.get("Eligibility", []),
        "Intervention": trial_data.get("Intervention", []),
        "Results": trial_data.get("Results", []),
        "Adverse Events": trial_data.get("Adverse_Events", [])
    }
    
    # 如果請求特定區段，則僅返回該區段
    if section_id in sections:
        section_data = sections[section_id]
        if isinstance(section_data, list):
            return "\n".join(str(item) for item in section_data)
        return str(section_data)
    
    # 否則返回所有區段
    result = []
    for section_name, section_data in sections.items():
        if section_data:
            result.append(f"{section_name}:")
            if isinstance(section_data, list):
                result.extend([f"  {item}" for item in section_data])
            else:
                result.append(f"  {section_data}")
    
    return "\n".join(result)

# 測試工具函式
sample_trial = load_clinical_trial("NCT00066573")
print(f"✅ 資料工具函式準備就緒。範例試驗: {sample_trial.get('Clinical Trial ID', '錯誤')}")

# 📋 函式說明：
# - load_clinical_trial(): 載入單一臨床試驗的完整資料
# - load_dataset(): 載入包含多個試驗的訓練/測試資料集（自動處理路徑）
# - extract_relevant_sections(): 提取試驗中特定區段的資料


# ## 模型配置
# 
# 配置Google Gemini模型：
# 
# > 🤖 **技術說明**: 我們使用Google Gemini 2.5 Flash模型，這是一個高效能且成本效益高的大型語言模型，特別適合多代理系統。

# In[ ]:


# 配置 Google Gemini 模型
# 設置 API 金鑰和模型參數

# 支援多種環境變數名稱
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("⚠️ 請設定 GEMINI_API_KEY 或 GOOGLE_API_KEY 環境變數")
    print("可以在 Colab 左側面板的 'Secrets' 中設定，或使用以下方式:")
    print("import os")
    print("os.environ['GEMINI_API_KEY'] = '您的API金鑰'")
    print("或")
    print("os.environ['GOOGLE_API_KEY'] = '您的API金鑰'")
    raise ValueError("缺少 API 金鑰")
else:
    print(f"✅ 找到 API 金鑰: {api_key[:8]}...{api_key[-4:]}")

genai.configure(api_key=api_key)

# 測試 API 連接
try:
    test_model = genai.GenerativeModel("gemini-2.5-flash")
    test_response = test_model.generate_content("Hello, respond with 'API test successful'")
    print(f"✅ API 連接測試成功: {test_response.text[:50]}...")
except Exception as e:
    print(f"❌ API 連接測試失敗: {e}")
    raise

# 創建 Gemini 模型實例
model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    generation_config=genai.types.GenerationConfig(
        temperature=0.1,  # 低溫度確保一致的結果
        max_output_tokens=4096,
        top_p=1,
        top_k=1
    )
)

# 模型配置
MODEL_NAME = "gemini-2.5-flash"

print(f"✅ Google Gemini模型配置完成，使用模型: {MODEL_NAME}")

# 💡 模型說明：Gemini 2.5 Flash是Google最新的高效能模型，提供優秀的推理能力和快速回應


# ## 代理定義
# 
# 現在讓我們使用Atomic Agents框架定義每個專門的代理。每個代理都有特定的角色和專業領域：
# 
# > 🎯 **設計原則**: 每個代理都專注於特定領域的分析，透過分工合作來提高整體分析品質。這是多代理系統的核心優勢。

# In[ ]:


# 1. 醫療專家代理
# 這個代理專門負責從醫學角度分析陳述的準確性

# 創建適用於Gemini的分析函數
def analyze_with_gemini(prompt: str, context: str) -> str:
    """使用Gemini模型進行分析"""
    full_prompt = f"{prompt}\n\nContext:\n{context}"
    try:
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

def medical_expert_analysis(context: str) -> str:
    """醫療專家代理分析函數"""
    prompt = """You are a Medical Expert Agent specializing in clinical trial analysis.
You have deep knowledge of medical terminology, clinical concepts, and trial procedures.
Your role is to analyze statements from a medical perspective and identify relevant clinical insights.

Steps:
1. Analyze the medical terminology and concepts in the statement
2. Identify key clinical elements and their significance
3. Review the clinical trial data for medical accuracy
4. Assess whether the medical claims align with the trial evidence
5. Provide medical reasoning and clinical context

Provide a clear medical analysis focusing on:
- Medical terminology accuracy
- Clinical relevance and significance
- Alignment with medical evidence in the trial
- Any medical concerns or considerations

End with: MEDICAL_ASSESSMENT: [SUPPORTS/CONTRADICTS/UNCLEAR] based on medical evidence"""
    
    return analyze_with_gemini(prompt, context)

print("✅ 醫療專家代理建立完成")

# 🩺 代理說明：醫療專家代理專門分析醫學術語的準確性和臨床相關性


# In[ ]:


# 2. 數值分析代理
def numerical_analyzer_analysis(context: str) -> str:
    """數值分析代理分析函數"""
    prompt = """You are a Numerical Analyzer Agent specializing in quantitative analysis of clinical trials.
You excel at processing numbers, statistics, percentages, and numerical relationships.
Your role is to verify numerical claims and perform statistical analysis.

Steps:
1. Extract all numerical values, percentages, and statistics from the statement
2. Identify corresponding numbers in the clinical trial data
3. Perform calculations to verify numerical relationships
4. Check for statistical significance and clinical meaningfulness
5. Identify any numerical inconsistencies or errors

Provide a detailed numerical analysis including:
- All numerical values extracted from the statement
- Corresponding values found in trial data
- Calculations performed to verify claims
- Assessment of numerical accuracy

End with: NUMERICAL_ASSESSMENT: [ACCURATE/INACCURATE/PARTIALLY_ACCURATE] with confidence level"""
    
    return analyze_with_gemini(prompt, context)

print("✅ 數值分析代理建立完成")


# In[ ]:


# 3. 邏輯檢查代理
def logic_checker_analysis(context: str) -> str:
    """邏輯檢查代理分析函數"""
    prompt = """You are a Logic Checker Agent responsible for validating logical reasoning and consistency.
You specialize in identifying logical relationships, contradictions, and reasoning patterns.
Your role is to ensure logical soundness and coherence in claims and evidence.

Steps:
1. Analyze the logical structure of the statement
2. Identify cause-and-effect relationships and implications
3. Check for internal consistency and coherence
4. Evaluate the validity of inferences and conclusions
5. Detect any logical fallacies or contradictions

Provide a logical analysis focusing on:
- Logical structure and reasoning patterns
- Consistency between claims and evidence
- Validity of inferences and implications
- Any logical issues or contradictions found

End with: LOGICAL_ASSESSMENT: [VALID/INVALID/QUESTIONABLE] with reasoning"""
    
    return analyze_with_gemini(prompt, context)

print("✅ 邏輯檢查代理建立完成")


# In[ ]:


# 4. 聚合代理
def aggregator_analysis(medical_analysis: str, numerical_analysis: str, logic_analysis: str, statement: str) -> str:
    """聚合代理分析函數"""
    prompt = f"""You are the Aggregator Agent responsible for making final entailment decisions.
You synthesize analyses from the Medical Expert, Numerical Analyzer, and Logic Checker.
Your role is to weigh different evidence types and make the final classification decision.

ORIGINAL STATEMENT: "{statement}"

MEDICAL EXPERT ANALYSIS:
{medical_analysis}

NUMERICAL ANALYZER ANALYSIS:
{numerical_analysis}

LOGIC CHECKER ANALYSIS:
{logic_analysis}

Steps:
1. Review the medical assessment for clinical accuracy
2. Consider the numerical analysis for quantitative validity
3. Evaluate the logical assessment for reasoning soundness
4. Weigh all evidence types appropriately
5. Make the final Entailment vs Contradiction decision

Based on all specialist analyses, determine if the statement is:
- ENTAILMENT: Statement is directly supported by the trial data
- CONTRADICTION: Statement is refuted by the trial data

Provide brief reasoning and then output exactly one of:
FINAL_DECISION: Entailment
FINAL_DECISION: Contradiction"""
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

print("✅ 聚合代理建立完成")


# ## 多代理分析管道
# 
# 現在讓我們創建遵循教學圖表架構、協調所有代理的結構化管道：
# 
# > ⚙️ **工作流程說明**: 這個管道將按順序執行每個專業代理，最終由聚合代理做出決策。每個步驟都會產生專業的分析結果。

# In[ ]:


def atomic_agents_pipeline(statement: str, primary_id: str, secondary_id: Optional[str] = None, 
                          section_id: Optional[str] = None, verbose: bool = False) -> str:
    """
    Run the complete Atomic Agents analysis pipeline using Google Gemini.
    
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
        # Step 1: Load clinical trial data
        primary_data = load_clinical_trial(primary_id)
        secondary_data = None
        if secondary_id:
            secondary_data = load_clinical_trial(secondary_id)
        
        # Step 2: Extract relevant sections
        primary_sections = extract_relevant_sections(primary_data, section_id or "All")
        secondary_sections = None
        if secondary_data:
            secondary_sections = extract_relevant_sections(secondary_data, section_id or "All")
        
        # Step 3: Prepare input context for agents
        input_context = f"""
STATEMENT TO ANALYZE: "{statement}"

PRIMARY TRIAL ({primary_id}):
{primary_sections}

{f'SECONDARY TRIAL ({secondary_id}):\n{secondary_sections}' if secondary_sections else ''}

TASK: Analyze this statement against the clinical trial evidence.
        """.strip()
        
        if verbose:
            print(f"📄 Analyzing: {statement[:100]}...")
            print(f"🏥 Primary Trial: {primary_id}")
            if secondary_id:
                print(f"🏥 Secondary Trial: {secondary_id}")
        
        # Step 4: Medical Expert Analysis
        medical_analysis = medical_expert_analysis(input_context)
        
        if verbose:
            print(f"🩺 Medical Expert: {medical_analysis.split('MEDICAL_ASSESSMENT:')[-1].strip() if 'MEDICAL_ASSESSMENT:' in medical_analysis else 'Analysis complete'}")
        
        # Step 5: Numerical Analyzer Analysis
        numerical_analysis = numerical_analyzer_analysis(input_context)
        
        if verbose:
            print(f"🔢 Numerical Analyzer: {numerical_analysis.split('NUMERICAL_ASSESSMENT:')[-1].strip() if 'NUMERICAL_ASSESSMENT:' in numerical_analysis else 'Analysis complete'}")
        
        # Step 6: Logic Checker Analysis
        logic_analysis = logic_checker_analysis(input_context)
        
        if verbose:
            print(f"🧠 Logic Checker: {logic_analysis.split('LOGICAL_ASSESSMENT:')[-1].strip() if 'LOGICAL_ASSESSMENT:' in logic_analysis else 'Analysis complete'}")
        
        # Step 7: Aggregator Decision
        final_analysis = aggregator_analysis(medical_analysis, numerical_analysis, logic_analysis, statement)
        
        # Step 8: Extract final decision
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
            print(f"❌ Error in pipeline: {e}")
        return "Contradiction"  # Conservative fallback

print("✅ Atomic Agents pipeline ready (using Google Gemini)")


# ## 測試範例
# 
# 讓我們測試改進的Atomic Agents系統：
# 
# > 🧪 **測試說明**: 這個測試將展示完整的多代理分析流程，您可以看到每個代理如何協作處理臨床試驗陳述。

# In[ ]:


# Test with a sample statement
test_statement = "there is a 13.2% difference between the results from the two the primary trial cohorts"
test_primary_id = "NCT00066573"

print(f"Testing Atomic Agents with statement:")
print(f"'{test_statement}'")
print(f"Primary trial: {test_primary_id}")
print("\n" + "="*80)

# Run the analysis with verbose output
result = atomic_agents_pipeline(
    statement=test_statement,
    primary_id=test_primary_id,
    section_id="Results",
    verbose=True
)

print(f"\n🎯 ATOMIC AGENTS RESULT: {result}")
print("="*80)


# ## 在訓練資料上的評估
# 
# 讓我們在訓練資料上評估改進的Atomic Agents系統：
# 
# > 📊 **評估說明**: 這個部分將測試我們的多代理系統在實際資料上的表現，並計算準確率等關鍵指標。

# In[ ]:


# Load training data
train_data = load_dataset("training_data/train.json")
print(f"Loaded {len(train_data)} training examples")

# Evaluate on a sample (adjust sample_size as needed)
sample_size = 30
examples = list(train_data.items())[:sample_size]

print(f"\nEvaluating Atomic Agents on {len(examples)} examples...")

results = []
correct = 0

for i, (uuid, example) in enumerate(tqdm(examples, desc="Atomic Agents")):
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
        
        # Get prediction from Atomic Agents pipeline
        predicted = atomic_agents_pipeline(
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
print(f"\n📊 Atomic Agents Results:")
print(f"Accuracy: {accuracy:.2%} ({correct}/{len(examples)})")

# Store results for later comparison
atomic_agents_results = results.copy()


# ## Error Analysis
# 
# Let's analyze the errors to understand areas for improvement:

# In[ ]:


# Analyze incorrect predictions
incorrect_results = [r for r in results if not r["correct"] and r["predicted"] not in ["SKIPPED", "ERROR"]]

print(f"\n🔍 Error Analysis ({len(incorrect_results)} incorrect predictions):")
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


# ## Generate Submission File
# 
# Let's generate a submission file using our Atomic Agents system:

# In[ ]:


def generate_atomic_agents_submission(test_file="test.json", output_file="atomic_agents_submission.json", sample_size=None):
    """
    Generate submission file using Atomic Agents system.
    
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
        
    print(f"🚀 Generating Atomic Agents predictions for {len(examples)} examples...")
    
    submission = {}
    
    for i, (uuid, example) in enumerate(tqdm(examples, desc="Atomic Agents Processing")):
        try:
            statement = example.get("Statement")
            primary_id = example.get("Primary_id")
            secondary_id = example.get("Secondary_id")
            section_id = example.get("Section_id")
            
            if not statement or not primary_id:
                submission[uuid] = {"Prediction": "Contradiction"}  # Default fallback
                continue
                
            # Get prediction from Atomic Agents system
            prediction = atomic_agents_pipeline(
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
    
    print(f"✅ Atomic Agents submission saved to {output_file}")
    return submission

# Generate submission for a small sample
atomic_submission = generate_atomic_agents_submission(
    test_file="test.json", 
    output_file="atomic_agents_submission.json",
    sample_size=10  # Adjust as needed
)

print(f"Generated predictions for {len(atomic_submission)} examples")


# ## Performance Analysis and Comparison
# 
# Let's analyze the performance of our improved Atomic Agents implementation:

# In[ ]:


import time
from collections import Counter

# Performance metrics
def analyze_performance(results, framework_name):
    """
    Analyze performance metrics for a framework.
    """
    valid_results = [r for r in results if r["predicted"] not in ["SKIPPED", "ERROR"]]
    
    if not valid_results:
        print(f"No valid results for {framework_name}")
        return
    
    # Basic metrics
    total = len(valid_results)
    correct = sum(1 for r in valid_results if r["correct"])
    accuracy = correct / total if total > 0 else 0
    
    # Confusion matrix
    true_pos = sum(1 for r in valid_results if r["expected"] == "Entailment" and r["predicted"] == "Entailment")
    true_neg = sum(1 for r in valid_results if r["expected"] == "Contradiction" and r["predicted"] == "Contradiction")
    false_pos = sum(1 for r in valid_results if r["expected"] == "Contradiction" and r["predicted"] == "Entailment")
    false_neg = sum(1 for r in valid_results if r["expected"] == "Entailment" and r["predicted"] == "Contradiction")
    
    # Calculate metrics
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n📊 {framework_name} Performance Analysis:")
    print(f"{'='*50}")
    print(f"Total examples: {total}")
    print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-Score: {f1_score:.3f}")
    print(f"\nConfusion Matrix:")
    print(f"True Positives (Entailment): {true_pos}")
    print(f"True Negatives (Contradiction): {true_neg}")
    print(f"False Positives: {false_pos}")
    print(f"False Negatives: {false_neg}")
    
    # Label distribution
    expected_dist = Counter(r["expected"] for r in valid_results)
    predicted_dist = Counter(r["predicted"] for r in valid_results)
    
    print(f"\nLabel Distribution:")
    print(f"Expected - Entailment: {expected_dist.get('Entailment', 0)}, Contradiction: {expected_dist.get('Contradiction', 0)}")
    print(f"Predicted - Entailment: {predicted_dist.get('Entailment', 0)}, Contradiction: {predicted_dist.get('Contradiction', 0)}")
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }

# Analyze current results
atomic_metrics = analyze_performance(atomic_agents_results, "Atomic Agents")


# ## 結論與洞察
# 
# ### Atomic Agents 框架優勢：
# 1. **超輕量級**: 最小開銷和極快啟動時間（~3μs）
# 2. **模組化架構**: 易於組合和修改代理
# 3. **生產就緒**: 專為實際部署場景而建構
# 4. **記憶體管理**: 內建的上下文和記憶體處理
# 5. **簡單API**: 直觀的代理創建和協調
# 6. **模型彈性**: 支援多種LLM後端（本示例使用Google Gemini 2.5 Flash）
# 
# ### 關鍵改進：
# - **結構化管道**: 代理間職責清晰分工
# - **專業角色**: 醫療專家、數值分析員、邏輯檢查員、聚合員
# - **更好的協調**: 系統化的資訊流動
# - **錯誤處理**: 強健的回退機制
# - **區段提取**: 針對相關試驗區段的精準分析
# - **Gemini整合**: 使用Google最新的高效能語言模型
# 
# ### 架構優勢：
# - **快速執行**: 高吞吐量場景的最小開銷
# - **易於除錯**: 清晰的代理邊界便於排錯
# - **可擴展設計**: 簡單添加新的專業代理
# - **記憶體效率**: 輕量級代理實例
# - **生產部署**: 實際應用準備就緒
# - **模型無關**: 易於切換不同的LLM提供商
# 
# ### 優化機會：
# 1. **提示工程**: 微調個別代理提示以適應Gemini
# 2. **代理協調**: 改善代理間資訊傳遞
# 3. **錯誤分析**: 利用失敗案例改進代理推理
# 4. **效能調整**: 速度與準確性權衡最佳化
# 5. **領域知識**: 增強代理的醫療專業知識
# 
# ### 何時使用 Atomic Agents：
# - 高效能生產環境
# - 需要快速啟動和執行的應用
# - 重視簡潔性和模組化的場景
# - 需要輕量級代理協調的系統
# - 優先考慮部署效率的專案
# - 希望靈活切換LLM提供商的系統
# 
# ## 🎓 學習重點總結
# - **原子性概念**: 每個代理都是獨立的最小功能單元
# - **分工合作**: 不同專業領域的代理協同工作
# - **管道設計**: 結構化的資料流和決策流程
# - **實際應用**: 臨床試驗NLP的實戰案例
# - **模型整合**: 如何將Atomic Agents與Google Gemini整合
# 
# Atomic Agents 在效能、簡潔性和模組化之間提供了絕佳平衡，結合Google Gemini 2.5 Flash的強大能力，使其成為需要速度、可靠性和成本效益的生產臨床NLP應用的理想選擇。
