#!/usr/bin/env python
# coding: utf-8

# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/0x-yuan/clintrial-nlp/blob/main/langchain_baseline.ipynb)
# 
# # LangChain/LangGraph 框架基線 - 臨床試驗 NLP
# 
# ## 概述
# 
# 本notebook展示如何使用LangChain和LangGraph建構一個有狀態的、基於圖的代理系統，用於臨床試驗自然語言推理(NLI)。LangChain提供了最成熟和功能豐富的LLM應用生態系統。
# 
# ## 📚 學習目標
# 完成本教學後，您將學會：
# - 理解 LangChain 生態系統和 LangGraph 的狀態管理
# - 建構基於圖的工作流程和狀態管理
# - 實作複雜的多步驟推理流程
# - 使用 SQLite 檢查點進行對話持續性
# 
# ### 為什麼選擇 LangChain/LangGraph？
# - **成熟生態系統**: 最完善的LLM應用框架
# - **豐富整合**: 廣泛的工具和服務整合
# - **有狀態工作流程**: LangGraph支援複雜的有狀態代理互動
# - **進階模式**: 支援複雜的推理和決策模式  
# - **社群支援**: 龐大社群和豐富文檔
# - **生產就緒**: 在眾多實際應用中經過實戰驗證
# 
# ### 🔄 使用 LangGraph 的代理架構
# 我們將實作一個基於圖的工作流程，包含有狀態代理：
# 1. **臨床資料提取器**: 處理和結構化試驗資料
# 2. **醫療分析節點**: 專業醫療推理
# 3. **統計分析節點**: 數值和統計驗證
# 4. **邏輯驗證節點**: 邏輯一致性檢查
# 5. **決策綜合節點**: 最終蘊含分類
# 6. **狀態管理**: 整個分析工作流程的持久狀態
# 
# > 🔗 **關鍵概念**: LangGraph 將工作流程視為有向圖，其中每個節點都是一個功能，邊代表資料流動。這使得複雜的多步驟推理變得可管理和可追蹤。

# In[ ]:


# 🔧 Colab 環境設置 - 一鍵安裝 LangChain/LangGraph 相關套件
# 這個cell會靜默安裝所有LangChain生態系統所需的套件
get_ipython().system('pip install -q langchain langchain-google-genai langgraph python-dotenv pandas tqdm')
get_ipython().system('pip install -q langchain-core langchain-community gdown')

print("✅ LangChain/LangGraph 生態系統安裝完成！可以開始建構有狀態的代理工作流程了")


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
# 首先，讓我們設置環境並匯入必要的函式庫：
# 
# > 📝 **說明**: LangChain 生態系統包含多個組件，我們需要匯入核心功能、Google Gemini整合以及LangGraph狀態管理。

# In[ ]:


# Load environment variables
from dotenv import load_dotenv
import os

load_dotenv()
print("✅ Environment loaded")


# In[ ]:


# 匯入必要的函式庫
# LangChain 生態系統提供豐富的LLM應用開發工具
import json
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Any, Optional, TypedDict, Annotated
import warnings
warnings.filterwarnings('ignore')

# LangChain 核心組件
from langchain_google_genai import ChatGoogleGenerativeAI  # Google Gemini整合
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage  # 訊息類型
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate  # 提示模板
from langchain_core.output_parsers import StrOutputParser  # 輸出解析器
from langchain_core.runnables import RunnablePassthrough  # 可運行鏈
from langchain.schema import Document  # 文檔結構
from langchain.text_splitter import RecursiveCharacterTextSplitter  # 文字分割器

# LangGraph 狀態管理組件
from langgraph.graph import StateGraph, END  # 狀態圖和結束節點
from langgraph.checkpoint.sqlite import SqliteSaver  # SQLite檢查點保存器
from langgraph.prebuilt import ToolExecutor  # 工具執行器
import operator  # 運算符

print("✅ 所有LangChain/LangGraph組件匯入成功")

# 🧩 架構說明：LangChain提供模組化組件，LangGraph增加狀態管理和圖執行能力


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

def create_trial_documents(trial_data: Dict[str, Any]) -> List[Document]:
    """Create LangChain documents from trial data for better processing.
    
    Args:
        trial_data: Clinical trial data dictionary
        
    Returns:
        List of Document objects for LangChain processing
    """
    if "error" in trial_data:
        return [Document(page_content=f"Error: {trial_data['error']}", metadata={"section": "error"})]
    
    documents = []
    trial_id = trial_data.get("Clinical Trial ID", "Unknown")
    
    # Create documents for each section
    sections = {
        "Eligibility": trial_data.get("Eligibility", []),
        "Intervention": trial_data.get("Intervention", []),
        "Results": trial_data.get("Results", []),
        "Adverse_Events": trial_data.get("Adverse_Events", [])
    }
    
    for section_name, section_data in sections.items():
        if section_data:
            if isinstance(section_data, list):
                content = "\n".join(str(item) for item in section_data)
            else:
                content = str(section_data)
            
            documents.append(Document(
                page_content=content,
                metadata={
                    "trial_id": trial_id,
                    "section": section_name
                }
            ))
    
    return documents

# Test utilities
sample_trial = load_clinical_trial("NCT00066573")
sample_docs = create_trial_documents(sample_trial)
print(f"✅ Data utilities ready. Sample trial: {sample_trial.get('Clinical Trial ID', 'Error')}")
print(f"Created {len(sample_docs)} documents from sample trial")


# ## Model Configuration
# 
# Set up the ChatGoogleGenerativeAI model for LangChain:

# In[ ]:


# Initialize ChatGoogleGenerativeAI model with flexible API key support
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

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1,  # Low temperature for consistent results
    google_api_key=api_key
)

# Initialize checkpointer for state persistence
checkpointer = SqliteSaver.from_conn_string(":memory:")

print("✅ Model and checkpointer configured")


# ## 狀態定義
# 
# 為我們的LangGraph工作流程定義狀態結構：
# 
# > 🔄 **狀態管理說明**: LangGraph的核心概念是維護一個在整個工作流程中持續存在的狀態。這個狀態包含所有分析步驟的輸入、中間結果和最終輸出。

# In[ ]:


class ClinicalAnalysisState(TypedDict):
    """臨床試驗分析工作流程的狀態架構。"""
    
    # 輸入資料
    statement: str  # 要分析的陳述
    primary_trial_id: str  # 主要試驗ID
    secondary_trial_id: Optional[str]  # 次要試驗ID（比較時使用）
    focus_section: Optional[str]  # 關注的試驗區段
    
    # 試驗資料
    primary_trial_data: Dict[str, Any]  # 主要試驗的完整資料
    secondary_trial_data: Optional[Dict[str, Any]]  # 次要試驗資料
    trial_documents: List[Document]  # LangChain文檔格式的試驗資料
    
    # 分析結果
    medical_analysis: Optional[str]  # 醫療專家分析結果
    statistical_analysis: Optional[str]  # 統計分析結果
    logical_analysis: Optional[str]  # 邏輯分析結果
    
    # 最終決策
    final_decision: Optional[str]  # 最終的蘊含/矛盾決策
    confidence_score: Optional[float]  # 信心分數(0-1)
    
    # 工作流程控制
    next_action: Optional[str]  # 下一個要執行的動作
    error_messages: Annotated[List[str], operator.add]  # 錯誤訊息累積

print("✅ 狀態架構定義完成")

# 📊 狀態說明：這個TypedDict定義了工作流程中每個節點可以讀取和修改的資料結構


# ## Node Definitions
# 
# Define the analysis nodes for our LangGraph workflow:

# In[ ]:


def clinical_data_extractor(state: ClinicalAnalysisState) -> ClinicalAnalysisState:
    """Extract and structure clinical trial data."""
    
    try:
        # Load primary trial data
        primary_data = load_clinical_trial(state["primary_trial_id"])
        state["primary_trial_data"] = primary_data
        
        # Load secondary trial data if provided
        if state["secondary_trial_id"]:
            secondary_data = load_clinical_trial(state["secondary_trial_id"])
            state["secondary_trial_data"] = secondary_data
        
        # Create documents for processing
        documents = create_trial_documents(primary_data)
        if state["secondary_trial_data"]:
            documents.extend(create_trial_documents(state["secondary_trial_data"]))
        
        state["trial_documents"] = documents
        state["next_action"] = "medical_analysis"
        
    except Exception as e:
        state["error_messages"].append(f"Data extraction error: {str(e)}")
        state["next_action"] = "end"
    
    return state

print("✅ Data extractor node defined")


# In[ ]:


def medical_analysis_node(state: ClinicalAnalysisState) -> ClinicalAnalysisState:
    """Perform medical analysis of the statement against trial data."""
    
    try:
        # Create medical analysis prompt
        medical_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""
You are a Medical Expert specializing in clinical trial analysis.
Your role is to analyze statements from a medical perspective and assess their accuracy against clinical trial evidence.

Focus on:
- Medical terminology accuracy
- Clinical relevance and significance
- Medical plausibility of claims
- Clinical context and implications
- Medical evidence alignment

Provide a thorough medical analysis and end with:
MEDICAL_VERDICT: [SUPPORTS/CONTRADICTS/UNCLEAR] - brief reasoning
            """.strip()),
            HumanMessage(content="""
STATEMENT TO ANALYZE: "{statement}"

CLINICAL TRIAL EVIDENCE:
{trial_evidence}

Please provide your medical analysis of this statement against the trial evidence.
            """.strip())
        ])
        
        # Prepare trial evidence
        trial_evidence = ""
        for doc in state["trial_documents"]:
            if state["focus_section"] and doc.metadata.get("section") != state["focus_section"]:
                continue
            trial_evidence += f"\n{doc.metadata.get('section', 'Unknown')}:\n{doc.page_content}\n"
        
        # Run medical analysis
        medical_chain = medical_prompt | llm | StrOutputParser()
        medical_result = medical_chain.invoke({
            "statement": state["statement"],
            "trial_evidence": trial_evidence
        })
        
        state["medical_analysis"] = medical_result
        state["next_action"] = "statistical_analysis"
        
    except Exception as e:
        state["error_messages"].append(f"Medical analysis error: {str(e)}")
        state["next_action"] = "statistical_analysis"  # Continue with next analysis
    
    return state

print("✅ Medical analysis node defined")


# In[ ]:


def statistical_analysis_node(state: ClinicalAnalysisState) -> ClinicalAnalysisState:
    """Perform statistical and numerical analysis."""
    
    try:
        # Create statistical analysis prompt
        statistical_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""
You are a Statistical Analyst specializing in clinical trial data analysis.
Your role is to analyze numerical claims, statistics, and quantitative relationships in clinical trials.

Focus on:
- Numerical accuracy and verification
- Statistical significance and validity
- Quantitative relationships and comparisons
- Data calculations and mathematical reasoning
- Confidence intervals and error margins

Perform detailed calculations and end with:
STATISTICAL_VERDICT: [ACCURATE/INACCURATE/PARTIALLY_ACCURATE] - numerical reasoning
            """.strip()),
            HumanMessage(content="""
STATEMENT TO ANALYZE: "{statement}"

CLINICAL TRIAL DATA:
{trial_evidence}

Please perform statistical analysis of the numerical claims in this statement.
            """.strip())
        ])
        
        # Prepare trial evidence (focus on Results section for statistical data)
        trial_evidence = ""
        for doc in state["trial_documents"]:
            # Prioritize Results and statistical sections
            if doc.metadata.get("section") in ["Results", "Adverse_Events"] or not state["focus_section"]:
                trial_evidence += f"\n{doc.metadata.get('section', 'Unknown')}:\n{doc.page_content}\n"
        
        # Run statistical analysis
        statistical_chain = statistical_prompt | llm | StrOutputParser()
        statistical_result = statistical_chain.invoke({
            "statement": state["statement"],
            "trial_evidence": trial_evidence
        })
        
        state["statistical_analysis"] = statistical_result
        state["next_action"] = "logical_analysis"
        
    except Exception as e:
        state["error_messages"].append(f"Statistical analysis error: {str(e)}")
        state["next_action"] = "logical_analysis"  # Continue with next analysis
    
    return state

print("✅ Statistical analysis node defined")


# In[ ]:


def logical_analysis_node(state: ClinicalAnalysisState) -> ClinicalAnalysisState:
    """Perform logical consistency analysis."""
    
    try:
        # Create logical analysis prompt
        logical_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""
You are a Logic Analyst specializing in reasoning validation and consistency checking.
Your role is to validate logical relationships, consistency, and reasoning soundness.

Focus on:
- Logical structure and coherence
- Cause-and-effect relationships
- Internal consistency
- Validity of inferences
- Detection of logical fallacies
- Reasoning pattern analysis

Provide logical analysis and end with:
LOGICAL_VERDICT: [SOUND/UNSOUND/QUESTIONABLE] - logical reasoning
            """.strip()),
            HumanMessage(content="""
STATEMENT TO ANALYZE: "{statement}"

EVIDENCE CONTEXT:
{trial_evidence}

Please analyze the logical consistency and reasoning of this statement.
            """.strip())
        ])
        
        # Prepare trial evidence
        trial_evidence = ""
        for doc in state["trial_documents"]:
            if state["focus_section"] and doc.metadata.get("section") != state["focus_section"]:
                continue
            trial_evidence += f"\n{doc.metadata.get('section', 'Unknown')}:\n{doc.page_content[:500]}...\n"
        
        # Run logical analysis
        logical_chain = logical_prompt | llm | StrOutputParser()
        logical_result = logical_chain.invoke({
            "statement": state["statement"],
            "trial_evidence": trial_evidence
        })
        
        state["logical_analysis"] = logical_result
        state["next_action"] = "decision_synthesis"
        
    except Exception as e:
        state["error_messages"].append(f"Logical analysis error: {str(e)}")
        state["next_action"] = "decision_synthesis"  # Continue to final decision
    
    return state

print("✅ Logical analysis node defined")


# In[ ]:


def decision_synthesis_node(state: ClinicalAnalysisState) -> ClinicalAnalysisState:
    """Synthesize all analyses and make final decision."""
    
    try:
        # Create decision synthesis prompt
        decision_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""
You are the Decision Synthesizer responsible for making final entailment classifications.
Your role is to synthesize expert analyses and determine the final verdict.

Classification Rules:
- ENTAILMENT: Statement is directly supported by the trial evidence
- CONTRADICTION: Statement is refuted or contradicted by the trial evidence

Weigh all evidence types:
- Medical expert analysis (clinical accuracy)
- Statistical analysis (numerical validity)
- Logical analysis (reasoning soundness)

Provide reasoning and confidence, then end with:
FINAL_DECISION: [Entailment/Contradiction]
CONFIDENCE: [0.0-1.0]
            """.strip()),
            HumanMessage(content="""
ORIGINAL STATEMENT: "{statement}"

MEDICAL ANALYSIS:
{medical_analysis}

STATISTICAL ANALYSIS:
{statistical_analysis}

LOGICAL ANALYSIS:
{logical_analysis}

Based on these expert analyses, provide your final entailment decision.
            """.strip())
        ])
        
        # Run decision synthesis
        decision_chain = decision_prompt | llm | StrOutputParser()
        decision_result = decision_chain.invoke({
            "statement": state["statement"],
            "medical_analysis": state.get("medical_analysis", "Not available"),
            "statistical_analysis": state.get("statistical_analysis", "Not available"),
            "logical_analysis": state.get("logical_analysis", "Not available")
        })
        
        # Parse decision and confidence
        if "FINAL_DECISION: Entailment" in decision_result:
            final_decision = "Entailment"
        elif "FINAL_DECISION: Contradiction" in decision_result:
            final_decision = "Contradiction"
        else:
            # Fallback parsing
            if "entailment" in decision_result.lower() and "contradiction" not in decision_result.lower():
                final_decision = "Entailment"
            else:
                final_decision = "Contradiction"
        
        # Extract confidence score
        confidence = 0.5  # Default
        try:
            if "CONFIDENCE:" in decision_result:
                confidence_str = decision_result.split("CONFIDENCE:")[1].strip().split()[0]
                confidence = float(confidence_str)
        except:
            pass
        
        state["final_decision"] = final_decision
        state["confidence_score"] = confidence
        state["next_action"] = "end"
        
    except Exception as e:
        state["error_messages"].append(f"Decision synthesis error: {str(e)}")
        state["final_decision"] = "Contradiction"  # Conservative fallback
        state["confidence_score"] = 0.1
        state["next_action"] = "end"
    
    return state

print("✅ Decision synthesis node defined")


# ## 工作流程定義
# 
# 透過連接所有節點來創建LangGraph工作流程：
# 
# > 🔗 **圖結構說明**: LangGraph工作流程就像一個流程圖，每個節點代表一個分析步驟，邊緣定義資料如何在節點間流動。這確保了可追蹤和可重現的分析過程。

# In[ ]:


def create_clinical_analysis_workflow():
    """Create the clinical analysis workflow using LangGraph."""
    
    # Create the StateGraph
    workflow = StateGraph(ClinicalAnalysisState)
    
    # Add nodes
    workflow.add_node("data_extraction", clinical_data_extractor)
    workflow.add_node("medical_analysis", medical_analysis_node)
    workflow.add_node("statistical_analysis", statistical_analysis_node)
    workflow.add_node("logical_analysis", logical_analysis_node)
    workflow.add_node("decision_synthesis", decision_synthesis_node)
    
    # Add edges
    workflow.set_entry_point("data_extraction")
    workflow.add_edge("data_extraction", "medical_analysis")
    workflow.add_edge("medical_analysis", "statistical_analysis")
    workflow.add_edge("statistical_analysis", "logical_analysis")
    workflow.add_edge("logical_analysis", "decision_synthesis")
    workflow.add_edge("decision_synthesis", END)
    
    # Compile the workflow
    app = workflow.compile(checkpointer=checkpointer)
    
    return app

# Create the workflow
clinical_workflow = create_clinical_analysis_workflow()

print("✅ LangGraph workflow created")


# ## Analysis Pipeline
# 
# Create the main pipeline function that uses our LangGraph workflow:

# In[ ]:


def langchain_analysis_pipeline(statement: str, primary_id: str, secondary_id: Optional[str] = None, 
                               section_id: Optional[str] = None, verbose: bool = False) -> str:
    """
    Run the complete LangChain/LangGraph analysis pipeline.
    
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
        if verbose:
            print(f"📄 Analyzing: {statement[:100]}...")
            print(f"🏥 Primary Trial: {primary_id}")
            if secondary_id:
                print(f"🏥 Secondary Trial: {secondary_id}")
        
        # Create initial state
        initial_state = {
            "statement": statement,
            "primary_trial_id": primary_id,
            "secondary_trial_id": secondary_id,
            "focus_section": section_id,
            "primary_trial_data": {},
            "secondary_trial_data": None,
            "trial_documents": [],
            "medical_analysis": None,
            "statistical_analysis": None,
            "logical_analysis": None,
            "final_decision": None,
            "confidence_score": None,
            "next_action": None,
            "error_messages": []
        }
        
        # Run the workflow
        config = {"configurable": {"thread_id": f"analysis_{hash(statement)}"[:10]}}
        result = clinical_workflow.invoke(initial_state, config)
        
        if verbose:
            print(f"🩺 Medical Analysis: {'✅' if result.get('medical_analysis') else '❌'}")
            print(f"📊 Statistical Analysis: {'✅' if result.get('statistical_analysis') else '❌'}")
            print(f"🧠 Logical Analysis: {'✅' if result.get('logical_analysis') else '❌'}")
            print(f"⚖️ Final Decision: {result.get('final_decision', 'Unknown')}")
            print(f"🎯 Confidence: {result.get('confidence_score', 0.0):.2f}")
            
            if result.get("error_messages"):
                print(f"⚠️ Errors: {len(result['error_messages'])}")
            print("-" * 50)
        
        return result.get("final_decision", "Contradiction")
        
    except Exception as e:
        if verbose:
            print(f"❌ Error in LangChain pipeline: {e}")
        return "Contradiction"  # Conservative fallback

print("✅ LangChain analysis pipeline ready")


# ## Test Example
# 
# Let's test our LangChain/LangGraph system:

# In[ ]:


# Test with a sample statement
test_statement = "there is a 13.2% difference between the results from the two the primary trial cohorts"
test_primary_id = "NCT00066573"

print(f"Testing LangChain/LangGraph system with statement:")
print(f"'{test_statement}'")
print(f"Primary trial: {test_primary_id}")
print("\n" + "="*80)

# Run the analysis with verbose output
result = langchain_analysis_pipeline(
    statement=test_statement,
    primary_id=test_primary_id,
    section_id="Results",
    verbose=True
)

print(f"\n🎯 LANGCHAIN RESULT: {result}")
print("="*80)


# ## Evaluation on Training Data
# 
# Let's evaluate our LangChain/LangGraph system on training data:

# In[ ]:


# Load training data
train_data = load_dataset("training_data/train.json")
print(f"Loaded {len(train_data)} training examples")

# Evaluate on a sample (adjust sample_size as needed)
sample_size = 20
examples = list(train_data.items())[:sample_size]

print(f"\nEvaluating LangChain/LangGraph system on {len(examples)} examples...")

results = []
correct = 0

for i, (uuid, example) in enumerate(tqdm(examples, desc="LangChain Processing")):
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
        
        # Get prediction from LangChain/LangGraph system
        predicted = langchain_analysis_pipeline(
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
print(f"\n📊 LangChain/LangGraph Results:")
print(f"Accuracy: {accuracy:.2%} ({correct}/{len(examples)})")

# Store results for comparison
langchain_results = results.copy()


# ## State Inspection
# 
# Let's inspect the stateful capabilities of our LangGraph workflow:

# In[ ]:


# Demonstrate state persistence and inspection
print("🔍 LangGraph State Inspection:")
print("=" * 50)

# Run a simple analysis and inspect intermediate states
test_config = {"configurable": {"thread_id": "demo_analysis"}}
test_state = {
    "statement": "The primary endpoint was met",
    "primary_trial_id": "NCT00066573",
    "secondary_trial_id": None,
    "focus_section": "Results",
    "primary_trial_data": {},
    "secondary_trial_data": None,
    "trial_documents": [],
    "medical_analysis": None,
    "statistical_analysis": None,
    "logical_analysis": None,
    "final_decision": None,
    "confidence_score": None,
    "next_action": None,
    "error_messages": []
}

try:
    # Stream the workflow execution to see intermediate states
    print("Streaming workflow execution:")
    for step in clinical_workflow.stream(test_state, test_config):
        for node_name, node_output in step.items():
            print(f"\n📍 Node: {node_name}")
            if node_output.get("final_decision"):
                print(f"   Decision: {node_output['final_decision']}")
                print(f"   Confidence: {node_output.get('confidence_score', 'N/A')}")
            if node_output.get("error_messages"):
                print(f"   Errors: {len(node_output['error_messages'])}")
            print(f"   Next: {node_output.get('next_action', 'N/A')}")
    
    # Get final state
    final_state = clinical_workflow.get_state(test_config)
    print(f"\n🎯 Final State Summary:")
    print(f"   Thread ID: {test_config['configurable']['thread_id']}")
    print(f"   Final Decision: {final_state.values.get('final_decision')}")
    print(f"   Total Errors: {len(final_state.values.get('error_messages', []))}")
    
except Exception as e:
    print(f"Error in state inspection: {e}")

print("\n✅ State inspection completed")


# ## Generate Submission File
# 
# Let's generate a submission file using our LangChain/LangGraph system:

# In[ ]:


def generate_langchain_submission(test_file="test.json", output_file="langchain_submission.json", sample_size=None):
    """
    Generate submission file using LangChain/LangGraph system.
    
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
        
    print(f"🚀 Generating LangChain/LangGraph predictions for {len(examples)} examples...")
    
    submission = {}
    
    for i, (uuid, example) in enumerate(tqdm(examples, desc="LangChain Processing")):
        try:
            statement = example.get("Statement")
            primary_id = example.get("Primary_id")
            secondary_id = example.get("Secondary_id")
            section_id = example.get("Section_id")
            
            if not statement or not primary_id:
                submission[uuid] = {"Prediction": "Contradiction"}  # Default fallback
                continue
                
            # Get prediction from LangChain/LangGraph system
            prediction = langchain_analysis_pipeline(
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
    
    print(f"✅ LangChain submission saved to {output_file}")
    return submission

# Generate submission for a small sample
langchain_submission = generate_langchain_submission(
    test_file="test.json", 
    output_file="langchain_submission.json",
    sample_size=10  # Adjust as needed
)

print(f"Generated predictions for {len(langchain_submission)} examples")


# ## Workflow Visualization
# 
# Let's visualize our LangGraph workflow structure:

# In[ ]:


# Display workflow information
print("🔄 LangGraph Workflow Structure:")
print("=" * 50)

workflow_steps = [
    "1. Data Extraction → Load and structure clinical trial data",
    "2. Medical Analysis → Expert medical reasoning and assessment",
    "3. Statistical Analysis → Numerical validation and calculations",
    "4. Logical Analysis → Reasoning consistency and soundness",
    "5. Decision Synthesis → Final entailment classification"
]

for step in workflow_steps:
    print(f"   {step}")

print("\n📊 State Management Features:")
state_features = [
    "• Persistent state across workflow steps",
    "• Error tracking and recovery mechanisms",
    "• Confidence scoring and decision rationale",
    "• Intermediate result storage and inspection",
    "• Thread-based conversation management"
]

for feature in state_features:
    print(f"   {feature}")

print("\n✅ Workflow visualization complete")


# ## 結論與洞察
# 
# ### LangChain/LangGraph 框架優勢：
# 1. **成熟生態系統**: 最完善且功能豐富的LLM應用框架
# 2. **有狀態工作流程**: LangGraph支援複雜的有狀態代理互動
# 3. **豐富整合**: 廣泛的工具生態系統和服務整合
# 4. **進階模式**: 支援精密的推理和決策模式
# 5. **社群支援**: 龐大社群、豐富文檔和實際範例
# 6. **生產就緒**: 在眾多企業應用中經過實戰考驗
# 
# ### 關鍵功能展示：
# - **基於圖的工作流程**: 結構化、有狀態的分析管道
# - **狀態持久性**: SQLite檢查點用於對話持續性
# - **節點專業化**: 針對不同領域的專門分析節點
# - **錯誤處理**: 強健的錯誤追蹤和恢復機制
# - **串流支援**: 即時工作流程執行監控
# - **配置管理**: 基於線程的狀態管理
# 
# ### 架構優勢：
# - **彈性工作流程**: 易於修改和擴展分析管道
# - **狀態管理**: 跨複雜多步驟流程的持久上下文
# - **除錯支援**: 清晰的狀態檢查和中間結果追蹤
# - **可擴展設計**: 企業部署的生產就緒架構
# - **整合就緒**: 與外部工具和服務的無縫整合
# 
# ### 優化機會：
# 1. **提示工程**: 微調每個分析節點的提示
# 2. **條件路由**: 為不同陳述類型添加條件邏輯
# 3. **平行處理**: 在可能的地方實作平行分析節點
# 4. **工具整合**: 利用LangChain廣泛的工具生態系統
# 5. **進階模式**: 實作自我反思和迭代改進
# 
# ### 何時使用 LangChain/LangGraph：
# - 需要狀態管理的複雜多步驟推理工作流程
# - 需要廣泛工具和服務整合的應用
# - 需要強健生產就緒框架的企業系統
# - 社群支援和文檔至關重要的專案
# - 需要精密工作流程模式和客製化的場景
# 
# ### 框架比較總結：
# - **vs AutoGen**: 更適合結構化工作流程，AutoGen更適合自由形式對話
# - **vs Atomic Agents**: 更全面但較重，Atomic更適合純效能
# - **vs Agno**: 更廣泛的生態系統，Agno在內建記憶體和知識方面更佳
# 
# ### LangChain/LangGraph 獨特優勢：
# 1. **基於圖的推理**: 原生支援複雜的條件工作流程
# 2. **狀態持久性**: 內建檢查點和對話持續性
# 3. **生態系統成熟度**: 廣泛的工具、整合和社群支援
# 4. **企業功能**: 具備監控、記錄和除錯的生產就緒
# 5. **彈性**: 高度可客製化的工作流程和整合模式
# 
# ## 🎓 學習重點總結
# - **狀態圖概念**: 工作流程作為有向圖，節點是功能，邊是資料流
# - **持久狀態**: 在整個分析過程中維護上下文和中間結果
# - **模組化設計**: 每個節點專注於特定分析任務
# - **企業就緒**: 生產環境的監控、錯誤處理和可擴展性
# 
# LangChain/LangGraph在複雜的生產環境中表現卓越，特別適合需要結構化工作流程、狀態管理和廣泛整合的臨床NLP應用。它是需要精密推理模式和強健基礎設施支援的企業級應用的理想選擇。
