# 臨床試驗報告的生醫自然語言推理

這是一個臨床試驗報告自然語言推理任務，目標是開發能夠自動判斷自然語言陳述與臨床試驗報告間邏輯關係（蘊含或矛盾）的系統。

## 目錄
1. [背景與意義](#背景與意義)
2. [任務概述](#任務概述)
3. [資料描述](#資料描述)
4. [資料格式](#資料格式)
5. [提交格式](#提交格式)
6. [評估指標](#評估指標)
7. [最佳實踐](#最佳實踐)
8. [資源](#資源)

---

## 背景與意義

### 什麼是臨床試驗報告自然語言推理？

臨床試驗報告（Clinical Trial Reports, CTRs）是詳細記錄臨床試驗方法、結果和發現的綜合文件。這些報告包含以下重要資訊：
- **個人化醫療發展**：了解不同患者群體的治療有效性
- **實證臨床決策**：支持醫療提供者選擇治療方案
- **醫學研究**：推進疾病治療的科學知識

### 挑戰

目前有超過400,000份臨床試驗報告，醫療專業人員不可能手動審查每個病例的所有相關研究。這在以下方面造成了重大瓶頸：
- **臨床決策制定**：尋找最佳治療方案的延遲
- **研究效率**：由於缺乏全面的文獻回顧而導致重複研究
- **患者照護**：由於證據不完整而導致次優治療選擇

### 為什麼需要自然語言推理？

自然語言推理（Natural Language Inference, NLI）透過自動判斷以下之間的邏輯關係，提供了一個可擴展的解決方案：
- **陳述**：關於臨床試驗或治療的聲明
- **臨床試驗報告**：包含試驗資料的來源文件

這種自動化能夠實現：
- **快速證據整合**：快速識別相關試驗資訊
- **可擴展醫學研究**：大型臨床數據集的自動處理
- **增強臨床決策支持**：即時存取試驗證據

---

## 任務概述

### 主要目標

開發能夠：
- **準確分類**自然語言陳述與臨床試驗報告之間的蘊含關係
- **保持一致性**在語義相似的輸入上
- **展現忠實度**對基礎臨床數據
- **顯示穩健性**對抗各種輸入擾動

### 主要挑戰

1. **生物醫學複雜性**：理解專業醫學術語和概念
2. **數值推理**：處理定量臨床數據和統計資料
3. **多跳推理**：結合試驗報告多個部分的資訊
4. **干預穩健性**：在不同陳述變化中保持性能

---

## 資料描述

### 數據集組成

NLI4CT數據集包含**2,400個陳述-CTR對**，涵蓋：

#### 臨床試驗領域
- **主要焦點**：乳癌臨床試驗
- **試驗階段**：從早期階段到第三期研究的各個階段
- **試驗類型**：具有不同治療方法的介入性研究

#### 陳述類型
1. **單一陳述**：關於個別臨床試驗的聲明
2. **比較陳述**：比較兩個不同臨床試驗的聲明

#### 臨床試驗部分
- **資格條件**：患者納入/排除標準
- **介入措施**：治療方案和程序
- **結果**：結果測量和統計發現
- **不良事件**：安全性概況和副作用

### 數據分割

- **訓練數據**：`training_data/train.json`（用於模型訓練的標註範例）
- **開發數據**：`training_data/dev.json`（用於模型調整的驗證集）
- **測試數據**：`test.json`（用於最終評估的評估集）
- **黃金標準**：`gold_test.json`（帶有干預元數據的真實標籤）

---

## 資料格式

### 測試資料格式（`test.json`）

```json
{
    "uuid": {
        "Type": "Single|Comparison",
        "Section_id": "Eligibility|Intervention|Results|Adverse Events",
        "Primary_id": "NCT_identifier",
        "Secondary_id": "NCT_identifier",
        "Statement": "Natural language claim about the trial(s)"
    }
}
```

#### 欄位描述

- **`uuid`**：每個陳述-試驗對的唯一識別符
- **`Type`**：
  - `Single`：關於一個臨床試驗的陳述
  - `Comparison`：比較兩個臨床試驗的陳述
- **`Section_id`**：陳述所指的臨床試驗部分
- **`Primary_id`**：主要臨床試驗的NCT識別符
- **`Secondary_id`**：次要試驗的NCT識別符（僅比較陳述）
- **`Statement`**：需要蘊含分類的自然語言聲明

### 黃金標準格式（`gold_test.json`）

```json
{
    "uuid": {
        "Type": "Single|Comparison",
        "Section_id": "Section_identifier",
        "Primary_id": "NCT_identifier",
        "Secondary_id": "NCT_identifier",
        "Statement": "Natural language statement",
        "Label": "Entailment|Contradiction",
        "Intervention": "Paraphrase|Contradiction|Numerical_paraphrase|Numerical_contradiction|Text_appended",
        "Causal_type": ["Preserving|Altering", "reference_uuid"]
    }
}
```

#### 黃金標準中的附加欄位

- **`Intervention`**：應用於創建對比集的文本干預類型
  - `Paraphrase`：保持意義的語義改寫
  - `Contradiction`：邏輯否定或對立
  - `Numerical_paraphrase`：數值改寫
  - `Numerical_contradiction`：數值矛盾
  - `Text_appended`：添加額外解釋文本

- **`Causal_type`**：因果關係分類
  - `Preserving`：干預保持原始意義
  - `Altering`：干預改變原始意義
  - 連結到原始陳述的參考UUID

### 臨床試驗資料格式

臨床試驗資訊儲存在`training_data/CT json/`中的個別JSON檔案中：

```json
{
    "Clinical_Trial_ID": "NCT_identifier",
    "Brief_Title": "Trial title",
    "Conditions": ["Disease conditions"],
    "Interventions": ["Treatment interventions"],
    "Primary_Outcome": "Primary endpoint description",
    "Secondary_Outcome": "Secondary endpoint description",
    "Eligibility": {
        "Inclusion_Criteria": ["Criteria list"],
        "Exclusion_Criteria": ["Criteria list"]
    },
    "Results": {
        "Primary_Outcome_Measure": "Outcome details",
        "Secondary_Outcome_Measure": "Outcome details",
        "Statistical_Analysis": "Analysis methods"
    },
    "Adverse_Events": ["Safety information"]
}
```

---

## 提交格式

### 必需檔案：`submission.json`

```json
{
    "uuid_1": {
        "Prediction": "Entailment"
    },
    "uuid_2": {
        "Prediction": "Contradiction"
    },
    "uuid_3": {
        "Prediction": "Entailment"
    }
}
```

### 提交要求

1. **完整性**：必須包含所有測試UUID的預測
2. **有效標籤**：只接受`"Entailment"`或`"Contradiction"`
3. **一致格式**：每個UUID必須映射到帶有`"Prediction"`鍵的字典
4. **檔案格式**：有效的JSON與適當的UTF-8編碼

---

## 評估指標

### 主要指標

#### 1. 基本性能指標
- **F1分數**：精確率和召回率的調和平均
- **精確率**：預測正確的比例
- **召回率**：實際正確識別的比例

#### 2. 穩健性指標
- **一致性**：在相似輸入上保持相同預測的能力
- **忠實度**：模型預測是否基於實際證據

### 性能等級

- **優秀（>0.70）**：表現優異，適合實際應用
- **良好（0.60-0.70）**：表現良好，仍有改進空間
- **中等（0.50-0.60）**：基本可用，需要進一步改進
- **需要改進（<0.50）**：表現不佳，需要重新設計

---

## 最佳實踐

### 資料處理建議

1. **理解醫學術語**：
   - 熟悉常見的臨床試驗術語
   - 注意數值和統計資料的含義
   - 理解不同試驗階段的差異

2. **分析陳述類型**：
   - 區分單一試驗陳述和比較陳述
   - 識別陳述涉及的具體試驗部分
   - 注意陳述中的關鍵詞和邏輯關係

3. **運用AI Agent方法**：
   - 使用AI Agent框架來自動化分析過程
   - 構建專門的Agent來處理不同類型的任務
   - 讓Agent協作來提高準確性

### 推薦的AI Agent框架

針對此任務，我們特別推薦以下四個優秀的AI Agent框架：

#### 🎯 **核心推薦框架**

1. **[AutoGen](https://github.com/microsoft/autogen)**
   - 微軟開發的多Agent對話系統
   - 優秀的Agent間協作機制
   - 適合複雜的多步驟推理任務

2. **[Atomic Agents](https://github.com/BrainBlend-AI/atomic-agents)**
   - 輕量級且模組化的Agent框架
   - 極高的性能（~3μs啟動時間）
   - 非常適合快速原型開發

3. **[Agno](https://docs.agno.com/)**
   - 功能強大的全棧Agent框架
   - 內建記憶、知識和推理能力
   - 支援多模態和高性能應用

4. **[LangChain](https://docs.langchain.com/)**
   - 最成熟和功能豐富的LLM應用框架
   - 豐富的工具和整合選項
   - 強大的社群支援

### 推薦的開發流程

1. **資料探索**：仔細研讀訓練資料，理解問題本質
2. **Agent設計**：選擇合適的框架設計專門的AI Agent
3. **模型選擇**：選擇適合生物醫學領域的預訓練模型
4. **驗證測試**：使用開發數據驗證Agent的性能
5. **迭代改進**：根據結果不斷優化Agent設計

---

## 資源

### 推薦AI Agent框架官方文檔

#### 🚀 **主要框架資源**

**AutoGen - 微軟多Agent系統**
- **官方文檔**：[Microsoft AutoGen Documentation](https://microsoft.github.io/autogen/)
- **GitHub**：[microsoft/autogen](https://github.com/microsoft/autogen)
- **特色**：多Agent對話、協作推理、工作流程自動化

**Atomic Agents - 輕量級高性能框架**
- **官方文檔**：[Atomic Agents Documentation](https://brainblend-ai.github.io/atomic-agents/)
- **GitHub**：[BrainBlend-AI/atomic-agents](https://github.com/BrainBlend-AI/atomic-agents)
- **特色**：極高性能、模組化設計、生產就緒

**Agno - 全棧Agent平台**
- **官方文檔**：[Agno Documentation](https://docs.agno.com/)
- **PyPI**：[agno](https://pypi.org/project/agno/)
- **特色**：內建記憶、知識管理、推理能力

**LangChain - 綜合LLM框架**
- **官方文檔**：[LangChain Documentation](https://docs.langchain.com/)
- **GitHub**：[langchain-ai/langchain](https://github.com/langchain-ai/langchain)
- **特色**：豐富生態系統、廣泛整合、成熟穩定

#### 📚 **補充學習資源**

**多Agent協作框架**
- **[CrewAI](https://docs.crewai.com/)**：專為多Agent協作設計的框架
- **[LangGraph](https://github.com/langchain-ai/langgraph)**：用於構建有狀態Agent應用

**輕量級工具**
- **[AutoChain](https://github.com/Forethought-Technologies/AutoChain)**：輕量級Agent構建工具

### 學習指南和教學

#### 技術文獻
- **[AI Agent綜合指南](https://medium.com/ai-simplified-in-plain-english/a-comprehensive-guide-to-top-ai-agent-frameworks-8dc0652891d7)**：主流Agent框架比較分析
- **[多Agent工作流程設計](https://www.marktechpost.com/2025/05/23/a-comprehensive-coding-guide-to-crafting-advanced-round-robin-multi-agent-workflows-with-microsoft-autogen/)**：使用AutoGen的進階模式

---

## 重要提醒

### 學習指導

作為參與者，您需要：

1. **了解任務目標**：
   - 判斷陳述與臨床試驗報告的邏輯關係
   - 學會區分「蘊含」和「矛盾」關係
   - 理解醫學文本的基本概念

2. **掌握檔案操作**：
   - 訓練資料：`training_data/train.json`
   - 測試資料：`test.json`
   - 提交檔案：`submission.json`
   - 評估檔案：`gold_test.json`

3. **選擇合適的AI Agent框架**：
   - **初學者推薦**：從Agno或Atomic Agents開始，易於上手
   - **進階用戶**：使用AutoGen進行複雜的多Agent協作
   - **企業應用**：選擇LangChain構建大型系統

4. **確保正確提交**：
   - 檢查所有測試UUID都有預測
   - 確保只使用"Entailment"或"Contradiction"標籤
   - 驗證JSON格式正確無誤

5. **評估標準理解**：
   - 主要關注F1分數、一致性和忠實度
   - 目標達到F1 > 0.60即為良好表現
   - 理解不同指標的意義和重要性

### 建議的學習路徑

1. **基礎學習**：閱讀訓練資料，理解任務要求
2. **框架選擇**：根據經驗水平選擇適合的AI Agent框架
3. **實作練習**：設計簡單的Agent來處理部分數據
4. **性能優化**：根據評估結果改進Agent設計
5. **完整提交**：生成並驗證最終提交檔案

### 技術提示

- **開始簡單**：先用單一Agent處理基本任務
- **逐步複雜化**：根據需要增加多Agent協作
- **重視評估**：頻繁使用評估腳本檢查進展
