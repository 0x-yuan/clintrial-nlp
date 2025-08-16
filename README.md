# 臨床試驗 NLP AI 代理框架基準測試

針對臨床試驗自然語言推理 (NLI) 任務的三個主要 AI 代理框架的完整基準實作。所有框架都配置為使用 **Google Gemini 2.5 Flash** 以獲得最佳的成本效益平衡。

## 概覽

本儲存庫包含使用以下框架的臨床試驗 NLP 多代理系統完整實作：
- **Atomic Agents** - 輕量、高效能模組化代理 (Google Generative AI)
- **Agno (Phidata)** - 具備記憶與知識功能的全端代理平台 (Gemini 原生支援)
- **LangChain/LangGraph** - 狀態化圖形代理工作流程 (ChatGoogleGenerativeAI)

## 任務描述

**目標**：判斷關於臨床試驗的自然語言陳述是否為 **蘊含** (有證據支持) 或 **矛盾** (被證據駁斥)。

**資料集**：涵蓋資格、介入措施、結果和不良事件的臨床試驗報告陳述。

## 快速開始

### 安裝

```bash
# 安裝必要套件
pip install phidata google-generativeai langchain langchain-google-genai langgraph atomic-agents

# 或一次安裝所有相依套件
pip install phidata google-generativeai langchain langchain-google-genai langgraph atomic-agents pandas jupyter tqdm python-dotenv gdown
```

### 環境設定

建立包含 Google Gemini API 金鑰的 `.env` 檔案：
```bash
GOOGLE_API_KEY=your_gemini_api_key_here
```

從 [Google AI Studio](https://ai.google.dev/gemini-api/docs/api-key) 取得您的 API 金鑰。

### 執行基準測試

選擇任一框架並執行對應的 notebook：

```bash
# Atomic Agents - 輕量且快速 (Google Generative AI)
jupyter notebook atomic_agents_baseline.ipynb  

# Agno - 具備記憶功能的全端 (Gemini 2.5 Flash)
jupyter notebook agno_baseline.ipynb

# LangChain - 狀態化工作流程 (ChatGoogleGenerativeAI)
jupyter notebook langchain_baseline.ipynb
```

所有 notebook 都預先配置使用 **Google Gemini 2.5 Flash** 以獲得最佳效能和成本效益。每個 notebook 都會自動從 Google Drive 下載臨床試驗資料集，實現一鍵執行。

## 框架比較

| 框架 | 優勢 | 最適用於 | 學習曲線 |
|-----------|-----------|----------|----------------|
| **Atomic Agents** | 超輕量、模組化、快速啟動 | 生產部署、高速需求 | 簡單 ⭐⭐ |
| **Agno** | 內建記憶、知識管理、RAG | 企業應用、持久化 | 簡單 ⭐⭐ |
| **LangChain** | 成熟生態系、豐富整合 | 複雜工作流程、工具整合 | 較高 ⭐⭐⭐⭐ |

## 架構

每個基準實作都使用結構化多代理管線：

1. **資料處理** - 載入並結構化臨床試驗資料
2. **醫學分析** - 專業醫學推理 
3. **數值分析** - 統計驗證
4. **邏輯檢查** - 一致性驗證
5. **決策制定** - 最終蘊含分類

## 專案結構

```
├── atomic_agents_baseline.ipynb   # Atomic Agents 實作
├── agno_baseline.ipynb           # Agno 全端系統
├── langchain_baseline.ipynb      # LangChain/LangGraph 工作流程
├── requirements.txt             # 所有框架相依套件
├── training_data/              # 訓練和開發資料
│   ├── train.json
│   ├── dev.json
│   └── CT json/               # 臨床試驗資料
└── README.md                  # 專案說明文件
```

## 效能表現

### 執行時間基準 (處理單筆資料)
- **Atomic Agents**: 1.75 秒 - 最快速
- **Phidata (Agno)**: 4.34 秒 - 平衡效能
- **LangChain**: 14.71 秒 - 功能最豐富

### 目標效能水準
- **優秀 (>70%)**: 可投入生產
- **良好 (60-70%)**: 強力基準
- **中等 (50-60%)**: 基本功能
- **需改進 (<50%)**: 需要優化

## 主要特色

### Atomic Agents 亮點
- ✅ ~3μs 啟動時間
- ✅ 模組化代理組合
- ✅ 生產就緒架構
- ✅ 最小資源負擔

### Agno 亮點
- ✅ 內建對話記憶
- ✅ RAG 與向量知識庫
- ✅ SQLite 狀態持久化
- ✅ 企業級功能

### LangChain 亮點
- ✅ 圖形化狀態工作流程
- ✅ SQLite 檢查點
- ✅ 豐富的工具生態系
- ✅ 串流執行監控

## 開發建議

1. **從簡單開始**：以 Atomic Agents 或 Agno 開始較容易上手
2. **擴展規模**：使用 LangChain 處理企業工作流程
3. **生產部署**：根據速度需求選擇 Atomic Agents
4. **反覆改進**：使用評估結果改善代理推理

## 檔案格式

**提交格式** (`submission.json`)：
```json
{
  "uuid": {"Prediction": "Entailment"},
  "uuid2": {"Prediction": "Contradiction"}
}
```

**測試資料** (`test.json`)：
```json
{
  "uuid": {
    "Statement": "要分析的臨床陳述",
    "Primary_id": "NCT00000000", 
    "Section_id": "Results"
  }
}
```

## 疑難排解

**常見問題：**
- 缺少 API 金鑰 → 檢查 `.env` 檔案中的 `GOOGLE_API_KEY`
- 匯入錯誤 → 執行 `pip install -r requirements.txt`
- 記憶體問題 → 減少 notebook 中的樣本大小
- 執行緩慢 → 使用適當的樣本大小參數

**效能建議：**
- 使用 `uv` 進行更快的套件管理
- 為測試設定適當的樣本大小
- 使用 Google AI Studio 監控 token 使用量
- 在可用之處利用快取功能
- Gemini 2.5 Flash 相較於先前模型提供 22% 效率改進

## 貢獻

1. 使用多個框架測試您的變更
2. 更新評估指標
3. 加入完整文件
4. 遵循現有的程式碼模式

## 引用

如果您在研究中使用這些基準，請引用：
```
Clinical Trial NLP Agent Framework Baselines
https://github.com/your-repo/clintrial-nlp-baseline
```

---

**準備開始了嗎？** 選擇一個框架 notebook 並執行！每個都包含詳細說明和逐步指導。