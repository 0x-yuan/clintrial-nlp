# Clinical Trial NLP Agent Framework Baselines

Comprehensive baselines for four major AI agent frameworks applied to clinical trial natural language inference (NLI). All frameworks are configured to use **Google Gemini 2.5 Flash** for optimal cost-performance balance.

## Overview

This repository contains complete implementations of multi-agent systems for clinical trial NLP using:
- **AutoGen** - Microsoft's multi-agent conversation framework (with Gemini integration)
- **Atomic Agents** - Lightweight, high-performance modular agents (Google Generative AI)
- **Agno (Phidata)** - Full-stack agent platform with memory and knowledge (Gemini native support)
- **LangChain/LangGraph** - Stateful graph-based agent workflows (ChatGoogleGenerativeAI)

## Task Description

**Objective**: Determine whether natural language statements about clinical trials represent **Entailment** (supported by evidence) or **Contradiction** (refuted by evidence).

**Dataset**: Clinical trial reports with statements covering eligibility, interventions, results, and adverse events.

## Quick Start

### Installation

```bash
# Install required packages
pip install phidata google-generativeai langchain langchain-google-genai langgraph pyautogen

# Or install all dependencies at once
pip install phidata google-generativeai langchain langchain-google-genai langgraph pyautogen pandas jupyter tqdm python-dotenv
```

### Environment Setup

Create `.env` file with your Google Gemini API key:
```bash
GEMINI_API_KEY=your_gemini_api_key_here
```

Get your API key from [Google AI Studio](https://ai.google.dev/gemini-api/docs/api-key).

### Run a Baseline

Choose any framework and run the corresponding notebook:

```bash
# AutoGen - Multi-agent conversations (with Gemini)
jupyter notebook autogen_baseline.ipynb

# Atomic Agents - Lightweight & fast (Google Generative AI)
jupyter notebook atomic_agents_baseline.ipynb  

# Agno - Full-stack with memory (Gemini 2.5 Flash)
jupyter notebook agno_baseline.ipynb

# LangChain - Stateful workflows (ChatGoogleGenerativeAI)
jupyter notebook langchain_baseline.ipynb
```

All notebooks are pre-configured to use **Google Gemini 2.5 Flash** for optimal performance and cost efficiency.

## Framework Comparison

| Framework | Strengths | Best For | Learning Curve |
|-----------|-----------|----------|----------------|
| **AutoGen** | Multi-agent dialogue, role specialization | Complex collaborative reasoning | Medium ⭐⭐⭐ |
| **Atomic Agents** | Ultra-lightweight, modular, fast startup | Production deployment, speed | Simple ⭐⭐ |
| **Agno** | Built-in memory, knowledge management, RAG | Enterprise applications, persistence | Simple ⭐⭐ |
| **LangChain** | Mature ecosystem, extensive integrations | Complex workflows, tool integration | Higher ⭐⭐⭐⭐ |

## Architecture

Each baseline implements a structured multi-agent pipeline:

1. **Data Processing** - Load and structure clinical trial data
2. **Medical Analysis** - Expert medical reasoning 
3. **Numerical Analysis** - Statistical validation
4. **Logic Checking** - Consistency verification
5. **Decision Making** - Final entailment classification

## Project Structure

```
├── autogen_baseline.ipynb         # AutoGen multi-agent implementation
├── atomic_agents_baseline.ipynb   # Improved Atomic Agents pipeline
├── agno_baseline.ipynb           # Agno full-stack system
├── langchain_baseline.ipynb      # LangChain/LangGraph workflow
├── evaluate_model.py            # Comprehensive evaluation script
├── requirements.txt             # All framework dependencies
├── training_data/              # Training and development data
│   ├── train.json
│   ├── dev.json
│   └── CT json/               # Clinical trial data
├── test.json                  # Test data for predictions
└── gold_test.json            # Gold standard for evaluation
```

## Evaluation

Run evaluation on any framework's predictions:

```bash
python evaluate_model.py your_predictions.json
```

**Metrics calculated:**
- F1 Score, Precision, Recall
- Consistency and Faithfulness
- Intervention-specific performance

## Expected Performance

Target performance levels:
- **Excellent (>70%)**: Production ready
- **Good (60-70%)**: Strong baseline
- **Moderate (50-60%)**: Basic functionality
- **Needs Improvement (<50%)**: Requires optimization

## Key Features

### AutoGen Highlights
- ✅ Structured multi-agent conversations
- ✅ Role-based specialization
- ✅ Group chat coordination
- ✅ Transparent reasoning process

### Atomic Agents Highlights  
- ✅ ~3μs startup time
- ✅ Modular agent composition
- ✅ Production-ready architecture
- ✅ Minimal resource overhead

### Agno Highlights
- ✅ Built-in conversation memory
- ✅ RAG with vector knowledge base
- ✅ SQLite state persistence
- ✅ Enterprise-grade features

### LangChain Highlights
- ✅ Graph-based stateful workflows
- ✅ SQLite checkpointing
- ✅ Rich tool ecosystem
- ✅ Streaming execution monitoring

## Development Tips

1. **Start Simple**: Begin with Atomic Agents or Agno for easier onboarding
2. **Scale Up**: Use AutoGen for complex multi-agent collaboration
3. **Production**: Choose LangChain for enterprise workflows
4. **Iterate**: Use evaluation results to improve agent reasoning

## File Formats

**Submission Format** (`submission.json`):
```json
{
  "uuid": {"Prediction": "Entailment"},
  "uuid2": {"Prediction": "Contradiction"}
}
```

**Test Data** (`test.json`):
```json
{
  "uuid": {
    "Statement": "Clinical statement to analyze",
    "Primary_id": "NCT00000000", 
    "Section_id": "Results"
  }
}
```

## Troubleshooting

**Common Issues:**
- Missing API key → Check `.env` file for `GEMINI_API_KEY`
- Import errors → Run `uv pip install -r requirements.txt`
- Memory issues → Reduce sample sizes in notebooks
- Slow execution → Use `--sample-size` parameter

**Performance Tips:**
- Use `uv` for faster package management
- Set appropriate sample sizes for testing
- Monitor token usage with Google AI Studio
- Leverage caching where available
- Gemini 2.5 Flash offers 22% efficiency improvements over previous models

## Contributing

1. Test your changes with multiple frameworks
2. Update evaluation metrics
3. Add comprehensive documentation
4. Follow the existing code patterns

## Citation

If you use these baselines in your research, please cite:
```
Clinical Trial NLP Agent Framework Baselines
https://github.com/your-repo/clintrial-nlp-baseline
```

---

**Ready to start?** Pick a framework notebook and run it! Each contains detailed explanations and step-by-step guidance.