# GenAI Bootcamp Project

## 1. 项目简介
- **目标**：基于 Transformers + LangChain + LlamaIndex，构建从 Pipeline Demo 到 RAG+Agent 的学习闭环。
- **阶段**：
  1. 初级：Transformers + LangChain + LlamaIndex 入门 
  2. 进阶：Memory、RAG、Agent 开发  
  3. 高级：LangGraph 多 Agent + vLLM 部署

## 2. 环境准备
```bash
conda activate genai
pip install -r requirements.txt
```

## 3. 初级阶段 
- 环境：conda create -n genai-day7 python=3.11
- 安装：pip install torch transformers langchain llama-index faiss-cpu
- Transformers Demo: <br> `python ./scripts/level_0/demo_transformers.py`
- LangChain Demo:   <br> `python ./scripts/level_0/demo_langchain.py`
- LlamaIndex Demo:  <br> `python ./scripts/level_0/demo_llama_index.py`
