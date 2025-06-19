# ArXiv-Powered RAG Assistant

A command-line research companion that:

1. Searches **arXiv** for the most relevant paper to your question  
2. Downloads the PDF  
3. Parses every page into text  
4. Builds a **Retrieval-Augmented Generation (RAG)** index with FAISS + OpenAI embeddings  
5. Uses an LLM (default: `gpt-4o-mini`) to answer your question, grounded in the paper’s content  
6. Streams reasoning steps to the console via a **ReAct** agent

---

## Features

- **End-to-end pipeline** from keyword search to answer generation  
- **LangChain** abstractions throughout (Retriever, Chain, Agent)  
- **Streaming output** for step-by-step transparency  
- Falls back to direct LLM answers when no paper is found  
- Small, dependency-light codebase—easy to extend or embed in other projects  

---

## Quick Start

### 1 . Install dependencies

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

<details>
<summary><code>requirements.txt</code></summary>

```
langchain>=0.1.16
langchain-openai>=0.0.8
langchain-community>=0.0.28
langchain-text-splitters>=0.0.1
faiss-cpu
openai
PyPDF2
requests
```
</details>

### 2 . Set your OpenAI key

```bash
export OPENAI_API_KEY="sk-..."  
```

(Optional) enable LangSmith tracing:

```bash
export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
export LANGCHAIN_API_KEY="<your_langsmith_key>"
```

### 3 . Run

#### Interactive REPL

```bash
python paper_bot.py
```

```
Query ('exit' to quit): why do transformers work so well?
2025-06-20 21:53:12 INFO | Thinking...
> Found paper: Attention Is All You Need
> PDF saved to downloads/1706.03762.pdf
> Text length: 142 374 chars
> RAG built; answering question...
▌Step 1 ...
```

Ask more questions, type `exit` to quit.

#### One-shot CLI

```bash
python paper_bot.py --query "graph neural network applications"
```

---

## How It Works

```
               ↓ keyword search (arXiv API)
user question ─┐
               ├──► first PDF ► text ► chunk+embed ► FAISS
               └──► ReAct agent (LangChain) ──► RetrievalQA chain ──► answer
```

* **Search & Download** – simple HTTP calls, Atom XML parsing  
* **Chunk & Embed** – `RecursiveCharacterTextSplitter` + `OpenAIEmbeddings` (`all-MiniLM` available if you prefer HF models)  
* **Vector Store** – in-memory FAISS (swap for Milvus, Pinecone, etc.)  
* **Reasoning Engine** – ReAct agent orchestrates the pipeline as a `Tool`  

---

## Configuration

| Parameter                | Default               | Notes                                   |
|--------------------------|-----------------------|-----------------------------------------|
| LLM model                | `gpt-4o-mini`         | change in `LLM = ChatOpenAI(...)`       |
| arXiv max results        | `2`                   | tweak in `search_arxiv()`               |
| Chunk size / overlap     | `800 / 200` chars     | change `CHUNK_SIZE`, `CHUNK_OVERLAP`    |
| Vector DB                | `faiss-cpu` in-proc   | replace with any LangChain `VectorStore`|
| Max agent iterations     | `5`                   | adjust in `AgentExecutor`               |

---

## Extending

* **Local inference** – drop in `ChatOllama` or connect to a vLLM server  
* **Multi-paper RAG** – aggregate top-N PDFs before building the index  
* **Web UI** – wrap `qa_pipeline()` in FastAPI or Streamlit for a browser front-end  
* **Caching** – persist downloaded PDFs & FAISS indices to speed up repeat queries  
* **Advanced tools** – add Math/Code execution tools to the ReAct toolbox for richer reasoning  

---

## Troubleshooting

| Issue                                 | Resolution                                                   |
|---------------------------------------|--------------------------------------------------------------|
| `OpenAIAuthenticationError`           | Check `OPENAI_API_KEY` and model access/billing status       |
| `faiss` install problems (Apple Silicon) | `brew install faiss` then `pip install faiss-cpu`            |
| PDF text extraction returns empty     | Some PDFs are scans—integrate `pdfminer.six` + OCR (Tesseract) |
| Agent loops or exceeds iterations     | Increase `max_iterations` or refine the prompt/tool desc.    |

---

## License

MIT – free for personal and commercial use. See `LICENSE` for details.
