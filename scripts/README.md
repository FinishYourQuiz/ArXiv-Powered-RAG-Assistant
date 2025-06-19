# GenAI Projects

## AI Paper Q\&A Pipeline 📄🤖

### 1 . What does this project do?

A single-file prototype that **finds a relevant arXiv paper on-the-fly** and turns it into a mini Retrieval-Augmented Generation (RAG) system so you can ask questions about the paper in natural language.

Flow:

```
[query] → arXiv search → PDF download → text extraction
→ chunk & embed → FAISS vector store → LLM answers your question
```

The whole flow is wrapped in a simple REPL powered by a ReAct agent.
If arXiv returns nothing, it falls back to answering with the LLM directly.

---

### 2 . Key Features

| Feature                                   | Tech behind it                                                  |
| ----------------------------------------- | --------------------------------------------------------------- |
| Keyword–based arXiv search                | `requests` + Atom feed parsing                                  |
| PDF download & OCR-free text extraction   | `PyPDF2`                                                        |
| Chunking & embeddings                     | LangChain `RecursiveCharacterTextSplitter` + `OpenAIEmbeddings` |
| In-memory vector DB                       | `FAISS` (CPU-only)                                              |
| RAG answer generation                     | `RetrievalQA` chain                                             |
| Agent wrapper with step-by-step reasoning | ReAct agent (`langchain.agents.create_react_agent`)             |
| Streaming answers to the console          | `StreamingStdOutCallbackHandler`                                |

---

### 3 . Prerequisites

| Item                                | Notes                                           |
| ----------------------------------- | ----------------------------------------------- |
| **Python ≥ 3.10**                   | Typing & `langchain` requirements               |
| **OpenAI account & API key**        | Used by `ChatOpenAI` + `OpenAIEmbeddings`       |
| (Optional) **Semantic Scholar key** | Imported but *not* required in the default flow |

---

### 4 . Installation

```bash
git clone <this-repo>
cd <repo>
python -m venv venv && source venv/bin/activate    # or your favourite env mgr
pip install -r requirements.txt
```

`requirements.txt` (minimal):

```
langchain>=0.1.16
langchain-openai>=0.0.8
langchain-community>=0.0.28
langchain-text-splitters>=0.0.1
faiss-cpu
PyPDF2
requests
arxiv
```

---

### 5 . Environment variables

```bash
export OPENAI_API_KEY="sk-..."
# optional
export LANGCHAIN_TRACING_V2="true"     # if you use LangSmith
export LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
export LANGCHAIN_API_KEY="<your_langsmith_key>"
```

---

### 6 . Running the demo

```bash
python paper_bot_pipeline.py   # or whatever you saved the file as
```

You will see:

```
Enter your query: what is a transformer and why does it work so well?
Start thinking....
> Found paper: Attention Is All You Need
> Downloaded PDF to downloads/...
> Preview ...
> Index built with 120 chunks
> Answering question ...
```

Type questions until you enter `exit`.

---

### 7 . How it works (code tour)

1. **`search_arxiv(question)`**

   * Builds a simple “all\:keyword1+keyword2…” query.
   * Calls arXiv’s Atom API → parses XML → returns the first hit.

2. **`download_pdf(info)`**

   * Saves the PDF to `downloads/` with a deterministic filename.

3. **`parse_pdf(pdf_path)`**

   * Reads every page’s text into one long string.

4. **`build_retriever(text)`**

   * Splits text (800 chars / 200 overlap).
   * Embeds with OpenAI, stores in an in-memory FAISS index.
   * Wraps in LangChain Retriever + “stuff” combine-docs chain.

5. **`answer_question(rag_chain, question)`**

   * Invokes the chain → LLM looks at retrieved chunks → returns answer.

6. **Agent glue** (`qa_pipeline` + ReAct)

   * Creates a LangChain `Tool` around the pipeline.
   * ReAct agent decides to call that one tool; reasoning steps are streamed.

---

### 8 . Extending / hardening

* **Add retries / timeout handling** for slow arXiv responses.
* **Cache PDFs & vector stores** to avoid repeated downloads / embeds.
* **Swap in vLLM / Ollama** if you want local inference—just replace `ChatOpenAI`.
* **Multiple-paper RAG**: search top-N, build a single combined store for broader context.
* **UI**: wrap the REPL in Streamlit or FastAPI if you need a web front-end.

---

### 9 . Troubleshooting

| Symptom                               | Fix                                                                            |
| ------------------------------------- | ------------------------------------------------------------------------------ |
| `OpenAIAuthenticationError`           | Check `OPENAI_API_KEY`, billing, model name (`gpt-4o-mini`).                   |
| `FAISS` install fails (Apple Silicon) | `brew install faiss` then `pip install faiss-cpu` or use Docker.               |
| Empty PDF text                        | arXiv sometimes ships scanned PDFs; integrate `pdfminer.six` or Tesseract OCR. |

---

### 10 . License

MIT — do whatever you want, no warranties.
