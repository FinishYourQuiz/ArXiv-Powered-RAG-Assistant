#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paper_bot.py
------------

On-demand arXiv → RAG pipeline with LangChain + OpenAI.

Usage
~~~~~
$ export OPENAI_API_KEY="sk-..."
$ python paper_bot.py --query "what is a transformer"

Author: <Your Name>
License: MIT
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from langchain import hub
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.summarize import load_summarize_chain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.vectorstores import FAISS
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
ARXIV_API_URL = "https://export.arxiv.org/api/query"
DOWNLOAD_DIR = Path("downloads")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# LangChain objects prepared once and reused
LLM = ChatOpenAI(
    model="gpt-4o-mini",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)
EMBEDDINGS = OpenAIEmbeddings()
SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
)
REACT_PROMPT = hub.pull("hwchase17/react")
QA_PROMPT = hub.pull("langchain-ai/retrieval-qa-chat")


# --------------------------------------------------------------------------- #
# ArXiv Helpers
# --------------------------------------------------------------------------- #
def _parse_arxiv_response(resp: requests.Response) -> List[Dict[str, str]]:
    """Parse Atom XML and return [{title, pdf_url}, ...]."""
    if resp.status_code != 200:
        raise RuntimeError(f"arXiv HTTP {resp.status_code}")
    if not resp.text:
        raise ValueError("Blank response from arXiv")

    root = ET.fromstring(resp.text)
    out: List[Dict[str, str]] = []

    for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
        title_el = entry.find("{http://www.w3.org/2005/Atom}title")
        if title_el is None:
            continue
        title = title_el.text or "Untitled"

        pdf_url: Optional[str] = None
        for link in entry.findall("{http://www.w3.org/2005/Atom}link"):
            href = link.attrib.get("href", "")
            if "pdf" in href:
                pdf_url = href
                break

        if pdf_url:
            out.append({"title": title.strip(), "pdf_url": pdf_url})

    return out


def search_arxiv(query: str, max_results: int = 2) -> Optional[Dict[str, str]]:
    """Return the first arXiv hit as dict(title, pdf_url) or None."""
    keywords = "all:" + "+".join(query.split()[:5])
    params = {"search_query": keywords, "start": 0, "max_results": max_results}

    try:
        resp = requests.get(ARXIV_API_URL, params=params, timeout=20)
        papers = _parse_arxiv_response(resp)
        return papers[0] if papers else None
    except Exception as exc:  # noqa: BLE001
        logger.error("ArXiv search failed: %s", exc)
        return None


def download_pdf(pdf_url: str, dest_dir: Path = DOWNLOAD_DIR) -> Path:
    """Download PDF and return local path."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    fname = pdf_url.split("/")[-1] + ".pdf"
    out_path = dest_dir / fname

    try:
        resp = requests.get(pdf_url, timeout=30)
        resp.raise_for_status()
        out_path.write_bytes(resp.content)
        logger.info("PDF saved to %s", out_path)
        return out_path
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"PDF download failed: {exc}") from exc


def parse_pdf(pdf_path: Path) -> str:
    """Extract plain text from PDF."""
    reader = PdfReader(str(pdf_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


# --------------------------------------------------------------------------- #
# RAG Construction
# --------------------------------------------------------------------------- #
def build_rag_chain(text: str):
    chunks = SPLITTER.split_text(text)
    if not chunks:
        raise ValueError("No text extracted from PDF.")

    vector_db = FAISS.from_texts(chunks, EMBEDDINGS)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    combine_chain = create_stuff_documents_chain(LLM, QA_PROMPT)
    return create_retrieval_chain(retriever, combine_chain)


def summarize_text(text: str) -> str:
    docs = SPLITTER.split_text(text)
    chain = load_summarize_chain(llm=LLM, chain_type="map_reduce")
    return chain.run(docs)


# --------------------------------------------------------------------------- #
# End-to-end Pipeline
# --------------------------------------------------------------------------- #
def qa_pipeline(question: str) -> str:
    """Search arXiv, build RAG, and answer the original question."""
    paper = search_arxiv(question)
    if not paper:
        logger.warning("No paper found; falling back to LLM only.")
        return str(LLM.invoke(question))

    logger.info("Found paper: %s", paper["title"])
    pdf_path = download_pdf(paper["pdf_url"])
    raw_text = parse_pdf(pdf_path)
    logger.info("Text length: %s chars", len(raw_text))

    rag_chain = build_rag_chain(raw_text)
    logger.info("RAG built; answering question...")
    answer = rag_chain.invoke({"input": question})
    return str(answer)


# --------------------------------------------------------------------------- #
# Agent Setup
# --------------------------------------------------------------------------- #
SEARCH_TOOL = Tool.from_function(
    qa_pipeline,
    name="SearchArXiv",
    description="Answer a question by searching arXiv and reading the first PDF.",
)

AGENT = AgentExecutor(
    agent=create_react_agent(LLM, [SEARCH_TOOL], REACT_PROMPT),
    tools=[SEARCH_TOOL],
    verbose=True,
    max_iterations=5,
)


# --------------------------------------------------------------------------- #
# CLI / REPL
# --------------------------------------------------------------------------- #
def interactive_loop() -> None:
    """Simple REPL; type 'exit' to quit."""
    try:
        while True:
            query = input("Query ('exit' to quit): ").strip()
            if query.lower() == "exit":
                break
            if not query:
                continue
            logger.info("Thinking...")
            _ = AGENT.invoke({"input": query})
    except (KeyboardInterrupt, EOFError):
        logger.info("Interrupted, exiting.")


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="arXiv-powered RAG assistant")
    parser.add_argument(
        "--query",
        help="Run a single query and exit (skips REPL)",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv or sys.argv[1:])
    if args.query:
        logger.info("Single-shot mode")
        result = qa_pipeline(args.query)
        print(result)
    else:
        interactive_loop()


if __name__ == "__main__":
    main()
