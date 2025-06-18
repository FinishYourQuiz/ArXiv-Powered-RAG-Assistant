import requests
import gradio as gr
from langchain import hub
from semanticscholar import SemanticScholar 
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.agents import Tool, AgentExecutor, create_react_agent 
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.retrievers import ArxivRetriever 
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
import os
from PyPDF2 import PdfReader
from langchain_core.runnables import Runnable
import xml.etree.ElementTree as ET
from langchain.chains.retrieval import create_retrieval_chain

client = SemanticScholar()
llm = ChatOpenAI(
    model="gpt-4o-mini", 
    streaming=True, 
    callbacks=[StreamingStdOutCallbackHandler()]
)
react_agent_prompt = hub.pull("hwchase17/react")
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
embeddings = OpenAIEmbeddings()
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
 
def parse_arxiv_response(response):
    """
    Parses the arXiv API response and extracts relevant information.
    """
    if response.status_code != 200:
        raise Exception(f"Error fetching data from arXiv: {response.status_code}")
    if not response.text:
        raise ValueError("Empty response from arXiv API.")
    root = ET.fromstring(response.text)
    entries = []
    for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
        title = entry.find('{http://www.w3.org/2005/Atom}title').text
        pdf_url = None
        for link in entry.findall('{http://www.w3.org/2005/Atom}link'):
            print(link.attrib)
            if 'pdf' in link.attrib.get('href', ''):
                pdf_url = link.attrib['href']
        entries.append({'title': title, 'pdf_url': pdf_url})
    return entries

def search_arxiv(question: str):
    """
    Searches arXiv for papers related to the question and returns the first paper found.
    """
    keywords = "all:" + "+".join(question.split()[:5])
    url = "https://export.arxiv.org/api/query" 
    params = {
        "search_query": keywords,
        "start": 0,
        "max_results": 2
    }
    resp = requests.get(url, params=params)
    resp = parse_arxiv_response(resp)  # Get the first paper
    if not resp:
        raise ValueError("No papers found for the given query.")
    return resp[0]

def download_pdf(paper: dict, save_dir: str = "downloads"):
    """
    Downloads the PDF of the paper from arXiv and saves it to the specified directory.
    """
    save_dir = "downloads"
    pdf_url = paper['pdf_url']
    if not os.path.exists(path=save_dir):
        os.makedirs(save_dir)
    pdf_path = os.path.join(save_dir, pdf_url.split('/')[-1]+".pdf")
    resp = requests.get(pdf_url, timeout=30)
    with open(pdf_path, "wb") as f:
        f.write(resp.content)
    return pdf_path

def parse_pdf(pdf_path: str):
    """
    Parses the PDF file and extracts text from all pages.
    """
    reader = PdfReader(pdf_path)
    pages_ = [page.extract_text() or "" for page in reader.pages]
    pages = "\n".join(pages_)
    return pages

def build_retriever(pages: str):
    """
    Builds a retriever from the parsed text of the PDF.
    """
    chunks = splitter.split_text(pages)
    vector_index = FAISS.from_texts(chunks, embeddings)
    retriever = vector_index.as_retriever(search_kwargs={"k": 3})
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
    return rag_chain

def answer_question(rag_chain: Runnable, question: str): 
    """
    Answers the question using the RAG chain built from the PDF text.
    """
    result = rag_chain.invoke({"input": question})
    return result

def qa_pipeline(question: str):
    # 1) 搜索
    info = search_arxiv(question)
    print(f"[1] 找到论文：{info}")
    
    # 2) 下载
    pdf_path = download_pdf(info)
    print(f"[2] 下载完成：{pdf_path}")
    
    # 3) 解析
    full_text = parse_pdf(pdf_path)
    print(f"[3] 解析完成，文本长度：{len(full_text)} 字符")
    
    # 4) 索引构建
    retriever = build_retriever(full_text)
    print(f"[4] 索引构建完成，检索器已就绪")
    
    # 5) 检索问答
    print(f"[5] 提问：\n{question}")
    answer = answer_question(retriever, question) 

    # 6) 返回答案
    print(f"[6] 回答：\n{answer}")
    return answer

def summarize_paper(pages: str):
    docs = splitter.split_text(pages) 
    summarizer = load_summarize_chain(llm=llm, chain_type="map_reduce")
    return summarizer.run(docs)
  
search_tool = Tool.from_function(
    qa_pipeline,
    name="Search arXiv", 
    description="Search arXiv for papers based on question, and then answer the question"
)

tools = [search_tool]
 
agent_ = create_react_agent(llm, tools, react_agent_prompt)
agent = AgentExecutor(agent=agent_, tools=tools, verbose=True, max_iterations=5)

query_qa = "search from arxiv, what is transformer and why it yields good result in llm?"
response = agent.invoke({"input": query_qa})