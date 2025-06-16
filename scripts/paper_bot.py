import requests
import gradio as gr
from langchain import hub
from semanticscholar import SemanticScholar 
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.retrievers import ArxivRetriever 
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.chains.summarize import load_summarize_chain
import os
from PyPDF2 import PdfReader

client = SemanticScholar()
llm = ChatOpenAI(
    model="gpt-4o-mini", 
    streaming=True, 
    callbacks=[StreamingStdOutCallbackHandler()]
)
react_agent_prompt = hub.pull("hwchase17/react")
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

def search_arxiv(query: str):
    arxiv_retriever = ArxivRetriever(load_max_docs=1, load_all_available_meta=True, get_full_documents=True)
    results = arxiv_retriever.invoke(query)
    paper = {"title": results.metadata['Title'], "date": results.metadata['Published'], "pdf_url": results.metadata['links'][-1]}
    return paper

def download_pdf(paper: dict, save_dir: str = "downloads"):
    pdf_url = paper['pdf_url']
    if not os.path.exists(path=save_dir):
        os.makedirs(save_dir)
    local_path = os.path.join(save_dir, pdf_url.split('/')[-1]+".pdf")
    resp = requests.get(pdf_url, timeout=30)
    with open(local_path, "wb") as f:
        f.write(resp.content)
    return local_path

def parse_pdf(pdf_path: str):
    reader = PdfReader(pdf_path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)

def get_retriever(pages): 
    embeddings = OpenAIEmbeddings()
    vector_index = FAISS.from_documents(pages, embeddings)
    retriever = vector_index.as_retriever(search_kwargs={"k": 3})
    return retriever

def answer_question(question: str, retriever: VectorStoreRetriever):
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    rag_chain = create_react_agent(retriever, combine_docs_chain)
    result = rag_chain({"query": question})
    return result['answer']

def summarize_paper(question: str, retriever: VectorStoreRetriever):
    docs = retriever.get_relevant_documents(question)
    summarizer = load_summarize_chain(llm=llm, chain_type="map_reduce")
    return summarizer.run(docs)

def make_answer_tool(retriever):
    return Tool(
        name="answer_question",
        func=lambda q: answer_question(q, retriever),
        description="Answer based on querying the document's vector index"
    )

def make_summarize_tool(retriever):
    return Tool(
        name="summarize_paper",
        func=lambda _: summarize_paper(_, retriever),
        description="Summarize the paper in its vectors store"
    )

tools = [
    Tool.from_function(
        search_arxiv,
        name="Search arXiv", 
        description="Search arXiv for papers based on question and required number of papers will be searched for, and return title, published data, and pdf link"
    ),
    Tool.from_function(
        download_pdf, 
        name="Download pdf", 
        description="Download pdf searched from arxiv and return the local path where the pdf is downloaded at"
    ), 
    Tool.from_function(
        parse_pdf, 
        name="Parse pdf", 
        description="General chat"
    ) 
]

cot_template = PromptTemplate.from_template( 
    """
    You are PaperBuddy, an AI research assistant that thinks step by step.\n
    Answer the following questions as best you can. You have access to the following tools:
    {tools}
    Use the following format:
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought:{agent_scratchpad}\n
    """
)
agent_ = create_react_agent(llm, tools, cot_template)
agent = AgentExecutor(agent=agent_, tools=tools, verbose=True, max_iterations=5)

papers = search_arxiv("chain-of-thought prompting survey")
print("Arxiv Search Results:")
for paper in papers: 
    print(parse_pdf(download_pdf(paper['pdf_url'])))
  