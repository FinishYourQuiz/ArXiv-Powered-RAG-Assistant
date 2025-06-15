import requests
import gradio as gr
from langchain import hub
from semanticscholar import SemanticScholar
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.retrievers.arxiv import ArxivRetriever
from langchain_community.document_loaders.pdf import OnlinePDFLoader

arxiv_tool = ArxivRetriever(top_k_results=5)
client = SemanticScholar()
llm = ChatOpenAI(
    model="gpt-4o-mini", 
    streaming=True, 
    callbacks=StreamingStdOutCallbackHandler()
)
react_agent_prompt = hub.pull("hwchase17/react")
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

def fetch_metadata(paper_id: str) -> str:
    base_url = "https://api.semanticscholar.org/graph/v1/paper/"
    url = f"{base_url}ARXIV:{paper_id}?fields=title,abstract,authors,citationCount"
    headers = {}
    api_key = ""
    if api_key:
        headers["x-api-key"] = api_key
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

# def get_bibtex(paper_id: str) -> str:
#     paper = client.get_paper(paper_id)
#     return paper.citationStyles["bibtext"]

def web_chat(text: str) -> str:
    return llm.invoke(text).content

pdf_retriever = None

def load_pdf_and_embed(pages): 
    embeddings = OpenAIEmbeddings()
    vector_index = FAISS.from_documents(pages, embeddings)
    pdf_retriever = vector_index.as_retriever(search_kwargs={"k": 3})
    return pdf_retriever

def run_rag(query: str) -> str:
    url = f"https://arxiv.org/pdf/{paper_id}"
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 100
    )
    pages = OnlinePDFLoader(url).load_and_split(splitter)
    pdf_retriever = load_pdf_and_embed(pages)
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    rag_chain = create_react_agent(pdf_retriever, combine_docs_chain)
    result = rag_chain({"query": query})
    return result["answer"]

tools = [
    arxiv_tool,
    Tool.from_function(
        fetch_metadata, 
        name="PaperMetadataTool", 
        description="Fetch metadata from Semantic Scholar"
    ),
    # Tool.from_function(
    #     get_bibtex, 
    #     name="BibTeXTool", 
    #     description="Get BibTeX citation"
    # ),
    Tool.from_function(
        fetch_metadata, 
        name="ChatTool", 
        description="General chat"
    ),
    Tool.from_function(
        func=run_rag,
        name="RAGTool",
        description="Use this to answer questions from the uploaded paper"
    )
]

cot_template = PromptTemplate(
    input_variables=["chat_history", "input", "agent_scratchpad"],

    template="""
    You are PaperBuddy, an AI research assistant that thinks step by step.\n
    {chat_history}\n
    User: {input}\n
    Thought: I will break down the problem into substeps.\n
    {agent_scratchpad}\n
    """
)
agent = create_react_agent(llm, tools, cot_template)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=5)

# def chat_interface(user_input, chat_history):
#     if "pdf:" in user_input.lower() and pdf_retriever:
#         rag = 

