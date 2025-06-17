import requests
import gradio as gr
from langchain import hub
from semanticscholar import SemanticScholar 
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.agents import Tool, AgentExecutor, create_react_agent, AgentType
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.retrievers import ArxivRetriever 
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.sequential import SequentialChain
from langchain.chains import LLMChain
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
embeddings = OpenAIEmbeddings()
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)

def search_arxiv(question: str):
    arxiv_retriever = ArxivRetriever(load_max_docs=1, load_all_available_meta=True, get_full_documents=True)
    result = arxiv_retriever.invoke(question)[0]
    paper = {"title": result.metadata['Title'], "pdf_url": result.metadata['links'][-1]}
    print(f"\n[1] 找到论文：{paper}")
    #     return paper

    # def download_pdf(paper: dict, save_dir: str = "downloads"):
    save_dir = "downloads"
    pdf_url = paper['pdf_url']
    if not os.path.exists(path=save_dir):
        os.makedirs(save_dir)
    pdf_path = os.path.join(save_dir, pdf_url.split('/')[-1]+".pdf")
    resp = requests.get(pdf_url, timeout=30)
    with open(pdf_path, "wb") as f:
        f.write(resp.content)
    print(f"\n[2] 下载完成：{pdf_path}")
#     return pdf_path

# def parse_pdf(pdf_path: str):
    reader = PdfReader(pdf_path)
    pages_ = [page.extract_text() or "" for page in reader.pages]
    pages = "\n".join(pages_)
    print(f"\n[3] 解析完成，文本长度：{len(pages)} 字符")

    # return "\n".join(pages)

# def answer_question(pages: str, question: str):
    chunks = splitter.split_text(pages)
    vector_index = FAISS.from_texts(chunks, embeddings)
    retriever = vector_index.as_retriever(search_kwargs={"k": 3})
    print(f"\n[4] 索引构建完成，检索器已就绪")

    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    template_ = """
        你是一个智能论文助手。你可以使用以下工具来帮助用户完成任务。

        {tools}

        当你思考时，请把思考过程写进下面的 Scratchpad：
        {agent_scratchpad}

        用户输入：
        {input}

        请开始行动。
    """
    p = PromptTemplate(
        template=template_,
        input_variables=["input", "agent_scratchpad", "tools"]
    )
    rag_chain = create_react_agent(retriever, combine_docs_chain, prompt=retrieval_qa_chat_prompt)
    result = rag_chain.invoke({"input": question})
    print(f"\n[5] 提问：{question}")
    return result

def summarize_paper(pages: str):
    docs = splitter.split_text(pages) 
    summarizer = load_summarize_chain(llm=llm, chain_type="map_reduce")
    return summarizer.run(docs)
 
search_tool = Tool.from_function(
    search_arxiv,
    name="Search arXiv", 
    description="Search arXiv for papers based on question, and then answer the question"
)

# download_tool = Tool.from_function(
#         download_pdf, 
#         name="Download pdf", 
#         description="Download pdf searched from arxiv and return the local path where the pdf is downloaded at"
#     )

# parse_pdf_tool = Tool.from_function(
#         parse_pdf, 
#         name="Parse pdf", 
#         description="Extract and return all text from the given PDF file path."
#     ) 

# summarize_tool = Tool(
#     name="Summarize paper by pdf",
#     func=summarize_paper,
#     description="Given the full text, generate a concise, structured summary."
# )

# answer_tool = Tool(
#     name="Answer question from paper by pdf",
#     func=answer_question,
#     description="Given the full text and a question, return a precise answer using RAG."
# )

# tools = [search_tool, download_tool, parse_pdf_tool, answer_tool]
tools = [search_tool]
 

agent_ = create_react_agent(llm, tools, react_agent_prompt)
agent = AgentExecutor(agent=agent_, tools=tools, verbose=True, max_iterations=5)

# llm_with_tools = llm.bind_tools(tools)

# print("=== QA 开始输出 ===")
# for ch in agent.stream({'topic': 'transformer survey', 'question': 'what is transformer? what is the advantage of using it in llm?'}):
#     print(ch, end="|")
# print("=== QA 输出结束 ===")

query_qa = "search from arxiv, what is transformer and why it yields good result in llm?"
response = agent.invoke({"input": query_qa})
print("\n=== 回答 ===")

print(response)