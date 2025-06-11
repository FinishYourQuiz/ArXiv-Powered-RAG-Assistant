

"""
1. 检索(Retrieval)
- 目的：为 LLM 提供与问题最相关的“上下文”, 弥补模型单纯语言生成时对外部知识盲区的不足。
- 步骤：
    - 文档切块(Chunk)→ 向量化(Embeddings)→ 建索引(Vector Store)
- 实现：
    - 把文档切块(chunk)
    - 将每个 chunk 编码成向量(embeddings)
    - 把向量存入向量数据库(vector store, 如 FAISS)
    - 给定查询, 将查询编码成向量
    - 在向量库中检索出与之最相似的若干 chunk。
2. 增强(Augmented)
- 拼接上下文：根据用户问题检索到 top-k 个文档块, 然后连同用户原始问题一起, 按预定义的 Prompt 模板拼成 LLM 的“提示上下文”(prompt context)。
3. 生成(Generation) 
- LLM 推理：将拼好的上下文发给 LLM, 生成最终回答。
"""


#### === 1. Introduction ===
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
import asyncio

# Create a RAG tool using LlamaIndex
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
print("Query Engine Created!")

# Define two functions
def kkl(a: float, b: float) -> float:
    return a * b + a + b

async def search_documents(query: str) -> str:
    response = await query_engine.query(query) 
    return str(response)

# Define agent
agent = FunctionAgent(
    tools = [kkl, search_documents],
    llm = OpenAI(model="gpt-4o-mini"),
    system_prompt = "You are a helpful assistant that can perform calculations and search through documents to answer questions."
)
print("Agent Created!")


# Now we can ask questions about the documents or do calculations
async def main():
    response = await agent.run(
        "Tell me about Stella. Also, what's kkl of 7 and 8?"
    )
    print(response)

# Run the agent
if __name__ == "__main__":
    asyncio.run(main())

#####################################################################################################
########## ATTENTION - Most of the following usage have been depretiated by Settings ###############
#####################################################################################################

#### === 2. Integratet with LangChain ===
from llama_index import (
    download_loader,
    GPTVectorStoreIndex,
    ServiceContext,
    PromptHelper
)
from langchain.chat_models import ChatOpenAI

# 1. 文档加载：把PDF解析成RawDocument对象
def load_document(path="data/sample.pdf"): 
    PDFReader = download_loader("PDFReader")
    documents = PDFReader().load_data(file=path)
    return documents

# 2. 建立PromptHelper：决定如何把文档切块并重组上下文
def create_prompt_helper():
    prompt_helper = PromptHelper(
        max_input_size = 4096,      # 最大上下文容量
        num_output = 512,           # 预估回答长度 
        chunk_size_limit = 1024,    # 文本块最大tokens数
        max_chunk_overlap = 200     # 保留跨句连贯性
    )
    return prompt_helper

# 3. 封装 PromptHelper 和 Embedding 模型
def build_service_context(model="gpt-4o-mini", temp=0.0):
    service_context = ServiceContext.from_defaults(
        # llm_model = ...
        # 如果只传了embed_mode, 同一个对象也会被当作生成模型来用
        embed_model = ChatOpenAI(model=model, temperature=temp),
        prompt_helper = create_prompt_helper()
    )
    return service_context

# 4. 对于每个文档, chunk编码为向量, 并存入内存中
def build_vector_index(documents, service_ctx):
    return GPTVectorStoreIndex.from_documents(
        documents,
        service_context = service_ctx
    )

# 5. 提供 Query接口, 用来根据问题检索top_k chunks + 拼接prompt + 调用LLM
def create_query_engine(index):
    return index.as_query_engine()

# 6. 执行查询并输出结果
def perform_query(engine, question: str):
    response = engine.query(question)
    print("Answer: ", response)
    for node in response.source_nodes:
        print(f"- Page {node.node.metadata.get('page', '?')}: {node.node.get_text()[:100]}...")
    return response

def main():
    docs = load_document()
    service_ctx = build_service_context()
    index = build_service_context(docs, service_ctx)
    query_engine = create_query_engine(index)

    question = "请简述 sample.pdf 中的主要观点。"
    perform_query(query_engine, question)