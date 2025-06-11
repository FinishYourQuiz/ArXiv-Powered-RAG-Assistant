import os
from dotenv import load_dotenv

# === 1. Set up Environment === #
load_dotenv()

# === 2. LlamaIndex for vector store index === #
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding

def setting():
    Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-small", embed_batch_size=100
    )
    Settings.context_window = 4096
    Settings.num_output = 256
    Settings.chunk_size = 512
    Settings.chunk_overlap = 20

def create_query_engine(path='data'):
    docs = SimpleDirectoryReader(path).load_data() 
    setting()
    index = VectorStoreIndex.from_documents(docs)
    query_engine = index.as_query_engine()
    return query_engine

# === 3. CLI 交互 === #
def main():
    engine = create_query_engine('data/docs')
    print("=== Demo Bot V1.0 ===\n")
    print("输入 'exit' 退出此程序！")
    while True:
        q = input("\n输入: ")
        if q == "" or q.strip("\n").strip().lower() in ('q','quit','exit'):
            print("\n再见!")
            break
        response = engine.query(q)
        print(f"答案: {response}\n")
        print("引用: ")
        for node in response.source_nodes:
            print(f"- {node.get_content()[:20].strip()}...")

if __name__ == "__main__":
    main()
    # 答案: 给我介绍下悉尼
    # 引用:
    # - Sydney is the largest city in Australia and the state capital of New South Wales....
    # - The programming language Python was created by Guido van Rossum and first released in 1991....