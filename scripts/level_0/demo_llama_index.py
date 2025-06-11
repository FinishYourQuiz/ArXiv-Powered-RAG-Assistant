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
def cli(md_path="results.md"):
    questions = []
    responses = []
    quotes = []
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
        questions.append(q)
        responses.append(response)
        print("引用: ")
        quote = []
        for node in response.source_nodes:
            curr_quote = node.get_content()[:20].strip()
            print(f"- {curr_quote}...")
            quote.append(curr_quote)
        quotes.append(quote)
    export_to_md(questions, responses, quotes, md_path)

# === 4.生成md表格 === #
def export_to_md(questions, responses, quotes, md_path): 
    parent_dir = os.path.dirname(md_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    md_lines = [
        "| 问题 | 回答 | 引用 |",
        "| --- | --- | --- |"
    ]

    for q, r, qs in zip(questions, responses, quotes):
        quote_str = ";".join(qs).replace("\n", " ").replace("|", "\\|")
        q_esc = q.replace("\n", " ").replace("|", "\\|")
        r_esc = str(r).replace("\n", " ").replace("|", "\\|")
        md_lines.append(f"| {q_esc} | {r_esc} | {quote_str} |")

    # 写入文件
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"已生成 Markdown 表格：{md_path}")

if __name__ == "__main__":
    cli('results/cli.md')
