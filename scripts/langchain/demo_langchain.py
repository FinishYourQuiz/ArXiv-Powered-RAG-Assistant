from dotenv import load_dotenv
load_dotenv()
 
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
# from langchain_community.llms import HuggingFaceHub

system_template = (
    '请将下面的句子从 {source_lang} 翻译为 {target_lang}：'
    ' "{text}"'
)

prompt_template = PromptTemplate(
    template=system_template,
    input_variables=["source_lang", "target_lang", "text"]
)

prompt = {
    "source_lang": "中文",
    "target_lang": "英文",
    "text": "很高兴认识你"
}

### ====== 1. Chain/Bind a PromptTemplate + a LLM  ======
from langchain.chains import LLMChain

llm_openai = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
# chain_openai = LLMChain(llm = llm_openai, prompt=prompt_template)
# response_openai = chain_openai.run(prompt)
# print("OpenAI: ", response_openai)
# > OpenAI:  Nice to meet you.

### ====== 2. Chain multiple steps as sequential ======
from langchain.chains import SequentialChain 
summarize_prompt = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following:\n\n{text}"
)
sum_chain = LLMChain(llm=llm_openai, prompt=summarize_prompt, output_key="summary")

translate_prompt = PromptTemplate(
    input_variables=["summary"],
    template="Translate this to Chinese:\n\n{summary}"
)
# can be different llm models
trans_chain = LLMChain(llm=llm_openai, prompt=translate_prompt, output_key="translation")

seq = SequentialChain(
    chains = [sum_chain, trans_chain],
    input_variables = ["text"],
    output_variables = ["summary", "translation"],
)

document = "The definite article is sometimes also used with proper names, which are already specified by definition (there is just one of them). For example: the Amazon, the Hebrides. In these cases, the definite article may be considered superfluous. Its presence can be accounted for by the assumption that they are shorthand for a longer phrase in which the name is a specifier, i.e. the Amazon River, the Hebridean Islands.[citation needed] Where the nouns in such longer phrases cannot be omitted, the definite article is universally kept: the United States, the People's Republic of China."
prompt = {
    "text": document
}

# results = seq(prompt)
# print("Summary:", results["summary"])
# print("Translation:", results["translation"])
"""
Summary: The definite article is sometimes used with proper names, even though they are already specified by definition. This can be seen in examples such as "the Amazon" and "the Hebrides." The presence of the definite article in these cases may be considered unnecessary, but it is often used as shorthand for longer phrases in which the name is a specifier. In cases where the nouns in these longer phrases cannot be omitted, the definite article is always kept, as seen in examples like "the United States" and "the People's Republic of China."
Translation: 定冠词有时与专有名词一起使用，尽管它们已经通过定义指定。这可以在“亚马逊”和“赫布里底群岛”等示例中看到。在这些情况下，定冠词的存在可能被认为是不必要的，但通常用作名称是特定词组的缩写。在这些较长 短语中不能省略名词的情况下，定冠词总是保留的，如“美利坚合众国”和“中华人民共和国”等示例。
"""

### ====== 3. Chain conversation with built-in memory settings ======
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
conversations = ConversationChain(
    llm = llm_openai,
    memory = ConversationBufferMemory()
)
# print(conversations.predict(input="hi, who are you?"))
# print(conversations.predict(input="Waht did I just say?"))
"""
Hello! I am an artificial intelligence designed to assist with answering questions and providing information. How can I help you today?
You just asked me who I am.
"""


### ====== 4.RAG ======
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings


embedding = OpenAIEmbeddings()
docs = [Document(page_content=text) for text in document]
vectorstore = FAISS.from_documents(docs, embedding)

rag = RetrievalQA.from_chain_type(
    llm = llm_openai,
    chain_type = "stuff",
    retriever=vectorstore.as_retriever()
)

answer = rag.run("What is definite article?")
print(answer)

# > A definite article is a word used before a noun to indicate that the noun is a particular person or thing that is known to the reader or listener. In English, the definite article is "the."