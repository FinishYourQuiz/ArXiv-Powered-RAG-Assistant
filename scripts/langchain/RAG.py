
##################################################
############### Retreival + QA ###################
##################################################

from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = DirectoryLoader("data\docs", glob="**/*.txt")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = splitter.split_documents(docs)


from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings

store = FAISS.from_documents(splits, OpenAIEmbeddings())
# FAISS(Facebook AI Similarity Search)
# - a library for efficient similarity search and clustering of dense vectors
# - contains algorithms that search in sets of vectors of any size
retriever = store.as_retriever()

from langchain import hub
from langchain_openai.chat_models import ChatOpenAI

base_prompt = hub.pull("langchain-ai/retrieval-qa-chat") # Official Query prompt
llm = ChatOpenAI(model="gpt-4o-mini")

############ High-level Chain Create #############
from langchain.chains.combine_documents import create_stuff_documents_chain 
from langchain.chains.retrieval import create_retrieval_chain

document_chain = create_stuff_documents_chain(llm, base_prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

result = retrieval_chain.invoke({"input": "Who is Amanda? And where is Sydney?"})
print(result["answer"])
# > Amanda Feng is applying for a Machine Learning Engineer role focused on autonomous systems, and she has a background in model training and deployment, particularly in machine learning and AI projects. Sydney is the largest city in Australia and the state capital of New South Wales.

############### LCEL Simple Usage ################
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

start = {
    "context": retriever | format_docs,
    "input": RunnablePassthrough() # to passthrough inputs
}
chain = start | base_prompt | llm | StrOutputParser()

result = chain.invoke("Who is Amanda? And where is Sydney?")
print(result)
# > Amanda Feng is applying for a Machine Learning Engineer role focused on autonomous systems, and she has a background in model training and deployment, particularly in machine learning and AI projects. Sydney is the largest city in Australia and the state capital of New South Wales.
