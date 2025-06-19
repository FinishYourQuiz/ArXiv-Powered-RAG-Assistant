#####################################
############## Tools ################
#####################################

### 0. Using tools in chatmodel
def tool_in_chat():
    from pydantic import BaseModel, Field
    from langchain.chat_models import init_chat_model
    from langchain_core.output_parsers import PydanticOutputParser

    class add(BaseModel):
        """Add two integers."""
        a: int = Field(..., description="First integer")
        b: int = Field(..., description="Second integer")

    class multiply(BaseModel):
        """Multiply two integers."""
        a: int = Field(..., description="First integer")
        b: int = Field(..., description="Second integer")

    tools = [add, multiply]
    llm = init_chat_model("gpt-4o-mini", model_provider="openai")
    llm_with_tools = llm.bind_tools(tools)
    chain = llm_with_tools | PydanticOutputParser(tools)

    query = "What is 3 * 12?"
    print(llm_with_tools.invoke(query))
    # > [multiply(a=3, b=12), add(a=11, b=49)]

### 1. Using the built-in Toolkits with customizations
def builtin_toolkit():
    from langchain_community.tools import WikipediaQueryRun
    from langchain_community.utilities import WikipediaAPIWrapper
    from pydantic import BaseModel, Field

    class WikiInputs(BaseModel):
        """Inputs too the wikipedia tool."""
        query: str = Field(
            description="query to look up in Wikipedia, should be 3 or less words"
        )

    api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)

    tool = WikipediaQueryRun(
        name="wiki-tool",
        description="look up things in wiki",
        args_schema=WikiInputs,
        api_wrapper=api_wrapper,
        return_direct=True
    )

    response = tool.run("what is bird")

    print(f"Response: {response}")
    print(f"Name: {tool.name}")
    print(f"Description: {tool.description}")
    print(f"args schema: {tool.args}")
    print(f"returns directly?: {tool.return_direct}")

### 2. Involving Human Interactions with tools
def tool_with_human_interaction():
     
    from langchain.chat_models import init_chat_model
    from typing import Dict, List
    from langchain_core.messages import AIMessage
    from langchain_core.tools import tool
    import json

    llm = init_chat_model("gpt-4o-mini", model_provider="openai")
    @tool
    def count_emails(last_n_days: int) -> int:
        """Dummy function to count number of e-mails. Returns 2 * last_n_days."""
        return last_n_days * 2
    @tool
    def send_email(message: str, recipient: str) -> str:
        """Dummy function for sending an e-mail."""
        return f"Successfully sent email to {recipient}."

    tools = [count_emails, send_email]
    llm_with_tools = llm.bind_tools(tools)

    def call_tools(msg: AIMessage) -> List[Dict]:
        tool_map = {tool.name : tool for tool in tools}
        tool_calls = msg.tool_calls.copy()
        for tool_call in tool_calls:
            tool_calls["output"] = tool_map[tool_call["name"]].invoke(tool_call["args"])
        return tool_calls
    
    def human_approval(msg: AIMessage) -> AIMessage:
        tool_strs = "\n\n".join(
            json.dumps(tool_call, indent=2) for tool_call in msg.tool_calls
        )
        input_msg = (
            f"Do you approve of the following tool invocations\n\n{tool_strs}\n\n"
            "Anything except 'Y'/'Yes' (case-insensitive) will be treated as a no.\n >>>"
        )
        resp = input(input_msg)
        if resp.lower() not in ("yes", "y"):
            return False
        return True

    chain = llm_with_tools | human_approval | call_tools
    print(chain.invoke("how many emails did i get in the last 5 days?"))
    """
    Do you approve of the following tool invocations

    {
    "name": "count_emails",
    "args": {
        "last_n_days": 5
    },
    "id": "toolu_01WbD8XeMoQaRFtsZezfsHor"
    }

    Anything except 'Y'/'Yes' (case-insensitive) will be treated as a no.
    >>> yes

    [{'name': 'count_emails',
    'args': {'last_n_days': 5},
    'id': 'toolu_01WbD8XeMoQaRFtsZezfsHor',
    'output': 10}]
    """

    print(chain.invoke("Send sally@gmail.com an email saying 'What's up homie'"))
    """
    Do you approve of the following tool invocations

    {
    "name": "send_email",
    "args": {
        "recipient": "sally@gmail.com",
        "message": "What's up homie"
    },
    "id": "toolu_014XccHFzBiVcc9GV1harV9U"
    }

    Anything except 'Y'/'Yes' (case-insensitive) will be treated as a no.
    >>> no
    ``````output

    Tool invocations not approved:

    {
    "name": "send_email",
    "args": {
        "recipient": "sally@gmail.com",
        "message": "What's up homie"
    },
    "id": "toolu_014XccHFzBiVcc9GV1harV9U"
    }
    """

#####################################
############## Agents ############### 
#####################################

### 0. Using AgentExecutor in LangChain
def build_agentExecutor_agent():
    #### 1. Define Tools
    import os  
    os.environ["TAVILY_API_KEY"] = "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"

    from langchain_community.tools.tavily_search import TavilySearchResults
    search = TavilySearchResults(max_results=2)

    from langchain_community.document_loaders import WebBaseLoader
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain.tools.retriever import create_retriever_tool

    docs = WebBaseLoader("https://docs.smith.langchain.com/overview").load()
    documents = RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=200
                ).split_documents(docs)
    vector = FAISS.from_documents(documents, OpenAIEmbeddings())
    retriever = vector.as_retriever()
    retriever_tool = create_retriever_tool(
                        retriever,
                        "langchain_search",
                        "Search for information about langchain. For any questions about langchain, you must use this tool!",
                    )

    tools = [search, retriever_tool]

    #### 2. Define Models annd Prompt
    from langchain.chat_models import init_chat_model
    from langchain import hub

    model = init_chat_model("gpt-4o-mini", model_provider="openai")
    prompt = hub.pull("hwchase17/openai-functions-agent")

    #### 3. Create Agent
    from langchain.agents import create_tool_calling_agent
    from langchain.agents import AgentExecutor
    from langchain_core.runnables.history import RunnableWithMessageHistory
    from langchain_community.chat_message_histories import ChatMessageHistory
    from langchain_core.chat_history import BaseChatMessageHistory

    store = {}
    def get_session_hisotry(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    agent = create_tool_calling_agent(model, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        get_session_hisotry, # easy with InMemoryChatMessageHistory
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    #### 4. Run the Agent
    agent_with_chat_history.invoke(
        {"input": "hi! I'm bob"},
        config={"configurable": {"session_id": "0"}},
    )
    """
    {'input': "hi! I'm bob",
    'chat_history': [],
    'output': 'Hello Bob! How can I assist you today?'}
    """
    agent_with_chat_history.invoke(
        {"input": "what's my name?"},
        config={"configurable": {"session_id": "0"}},
    )
    """
    {'input': "what's my name?",
    'chat_history': [HumanMessage(content="hi! I'm bob"),
    AIMessage(content='Hello Bob! How can I assist you today?')],
    'output': 'Your name is Bob.'}
    """
    agent_with_chat_history.invoke(
        {"how can langsmith help with testing?"},
        config={"configurable": {"session_id": "1"}},
    )
    # > 'output': 'LangSmith is a platform that aids in building production-grade Language Learning Model (LLM) applications. It can assist with testing in several ways:\n\n1. **Monitoring and Evaluation**: LangSmith allows close monitoring and evaluation of your application. This helps you to ensure the quality of your application and deploy it with confidence.\n\n2. **Tracing**: LangSmith has tracing capabilities that can be beneficial for debugging and understanding the behavior of your application.\n\n3. **Evaluation Capabilities**: LangSmith has built-in tools for evaluating the performance of your LLM. \n\n4. **Prompt Hub**: This is a prompt management tool built into LangSmith that can help in testing different prompts and their responses.\n\nPlease note that to use LangSmith, you would need to install it and create an API key. The platform offers Python and Typescript SDKs for utilization. It works independently and does not require the use of LangChain.'}

### 1. Using create_react_agent in LangGraph
def build_langGraph_agent():
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(model="gpt-4o") 

    system_message = "You are a helpful assistant."

    @tool
    def magic_function(input: int) -> int:
        """Applies a magic function to an input."""
        return input + 2

    tools = [magic_function]

    from langgraph.checkpoint.memory import MemorySaver  # an in-memory checkpointer
    from langgraph.prebuilt import create_react_agent
 
    memory = MemorySaver()
    langgraph_agent_executor = create_react_agent(
        model, tools, prompt=system_message, checkpointer=memory
    )

    config = {"configurable": {"thread_id": "test-thread"}}
    print(
        langgraph_agent_executor.invoke(
            {
                "messages": [
                    ("user", "Hi, I'm polly! What's the output of magic_function of 3?")
                ]
            },
            config,
        )["messages"][-1].content
    )
    # > The output of the magic function for the input 3 is 5.
    print(
        langgraph_agent_executor.invoke(
            {"messages": [("user", "Remember my name?")]}, config
        )["messages"][-1].content
    )
    # > Yes, you mentioned that your name is Polly.
    print(
        langgraph_agent_executor.invoke(
            {"messages": [("user", "what was that output again?")]}, config
        )["messages"][-1].content
    )
    # > The output of the magic function for the input 3 is 5.

########################################
############## Callbacks ############### 
########################################

### 0. Inherient BaseCallbackHandler
def build_baseCallbackHandler():
    from typing import Any, Dict, List
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import LLMResult
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    class LoggingHandler(BaseCallbackHandler):
        def on_chat_model_start(
            self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kwargs
        ) -> None:
            print("Chat model started")

        def on_llm_end(self, response: LLMResult, **kwargs) -> None:
            print(f"Chat model ended, response: {response}")

        def on_chain_start(
            self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs
        ) -> None:
            print(f"Chain {serialized.get('name')} started")

        def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
            print(f"Chain ended, outputs: {outputs}")

    callbacks = [LoggingHandler()]
    llm = ChatOpenAI(model="gpt-4o") 
    prompt = ChatPromptTemplate.from_template("What is 1 + {number}?")
    chain = prompt | llm
    chain_with_callbacks = chain.with_config(callbacks=callbacks)

    response = chain_with_callbacks.invoke({"number": "2"})
    """
    Chain RunnableSequence started
    Chain ChatPromptTemplate started
    Chain ended, outputs: messages=[HumanMessage(content='What is 1 + 2?')]
    Chat model started
    Chat model ended, response: generations=[[ChatGeneration(text='1 + 2 = 3', message=AIMessage(content='1 + 2 = 3', response_metadata={'id': 'msg_01NTYMsH9YxkoWsiPYs4Lemn', 'model': 'claude-3-sonnet-20240229', 'stop_reason': 'end_turn', 'stop_sequence': None, 'usage': {'input_tokens': 16, 'output_tokens': 13}}, id='run-d6bcfd72-9c94-466d-bac0-f39e456ad6e3-0'))]] llm_output={'id': 'msg_01NTYMsH9YxkoWsiPYs4Lemn', 'model': 'claude-3-sonnet-20240229', 'stop_reason': 'end_turn', 'stop_sequence': None, 'usage': {'input_tokens': 16, 'output_tokens': 13}} run=None
    """
    print(response)
    # > AIMessage(content='1 + 2 = 3', response_metadata={'id': 'msg_01NTYMsH9YxkoWsiPYs4Lemn', 'model': 'claude-3-sonnet-20240229', 'stop_reason': 'end_turn', 'stop_sequence': None, 'usage': {'input_tokens': 16, 'output_tokens': 13}}, id='run-d6bcfd72-9c94-466d-bac0-f39e456ad6e3-0')


########################################
############## Memrory ################# 
########################################

### 0. Automatic Hisotry Menagement
def auto_memory_chat():
    from langchain_openai import OpenAI
    from langchain_core.messages import SystemMessage, HumanMessage
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import START, MessagesState, StateGraph

    model = OpenAI(nanme="gpt-4o-mini")
    workflow = StateGraph(state_schema=MessagesState)

    def call_model(state: MessagesState):
        system_propmt = (
            "You are a helpful assistant. "
            "Answer all questions to the best of your ability."
        ) 
        messages = [SystemMessage(content=system_propmt)] + state["messages"]
        response = model.invoke(messages)
        return {"messages": response}

    workflow.add_node("model", call_model)
    workflow.add_edge(START, "model")

    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    response = app.invoke(
        {"messages": [HumanMessage(content="Translate to French: I love programming.")]},
        config={"configurable": {"thread_id": "1"}},
    )
    print(response)
    """
    {'messages': 
        [
            HumanMessage(content='Translate to French: I love programming.', additional_kwargs={}, response_metadata={}, id='be5e7099-3149-4293-af49-6b36c8ccd71b'),
            AIMessage(content="J'aime programmer.", additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 4, 'prompt_tokens': 35, 'total_tokens': 39, 'completion_tokens_details': {'reasoning_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_e9627b5346', 'finish_reason': 'stop', 'logprobs': None}, id='run-8a753d7a-b97b-4d01-a661-626be6f41b38-0', usage_metadata={'input_tokens': 35, 'output_tokens': 4, 'total_tokens': 39})
        ]
    }
    """
    app.invoke(
        {"messages": [HumanMessage(content="What did I just ask you?")]},
        config={"configurable": {"thread_id": "1"}},
    )
    """
    {'messages': 
        [
            HumanMessage(content='Translate to French: I love programming.', additional_kwargs={}, response_metadata={}, id='be5e7099-3149-4293-af49-6b36c8ccd71b'),
            AIMessage(content="J'aime programmer.", additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 4, 'prompt_tokens': 35, 'total_tokens': 39, 'completion_tokens_details': {'reasoning_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_e9627b5346', 'finish_reason': 'stop', 'logprobs': None}, id='run-8a753d7a-b97b-4d01-a661-626be6f41b38-0', usage_metadata={'input_tokens': 35, 'output_tokens': 4, 'total_tokens': 39}),
            HumanMessage(content='What did I just ask you?', additional_kwargs={}, response_metadata={}, id='c667529b-7c41-4cc0-9326-0af47328b816'),
            AIMessage(content='You asked me to translate "I love programming" into French.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 13, 'prompt_tokens': 54, 'total_tokens': 67, 'completion_tokens_details': {'reasoning_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_1bb46167f9', 'finish_reason': 'stop', 'logprobs': None}, id='run-134a7ea0-d3a4-4923-bd58-25e5a43f6a1f-0', usage_metadata={'input_tokens': 54, 'output_tokens': 13, 'total_tokens': 67})]}
        ]
    }
    """

### 1. Summarize Hisotry 
def sum_memory_chat():
    from langchain_openai import OpenAI
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
    from langchain_core.messages import HumanMessage, RemoveMessage
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import START, MessagesState, StateGraph

    demo_ephemeral_chat_history = [
        HumanMessage(content="Hey there! I'm Nemo."),
        AIMessage(content="Hello!"),
        HumanMessage(content="How are you today?"),
        AIMessage(content="Fine thanks!"),
    ]

    model = OpenAI(nanme="gpt-4o-mini")
    workflow = StateGraph(state_schema=MessagesState)

    def call_model(state: MessagesState):
        system_prompt = (
            "You are a helpful assistant. "
            "Answer all questions to the best of your ability. "
            "The provided chat history includes a summary of the earlier conversation."
        )
        system_message = SystemMessage(content=system_prompt)
        message_history = state["messages"][:-1] 
        if len(message_history) >= 4:
            last_human_message = state["messages"][-1] 
            summary_prompt = (
                "Distill the above chat messages into a single summary message. "
                "Include as many specific details as you can."
            )
            summary_message = model.invoke(
                message_history + [HumanMessage(content=summary_prompt)]
            )
            delete_messages = [RemoveMessage(id=m.id) for m in state["messages"]]
            human_message = HumanMessage(content=last_human_message.content)
            response = model.invoke([system_message, summary_message, human_message])
            message_updates = [summary_message, human_message, response] + delete_messages
        else:
            message_updates = model.invoke([system_message] + state["messages"])

        return {"messages": message_updates}

    workflow.add_node("model", call_model)
    workflow.add_edge(START, "model")

    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    print(
        app.invoke(
            {
                "messages": demo_ephemeral_chat_history
                + [HumanMessage("What did I say my name was?")]
            },
            config={"configurable": {"thread_id": "4"}},
        )
    )
    """
    {
        'messages': [
            AIMessage(content="Nemo greeted me, and I responded positively, indicating that I'm doing well.", additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 16, 'prompt_tokens': 60, 'total_tokens': 76, 'completion_tokens_details': {'reasoning_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_1bb46167f9', 'finish_reason': 'stop', 'logprobs': None}, id='run-ee42f98d-907d-4bad-8f16-af2db789701d-0', usage_metadata={'input_tokens': 60, 'output_tokens': 16, 'total_tokens': 76}),
            HumanMessage(content='What did I say my name was?', additional_kwargs={}, response_metadata={}, id='788555ea-5b1f-4c29-a2f2-a92f15d147be'),
            AIMessage(content='You mentioned that your name is Nemo.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 8, 'prompt_tokens': 67, 'total_tokens': 75, 'completion_tokens_details': {'reasoning_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_1bb46167f9', 'finish_reason': 'stop', 'logprobs': None}, id='run-099a43bd-a284-4969-bb6f-0be486614cd8-0', usage_metadata={'input_tokens': 67, 'output_tokens': 8, 'total_tokens': 75})
        ]
    }
    """


#################################### 
############## RAG ################# 
#################################### 

def retrive():
    import bs4
    from langchain import hub
    from langchain_community.document_loaders import WebBaseLoader
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langgraph.graph import START, StateGraph
    from typing_extensions import List, TypedDict
    from langchain_core.vectorstores import InMemoryVectorStore
    from langchain_openai import OpenAIEmbeddings
    from langchain.chat_models import init_chat_model

    llm = init_chat_model("gpt-4o-mini", model_provider="openai")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = InMemoryVectorStore(embeddings)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    docs = WebBaseLoader(
                web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
                bs_kwargs=dict(
                    parse_only=bs4.SoupStrainer(
                        class_=("post-content", "post-title", "post-header")
                    )
                ),
            ).load()
    all_splits = text_splitter.split_documents(docs)
    _ = vector_store.add_documents(documents=all_splits)
 
    prompt = hub.pull("rlm/rag-prompt")
 
    class State(TypedDict):
        question: str
        context: List[Document]
        answer: str

    def retrieve(state: State):
        retrieved_docs = vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}

    def generate(state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke({"question": state["question"], "context": docs_content})
        response = llm.invoke(messages)
        return {"answer": response.content}

    # Compile application and test
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()
    response = graph.invoke({"question": "What is Task Decomposition?"})
    print(response["answer"])
    # > Task Decomposition is the process of breaking down a complicated task into smaller, manageable steps to facilitate easier execution and understanding. Techniques like Chain of Thought (CoT) and Tree of Thoughts (ToT) guide models to think step-by-step, allowing them to explore multiple reasoning possibilities. This method enhances performance on complex tasks and provides insight into the model's thinking process.