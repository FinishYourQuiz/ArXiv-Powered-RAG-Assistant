########################################################
################## LangChain Deep Dive #################
########################################################

##### 1. Tool callings ##### 
from langchain.chat_models import init_chat_model
llm = init_chat_model("gpt-4o-mini", model_provider="openai")

from langchain_core.tools import tool
@tool
def add(a: int, b: int) -> int:
    """Adds a and b.""" # Function must have a docstring if description not provided.
    return a+b

@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b.""" # Function must have a docstring if description not provided.
    return a*b

tools = [add, multiply]
llm_with_tools = llm.bind_tools(tools)

from langchain_core.messages import HumanMessage
query = "What is 3 * 12? Also, what is 11 + 49?"
messages = [HumanMessage(query)]
response = llm_with_tools.invoke(messages)
messages.append(response)
print(response.tool_calls)
# > [{'name': 'multiply', 'args': {'a': 3, 'b': 12}, 'id': 'call_bu5GeUsLx5XDeETuk1FpVvWy', 'type': 'tool_call'}, {'name': 'add', 'args': {'a': 11, 'b': 49}, 'id': 'call_HVC7podr2Bj9eH0u9PS49cpZ', 'type': 'tool_call'}]
print(response)
# > content='' additional_kwargs={'tool_calls': [{'id': 'call_bu5GeUsLx5XDeETuk1FpVvWy', 'function': {'arguments': '{"a": 3, "b": 12}', 'name': 'multiply'}, 'type': 'function'}, {'id': 'call_HVC7podr2Bj9eH0u9PS49cpZ', 'function': {'arguments': '{"a": 11, "b": 49}', 'name': 'add'}, 'type': 'function'}], 'refusal': None} response_metadata={'token_usage': {'completion_tokens': 50, 'prompt_tokens': 88, 'total_tokens': 138, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_34a54ae93c', 'id': 'chatcmpl-BhiMFIYyFCqFkP0rhzeWqQn8b9Cbh', 'service_tier': 'default', 'finish_reason': 'tool_calls', 'logprobs': None} id='run--19363fed-9741-47c1-9421-539906690550-0' tool_calls=[{'name': 'multiply', 'args': {'a': 3, 'b': 12}, 'id': 'call_bu5GeUsLx5XDeETuk1FpVvWy', 'type': 'tool_call'}, {'name': 'add', 'args': {'a': 11, 'b': 49}, 'id': 'call_HVC7podr2Bj9eH0u9PS49cpZ', 'type': 'tool_call'}] usage_metadata={'input_tokens': 88, 'output_tokens': 50, 'total_tokens': 138, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}
for tool_call in response.tool_calls:
    selected_tool = {"add": add, "multiply": multiply}[tool_call["name"].lower()]
    tool_message = selected_tool.invoke(tool_call)
    messages.append(tool_message)
response = llm_with_tools.invoke(messages)
print(response)
# > content='The result of \\( 3 \\times 12 \\) is 36, and \\( 11 + 49 \\) equals 60.' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 30, 'prompt_tokens': 154, 'total_tokens': 184, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_34a54ae93c', 'id': 'chatcmpl-BhiQGTaunTRlqH1cWQ4BSPfj0LkoI', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None} id='run--19e40fa5-695f-4169-b091-fd2959dd69ef-0' usage_metadata={'input_tokens': 154, 'output_tokens': 30, 'total_tokens': 184, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}

##### 2. Return structured data as output ##### 
from typing import Optional
from pydantic import BaseModel, Field

class Joke(BaseModel):
    setup: str = Field(description="The set up of the joke")
    punchline: str = Field(description="The punchline to the joke")
    rating : Optional[int] = Field(
        default=None, description="How funny the joke is, from 1 to 10"
    )
structured_llm = llm.with_structured_output(Joke)
response = structured_llm.invoke("How are you today?")
print(response)
# > setup='Why was the cat sitting on the computer?' punchline='Because it wanted to keep an eye on the mouse!' rating=7
# > setup='How do you organize a space party?' punchline='You planet!' rating=8

from typing import Union

class ConversationalResponse(BaseModel):
    response: str = Field(description="A conversational respose to user query")

class FinalResponse(BaseModel):
    response: Union[Joke, ConversationalResponse]

structured_llm = llm.with_structured_output(FinalResponse)
response = structured_llm.invoke("How are you today?")
print(response)
# > response=ConversationalResponse(response="I'm just a collection of algorithms, so I don't have feelings—but I'm here and ready to help you! How about you? How's your day going?")

##### 3. Debugging  ##### 
from langchain.globals import set_verbose, set_debug
verbose = True
set_verbose(verbose) # print statements for "important" events in your chain. 
set_debug(not verbose) # add logging statements for ALL events in your chain.

##### 4. Stream runnables  ##### 

### Stream Output as string ###
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

model = init_chat_model("gpt-4o-mini", model_provider="openai")
prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
parser = StrOutputParser()
chain = prompt | model | parser 
for chunk in chain.stream({"topic": "parrot"}):
    print(chunk, end="|", flush=True)
# |Why| did| the| par|rot| wear| a| rain|coat|?
# |Because| it| wanted| to| be| a| poly|uns|aturated|!||

### Stream Input as json ###

chain_2 = model | JsonOutputParser()
for chunk in chain_2.stream(
    "output a list of the countries france, spain and japan and their populations in JSON format. "
    'Use a dict with an outer key of "countries" which contains a list of countries. '
    "Each country should have the key `name` and `population`"
):
    print(chunk, flush=True)

"""
{'countries': []}
{'countries': [{}]}
{'countries': [{'name': ''}]}
{'countries': [{'name': 'France'}]}
{'countries': [{'name': 'France', 'population': 652}]}
{'countries': [{'name': 'France', 'population': 652735}]}
{'countries': [{'name': 'France', 'population': 65273511}]}
{'countries': [{'name': 'France', 'population': 65273511}, {}]}
{'countries': [{'name': 'France', 'population': 65273511}, {'name': ''}]}
{'countries': [{'name': 'France', 'population': 65273511}, {'name': 'Spain'}]}
{'countries': [{'name': 'France', 'population': 65273511}, {'name': 'Spain', 'population': 467}]}
{'countries': [{'name': 'France', 'population': 65273511}, {'name': 'Spain', 'population': 467547}]}
{'countries': [{'name': 'France', 'population': 65273511}, {'name': 'Spain', 'population': 46754778}]}
{'countries': [{'name': 'France', 'population': 65273511}, {'name': 'Spain', 'population': 46754778}, {}]}
{'countries': [{'name': 'France', 'population': 65273511}, {'name': 'Spain', 'population': 46754778}, {'name': ''}]}
{'countries': [{'name': 'France', 'population': 65273511}, {'name': 'Spain', 'population': 46754778}, {'name': 'Japan'}]}
{'countries': [{'name': 'France', 'population': 65273511}, {'name': 'Spain', 'population': 46754778}, {'name': 'Japan', 'population': 126}]}
{'countries': [{'name': 'France', 'population': 65273511}, {'name': 'Spain', 'population': 46754778}, {'name': 'Japan', 'population': 126476}]}
{'countries': [{'name': 'France', 'population': 65273511}, {'name': 'Spain', 'population': 46754778}, {'name': 'Japan', 'population': 126476461}]}
"""


