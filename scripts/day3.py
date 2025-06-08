
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

import getpass
import os

"""
===== Using Language Models ===== 
"""
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")


# return a BaseChatModel from langchain-core, where BaseChatModel is Base class for chat models. 
model = init_chat_model("gpt-4o-mini", model_provider="openai")

messages = [
    SystemMessage("Translate the following from English into Italian"),
    HumanMessage("Hi")
]

# invoke method input a string/list and output a BaseMessage
output = model.invoke(messages)
print(output)
# > content='Ciao' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 2, 'prompt_tokens': 19, 'total_tokens': 21, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_34a54ae93c', 'id': 'chatcmpl-BgHvZbgkB2g1Lp9Pas8E4DIC1oTQO', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None} id='run--432b3dae-d9f5-4d6a-a99e-f33a41bf555a-0' usage_metadata={'input_tokens': 19, 'output_tokens': 2, 'total_tokens': 21, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}

# stream method yielding output of invoke
for token in model.stream(messages):
    print(token.content, end="|")
# > |C|iao|!||


"""
===== Using Prompt Templates ===== 
"""
system_template = "Translate the following from English into {language}"
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)
prompt = prompt_template.invoke({"language": "Italian", "text": "hi!"})
print(prompt)
# > messages=[SystemMessage(content='Translate the following from English into Italian', additional_kwargs={}, response_metadata={}), HumanMessage(content='hi!', additional_kwargs={}, response_metadata={})]
messages = prompt.to_messages() 
print(messages)
# > [SystemMessage(content='Translate the following from English into Italian', additional_kwargs={}, response_metadata={}), HumanMessage(content='hi!', additional_kwargs={}, response_metadata={})]
response = model.invoke(prompt)
print(response.content)
# > Ciao!
