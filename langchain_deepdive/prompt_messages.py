from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

# prompt = ChatPromptTemplate.from_template(
#     "Tell me a {adjective} about {topic}"
# )

# messages = prompt.format_messages(adjective="funny", topic="chickens")
# print(messages)

# multi-message templates
prompt = ChatPromptTemplate.from_messages(
    [("system", "You are a helpful assistant that translates {input_language} to {output_language}"),
     ("human", "Translate the following text: {text}")]
)

messages = prompt.format_messages(
    input_language="en",
    output_language="fr",
    text="What is the capital of French?",
)

model = init_chat_model(
    model="qwen2.5:3b",
    model_provider="ollama",
    stream=True,
    temperature=0
)
response = model.invoke(messages)

# if isinstance(response, AIMessage):
print(response.content)

# Message types:
from langchain_core.messages import (
HumanMessage,
SystemMessage,
AIMessage,
ToolMessage,
ChatMessage
)

messages = [
    SystemMessage(content="Answer geography questions in one sentence."),
    HumanMessage(content="What is the capital of France?"),
    AIMessage(
        content="",
        tool_calls=[{
            "name": "lookup_capital",
            "args": {"country": "France"},
            "id": "call_1",
        }],
    ),
    ToolMessage(content="Paris", tool_call_id="call_1"),
    AIMessage(content="The capital of France is Paris."),
]

#few shot example
from langchain_core.prompts import FewShotChatMessagePromptTemplate
examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
]

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)
final_prompt = ChatPromptTemplate.from_messages([
    ("system", "Give the opposite of each word"),
    few_shot_prompt,
    ("human", "{input}")]
)
few_shot_response = model.invoke(
    final_prompt.invoke({"input": "far"})
)


print(f"Response by few shot prompt: \n{few_shot_response.content}")


# Reusable prompt

system_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}")
])
user_prompt = ChatPromptTemplate.from_messages([
    ("human", "{question}"),
])

full_prompt = system_prompt + user_prompt

fin = full_prompt.format_messages(
    role="helpful assistant",
    question="What is the capital of France?",
)