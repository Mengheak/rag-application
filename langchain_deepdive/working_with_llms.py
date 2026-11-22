from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
import os

from openai import max_retries


def demo_init_chat_model():

    #easy to switch model providers
    if os.getenv("ANTHROPIC_API_KEY"):
        claude = init_chat_model(
            model="claude-sonnet-4-5",
            model_provider="anthropic",
            temperature=0.7,
            streaming=True,
            max_retries=3
        )
        response = claude.invoke("What is the Capital of Cambodia?")
        print(f"Response from anthropic : {response.content}")

    chat_model = init_chat_model(
        model="qwen2.5:3b",
        model_provider="ollama",
        temperature=0.7,
        streaming=True,
        max_messages=3,
    )
    response = chat_model.invoke("What is the Capital of Cambodia?")
    print(f"Response : {response.content}")
    return chat_model

def demo_model_comparison():
    prompt = "Explain recursion in one sentence"

    models = {
        "gpt-4o-mini": init_chat_model(
            model="gpt-4o-mini",
            temperature=0.7,
            streaming=False,
        ),
        "gpt-4o": init_chat_model(
            model="gpt-4o",
            temperature=0.7,
            streaming=False,
        )
    }
    if os.getenv("ANTHROPIC_API_KEY"):
        models["claude-sonnet-4-5"] = init_chat_model(
            model="claude-sonnet-4-5",
            model_provider="anthropic",
            temperature=0.7,
            streaming=False,
        )
    for models_name, model in models.items():
        response = model.invoke(prompt)
        print(f"Response from {models_name} : {response.content}")

def demo_message():
    model = ChatOllama(
        model="qwen2.5:3b",
        temperature=0
    )

    # using message objects (more control over roles)
    messages = [
        SystemMessage(content="You are a helpful assistant"),
        HumanMessage(content="What is the capital of Cambodia? answer in one word"),
    ]

    print("Using message objects")
    print(f"Messages : {messages[0]} | {messages[1]}")

    response = model.invoke(messages)
    print(f"Response using message objects: {response.content}")

    # Multi-turn conversation using message objects
    messages.append(response)
    print("Continuing conversation")
    print(messages)
    messages.append(HumanMessage(content="What is the capital of French?"))

if __name__ == "__main__":
    # demo_init_chat_model()
    # demo_model_comparison()
    demo_message()