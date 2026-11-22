from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def demo_basic_chain():
    """Demonstrate a basic chain using LCEL and Runnable."""
    # Component 1: Define the prompt template using LCEL
    prompt = ChatPromptTemplate.from_template("You are a helpful assistant. Answer in one sentence: {question}")

    model = ChatOllama(model="qwen2.5:3b", temperature=0.7)
    parser = StrOutputParser()

    # Compose components with pipe operator
    chain = prompt | model | parser

    # Execute the chain with an input
    result = chain.invoke({"question": "What is LangChain?"})

    print(f"Response: {result}")
    return result

def demo_batch_execution():
    """Demonstrate batch execution for multiple inputs"""
    prompt = ChatPromptTemplate.from_template(
        "Translate to Khmer: {text}"
    )
    model = ChatOllama(model="qwen2.5:3b", temperature=0.7)
    parser = StrOutputParser()

    chain = prompt | model | parser
    inputs = [
        {"text": "What is LangChain?"},
        {"text": "Hello, how are you?"},
        {"text": "What is your name"},
    ]

    results = chain.batch(inputs)

    for text in zip(inputs, results):
        print(f"Input: {text[0]['text']} => Output: {text[1]}")

if __name__ == "__main__":
    # demo_basic_chain()
    demo_batch_execution()
