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

if __name__ == "__main__":
    demo_basic_chain()

