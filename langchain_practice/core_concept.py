from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model

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

def demo_streaming():
    """Demonstrate streaming for real-time output"""
    prompt = ChatPromptTemplate.from_template(
        "Write a haiku about: {topic}"
    )
    model = ChatOllama(model="qwen2.5:3b", temperature=0.7)
    parser = StrOutputParser()

    chain =  prompt | model | parser

    #Streaming - run with streaming enabled

    for chunk in chain.stream({"topic": "nature"}):
        print(chunk, end="", flush=True)

def demo_schema_inspection():
    prompt = ChatPromptTemplate.from_template(
        "Write a haiku about: {topic}"
    )
    model = ChatOllama(model="qwen2.5:3b", temperature=0.7)
    parser = StrOutputParser()

    chain = prompt | model | parser

    #inspect input and output schemas

    input_schema = chain.input_schema.model_json_schema()
    output_schema = chain.output_schema.model_json_schema()

    print(f"Input schema: {input_schema}")
    print(f"Output schema: {output_schema}")

def exercise_first_chain():
    """
    EXERCISE: Create a chain that:
    1. Take a product name and target audience
    2. Generates a marketing tagline
    3. Return just the tagline as a string

    Test with: product="AI Course", audience="developers"
    """

    prompt = ChatPromptTemplate.from_template(
        "Generate a marketing tagline for product: {product} and the target audience: {audience}"
    )
    model = ChatOllama(model="qwen2.5:3b")
    parser = StrOutputParser()
    chain = prompt | model | parser

    result = chain.invoke({"product": "AI Course", "audience": "developers"})
    print(f"Response: {result}")

def new_way():
    model = init_chat_model("qwen2.5:3b", temperature=0.7)
    prompt = ChatPromptTemplate.from_template()

if __name__ == "__main__":
    # demo_basic_chain()
    # demo_batch_execution()
    # demo_streaming()
    # demo_schema_inspection(
    exercise_first_chain()