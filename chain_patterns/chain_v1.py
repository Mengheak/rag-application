from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain.chat_models import init_chat_model


model = init_chat_model(
    model="qwen2.5:3b",
    model_provider="ollama",
    temperature=0
)

def demo_basic_chain():
    prompt = ChatPromptTemplate.from_template("Summarize the following text in one sentence: {text}")
    parser = StrOutputParser()

    chain = prompt | model | parser

    result = chain.invoke({"text": "LangChain is a framework for developing applications powered by language models."})

    print(f"Summary: {result}")

if __name__ == "__main__":
    demo_basic_chain()