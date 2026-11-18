from dotenv import load_dotenv
from importlib.metadata import version
load_dotenv()


core_version = version("langchain-core")
lg_version = version("langgraph")
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama


print(f"langchain version: {core_version}")
print(f"langgraph version: {lg_version}")

def main():
    # llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    # response = llm.invoke("Say 'setup complete' in one word")
    # print(f"Response from openai: {response}")
    print(f"langchain-core version: {version('langchain-core')}")
    print(f"langgraph version: {version('langgraph')}")

    llm = ChatOllama(model="qwen2.5:3b", temperature=0)
    response = llm.invoke("Reply with exactly one word: ready")
    print(f"Response from Ollama: {response.content}")
if __name__ == "__main__":
    main()