from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import (
    StrOutputParser,
)
from langchain.chat_models import init_chat_model

llm = init_chat_model(
    model="qwen2.5:3b",
    model_provider="ollama",
    temperature=0,
)
parser = StrOutputParser()
prompt = ChatPromptTemplate.from_template("Wire a short poem about {topic}")

chain = prompt | llm | parser

response = chain.invoke({"topic": "nature"})
print(type(response))

# Json output parser sample
