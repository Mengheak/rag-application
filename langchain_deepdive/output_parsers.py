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
    JsonOutputParser, PydanticOutputParser, format_instructions
)
from langchain.chat_models import init_chat_model

llm = init_chat_model(
    model="qwen2.5:3b",
    model_provider="ollama",
    temperature=0,
)

def demo_str_output_parser():
    parser = StrOutputParser()
    prompt = ChatPromptTemplate.from_template("Wire a short poem about {topic}")

    chain = prompt | llm | parser

    response = chain.invoke({"topic": "nature"})
    print(type(response))

# Json output parser sample

def demo_json_output_parser():
    parser = JsonOutputParser()

    prompt = ChatPromptTemplate.from_template(
        "Return a JSON object with 'name' and 'age' for: {description}"
    )

    chain = prompt | llm | parser
    response = chain.invoke({"description": "A 20 years old developer named Heak"})
    print(response)

# Pydantic output parser
def demo_pydantic_output_parser():
    from pydantic import BaseModel, Field

    class Person(BaseModel):
        name: str = Field(..., description="The person's name")
        age: int = Field(..., description="The person's age")
        occupation: str = Field(..., description="The person's occupation")
    parser = PydanticOutputParser(pydantic_object=Person)
    prompt = ChatPromptTemplate.from_template(
        "Return a JSON object with 'name', 'age' and 'occupation' for: {description}"
    ).partial(format_instructions=parser.get_format_instructions())
    chain = prompt | llm | parser
    result = chain.invoke({"description": "A 20 years old developer named Heak"})
    print(result)
if __name__ == "__main__":
    # demo_str_output_parser()
    # demo_json_output_parser()
    demo_pydantic_output_parser()