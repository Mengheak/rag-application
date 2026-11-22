from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    MessagesPlaceholder,
)

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
)

from langchain_ollama import ChatOllama

model = ChatOllama(model="qwen2.5:3b")

def demo_basic_template():
    simple = ChatPromptTemplate.from_template(
        "Translate '{text}' to {language}"
    )

    messages = simple.invoke({"text": "Hello World", "language": "Khmer"})
    # messages = simple.format_messages(text="Hello World", language="Khmer")
    print("Simple template: ")
    print(messages)

    # Multi-message template
    multi = ChatPromptTemplate.from_messages([
        ("system", "You are a translator. Be concise."),
        ("human", "Translate '{text}' to {language}"),
    ])
    messages = multi.format_messages(text="Good morning", language="Japanese")
    print("\nMulti-message template:")
    for msg in messages:
        print(f"  {type(msg).__name__}: {msg.content}")

def demo_message_types():
    model = ChatOllama(model="qwen2.5:3b", temperature=0)

    #Build conversation manually
    messages = [
        SystemMessage(content="You are a Math tutor. Be brief."),
        HumanMessage(content="What's 5 * 5?"),
        AIMessage(content="25"),
        HumanMessage(content="And what if I add 10"),
    ]

    prompt = ChatPromptTemplate.from_messages(messages)

    chain = prompt | model
    print(f"Conversation result: {chain.invoke(input = {}).content}")


def demo_messages_placeholder():
    """Use MessagesPlaceholder for dynamic conversation history."""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )

    # Simulate conversation history
    history = [
        HumanMessage(content="My name is Paulo"),
        AIMessage(content="Nice to meet you, Paulo!"),
    ]

    messages = prompt.format_messages(history=history, question="What's my name?")

    print("With history placeholder:")
    for msg in messages:
        print(f"  {type(msg).__name__}: {msg.content[:50]}...")

    # Execute
    model = ChatOllama(model="qwen2.5:3b", temperature=0)

    chain = prompt | model
    response = chain.invoke({"history": history, "question": "What's my name?"})
    print(f"\nResponse: {response.content}")

def demo_few_shot():
    """Few-shot prompting with examples."""

    # Define examples
    examples = [
        {"word": "happy", "opposite": "sad"},
        {"word": "fast", "opposite": "slow"},
        {"word": "big", "opposite": "small"},
    ]

    # Template for each example
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "What's the opposite of '{word}'?"),
            ("ai", "The opposite of '{word}' is '{opposite}'."),
        ]
    )

    # Few-shot wrapper
    few_shot = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )

    # Final prompt
    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You give the opposite of words. Follow the examples."),
            few_shot,
            ("human", "What's the opposite of '{word}'?"),
        ]
    )

    # Test
    model = ChatOllama(model="qwen2.5:3b", temperature=0)
    chain = final_prompt | model

    response = chain.invoke({"word": "bright"})
    print(f"Few-shot result: {response.content}")

def demo_prompt_composition():
    """Compose prompts from reusable parts."""

    # Reusable system prompt
    persona = ChatPromptTemplate.from_messages(
        [("system", "You are a {role}. Your tone is {tone}.")]
    )

    # Reusable task prompt
    task = ChatPromptTemplate.from_messages([("human", "{task}")])

    # Combine
    full_prompt = persona + task

    # Test different combinations
    model = ChatOllama(model="qwen2.5:3b", temperature=0.7)
    chain = full_prompt | model

    # As a pirate
    response = chain.invoke(
        {
            "role": "pirate captain",
            "tone": "adventurous",
            "task": "Tell me about your ship",
        }
    )
    print(f"Pirate: {response.content[:100]}...")

    # As a scientist
    response = chain.invoke(
        {
            "role": "scientist",
            "tone": "precise and academic",
            "task": "Explain photosynthesis",
        }
    )
    print(f"\nScientist: {response.content[:100]}...")


if __name__ == "__main__":
    # demo_basic_template()
    # demo_message_types()
    # demo_messages_placeholder()
    # demo_few_shot()
    demo_prompt_composition()

