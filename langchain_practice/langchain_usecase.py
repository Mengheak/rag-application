MODEL = "qwen2.5:3b"

guides = {
    "meeting": (
        "If meeting audio breaks up during a download, enable traffic "
        "prioritization on the router and give voice calls high priority."
    ),
    "password": (
        "To change the Wi-Fi password, open Wireless Settings, "
        "enter a new password, and save the changes."
    ),
}

def find_guide(question):
    for keyword, guide in guides.items():
        if keyword in question.lower():
            return guide
    return "No matching guide found."

question = "Why does my meeting audio break up during a download?"
instructions = "Answer using only the guide. If it doesn't answer the question, say so."

def without_langchain(question):
    import requests

    guide = find_guide(question)

    messages = [
        {"role": "system", "content":instructions},
        {"role": "user", "content":f"Guide:\n{guide}\n\nQuestion:\n{question}"},
    ]
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={"model": MODEL, "messages": messages, "stream": False},
        timeout=120
    )
    response.raise_for_status()

    answer = response.json()["message"]["content"]
    print(answer)

def with_langchain(question):
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_ollama import ChatOllama

    prompt = ChatPromptTemplate.from_messages([
        ("system", instructions),
        ("human", "Guide:\n{guide}\n\nQuestion:\n{question}"),
    ])
    model = ChatOllama(model=MODEL)
    chain = prompt | model

    response = chain.invoke({
        "guide":find_guide(question),
        "question":question,
    })
    print(response.content)
if __name__ == "__main__":
    without_langchain(question)
    print("="*50)
    with_langchain(question)