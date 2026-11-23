from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
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


def demo_parallel_chain():
    """Run multiple chains in parallel."""
    # define individual chain
    summarize_prompt = ChatPromptTemplate.from_template(
        "Summarize in two sentences: {text}",
    )
    keywords_prompt = ChatPromptTemplate.from_template(
        "Extract keywords in the following text: {text}\n Return as comma-separated list",
    )
    sentiment_prompt = ChatPromptTemplate.from_template(
        "What is the sentiment of the following text? {text}",
    )

    parser = StrOutputParser()
    # Parallel execution
    analysis_chain = RunnableParallel(
        summary=summarize_prompt | model | parser,
        keywords=keywords_prompt | model | parser,
        sentiment=sentiment_prompt | model | parser,
    )
    text = """
    The new AI features are absolutely incredible! Users are loving the
    faster response times and improved accuracy. However, some have noted
    that the pricing could be more competitive. Overall, the product
    launch has been a massive success with record-breaking adoption rates.
    """

    results = analysis_chain.invoke({"text": text})
    print("Analysis Results:")
    print("Parallel Analysis Results:")
    print(f"  Summary: {results['summary']}")
    print(f"  Keywords: {results['keywords']}")
    print(f"  Sentiment: {results['sentiment']}")


def demo_passthrough_chain():
    """A chain that demonstrates passthrough functionality."""
    prompt = ChatPromptTemplate.from_template(
        "Original question: {question}\n"
        "Context: {context}\n\n"
        "Answer the question based on the context."
    )

    # Simulate a retriever operation

    def fake_retriever(input_dict):
        return "LangChain was created by Harrison Chase in 2022."

    chain = (
            RunnableParallel(
                context=RunnableLambda(fake_retriever),
                question=RunnablePassthrough(),
            )
            | RunnableLambda(lambda x: {
        "context": x["context"],
        "question": x["question"],
    }) | prompt | model | StrOutputParser()
    )

    result = chain.invoke({"Who created LangChain?"})
    print(result)


if __name__ == "__main__":
    # demo_basic_chain()
    # demo_parallel_chain()
    demo_passthrough_chain()