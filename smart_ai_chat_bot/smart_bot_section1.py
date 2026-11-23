from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langsmith import traceable, Client
from langsmith.run_trees import get_cached_client
from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv
import os

load_dotenv()

# Langsmith configuration

if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ.setdefault("LANGSMITH_PROJECT", "Smart Q&A bot project")
    print(f"LangSmith is configured. - Project: {os.getenv('LANGSMITH_PROJECT')}")

# Schema Definition
class QAResponse(BaseModel):
    answer: str=Field(description="The answer to the question.")
    confidence: str=Field(description="Confidence level: high, medium, low")
    reasoning: str=Field(description="Reasoning of the answer")
    follow_up_questions: List[str]=Field(description="Follow up questions related to the topic", default_factory=list)
    sources_needed: bool = Field(
        description="Indicates whether sources are needed for the answer.",
        default=False,
    )

#Bot implementation
class SmartQABot:
    def __init__(self, model_name: str="qwen2.5:3b", temperature: float = 0.3):
        self.model = ChatOllama(
            model=model_name,
            temperature=temperature
        ).with_structured_output(QAResponse)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are a knowledgeable Q&A assistant.
            
            Your guidelines:
            - Answer questions accurately and concisely
            - Be honest about uncertainty - set confidence to 'low' if unsure
            - Provide clear reasoning for your answers
            - Suggest relevant follow-up questions
            - Indicate if external sources would help
            
            Always respond with accurate, helpful information.
            """),
            ('human', "{question}")
        ])
        self.chain = self.prompt | self.model

    @traceable(name="ask_question", run_type="chain")
    def ask(self, question: str) -> QAResponse:
        try:
            response = self.chain.invoke({"question": question})
            return response
        except Exception as e:
            return QAResponse(
                answer="I'm sorry, I couldn't process your question at this time.",
                confidence="low",
                reasoning=str(e),
                follow_up_questions=["Could you please try again later?"],
                sources_needed=True,
            )

    @traceable(name="ask_batch", run_type="chain")
    def ask_batch(self, questions: List[str]) -> List[QAResponse]:
        """Ask multiple questions in parallel."""
        inputs = [{"question": q} for q in questions]
        return self.chain.batch(inputs)

def demo_qa_bot():
    bot = SmartQABot()

    questions = [
        "What is the capital of France?",
        "Explain the theory of relativity.",
        "How does photosynthesis work?",
    ]

    print("=" * 60)
    print("SMART Q&A BOT DEMO")
    print("=" * 60)

    for question in questions:

        print(f"\n Question: {question}")
        print("-" * 40)

        response = bot.ask(question)

        print(f"Question: {question}")
        print(f"Answer: {response.answer}")
        print(f"Confidence: {response.confidence}")
        print(f"Reasoning: {response.reasoning}")
        print(f"Follow-up Questions: {response.follow_up_questions}")
        print(f"Sources Needed: {response.sources_needed}")
        print("-" * 60)
def demo_error_handling():
    """Demonstrate error handling."""

    bot = SmartQABot()

    print("\n" + "=" * 60)
    print("ERROR HANDLING DEMO")
    print("=" * 60)

    # Test with a very long question (edge case)
    long_question = "What is " + "very " * 100 + "important?"

    response = bot.ask(long_question)
    print(f"Handled gracefully: {response.confidence}")

def demo_batch_processing():
    """Demonstrate batch processing."""

    bot = SmartQABot()

    questions = [
        "What is Python?",
        "What is JavaScript?",
        "What is Rust?",
    ]

    print("\n" + "=" * 60)
    print("BATCH PROCESSING DEMO")
    print("=" * 60)

    responses = bot.ask_batch(questions)

    for q, r in zip(questions, responses):
        print(f"\n{q}")
        print(f"  -> {r.answer[:100]}...")
        print(f"  Confidence: {r.confidence}")


if __name__ == "__main__":

    try:
        demo_qa_bot()
        # demo_batch_processing()
        # demo_error_handling()
        print("\n" + "=" * 60)
    finally:
        get_cached_client().flush()