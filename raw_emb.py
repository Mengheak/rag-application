from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

conversation = client.chat.completions.create(
    model="qwen2.5:3b",
    messages=[{"role": "system", "content": "You are a helpful assistant"},
              {"role": "user", "content": "What is the capital of France?"}],
)

print(conversation.choices[0].message.content)