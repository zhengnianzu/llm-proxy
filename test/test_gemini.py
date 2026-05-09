from openai import OpenAI

client = OpenAI(
    base_url="http://1.95.199.64:8085/",
    api_key="sk-18842394415-key1",
)


model_name = "gemini-3-pro-preview"
model_name = "kimi-k2.6"

response = client.chat.completions.create(
    model=model_name,
    messages=[
        {"role": "user", "content": "Hello, say hi in one sentence."}
    ],
    max_tokens=100,
)

print(response.choices[0].message.content)
