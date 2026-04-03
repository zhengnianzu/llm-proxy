import os

import anthropic
import httpx
from urllib3 import disable_warnings
from urllib3.exceptions import InsecureRequestWarning


base_url = "127.0.0.1"
os.environ["no_proxy"] = base_url
os.environ["NO_PROXY"] = base_url

http_client = httpx.Client(verify=False)
disable_warnings(InsecureRequestWarning)
print(f"noproxy: {os.environ['no_proxy']}")

client = anthropic.Anthropic(
    base_url=f"http://{base_url}:4000",
    http_client=http_client,
)

model_name = "claude-sonnet-4-6"
client.api_key = "sk-213"


def extract_text(response):
    texts = []
    for block in response.content:
        if block.type == "text":
            texts.append(block.text)
    return "\n".join(texts).strip()


messages = []
questions = [
    "你是谁？",
    "你在哪里？",
    "你可以做什么？",
]

for i, question in enumerate(questions, start=1):
    messages.append({"role": "user", "content": question})

    response = client.messages.create(
        model=model_name,
        max_tokens=512,
        temperature=0.7,
        system="你是一位资深的 Python 开发者，请简洁回答并保持上下文一致。",
        messages=messages,
    )

    answer = extract_text(response)

    print(f"Q{i}: {question}")
    print(f"A{i}: {answer}")
    print()

    messages.append({"role": "assistant", "content": answer})
