import json

import httpx
from openai import OpenAI
from urllib3 import disable_warnings
from urllib3.exceptions import InsecureRequestWarning

# 不走代理、忽略自签证书告警
http_client = httpx.Client(verify=False)
disable_warnings(InsecureRequestWarning)

# base_url = 'https://tokenfly.com/'
# api_key = 'sk-fcqkMWw7NpXUkNw1ZybZpqvkl2ohDn68i1YkS8BFK50xfuxZ'
base_url = "http://1.95.199.64:8084/"
api_key = "sk-a1867b63"

client = OpenAI(
    base_url=base_url,
    api_key=api_key,
    http_client=http_client,
)

model_name = "gemini-3-pro-preview"
# model_name = "tokenfly-01/glm-5.2"
model_name = "OAI:tokenfly-01/glm-5.2"

# 1. 定义工具 (OpenAI function tools 格式)
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的当前天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "城市名称，例如：北京",
                    }
                },
                "required": ["location"],
            },
        },
    }
]

messages = [
    {"role": "system", "content": "你是一位资深的 Python 开发者。"},
    {"role": "user", "content": "北京现在的天气怎么样？"},
]

# 2. 第一轮：让模型决定是否调用工具
response = client.chat.completions.create(
    model=model_name,
    messages=messages,
    tools=tools,
    tool_choice="auto",
    max_tokens=512,
    temperature=0.7,
    # reasoning_effort='high'

)

print(response)
msg = response.choices[0].message
print("=== 第一轮响应 ===")
print("content:", msg.content)
print("tool_calls:", msg.tool_calls)

# 3. 如果模型请求调用工具，执行工具并把结果回传，做第二轮
if msg.tool_calls:
    messages.append(msg.model_dump(exclude_none=True))

    for tc in msg.tool_calls:
        args = json.loads(tc.function.arguments or "{}")
        print(f"\n调用工具 {tc.function.name}，参数：{args}")

        # 模拟工具执行结果
        tool_result = {"location": args.get("location"), "weather": "晴", "temp_c": 28}

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(tool_result, ensure_ascii=False),
            }
        )

    # 4. 第二轮：把工具结果交给模型生成最终回复
    final = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=512,
        temperature=0.7,
    )
    print("\n=== 最终回复 ===")
    print(final.choices[0].message.content)
