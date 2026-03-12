import httpx
from openai import OpenAI
import os
from urllib3 import disable_warnings
from urllib3.exceptions import InsecureRequestWarning
import anthropic

os.environ['no_proxy'] = "127.0.0.1"
# 创建不经过代理的HTTP客户端
http_client = httpx.Client(verify=False)
disable_warnings(InsecureRequestWarning)


# 1. 初始化客户端 (SDK 会默认读取环境变量 ANTHROPIC_API_KEY)
client = anthropic.Anthropic(
    base_url="http://127.0.0.1:4000",
    http_client=http_client)

model_name='claude-opus-4-6'
client.api_key = 'xx'


# 1. 定义工具 (Tools Schema)
tools = [
    {
        "name": "get_weather",
        "description": "获取指定城市的当前天气",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "城市名称，例如：北京"
                }
            },
            "required": ["location"]
        }
    }
]

# 2. 发送消息
response = client.messages.create(
    model=model_name,  # 指定模型版本
    max_tokens=512,                     # 最大输出长度
    temperature=0.7,                     # 随机性控制 (0-1)
    system="你是一位资深的 Python 开发者。", # 设置系统提示词（可选）
    tools=tools, # 传入工具定义
    messages=[
        {"role": "user", "content": "北京现在的天气怎么样？"}
    ],
    # stream=True
)

# 3. 打印回复内容
for line in response:
    print(line)
