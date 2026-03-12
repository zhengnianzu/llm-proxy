## 项目介绍

本项目对 Anthropic `/v1/messages`和 OpenAI `/v1/chat/completions` 请求进行了封装，解决了:

1. 日志自动记录
2. token数一键统计

## 配置

- 复制模板

```shell
cp .env.example .env
```

- 配置说明
```text
这两个key配置后强制使用给定值，当值为空时，才选择用户传入的模型
UPSTREAM_API_KEY=sk-xxx
MODEL_ID=gpt-5
```

```shell
# 转发前的地址
UPSTREAM_URL=https://yibuapi.com/v1  # 后缀写到v1
UPSTREAM_API_KEY="填入AK"  # 密钥
MODEL_ID=

# 转发后的地址
PROXY_HOST=127.0.0.1
PROXY_PORT=4000

# 其他参数
SSL_VERIFY=false
BAN_EXPLORE=false
BAN_STREAM=false

# 代理相关
TRUST_ENV=true  # 为true时使用下面全局环境变量，密码特殊字符记得转码
HTTP_PROXY=http://华为账号:华为密码@proxyhk.huawei.com:8080
HTTPS_PROXY=http://华为账号:华为密码@proxyhk.huawei.com:8080
NO_PROXY=localhost,127.0.0.1,*.huawei.com,*.local,*.lan,10.70.85.106
```

## 环境

```shell
pip install -r requirements.txt
```

## 启动

```shell
python app.py
```

## 测试

```
python test/test-api.py
```

## 统计token数

1. 基于web界面

```text
在代理服务启动后，访问 http://127.0.0.1:4000/
```

2. 基于函数调用

- 命令行调用

```shell
python print_stats_summary.py --date_start=2026-01-01 --date_end=2026-03-09
```

- 参数解释

```text
model: 过滤模型，忽略大小写，多个模型用,拼接
date_start: 过滤日期-开启，格式YYYY-MM-DD
date_end: 过滤日期-结束，格式YYYY-MM-DD
status: 过滤状态: 全部、成功、失败
```

## 多轮对话可视化

```text
在代理服务启动后，访问 http://127.0.0.1:4000/chat_viewer
```