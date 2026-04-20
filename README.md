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

# API_KEY 用于接口鉴权，支持多个 key，用逗号分隔
# 不配置则跳过鉴权，任何请求都能访问
API_KEY=key1,key2,key3
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

# 监控后台登录保护
# 配置这两个值后，会保护 /、/query、/history、/failures、/statistic、/metrics/*、/logs/*
MONITOR_USERNAME=admin
MONITOR_PASSWORD=请改成强密码

# Session Cookie 配置
MONITOR_SESSION_SECRET=请改成一串随机长字符串
MONITOR_COOKIE_SECURE=false
MONITOR_SESSION_MAX_AGE=43200

# 代理相关
TRUST_ENV=true  # 为true时使用下面全局环境变量，密码特殊字符记得转码
HTTP_PROXY=http://华为账号:华为密码@proxyhk.huawei.com:8080
HTTPS_PROXY=http://华为账号:华为密码@proxyhk.huawei.com:8080
NO_PROXY=localhost,127.0.0.1,*.huawei.com,*.local,*.lan,10.70.85.106
```

如果你想让日志目录能区分不同 `.env` 启动的实例，不需要手动配置额外变量。
使用 `./app start --env .env.xxx` 时，CLI 会自动把 env 文件名转换成启动时环境变量 `LOG_TASK_TAG`。

例如：

```text
./app start --env .env.prod
```

会自动生成类似：

```text
LOG_TASK_TAG=env-prod
```

因此日志目录会从：

```text
logs_anthropic_wy92_260407
```

变成：

```text
logs_anthropic_env-prod_wy92_260407
```

如果不是通过 `./app` CLI 启动，而是直接 `python app.py`，那就不会自动带这个标记，目录名仍保持原样。

### 监控后台登录保护

如果这个服务会被外网或其他人访问，建议开启监控后台登录保护。

启用方式：

```text
MONITOR_USERNAME=你的用户名
MONITOR_PASSWORD=你的密码
MONITOR_SESSION_SECRET=随机长字符串
```

启用后：

```text
1. 访问 /、/query、/history、/failures 会先跳转到 /login
2. /statistic、/metrics/*、/logs/*、/docs、/redoc、/openapi.json 也会被保护
3. /v1/messages 和 /chat/completions 仍然继续使用原来的 API_KEY 鉴权，不受这套网页登录影响
```

可选配置：

```text
MONITOR_COOKIE_SECURE=true      # 通过 HTTPS 暴露时建议开启
MONITOR_SESSION_MAX_AGE=43200   # 登录态有效期，单位秒，默认 12 小时
MONITOR_AUTH_ENABLED=true       # 显式开启；默认情况下只要配置了用户名和密码就会自动开启
```

安全建议：

```text
1. 不要只依赖登录页，最好同时加 Nginx/Caddy 反向代理和 HTTPS
2. 对公网暴露时，建议再配 IP 白名单
3. MONITOR_SESSION_SECRET 不要和弱口令一起使用，更不要直接提交到仓库
```

## 环境

```shell
pip install -r requirements.txt
```

## 启动与管理

推荐使用根目录下的 CLI，而不是手动执行 `bash server.sh start .env`。

先给 CLI 执行权限：

```shell
chmod +x app
```

### 1. 配置默认 env

```shell
./app config .env
```

这会把当前默认使用的环境文件写入 `.cli_state.yaml` 的 `source_env` 字段。

查看当前配置：

```shell
./app config
```

### 2. 启动 / 停止 / 重启

```shell
./app start
./app stop
./app restart
```

这些命令默认作用于当前 `source_env`。

### 3. 指定某个 env 操作

```shell
./app start --env .env.test
./app stop --env .env.test
./app restart --env .env.prod
./app logs --env .env.prod -f
```

### 4. 查看日志

```shell
./app logs
./app logs -f
./app logs --env .env.test -n 200
```

### 5. 查看服务状态

```shell
./app status
./app list
```

`status` 会显示当前默认 `source_env`，并列出所有已记录服务。  
`list` 会直接列出 `.cli_state.yaml` 中记录的所有 env 服务。

## 多个 .env 同时运行

CLI 支持多个 `.env` 同时启动，只要它们的 `PROXY_PORT` 不冲突。

例如：

```shell
./app start --env .env
./app start --env .env.test
./app start --env .env.prod
```

每个 env 都会单独记录到 `.cli_state.yaml` 的 `services` 中，字段包括：

```text
env_path
pid
host
port
pid_file
log_file
started_at
```

典型状态示例：

```yaml
source_env: .env
services:
  .env:
    pid: 1234
    host: 127.0.0.1
    port: 4000
    pid_file: logs/app-port4000.pid
    log_file: logs/app-port4000.log
  .env.test:
    pid: 2345
    host: 127.0.0.1
    port: 4001
    pid_file: logs/app-port4001.pid
    log_file: logs/app-port4001.log
```

说明：

```text
1. 不同 env 本质上是不同的 app.py 进程
2. CLI 按 env 文件区分服务
3. stop / restart / logs 可通过 --env 精确作用到某个服务
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
在代理服务启动后，访问：

- 监控总览: `http://127.0.0.1:4000/`
- 查询统计: `http://127.0.0.1:4000/query`
- 对话历史记录: `http://127.0.0.1:4000/history`
- 失败历史记录: `http://127.0.0.1:4000/failures`

如果开启了监控后台登录保护，上面这些页面会先跳转到：

- 登录页: `http://127.0.0.1:4000/login`
```
