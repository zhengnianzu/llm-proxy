"""
压力测试 & 长连接测试
用法:
  # 并发压力测试（默认 10 并发，每并发 5 次）
  python test/test_stress.py --host http://127.0.0.1:4001 --concurrency 10 --requests 5

  # 长连接测试（单连接跑 60 秒，连续发请求）
  python test/test_stress.py --host http://127.0.0.1:4001 --mode long --duration 60
"""

import argparse
import asyncio
import json
import time
import statistics
from dataclasses import dataclass, field
from typing import List

import httpx

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
ANTHROPIC_PAYLOAD = {
    "model": "h:claude-sonnet-4-6",
    "max_tokens": 64,
    "stream": True,
    "messages": [{"role": "user", "content": "用一句话介绍你自己"}],
}

HEADERS = {
    "Content-Type": "application/json",
    "x-api-key": "test-key",          # 与 .env 中 API_KEY 对应，未设置则随便填
    "anthropic-version": "2023-06-01",
}


# ---------------------------------------------------------------------------
# 结果收集
# ---------------------------------------------------------------------------
@dataclass
class RequestResult:
    success: bool
    duration_ms: float
    first_token_ms: float = 0.0   # 流式：第一个 data chunk 的延迟
    event_count: int = 0
    error: str = ""
    protocol_ok: bool = True      # SSE 事件顺序是否合法


@dataclass
class Stats:
    results: List[RequestResult] = field(default_factory=list)

    def add(self, r: RequestResult):
        self.results.append(r)

    def report(self):
        total = len(self.results)
        ok = [r for r in self.results if r.success]
        fail = [r for r in self.results if not r.success]
        protocol_err = [r for r in self.results if not r.protocol_ok]

        print(f"\n{'='*60}")
        print(f"总请求数:      {total}")
        print(f"成功:          {len(ok)}  ({100*len(ok)/total:.1f}%)")
        print(f"失败:          {len(fail)}  ({100*len(fail)/total:.1f}%)")
        print(f"SSE协议异常:   {len(protocol_err)}")

        if ok:
            durations = [r.duration_ms for r in ok]
            ftts = [r.first_token_ms for r in ok if r.first_token_ms > 0]
            print(f"\n耗时(ms) — 平均:{statistics.mean(durations):.0f}  "
                  f"P50:{statistics.median(durations):.0f}  "
                  f"P95:{_pct(durations, 95):.0f}  "
                  f"Max:{max(durations):.0f}")
            if ftts:
                print(f"首Token(ms)— 平均:{statistics.mean(ftts):.0f}  "
                      f"P50:{statistics.median(ftts):.0f}  "
                      f"Max:{max(ftts):.0f}")
        if fail:
            print("\n失败详情(前5条):")
            for r in fail[:5]:
                print(f"  {r.error}")
        if protocol_err:
            print("\nSSE协议异常详情(前5条):")
            for r in protocol_err[:5]:
                print(f"  {r.error}")
        print('='*60)


def _pct(data: List[float], p: int) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * p / 100)
    return sorted_data[min(idx, len(sorted_data) - 1)]


# ---------------------------------------------------------------------------
# 单次流式请求
# ---------------------------------------------------------------------------
async def do_stream_request(client: httpx.AsyncClient, url: str) -> RequestResult:
    t0 = time.monotonic()
    first_token_ms = 0.0
    event_count = 0
    message_start_seen = False
    message_stop_seen = False
    protocol_ok = True
    error_msg = ""

    try:
        async with client.stream("POST", url, headers=HEADERS, json=ANTHROPIC_PAYLOAD) as r:
            if r.status_code != 200:
                body = await r.aread()
                return RequestResult(
                    success=False,
                    duration_ms=(time.monotonic() - t0) * 1000,
                    error=f"HTTP {r.status_code}: {body[:200].decode(errors='replace')}",
                )

            async for line in r.aiter_lines():
                if not line:
                    continue
                if line.startswith("event:"):
                    event_type = line[6:].strip()
                    event_count += 1

                    if event_type == "message_start":
                        if message_start_seen:
                            protocol_ok = False
                            error_msg = "重复收到 message_start"
                        message_start_seen = True
                        if first_token_ms == 0.0:
                            first_token_ms = (time.monotonic() - t0) * 1000

                    elif event_type == "message_stop":
                        if not message_start_seen:
                            protocol_ok = False
                            error_msg = "收到 message_stop 但未见 message_start"
                        message_stop_seen = True

                    elif event_type == "error":
                        error_msg = f"上游 SSE error 事件 (第{event_count}个事件)"

        duration_ms = (time.monotonic() - t0) * 1000
        return RequestResult(
            success=True,
            duration_ms=duration_ms,
            first_token_ms=first_token_ms,
            event_count=event_count,
            protocol_ok=protocol_ok,
            error=error_msg,
        )

    except Exception as e:
        return RequestResult(
            success=False,
            duration_ms=(time.monotonic() - t0) * 1000,
            error=f"{type(e).__name__}: {e}",
        )


# ---------------------------------------------------------------------------
# 压力测试：N 并发，每并发 M 次
# ---------------------------------------------------------------------------
async def stress_test(host: str, concurrency: int, requests_per_worker: int):
    url = f"{host.rstrip('/')}/v1/messages"
    print(f"[压力测试] {url}  并发={concurrency}  每worker请求数={requests_per_worker}")

    stats = Stats()
    lock = asyncio.Lock()

    async def worker(worker_id: int):
        async with httpx.AsyncClient(verify=False, timeout=httpx.Timeout(120.0)) as client:
            for i in range(requests_per_worker):
                result = await do_stream_request(client, url)
                async with lock:
                    stats.add(result)
                status = "✓" if result.success else "✗"
                print(f"  worker{worker_id} req{i+1}: {status} {result.duration_ms:.0f}ms "
                      f"events={result.event_count} {'[协议异常]' if not result.protocol_ok else ''}")

    t0 = time.monotonic()
    await asyncio.gather(*[worker(i) for i in range(concurrency)])
    total_sec = time.monotonic() - t0
    print(f"\n总耗时: {total_sec:.1f}s  "
          f"吞吐: {len(stats.results)/total_sec:.2f} req/s")
    stats.report()


# ---------------------------------------------------------------------------
# 长连接测试：单 client，持续 duration 秒
# ---------------------------------------------------------------------------
async def long_connection_test(host: str, duration: int):
    url = f"{host.rstrip('/')}/v1/messages"
    print(f"[长连接测试] {url}  持续={duration}s")

    stats = Stats()
    t_end = time.monotonic() + duration
    req_num = 0

    async with httpx.AsyncClient(verify=False, timeout=httpx.Timeout(120.0)) as client:
        while time.monotonic() < t_end:
            req_num += 1
            result = await do_stream_request(client, url)
            stats.add(result)
            status = "✓" if result.success else "✗"
            elapsed = duration - (t_end - time.monotonic())
            print(f"  [{elapsed:.0f}s] req{req_num}: {status} {result.duration_ms:.0f}ms "
                  f"events={result.event_count} {'[协议异常]' if not result.protocol_ok else ''}"
                  f"{' ERR:'+result.error if result.error else ''}")

    stats.report()


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://127.0.0.1:4002")
    parser.add_argument("--mode", choices=["stress", "long"], default="stress")
    parser.add_argument("--concurrency", type=int, default=10,
                        help="[stress] 并发 worker 数")
    parser.add_argument("--requests", type=int, default=5,
                        help="[stress] 每 worker 请求数")
    parser.add_argument("--duration", type=int, default=60,
                        help="[long] 持续秒数")
    args = parser.parse_args()

    if args.mode == "stress":
        asyncio.run(stress_test(args.host, args.concurrency, args.requests))
    else:
        asyncio.run(long_connection_test(args.host, args.duration))
