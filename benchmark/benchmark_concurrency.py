# -*- coding: utf-8 -*-
"""
🚦 并发压测脚本 — 论文实验 D（系统性能）
========================================
用于量化"异步非阻塞改造"的收益：对比改造前后版本在并发下的
延迟（P50 / P95 / P99）、吞吐（req/s）、错误率。

核心思想（写论文时作为动机）：
  改造前：单个 LLM 请求（最长 60s）会阻塞 ASGI 事件循环，
          高并发时所有请求排队，P95 延迟飙升；
  改造后：YOLO 推理 / LLM 调用放入线程池，高并发仍可快速响应。

用法示例
--------
1) 启动被测服务（改造后版本）:
     uvicorn backend.main:app --host 127.0.0.1 --port 8000

2) 压测 detect 端点（多并发档位）:
     python benchmark/benchmark_concurrency.py --url http://127.0.0.1:8000 \
         --label after --endpoint detect --concurrency 1,5,20 --requests 10

3) 压测 qa 端点（会真实调用 LLM，最能体现差异，请调小 requests）:
     python benchmark/benchmark_concurrency.py --url http://127.0.0.1:8000 \
         --label after --endpoint qa --concurrency 1,5,10 --requests 3

4) 测"改造前"版本：用 git 检出旧提交到另一个目录/端口，跑同一组命令，
   加 --label before（保证 URL/并发/请求数完全一致才可对比）。

5) 生成对比表:
     python benchmark/benchmark_concurrency.py --compare \
         benchmark/reports/concurrency_after.json \
         benchmark/reports/concurrency_before.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import requests

try:
    import cv2
    import numpy as np
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

REPORT_DIR = Path(__file__).resolve().parent / "reports"


# ═══════════════════════════════════════════════════════════════
#  参数解析
# ═══════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="并发压测：对比服务在并发下的延迟/吞吐/错误率")
    p.add_argument("--url", default="http://127.0.0.1:8000", help="被测服务基础 URL")
    p.add_argument("--label", default="after", help="结果标签（如 after/before），用于命名输出文件")
    p.add_argument("--endpoint", choices=["detect", "qa", "all"], default="all",
                   help="压测端点：detect=检测 / qa=问答 / all=两者")
    p.add_argument("--concurrency", default="1,5,10,20", help="并发档位，逗号分隔")
    p.add_argument("--requests", type=int, default=5, help="每并发档位中每个 worker 的请求数")
    p.add_argument("--question", default="稻纵卷叶螟怎么防治？", help="qa 压测使用的问题")
    p.add_argument("--timeout", type=float, default=30.0, help="单请求超时（秒）")
    p.add_argument("--output", default=None, help="结果 JSON 输出路径（默认 reports/concurrency_{label}.json）")
    p.add_argument("--compare", nargs="+", default=None, help="对比模式：传入多个结果 JSON 路径，输出对比表")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════
#  工具
# ═══════════════════════════════════════════════════════════════

def make_test_image() -> bytes:
    """生成一张合成测试图（带随机斑点的彩色图），避免依赖外部图片文件"""
    if _HAS_CV2:
        rng = np.random.default_rng(42)
        img = rng.integers(60, 200, size=(640, 640, 3), dtype=np.uint8)
        for _ in range(20):  # 撒一些高亮斑点模拟虫体
            x, y = int(rng.integers(80, 560)), int(rng.integers(80, 560))
            cv2.circle(img, (x, y), int(rng.integers(4, 14)), (0, 255, 0), -1)
        ok, buf = cv2.imencode(".jpg", img)
        if ok:
            return buf.tobytes()
    # 兜底：纯色 JPEG
    return (b"\xff\xd8\xff\xe0" + b"\x00" * 1000)


def _send_one(endpoint: str, base_url: str, image_bytes: bytes,
              question: str, timeout: float) -> tuple[float, bool]:
    """发送单个请求，返回 (延迟 ms, 是否成功)"""
    t0 = time.perf_counter()
    try:
        if endpoint == "detect":
            resp = requests.post(
                f"{base_url}/detect/image",
                files={"file": ("bench.jpg", image_bytes, "image/jpeg")},
                timeout=timeout,
            )
        else:  # qa
            resp = requests.post(
                f"{base_url}/qa/ask",
                json={"question": question, "pest_name": "rice leaf roller"},
                timeout=timeout,
            )
        ok = 200 <= resp.status_code < 300
    except Exception:
        ok = False
    latency_ms = (time.perf_counter() - t0) * 1000
    return latency_ms, ok


def _run_worker(endpoint: str, base_url: str, image_bytes: bytes,
                question: str, per_worker: int, timeout: float,
                results: list) -> None:
    """单个 worker：连续发送 per_worker 个请求，结果写入共享列表"""
    for _ in range(per_worker):
        results.append(_send_one(endpoint, base_url, image_bytes, question, timeout))


def _percentile(sorted_times: list[float], p: float) -> float:
    """计算百分位（sorted_times 已升序）。p∈(0,1]"""
    if not sorted_times:
        return 0.0
    idx = int(math.ceil(p * len(sorted_times))) - 1
    idx = min(max(idx, 0), len(sorted_times) - 1)
    return sorted_times[idx]


def summarize(results: list[tuple[float, bool]], wall_s: float) -> dict:
    """汇总统计：延迟分位、吞吐、错误率"""
    n = len(results)
    success = sum(1 for _, ok in results if ok)
    errors = n - success
    times = sorted(lat for lat, _ in results)

    return {
        "samples": n,
        "success": success,
        "errors": errors,
        "error_rate_pct": round(errors / n * 100, 2) if n else 0.0,
        "avg_ms": round(sum(times) / n, 2) if n else 0.0,
        "p50_ms": round(_percentile(times, 0.50), 2),
        "p95_ms": round(_percentile(times, 0.95), 2),
        "p99_ms": round(_percentile(times, 0.99), 2),
        "max_ms": round(times[-1], 2) if times else 0.0,
        "throughput_rps": round(success / wall_s, 2) if wall_s > 0 else 0.0,
    }


def run_one_concurrency(endpoint: str, base_url: str, concurrency: int,
                        per_worker: int, image_bytes: bytes,
                        question: str, timeout: float) -> dict:
    """对单一并发档位跑一轮，返回统计 dict"""
    results: list[tuple[float, bool]] = []
    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(_run_worker, endpoint, base_url, image_bytes,
                        question, per_worker, timeout, results)
            for _ in range(concurrency)
        ]
        for f in futures:
            f.result()  # 等所有 worker 完成
    wall_s = time.perf_counter() - wall_start

    stats = summarize(results, wall_s)
    stats["concurrency"] = concurrency
    return stats


# ═══════════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════════

def run_benchmark(args: argparse.Namespace) -> dict:
    levels = [int(x.strip()) for x in args.concurrency.split(",") if x.strip()]
    image_bytes = make_test_image()
    endpoints = ["detect", "qa"] if args.endpoint == "all" else [args.endpoint]

    result: dict = {
        "label": args.label,
        "url": args.url,
        "endpoint": args.endpoint,
        "question": args.question,
        "timeout_s": args.timeout,
        "per_worker_requests": args.requests,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "endpoints": {},
    }

    for ep in endpoints:
        print(f"\n🚦 压测端点 [{ep}] 并发档位: {levels}")
        result["endpoints"][ep] = {}
        for c in levels:
            print(f"  → 并发 {c} ... ", end="", flush=True)
            stats = run_one_concurrency(
                ep, args.url, c, args.requests, image_bytes, args.question, args.timeout
            )
            result["endpoints"][ep][str(c)] = stats
            print(f"P50 {stats['p50_ms']:.0f}ms | P95 {stats['p95_ms']:.0f}ms | "
                  f"吞吐 {stats['throughput_rps']:.1f} req/s | 错误率 {stats['error_rate_pct']}%")

    # 打印汇总表
    for ep in endpoints:
        print(f"\n📊 汇总表 [{ep}]（label={args.label}）")
        print(f"{'并发':>5} | {'样本':>6} | {'平均ms':>8} | {'P50ms':>7} | "
              f"{'P95ms':>7} | {'P99ms':>7} | {'最大ms':>7} | {'req/s':>8} | {'错误%':>6}")
        print("-" * 80)
        for c in levels:
            s = result["endpoints"][ep][str(c)]
            print(f"{c:>5} | {s['samples']:>6} | {s['avg_ms']:>8.1f} | {s['p50_ms']:>7.1f} | "
                  f"{s['p95_ms']:>7.1f} | {s['p99_ms']:>7.1f} | {s['max_ms']:>7.1f} | "
                  f"{s['throughput_rps']:>8.1f} | {s['error_rate_pct']:>6.2f}")

    return result


def compare(args: argparse.Namespace) -> None:
    """对比模式：读取多个结果 JSON，按端点/并发档位输出并排对比表"""
    payloads = []
    for path in args.compare:
        with open(path, "r", encoding="utf-8") as f:
            payloads.append(json.load(f))

    # 收集所有出现的端点
    endpoints = sorted({ep for p in payloads for ep in p["endpoints"]})

    for ep in endpoints:
        # 收集所有出现的并发档位（按数值排序）
        levels = sorted({
            int(c) for p in payloads
            if ep in p["endpoints"]
            for c in p["endpoints"][ep]
        })
        print(f"\n📊 对比表 [{ep}]")
        header = f"{'并发':>5} | " + " | ".join(
            f"{p['label']:<18}" for p in payloads
        )
        print(header)
        print("-" * len(header))
        for c in levels:
            row = f"{c:>5} | "
            for p in payloads:
                s = p["endpoints"].get(ep, {}).get(str(c))
                if s is None:
                    row += f"{'—':<18} | "
                else:
                    row += (f"P95 {s['p95_ms']:>6.0f}ms 吞吐 {s['throughput_rps']:>6.1f} "
                            f"错 {s['error_rate_pct']:>4.1f}% | ")
            print(row)

    # 额外输出：P95 提升倍数
    if len(payloads) == 2 and endpoints:
        a, b = payloads
        print("\n📈 P95 延迟对比（后 vs 前）：")
        for ep in endpoints:
            for c in sorted({int(x) for x in a["endpoints"].get(ep, {})} & {int(x) for x in b["endpoints"].get(ep, {})}):
                pa = a["endpoints"][ep].get(str(c), {}).get("p95_ms")
                pb = b["endpoints"][ep].get(str(c), {}).get("p95_ms")
                if pa and pb:
                    print(f"  [{ep}] 并发 {c}: before {pb:.0f}ms → after {pa:.0f}ms "
                          f"(×{pa / pb:.2f})")


def main() -> int:
    args = parse_args()

    if args.compare:
        compare(args)
        return 0

    # 健康检查：确认服务可达
    try:
        r = requests.get(f"{args.url}/health", timeout=10)
        r.raise_for_status()
        print(f"✅ 服务可达: {args.url} → {r.json()}")
    except Exception as e:
        print(f"❌ 无法连接 {args.url}: {e}")
        print("   请先启动服务：uvicorn backend.main:app --host 127.0.0.1 --port 8000")
        return 1

    result = run_benchmark(args)

    # 保存 JSON
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = args.output or str(REPORT_DIR / f"concurrency_{args.label}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n💾 结果已保存: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
