"""重试机制 —— 只对 transient 失败重试，用指数退避 + 抖动。

Day1 给失败分了类，这个文件是用分类做决策的第一处。

    为什么不能"失败就立刻重试"？
        因为失败往往不是随机的，是一段时间内的状态：
        服务端在重启、限流窗口还没过、网络正在抖动。
        立刻重试大概率还是同一个结果，只是把 3 次机会浪费在 1 毫秒内。

    为什么要"退避"（越等越久）？
        给对面恢复的时间。1s → 2s → 4s，比每次都 1s 更容易等到它活过来。

    为什么要"抖动"（随机化）？
        如果没有抖动，100 个客户端会在同一秒一起重试，形成尖峰，
        把刚恢复的服务再打挂一次。这个现象叫"惊群"（thundering herd）。
        加随机量能让它们错开。

    为什么只重试 transient？
        permanent（权限、404）重试一万次也一样，纯浪费。
        model_fixable（参数错）原样重发也没用 —— 要回填让模型改。
        **重试只解决"环境暂时不好"，别指望它解决别的问题。**

⚠️ 最重要的一条，写在最前面：
    **只对【幂等】操作开启重试。**

    幂等 = 同一个请求执行多次，结果一样。
    - 查询、搜索、读文件 → 幂等，重试安全
    - 转账、发邮件、下单、写文件 → **不幂等**，重试可能重复扣款/重复发送

    这条是 Day3 的主题。今天先记住：**给工具配重试之前，先问它幂不幂等。**
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import dataclass
from typing import Callable

from agent.tools import execute_tool

# 默认参数。数值不是拍脑袋的，理由写在下面每一条后面。
DEFAULT_MAX_ATTEMPTS = 3          # 3 次 = 1 次原始 + 2 次重试。再多收益很小
DEFAULT_BASE_DELAY = 0.5          # 首次等 0.5s。太短等于没等，太长用户等不起
DEFAULT_MAX_DELAY = 8.0           # 单次等待上限，防止指数爆炸（2^10 秒就没法用了）
DEFAULT_MAX_TOTAL_SECONDS = 30.0  # 总时长上限：退避期间时间也在流逝，不能无限等


# ============================================================
# 结果：一次"带重试的执行"的完整记录
# ============================================================

@dataclass
class RetryOutcome:
    """一次带重试的工具执行结果。

    raw 是给模型看的（直接回填），其余是给你看的（观测、统计、调参）。
    """
    raw: str              # 工具最后返回的 JSON 字符串（原样回填给模型）
    attempts: int         # 实际尝试了几次（1 表示一次就成功）
    total_delay: float    # 总共退避了多久（秒）
    kind: str             # 最终结果的 kind：ok / transient / permanent / model_fixable
    succeeded: bool       # 最终成功了吗

    @property
    def retried(self) -> bool:
        """有没有发生过重试。"""
        return self.attempts > 1

    def summary(self) -> str:
        if self.succeeded:
            return f"✅ 成功（第 {self.attempts} 次尝试，累计退避 {self.total_delay:.2f}s）"
        return f"❌ 失败（{self.attempts} 次尝试，累计退避 {self.total_delay:.2f}s，最终 kind={self.kind}）"


# ============================================================
# 退避时长计算
# ============================================================

def compute_delay(
    attempt: int,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    jitter: str = "equal",
    retry_after: float | None = None,
) -> float:
    """算第 N 次重试前该等多久。

    参数：
        attempt      第几次尝试（1 表示第一次失败后、准备第 2 次之前）
        base_delay   基础间隔
        max_delay    单次上限
        jitter       抖动方式："none" / "full" / "equal"
        retry_after  服务端明确要求的等待秒数（429 的 Retry-After 头）

    退避序列（base=0.5, jitter="none"）：
        attempt=1 → 0.5
        attempt=2 → 1.0
        attempt=3 → 2.0
        attempt=4 → 4.0
        ……到 max_delay 就不再涨

    ---------- 三种抖动 ----------
    none   纯指数。好观察，但会造成惊群：所有客户端同一时刻重试
    full   在 [0, delay] 里随机。AWS 推荐的方案，总完成时间最短
           代价：可能抽到接近 0 的值，等于没退避
    equal  在 [delay/2, delay] 里随机。保留了一半退避效果，又错开了
           本项目默认它 —— 教学场景更可预测，生产也够用
    """
    # 服务端说了算：429 带 Retry-After 时，尊重它比你自己拍一个更靠谱。
    # 这是唯一一个"可以不听自己算法"的情况。
    if retry_after is not None:
        return min(float(retry_after), max_delay)

    # 指数增长：base * 2^(attempt-1)，封顶 max_delay
    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)

    if jitter == "none":
        return delay
    if jitter == "full":
        return random.uniform(0, delay)

    # equal（默认）
    half = delay / 2
    return half + random.uniform(0, half)


# ============================================================
# 带重试的执行
# ============================================================

async def execute_with_retry(
    name: str,
    arguments_json: str,
    *,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    jitter: str = "equal",
    max_total_seconds: float = DEFAULT_MAX_TOTAL_SECONDS,
    on_retry: Callable[[int, float, str], None] | None = None,
) -> RetryOutcome:
    """执行工具；如果是可重试的失败，就退避后重试。

    这是 loop 里第 ④ 步的替代品 —— 原来直接调 execute_tool，现在套一层重试。

    ⚠️ 只给【幂等】工具用它。对转账、发邮件这类操作开重试，会重复执行。

    参数：
        max_attempts       最多尝试几次（含第一次）
        base_delay         首次退避时长
        max_delay          单次退避上限
        jitter             抖动方式，见 compute_delay
        max_total_seconds  总时长上限：退避累计超过它就不再试了
        on_retry           每次准备重试时的回调 (第几次, 等待秒数, 错误信息)

    为什么是 async 而不是普通函数？
        因为退避要"等"。用 time.sleep 会占住一个线程；
        用 asyncio.sleep 只是让出事件循环，什么都不占。
        （这是 W1 Day2 那条"阻塞 vs 非阻塞"的直接应用）
    """
    started = time.perf_counter()
    total_delay = 0.0
    last_raw = ""
    last_kind = "unknown"
    attempt = 0

    for attempt in range(1, max_attempts + 1):
        # 真正执行工具。注意这里仍然是 to_thread：
        # execute_tool 是同步函数，工具内部可能有阻塞调用（读文件、发请求）。
        raw = await asyncio.to_thread(execute_tool, name, arguments_json)
        last_raw = raw

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # 工具返回的不是合法 JSON —— 这是工具自身的问题，重试也不会变好。
            # 直接返回，交给上层处理（正常情况下 execute_tool 不会这样）
            return RetryOutcome(raw, attempt, total_delay, "permanent", False)

        # ---------- 判断要不要停 ----------
        # 成功的，停；不可重试的，停。
        # 注意顺序：先看 ok，因为成功的结果里没有 retryable 字段。
        if data.get("ok"):
            return RetryOutcome(raw, attempt, total_delay, "ok", True)

        last_kind = data.get("kind", "permanent")

        # 这一步是整段代码的核心：
        # **重试的依据不是"抛异常了"，而是"结果是 transient"。**
        # 因为 execute_tool 从不抛异常，它把失败也变成了普通的返回值。
        if not data.get("retryable"):
            return RetryOutcome(raw, attempt, total_delay, last_kind, False)

        # ---------- 决定还要不要再来一次 ----------
        if attempt >= max_attempts:
            break      # 次数用完了

        # 算这次该等多久。429 的 Retry-After 会被优先采用
        delay = compute_delay(
            attempt, base_delay, max_delay, jitter,
            retry_after=data.get("retry_after"),
        )

        # 总时长预算：已经花的时间 + 这次要等的，超过上限就放弃
        elapsed = time.perf_counter() - started
        if elapsed + delay > max_total_seconds:
            break

        if on_retry:
            on_retry(attempt, delay, data.get("message", ""))

        await asyncio.sleep(delay)
        total_delay += delay

    return RetryOutcome(last_raw, attempt, total_delay, last_kind, False)


# ============================================================
# 同步版：给不方便用 async 的地方（比如一次性脚本）
# ============================================================

def execute_with_retry_sync(
    name: str,
    arguments_json: str,
    *,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    jitter: str = "equal",
) -> RetryOutcome:
    """同步版本，逻辑完全一样，只是用 time.sleep 等。

    ⚠️ 不要在 async 函数里调它 —— 它会阻塞事件循环（W1 Day2 的坑）。
    只在同步脚本或测试里用。
    """
    started = time.perf_counter()
    total_delay = 0.0
    last_raw = ""
    last_kind = "unknown"
    attempt = 0

    for attempt in range(1, max_attempts + 1):
        raw = execute_tool(name, arguments_json)
        last_raw = raw
        data = json.loads(raw)

        if data.get("ok"):
            return RetryOutcome(raw, attempt, total_delay, "ok", True)

        last_kind = data.get("kind", "permanent")
        if not data.get("retryable") or attempt >= max_attempts:
            break

        delay = compute_delay(
            attempt, base_delay, max_delay, jitter,
            retry_after=data.get("retry_after"),
        )
        time.sleep(delay)
        total_delay += delay

    return RetryOutcome(last_raw, attempt, total_delay, last_kind, False)
