"""幂等设计 —— 让"重试"变成真的安全。

Day2 留下的问题：
    重试只对【幂等】操作安全。转账重试会扣两次钱，发邮件重试会发两封。

这个文件提供三层东西：

    ① derive_key()          从"工具名 + 规范化参数"算一个去重键
    ② IdempotencyStore      key → 第一次的结果，带 TTL 和容量上限
    ③ execute_idempotent()  先查键，命中就返回旧结果；没命中才真执行

--------------------------------------------------------------------------
先分清两种"重复"，它们的解法完全不同
--------------------------------------------------------------------------

【场景 A：单次任务内的重试】
    execute_with_retry 重试 3 次，第 1 次其实成功了但响应超时。
    → 用"参数 hash"做键就够了，因为参数在这 3 次里完全一样。
    → 存的地方是进程内存，TTL 几秒到几分钟。

【场景 B：跨请求的重复提交】
    用户点了两次"转账"，或者客户端断线重发。
    → **参数 hash 不可靠**：用户真的可能想转两笔一样的钱。
    → 必须用业务键（订单号 request_id），由上游生成，持久化到数据库。
    → TTL 要长（几天），因为用户可能隔天重放。

⚠️ 这个文件实现的是【场景 A】。
   场景 B 需要业务层的订单表，不是这里能解决的 —— 但设计思路一样，
   只是 store 从内存换成数据库、key 从 hash 换成业务单号。

--------------------------------------------------------------------------
一个必须知道的局限
--------------------------------------------------------------------------
当前实现【没有并发保护】。两个相同 key 的请求同时到达时，都会执行。

要修，得加"进行中"状态：
    key → Future，第一个请求创建 Future 并执行，
    第二个请求 await 同一个 Future。
真实项目（如 Stripe 的幂等实现）就是这么做的。

今天先不做，因为学习阶段看清机制更重要。但你要知道这个洞在哪。
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field

from agent.retry import RetryOutcome, execute_with_retry

# 默认保留多久。1 小时够覆盖"一次任务内的重试"和"用户手抖点两次"。
DEFAULT_TTL_SECONDS = 3600.0

# 最多存多少条。防止长跑进程内存无限增长。
# 1000 条 × 平均 2KB ≈ 2MB，可以接受。
DEFAULT_MAX_ENTRIES = 1000


# ============================================================
# ① 键的派生
# ============================================================

def derive_key(name: str, arguments_json: str) -> str:
    """从"工具名 + 规范化参数"算一个稳定的去重键。

    为什么要"规范化"？因为下面两种写法是**同一个调用**：

        {"city": "北京", "unit": "celsius"}
        {"unit": "celsius", "city": "北京"}

    JSON 对象的键顺序不影响语义。如果不排序，模型换个顺序你就认不出来了。

    同理，空格也不影响语义：
        {"a": 1}  和  {"a":1}   应该算出同一个键

    所以用：
        sort_keys=True        → 键按字母序排
        separators=(",", ":") → 去掉多余空格

    为什么要 hash 而不是直接用参数字符串？
        因为参数可能很长（比如一整段文档内容），
        直接当 key 会占内存、日志也没法看。
        sha256 前 16 位足够避免碰撞（这个量级下碰撞概率可以忽略）。

    返回形如：transfer_money:9f2c1a3b4d5e6f70
    """
    try:
        args = json.loads(arguments_json) if arguments_json else {}
        normalized = json.dumps(
            args, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
    except json.JSONDecodeError:
        # 参数本身不是合法 JSON —— 这种调用会在 execute_tool 里直接失败，
        # 这里只要能算出个稳定的键就行，不用管格式
        normalized = arguments_json

    digest = hashlib.sha256(f"{name}:{normalized}".encode("utf-8")).hexdigest()[:16]
    return f"{name}:{digest}"


# ============================================================
# ② 存储
# ============================================================

@dataclass
class Entry:
    """一条幂等记录。"""
    raw: str                # 第一次执行的结果（原样保存，重放时直接返回）
    created_at: float       # 什么时候存的（用来判断 TTL）
    hits: int = 0           # 被重放了几次（观测用）


class IdempotencyStore:
    """key → 第一次执行结果的存储。

    为什么要 TTL？
        幂等窗口是有限的。一天前的"查北京天气"结果不该再返回，
        因为天气早变了。

    为什么要容量上限？
        长跑进程不清理内存会一直涨。满了就淘汰最旧的（FIFO）。

    ⚠️ 这是内存实现，进程重启就没了。
       需要跨重启的幂等，得换成 Redis / 数据库 —— 接口一样，换实现即可。
    """

    def __init__(
        self,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        max_entries: int = DEFAULT_MAX_ENTRIES,
    ):
        self._data: dict[str, Entry] = {}
        self._ttl = ttl_seconds
        self._max = max_entries

    def get(self, key: str) -> Entry | None:
        """查一个键。过期的当作不存在（顺手删掉）。"""
        entry = self._data.get(key)
        if entry is None:
            return None

        if time.time() - entry.created_at > self._ttl:
            del self._data[key]
            return None

        entry.hits += 1
        return entry

    def put(self, key: str, raw: str) -> None:
        """存一条记录。满了就淘汰最旧的。"""
        self._data[key] = Entry(raw=raw, created_at=time.time())

        # dict 在 Python 3.7+ 保插入顺序，所以第一个 key 就是最旧的
        while len(self._data) > self._max:
            oldest = next(iter(self._data))
            del self._data[oldest]

    def __len__(self) -> int:
        return len(self._data)

    def clear(self) -> None:
        self._data.clear()

    def stats(self) -> dict:
        """看一眼内部状态，调试用。"""
        return {
            "entries": len(self._data),
            "total_hits": sum(e.hits for e in self._data.values()),
        }


# 全局默认存储。不方便显式传 store 时用它
_DEFAULT_STORE = IdempotencyStore()


# ============================================================
# ③ 带幂等保护地执行
# ============================================================

@dataclass
class IdempotentOutcome:
    """一次带幂等保护的执行结果。

    和 RetryOutcome 的区别就多了一个 executed：
        executed=True  → 这次真的执行了工具
        executed=False → 命中缓存，直接返回了上次的结果
    """
    raw: str
    key: str
    executed: bool
    attempts: int
    total_delay: float
    kind: str
    succeeded: bool

    def summary(self) -> str:
        if not self.executed:
            return f"♻️ 命中缓存（key={self.key}），没有重复执行"
        flag = "✅" if self.succeeded else "❌"
        return (
            f"{flag} 实际执行（key={self.key}，{self.attempts} 次尝试，"
            f"退避 {self.total_delay:.2f}s）"
        )


async def execute_idempotent(
    name: str,
    arguments_json: str,
    *,
    key: str | None = None,
    store: IdempotencyStore | None = None,
    **retry_kwargs,
) -> IdempotentOutcome:
    """带幂等保护地执行工具：同一个键只真正执行一次。

    参数：
        name, arguments_json  和 execute_tool 一样
        key                   显式指定去重键。**不传就从参数派生**
                              ⚠️ 跨请求场景必须显式传业务键（订单号），
                              不要依赖参数 hash —— 用户真的可能想转两笔一样的钱
        store                 用哪个存储。不传就用全局默认的
        **retry_kwargs        透传给 execute_with_retry（max_attempts 等）

    ⚠️ 只缓存【成功】的结果。

        失败不缓存，因为失败可能是暂时的（transient），
        下次应该重新试，而不是永远返回那个失败结果。
    """
    store = store if store is not None else _DEFAULT_STORE
    actual_key = key or derive_key(name, arguments_json)

    # ---------- 先查：这个键做过没有 ----------
    cached = store.get(actual_key)
    if cached is not None:
        # 命中就原样返回第一次的结果。
        # 注意这里【不执行工具】—— 这就是幂等的全部意义。
        return IdempotentOutcome(
            raw=cached.raw,
            key=actual_key,
            executed=False,
            attempts=0,
            total_delay=0.0,
            kind="replayed",
            succeeded=True,
        )

    # ---------- 没做过：真的执行 ----------
    outcome: RetryOutcome = await execute_with_retry(
        name, arguments_json, **retry_kwargs
    )

    # 成功才存。失败存了的话，下次会直接返回失败结果，等于永久放弃
    if outcome.succeeded:
        store.put(actual_key, outcome.raw)

    return IdempotentOutcome(
        raw=outcome.raw,
        key=actual_key,
        executed=True,
        attempts=outcome.attempts,
        total_delay=outcome.total_delay,
        kind=outcome.kind,
        succeeded=outcome.succeeded,
    )
