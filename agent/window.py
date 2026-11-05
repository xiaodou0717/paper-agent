"""滑动窗口 —— 丢掉早期轮次，而不是把每条都压短。

--------------------------------------------------------------------------
为什么需要它？Day4 的基线里有个断崖
--------------------------------------------------------------------------
W4 Day4 跑 8 步任务，烧了 57537 token。这笔钱的结构是这样的：

    第 1 轮   598 tok
    第 2 轮  2076 tok
    ...
    第 8 轮 10975 tok

后一轮比前一轮贵，因为**每轮都在重发全部历史**。
而历史里有一大半是"用过就废"的东西：

    第 2 轮查到的搜索结果    → 第 3 轮读了正文之后就没用了
    第 3 轮读的文档全文      → 结论已经被模型写进第 4 轮的回复里
    第 5 轮的搜索过程        → 第 7 轮早就换了话题

它们还占着位置，但**模型不会再回头看**。

--------------------------------------------------------------------------
W4 Day3 的截断为什么不够
--------------------------------------------------------------------------
`budget.py` 的 `trim_tool_results()` 做的是"把长的变短"：

    1500 字符的文档全文  →  800 字符  →  400 字符  →  200 字符

但压到 200 字符时，那些内容已经没意义了（只剩标题），
模型看不懂还要重新查 —— **省下的 token 又花回去了**。

Day3 的压缩率曲线显示 400 是拐点，就是这个原因。

**正确的做法不是"压得更狠"，是"整段不要"。**

--------------------------------------------------------------------------
一个必须做对的细节：成对删除
--------------------------------------------------------------------------
messages 里 assistant 和 tool 是**成对**的：

    {"role": "assistant", "tool_calls": [{"id": "call_abc", ...}]}
    {"role": "tool", "tool_call_id": "call_abc", "content": "..."}

如果只删 assistant 保留 tool，API 会直接报错：
`tool_call_id "call_abc" not found`。

所以这个模块做了两件事：
    1. 以 assistant 为轮次边界切分 —— 天然保证成对
    2. 删完调 `validate_structure()` 校验一遍 —— **错了宁可别删**

第二条是安全网：如果校验不过，返回原 messages，让调用方知道"这次没删成功"。
**宁可多花 token，也不能发一个非法请求。**
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """兼容 dict 和 SDK 对象（和 context.py 里那个一样）。"""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _role(m: Any) -> str:
    return _get(m, "role") or ""


def _has_tool_calls(m: Any) -> bool:
    return bool(_get(m, "tool_calls"))


def _content_len(m: Any) -> int:
    c = _get(m, "content")
    return len(c) if isinstance(c, str) else 0


# ============================================================
# 结果
# ============================================================

@dataclass
class WindowResult:
    """一次滑窗操作的结果。"""
    messages: list[Any]              # 处理后的消息
    dropped_count: int = 0           # 丢了几条
    dropped_chars: int = 0           # 丢了多少字符
    dropped_rounds: int = 0          # 丢了几轮
    kept_rounds: int = 0             # 留了几轮
    ok: bool = True                  # 结构校验通过吗
    note: str = ""                   # 说明（失败原因 / 为什么没删）

    @property
    def changed(self) -> bool:
        return self.dropped_count > 0

    def summary(self) -> str:
        if not self.ok:
            return f"⚠️ 滑窗失败（{self.note}），保持原样"
        if not self.changed:
            return f"滑窗未触发（{self.note}）"
        return (
            f"🪟 丢弃 {self.dropped_rounds} 轮 / {self.dropped_count} 条"
            f"（{self.dropped_chars} 字符），保留最近 {self.kept_rounds} 轮"
        )


# ============================================================
# 结构校验 —— 安全网
# ============================================================

def validate_structure(messages: list[Any]) -> tuple[bool, str]:
    """检查消息结构是否合法：每条 tool 消息都能找到对应的 tool_call。

    为什么必须有这一步？
        删错一条会让整个请求被 API 拒绝，白花一轮（还拿不到任何结果）。
        删完 + 校验，成本几乎为零，但能避免"发出去才发现非法"。

    返回 (是否合法, 原因)。
    """
    call_ids: set[str] = set()
    for m in messages:
        for tc in (_get(m, "tool_calls") or []):
            cid = _get(tc, "id")
            if cid:
                call_ids.add(cid)

    for idx, m in enumerate(messages):
        if _role(m) != "tool":
            continue
        cid = _get(m, "tool_call_id")
        if cid and cid not in call_ids:
            return False, f"第 {idx} 条 tool 消息的 tool_call_id={cid} 找不到对应的 tool_calls"

    return True, "ok"


# ============================================================
# 滑动窗口
# ============================================================

def sliding_window(
    messages: list[Any],
    keep_head: int = 2,
    keep_tail_rounds: int = 4,
    validate: bool = True,
) -> WindowResult:
    """只保留"头部 + 最近 N 轮"，中间的整段丢掉。

    参数：
        messages         原始消息列表
        keep_head        头部保留几条。默认 2 = system + 第一条 user
                         **这两条绝对不能丢**：一个是行为约束，一个是任务目标
        keep_tail_rounds 保留最近几轮（一轮 = 一条 assistant 及其后的 tool）
        validate         删完是否校验结构。默认 True，**别关**

    返回 WindowResult（含统计和校验结果）。

    ---------- 为什么以 assistant 为边界 ----------
    因为 assistant 是一条"轮"的起点：
        assistant(tool_calls) → tool → tool → assistant(tool_calls) → ...
    从某个 assistant 开始切，它后面的 tool 自然跟着，不会出现"孤儿 tool"。
    """
    total = len(messages)

    # ---------- 太少就不用删 ----------
    if total <= keep_head:
        return WindowResult(messages, ok=True, note="消息太少")

    head = messages[:keep_head]
    body = messages[keep_head:]

    # ---------- 找轮次边界 ----------
    boundaries = [i for i, m in enumerate(body) if _role(m) == "assistant"]

    if len(boundaries) <= keep_tail_rounds:
        return WindowResult(
            messages, ok=True,
            note=f"只有 {len(boundaries)} 轮，不超过保留数 {keep_tail_rounds}",
        )

    # ---------- 从第 (len - keep) 个边界开始保留 ----------
    start = boundaries[-keep_tail_rounds]
    tail = body[start:]
    dropped = body[:start]

    if not dropped:
        return WindowResult(messages, ok=True, note="没有可丢弃的内容")

    candidate = head + tail

    # ---------- 安全网：校验结构 ----------
    if validate:
        ok, why = validate_structure(candidate)
        if not ok:
            # 宁可多花 token，也不发一个非法请求
            return WindowResult(messages, ok=False, note=why)

    kept_rounds = len([m for m in tail if _role(m) == "assistant"])
    return WindowResult(
        messages=candidate,
        dropped_count=len(dropped),
        dropped_chars=sum(_content_len(m) for m in dropped),
        dropped_rounds=len(boundaries) - keep_tail_rounds,
        kept_rounds=kept_rounds,
        ok=True,
    )


def auto_window(
    messages: list[Any],
    target_drop_chars: int,
    keep_head: int = 2,
    max_rounds_to_drop: int = 20,
) -> WindowResult:
    """按"要丢多少字符"自动决定丢几轮。

    为什么需要这个变体？
        sliding_window 是按"保留几轮"调用的，但预算器算出来的是
        "需要腾出多少字符"。两者单位不一样，需要换算。

    做法：从丢 1 轮开始试，直到丢够或者不能再丢。
    每试一次都校验结构（validate=True），所以不会出错。

    ⚠️ 这个函数可能调 use 多次 sliding_window，但每次都是纯计算，成本很低。
    """
    if target_drop_chars <= 0:
        return WindowResult(messages, ok=True, note="不需要丢")

    best = WindowResult(messages, ok=True, note="没找到合适的窗口")
    for drop_n in range(1, max_rounds_to_drop + 1):
        # 我们想丢 drop_n 轮 → 保留"总轮数 - drop_n"轮
        total_rounds = sum(1 for m in messages if _role(m) == "assistant")
        keep = total_rounds - drop_n
        if keep < 1:
            break

        result = sliding_window(messages, keep_head=keep_head, keep_tail_rounds=keep)
        if not result.changed:
            break

        best = result
        if result.dropped_chars >= target_drop_chars:
            break

    return best
