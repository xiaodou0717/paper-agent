"""上下文分层 —— 从"花了多少"到"谁占的"。

Day1 的 StepLog 告诉你"每一轮发过去多大"。
但那只回答了一半问题：**大头到底在哪一层？**

举个例子，一轮 3822 token 的调用，可能是：

    工具定义   505  (13%)   ← 不管你用不用，每轮都发
    用户问题    66  ( 2%)
    模型回复   800  (21%)
    工具结果  2347  (61%)   ← 真正的元凶
    ---------------------
    合计      3718

**如果只看总数，你会以为"上下文太长"；看了分层才知道该砍工具结果，不是砍历史。**

这就是这个模块的全部意义：**把不可操作的"总数"变成可操作的"分布"**。

--------------------------------------------------------------------------
三个设计决定
--------------------------------------------------------------------------

【一】为什么用估算而不是精确 token 数？

    精确切分 token 需要分词器（tiktoken），而不同模型的分词器不一样。
    但对"分层占比"这个用途，估算足够 —— 你关心的是 61% vs 13%，
    不是 2347 还是 2401。

    而且每天多一个依赖，就多一个可能装不上的东西。
    估算 + 实测校准（calibrate），精度够用且零依赖。

【二】为什么按"工具名"再细分工具结果？

    因为"工具结果占 61%" 这个信息还不够指导行动。
    你还得知道是哪个工具占的 —— 是 search_docs（短摘要）还是 read_doc（全文）？
    前者该改检索，后者该做截断。**动作完全不同。**

    所以要顺着 tool_call_id 回去查工具名。

【三】为什么不改变实际发送结构？

    这一层只做**度量**，不动 messages。
    真正的"分层装配"（按需组装、动态裁剪）是后续的事。

    **先看清楚，再动手** —— 顺序反了会改出无法验证的东西。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

# 估算系数：中文按 0.65 token/字，其他字符按 0.25 token/字符。
# 这两个数来自 W4 Day1 的实测（第 2 轮 0.750、第 3 轮 0.672），
# 取偏保守的值，宁可高估不要低估 —— 预算器低估会导致超支。
TOKENS_PER_CN_CHAR = 0.65
TOKENS_PER_OTHER_CHAR = 0.25

# 中文/汉字的 Unicode 范围
_CN_RE = re.compile(r"[\u4e00-\u9fff]")


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """统一取值：兼容 dict 和 SDK 对象。

    为什么需要？
        messages 里混着两种东西（W2 Day3 的"原样放回历史"）：
          - dict：我们自己 append 的 system / user / tool
          - SDK 对象：模型返回的 assistant 消息

    这个坑在 W4 Day1 已经踩过一次（观测代码抛 AttributeError，
    把整个任务搞挂了）。所以这里一开始就做对。
    """
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def estimate_tokens(text: str, json_like: bool = False) -> int:
    """粗估一段文本的 token 数。

    中文 0.65 token/字，其他 0.25 token/字符。
    +1 是因为每条消息都有几个 token 的结构开销（role、分隔符）。

    ---------- json_like 参数是踩坑后加的 ----------
    JSON / 代码的 token 密度比自然英文高得多。

    为什么？自然英文是"整词编码"（"context" 可能是 1 个 token），
    而 JSON 的结构字符（" { } : ,）几乎每个都独立成 token。

    实测：tools schema 791 字符，服务商算 505 token ——
    比值 0.64，和中文差不多，是纯英文 0.25 的 2.5 倍。

    第一版没区分，导致整体低估了 30%~48%（估算 2889 vs 实测 4057）。
    修法：JSON 类内容用 0.6 系数。

    ⚠️ 这是估算，误差 ±15% 都正常。
       要精确值就用服务商返回的 usage（那是真的），
       本模块的 breakdown() 支持传入 actual_tokens 做校准。
    """
    if not text:
        return 0
    cn = len(_CN_RE.findall(text))
    other = len(text) - cn
    other_rate = 0.60 if json_like else TOKENS_PER_OTHER_CHAR
    return int(cn * TOKENS_PER_CN_CHAR + other * other_rate) + 1


# ============================================================
# 分层结果
# ============================================================

@dataclass
class LayerStats:
    """一层上下文。"""
    name: str                 # 层名，如 "工具结果 · read_doc"
    chars: int                # 字符数
    est_tokens: int           # 估算 token
    n_items: int = 0          # 这一层有几条
    note: str = ""            # 补充说明（比如"每轮固定"）

    def percent_of(self, total: int) -> float:
        return self.est_tokens / total * 100 if total else 0.0


@dataclass
class ContextBreakdown:
    """一次调用的上下文分解。"""
    layers: list[LayerStats] = field(default_factory=list)
    total_chars: int = 0
    total_est: int = 0
    actual_tokens: int | None = None      # 服务商返回的真实 prompt_tokens

    @property
    def total(self) -> int:
        """优先用实测值，没有就用估算。"""
        return self.actual_tokens if self.actual_tokens else self.total_est

    @property
    def estimation_error(self) -> float | None:
        """估算误差（百分比）。正数表示低估，负数表示高估。"""
        if not self.actual_tokens or not self.total_est:
            return None
        return (self.total_est - self.actual_tokens) / self.actual_tokens * 100

    def top(self, n: int = 3) -> list[LayerStats]:
        """占比最大的前 n 层 —— 该先动的地方。"""
        return sorted(self.layers, key=lambda x: -x.est_tokens)[:n]


# ============================================================
# 分解
# ============================================================

def breakdown(
    messages: list[Any],
    tools_schema: list[dict] | None = None,
    actual_tokens: int | None = None,
) -> ContextBreakdown:
    """把上下文拆成层。

    分层规则：
        system 消息        → "system 提示词"
        tools_schema       → "工具定义"（**注意：它不在 messages 里**）
        role=user          → "用户输入"
        role=assistant 带 tool_calls → "模型决策"（那一轮它决定调什么）
        role=assistant 只有 content  → "模型回复"
        role=tool          → "工具结果 · <工具名>"（顺着 tool_call_id 查）

    为什么要单独把 tools_schema 列出来？
        因为它是一次性发过去的固定开销，但因为不在 messages 里，
        **只看 messages 的统计会漏掉它** —— 而它可能占 10%~30%。
    """
    bd = ContextBreakdown()

    # ---------- 第一步：建立 tool_call_id → 工具名 的映射 ----------
    # 为什么要这一步：tool 消息里只有 tool_call_id，没有工具名。
    # 不查回去，所有工具结果只能混成一堆，看不出是哪个工具吃掉的。
    call_names: dict[str, str] = {}
    for m in messages:
        if _get(m, "role") != "assistant":
            continue
        for tc in (_get(m, "tool_calls") or []):
            cid = _get(tc, "id")
            fn = _get(_get(tc, "function"), "name")
            if cid and fn:
                call_names[cid] = fn

    # ---------- 第二步：按层累加 ----------
    buckets: dict[str, LayerStats] = {}

    def add(layer_name: str, text: str, note: str = "", json_like: bool = False) -> None:
        item = buckets.setdefault(
            layer_name, LayerStats(name=layer_name, chars=0, est_tokens=0, note=note)
        )
        item.chars += len(text)
        item.est_tokens += estimate_tokens(text, json_like=json_like)
        item.n_items += 1

    for m in messages:
        role = _get(m, "role") or "unknown"
        content = _get(m, "content") or ""
        if not isinstance(content, str):
            content = str(content)

        if role == "system":
            add("system 提示词", content, note="每轮固定")
        elif role == "user":
            add("用户输入", content)
        elif role == "assistant":
            if _get(m, "tool_calls"):
                # ⚠️ 必须把 tool_calls 的 name + arguments 也算进来。
                #
                # 这是第一版漏掉的东西：模型决定调工具时，content 是空的，
                # 但 function.arguments 是一段 JSON 字符串，**要发过去、要付 token**。
                # 只算 content 会让"模型决策"这一层显示成几十 token，严重低估。
                extra = ""
                for tc in (_get(m, "tool_calls") or []):
                    fn = _get(_get(tc, "function"), "name") or ""
                    args = _get(_get(tc, "function"), "arguments") or ""
                    extra += f"{fn}{args}"
                add("模型决策（tool_calls）", content + extra, json_like=True)
            else:
                add("模型回复", content)
        elif role == "tool":
            cid = _get(m, "tool_call_id") or ""
            tool_name = call_names.get(cid, "未知工具")
            # 工具结果通常是 JSON（我们自己包的 {"ok":..., "result":...}）
            add(f"工具结果 · {tool_name}", content, json_like=True)
        else:
            add(f"其他（{role}）", content)

    # ---------- 第三步：工具定义单独一层 ----------
    if tools_schema:
        schema_text = json.dumps(tools_schema, ensure_ascii=False)
        buckets["工具定义"] = LayerStats(
            name="工具定义",
            chars=len(schema_text),
            est_tokens=estimate_tokens(schema_text, json_like=True),
            n_items=len(tools_schema),
            note=f"每轮固定 · {len(tools_schema)} 个工具",
        )

    # ---------- 第四步：排序（大的在前，一眼看到该动哪） ----------
    bd.layers = sorted(buckets.values(), key=lambda x: -x.est_tokens)
    bd.total_chars = sum(x.chars for x in bd.layers)
    bd.total_est = sum(x.est_tokens for x in bd.layers)
    bd.actual_tokens = actual_tokens
    return bd


# ============================================================
# 展示
# ============================================================

def format_report(bd: ContextBreakdown, width: int = 26) -> str:
    """把分解结果画成一张表 + 柱状图。

    为什么要有柱状图：
        数字看多了会麻木。柱子的长度是直觉，能让你一眼看出
        "工具结果是别人的五倍"这件事。
    """
    lines = []
    total = bd.total or 1

    head = f"上下文分解（实测 {bd.actual_tokens} tok / 估算 {bd.total_est} tok）" \
        if bd.actual_tokens else f"上下文分解（估算 {bd.total_est} tok）"
    lines.append(head)
    lines.append("-" * 66)

    max_tok = max((x.est_tokens for x in bd.layers), default=1)
    for layer in bd.layers:
        pct = layer.percent_of(bd.total)
        bar_len = int(layer.est_tokens / max_tok * 22) if max_tok else 0
        bar = "█" * max(bar_len, 1) if layer.est_tokens else ""
        name = layer.name[:width]
        lines.append(
            f"  {name:<{width}} {layer.est_tokens:>6} tok {pct:>5.1f}%  {bar}"
        )
        if layer.note:
            lines.append(f"  {'':<{width}} {'':>10}       └ {layer.note}")

    lines.append("-" * 66)

    if bd.actual_tokens:
        err = bd.estimation_error
        flag = "✅ 可用" if abs(err) < 20 else "⚠️ 偏差较大"
        lines.append(f"  估算误差：{err:+.1f}%   {flag}")
        lines.append("  （估算只用于看占比，绝对值以服务商返回的 usage 为准）")

    return "\n".join(lines)


def savings_if_cut(bd: ContextBreakdown, layer_prefix: str, keep_ratio: float) -> dict:
    """算一算"如果砍掉某一层的一部分，能省多少"。

    Day3 做预算器、W5 做压缩策略时，这个函数用来先估收益再动手。

    参数：
        layer_prefix  层名前缀，比如 "工具结果"
        keep_ratio    保留比例，0.3 表示只留 30%
    """
    target = [x for x in bd.layers if x.name.startswith(layer_prefix)]
    if not target:
        return {"found": False}

    before = sum(x.est_tokens for x in target)
    after = int(before * keep_ratio)
    return {
        "found": True,
        "layers": [x.name for x in target],
        "before": before,
        "after": after,
        "saved": before - after,
        "pct_of_total": (before - after) / bd.total * 100 if bd.total else 0,
    }
