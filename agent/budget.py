"""预算器 —— 从"花完了才知道"到"花之前先算"。

--------------------------------------------------------------------------
为什么需要这个模块？现在的问题在哪
--------------------------------------------------------------------------
W2 Day4 装的那道护栏（token_budget）是这样工作的：

    每轮开始前：花掉的 > 预算？ 是 → 停

它有两个局限：

【局限一】事后检查，只能"刹停"不能"省油"
    你发现超支的时候，钱已经花出去了。
    它的作用是"不再继续亏"，不是"让任务跑完"。

【局限二】它不知道"这次调用大概要花多少"
    所以没法做精细判断。比如预算还剩 5000，这次调用要 6000，
    它只能等到花完之后下次检查才发现 —— 中间那 1000 就是超出的部分。

--------------------------------------------------------------------------
Day3 做的事：调用前先算一笔账
--------------------------------------------------------------------------

    每轮开始前：
      1. 估算【这一次调用】要花多少（输入 + 预留输出）
      2. 看预算还够不够
      3. 不够 → 先压缩上下文，压完再算一遍
      4. 压完还不够 → 才停

区别在于：**给了"压缩"这个中间选项。**

    现在的护栏：  超预算 → 停
    Day3 的预算器：超预算 → 压缩 → 再判断 → 还不够才停

--------------------------------------------------------------------------
一个关键设计：压缩必须"说出来"
--------------------------------------------------------------------------
工具结果被截断后，如果悄悄塞回去，模型会以为自己看到了全文，
然后基于残缺信息给出自信的错误答案 —— 这比直接失败更糟（W3 Day4 讲过）。

所以截断后必须：
    1. 保持 JSON 结构合法（不能直接切字符串，模型解析不了）
    2. 带上 `_truncated` 标记和原始长度

**让它知道自己瞎了一只眼，比让它以为看得见更重要。**
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from agent.context import ContextBreakdown, breakdown, estimate_tokens

# 单条工具结果的保留上限（字符）。
# 为什么是 800：大约 500 token，够放一条检索结果的全部字段，
# 但挡得住 read_doc 那种全文（1500~3000 字）。
DEFAULT_TOOL_RESULT_LIMIT = 800

# 估算要留的安全余量。估算本身有 ±10% 误差，再留 10% 就够。
SAFETY_MARGIN = 1.10


@dataclass
class BudgetPlan:
    """一次调用的预算方案 —— 算完直接知道该怎么办。"""

    estimated_input: int = 0      # 估算的输入 token
    reserved_output: int = 0      # 给输出预留的 token（= max_tokens）
    estimated_total: int = 0      # 这次调用预计花多少
    remaining: int = 0            # 预算还剩多少
    fits: bool = True             # 装得下吗
    overshoot: int = 0            # 超出多少（fits=False 时才有意义）
    trim_chars_needed: int = 0    # 需要砍掉多少【字符】才能装下
    reason: str = ""              # 一句话说明决策依据

    def summary(self) -> str:
        if self.fits:
            return (
                f"✅ 装得下：预计 {self.estimated_total} ≤ 剩余 {self.remaining}"
            )
        return (
            f"⚠️ 装不下：预计 {self.estimated_total} > 剩余 {self.remaining}"
            f"（超 {self.overshoot}，需砍约 {self.trim_chars_needed} 字符）"
        )


def plan_call(
    messages: list[Any],
    tools_schema: list[dict] | None,
    budget_total: int | None,
    spent: int,
    max_output_tokens: int,
) -> BudgetPlan:
    """算一笔账：这次调用要花多少、装不装得下、装不下要砍多少。

    参数：
        messages           当前上下文
        tools_schema       工具定义
        budget_total       整个任务的预算（None = 不限）
        spent              已经花掉的 token
        max_output_tokens  max_tokens（要给输出预留）

    为什么要把输出也预留？因为预算是"输入+输出"合计。
        只算输入的话，你以为还剩 5000，实际调用完可能只剩 2000。
    """
    bd: ContextBreakdown = breakdown(messages, tools_schema)
    est_input = int(bd.total_est * SAFETY_MARGIN)
    est_total = est_input + max_output_tokens

    plan = BudgetPlan(
        estimated_input=est_input,
        reserved_output=max_output_tokens,
        estimated_total=est_total,
    )

    # 没设预算 → 永远装得下（但要记录估算值，方便观测）
    if budget_total is None:
        plan.fits = True
        plan.remaining = -1
        plan.reason = "未设预算"
        return plan

    plan.remaining = budget_total - spent
    plan.fits = est_total <= plan.remaining

    if plan.fits:
        plan.reason = "余量充足"
        return plan

    # ---------- 装不下：算算要砍多少 ----------
    plan.overshoot = est_total - plan.remaining

    # token 转字符：用 0.65 反算（中文场景的保守值）。
    # 砍字符数 = 超出 token / 0.65
    # 为什么用 0.65 而不是 0.25：我们砍的主要是中文内容 + JSON，
    # 用中文的比值反算，砍得更多一点，宁可多砍不要不够。
    plan.trim_chars_needed = int(plan.overshoot / 0.65)
    plan.reason = f"超出预算 {plan.overshoot} tok，需要压缩上下文"
    return plan


# ============================================================
# 压缩：目前只做"截断最长的工具结果"
# ============================================================

def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _trim_content(content: str, keep_chars: int) -> str:
    """把一条工具结果截短，**保持 JSON 结构合法**。

    为什么不能直接切字符串？
        我们的工具返回的是 {"ok": true, "result": ...}，
        从中间切一刀会得到非法 JSON，模型解析不了，等于把工具搞坏了。

    做法：如果是 JSON 且有 result 字段，只截 result，再补两个标记：
        _truncated: true        告诉模型"你看到的不是全部"
        _original_chars: 3200   告诉它原来有多长

    如果不是 JSON（纯文本工具），就切字符串 + 加一行说明。
    """
    if len(content) <= keep_chars:
        return content

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # 不是 JSON：切字符串，但必须说明
        return (
            content[:keep_chars]
            + f"\n\n...(内容已截断，原文 {len(content)} 字符。需要更多请用更精确的查询)"
        )

    if not isinstance(data, dict) or "result" not in data:
        return (
            content[:keep_chars]
            + f"\n\n...(内容已截断，原文 {len(content)} 字符)"
        )

    # 只截 result 字段，其余字段（ok / count / hint）保留
    result_text = (
        data["result"] if isinstance(data["result"], str)
        else json.dumps(data["result"], ensure_ascii=False)
    )
    original_chars = len(result_text)

    if original_chars > keep_chars:
        data["result"] = result_text[:keep_chars] + "…"
        data["_truncated"] = True
        data["_original_chars"] = original_chars

    return json.dumps(data, ensure_ascii=False)


def trim_tool_results(
    messages: list[Any],
    chars_to_cut: int,
    per_result_limit: int = DEFAULT_TOOL_RESULT_LIMIT,
) -> tuple[list[Any], int]:
    """从最长的工具结果开始截，直到省够 chars_to_cut。

    返回 (新的 messages, 实际省下的字符数)。

    为什么要"从最长的开始"？
        同样的压缩量，砍一刀最长的那条，损失的"条目数"最少。
        砍 5 条短的会把 5 个工具的结果都搞残，砍 1 条长的只残 1 个。

    ⚠️ 只改 role="tool" 的消息，且必须是 dict。
        assistant 消息是 SDK 对象，不动它（那是模型说的话，改了等于篡改历史）。
    """
    if chars_to_cut <= 0:
        return messages, 0

    # 复制一层，避免就地修改（调用方可能还要用原 messages 做对比）
    new_messages: list[Any] = []
    # 先找出所有可截断的工具消息：记录 (索引, 当前长度)
    candidates: list[tuple[int, int]] = []
    for idx, m in enumerate(messages):
        if not isinstance(m, dict) or m.get("role") != "tool":
            continue
        content = m.get("content") or ""
        if len(content) > per_result_limit:
            candidates.append((idx, len(content)))

    if not candidates:
        return messages, 0

    # 从最长的开始
    candidates.sort(key=lambda x: -x[1])

    # 逐个决定要砍到多少
    cut_plan: dict[int, str] = {}
    saved = 0
    for idx, cur_len in candidates:
        if saved >= chars_to_cut:
            break
        target_len = per_result_limit
        # 这一刀能省多少
        will_save = max(0, cur_len - target_len)
        cut_plan[idx] = target_len
        saved += will_save

    if not cut_plan:
        return messages, 0

    # 应用截断
    for idx, m in enumerate(messages):
        if idx in cut_plan:
            new_m = dict(m)      # 浅拷贝，不改原对象
            new_m["content"] = _trim_content(m.get("content") or "", cut_plan[idx])
            new_messages.append(new_m)
        else:
            new_messages.append(m)

    return new_messages, saved


def trim_summary(before: ContextBreakdown, after: ContextBreakdown) -> str:
    """压缩前后的对比，用于日志和实验。"""
    saved = before.total_est - after.total_est
    pct = saved / before.total_est * 100 if before.total_est else 0
    return (
        f"✂️ 压缩：{before.total_est} → {after.total_est} tok"
        f"（省 {saved}，{pct:.1f}%）"
    )
