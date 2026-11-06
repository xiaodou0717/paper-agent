"""摘要压缩 —— 不丢信息，把它压成一句话。

--------------------------------------------------------------------------
为什么需要它？滑动窗口留下的那个洞
--------------------------------------------------------------------------
W5 Day1 的滑窗解决"早期轮次占位置"的问题，用的是**直接丢**：

    第 2 轮读的文档全文（1500 字符）  →  删掉

省 token 很有效（实测 -21%），但它有个前提：**被丢的那些内容，以后不会再被用到。**

Day1 的审计任务恰好满足这个前提（读完第一篇读第二篇，线性推进），
所以丢早期的没出事。换成"对比 11 篇文档的异同"就立刻出问题：

    要写"哪几篇讨论了同一个话题"
    → 必须记得每篇讲了什么
    → 丢掉前 8 篇，只剩 3 篇，比不出来

--------------------------------------------------------------------------
今天的做法：换成"压成一段话"
--------------------------------------------------------------------------
    第 3 轮读了《上下文工程》全文（1500 字符）
      ↓ 摘要
    "第 3 轮：读了《上下文工程》，讲 4 种失效模式和 6 个技巧"

**30 个字代替 1500 字，而且模型还知道"我读过这个"。**

--------------------------------------------------------------------------
三个必须想清楚的设计点
--------------------------------------------------------------------------
【一】摘要是有成本的，而且这笔钱要算进预算

    每次摘要 = 一次完整的模型调用（输入 = 被摘要的内容，输出 = 摘要）。
    它不是免费的。所以本模块把这次调用的 token 用量返回出去，
    由 loop 记到总账里 —— **否则你会以为省了钱，其实是把钱挪到了看不见的地方。**

【二】摘要要能"滚动合并"

    第一次：把第 1 轮压成摘要 A
    第二次：不能只压第 2 轮（那样会有 A 和 B 两段摘要，越积越多）
            必须把"A + 第 2 轮"一起压成新摘要 B

    代码里没有专门写合并逻辑 —— 因为上一版摘要就在"要摘要的那一段"里，
    它自然被一起喂进去了。**结构选对了，合并是自动的。**

【三】摘要会丢细节，而且丢的是"最贵的那些"

    摘要模型自然会保留"做了什么、结论是什么"，丢掉的是：
        精确数字（500 字符 / 15%）、专有名词、原文措辞、代码块

    而这些恰恰是"引用原文"类任务最需要的。
    所以摘要不是"更好的滑窗"，是**另一种取舍**：
        滑窗 = 信息全丢，但保留的轮次完整
        摘要 = 信息保留骨架，但骨架之外的细节全丢

--------------------------------------------------------------------------
安全网：和滑窗一样，宁可多花 token，不发非法请求
--------------------------------------------------------------------------
摘要消息是插在"头部"和"最近 N 轮"之间的一条普通 user 消息，
理论上不会破坏 assistant/tool 的配对。但"理论上"不够 ——
插完照样跑一次 `validate_structure()`，不合法就返回原样。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import config
from agent.window import validate_structure

# 摘要消息的标记。为什么要标记？
#   模型看到一条 user 消息，会以为"用户又说话了"。
#   加上标记，它才知道这是"系统帮我记的笔记"，不是新指令。
#   这和 W4 Day3 给截断结果加 `_truncated` 是同一个思路：
#   **让模型知道这条信息的性质，比信息本身更重要。**
SUMMARY_MARKER = "[早期历史摘要·非用户发言]"


# ============================================================
# 摘要提示词
# ============================================================
# 为什么提示词要写成"必须保留 / 可以丢"两栏？
#   不这么写，模型会按自己的偏好压 —— 它偏爱"结论"，
#   容易把"查过哪几篇文档"这种流水账删掉。而 Agent 恰恰需要知道
#   "哪些已经查过了"，否则会重复查（W5 Day1 里 26 次工具调用就是这么来的）。
SUMMARY_SYSTEM_PROMPT = """你在压缩一段 AI Agent 的工作记录，供它之后继续完成任务时参考。

你不是在写总结报告，你是在帮它"记事儿"。所以：
- 用第三人称陈述（"已读取…""待确认…"）
- 只写事实，不写评价
- 不要编造记录里没有的内容"""

SUMMARY_INSTRUCTION = """把下面的 Agent 工作记录压缩成一段摘要，不超过 200 字。

【必须保留】
1. 已经查过/读过哪些对象（文档名、工具名），避免之后重复查
2. 得到的关键结论，**尤其是数字、比例、阈值、专有名词**——这类细节丢了就找不回来
3. 还没做完的事（下一步该干什么）
4. 用户提过的要求或约束

【可以丢】
- 原文段落、代码块、完整的 JSON 输出
- 工具的字段名、调用参数细节
- 重复出现的内容

【输出格式】
直接给摘要正文。不要写"以下是摘要"，不要超过 6 条。"""


# ============================================================
# 小工具（和 window.py 里那几个一样）
# ============================================================
# 为什么又写一遍，不从 window.py 里 import？
#   那两个函数（_get / _role）在 window.py 里是**私有**的（下划线开头），
#   跨模块 import 别人的私有函数，等于把两个模块焊死：
#   哪天 window.py 改了 _role 的语义，compactor 会静默出错。
#   三行的东西，各留一份更安全。
#
# 而 validate_structure 是公开函数（本来就是要被别处用的），所以直接复用。


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """兼容 dict 和 SDK 对象。"""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _role(m: Any) -> str:
    return _get(m, "role") or ""


def _chars(m: Any) -> int:
    c = _get(m, "content")
    return len(c) if isinstance(c, str) else 0


def _total_chars(messages: list[Any]) -> int:
    return sum(_chars(m) for m in messages)


# ============================================================
# 结果
# ============================================================

@dataclass
class CompactionResult:
    """一次摘要压缩的结果。

    除了"新的消息列表"，还带了三类信息，各有各的用处：

        压缩统计   compacted_rounds / before_chars / after_chars
                   → 回答"这次省了多少"
        调用成本   prompt_tokens / completion_tokens
                   → 回答"这次省的钱，是不是被摘要调用吃掉了"
        成败       ok / note
                   → 失败时必须能说清为什么（不然线上只能靠猜）
    """
    messages: list[Any]                 # 处理后的消息（失败时是原样）
    summary_text: str = ""              # 摘要正文
    compacted_rounds: int = 0           # 压掉了几轮
    before_chars: int = 0               # 压缩前总字符
    after_chars: int = 0                # 压缩后总字符
    prompt_tokens: int = 0              # 摘要调用花了多少输入 token
    completion_tokens: int = 0          # 摘要调用花了多少输出 token
    ok: bool = True                     # 成功了吗
    note: str = ""                      # 说明（失败原因 / 为什么没压）

    @property
    def changed(self) -> bool:
        return self.compacted_rounds > 0

    @property
    def saved_chars(self) -> int:
        return max(0, self.before_chars - self.after_chars)

    def summary(self) -> str:
        """一行日志，和 WindowResult.summary() 风格保持一致。"""
        if not self.ok:
            return f"⚠️ 摘要失败（{self.note}），保持原样"
        if not self.changed:
            return f"摘要未触发（{self.note}）"
        return (
            f"📝 摘要 {self.compacted_rounds} 轮："
            f"{self.before_chars} → {self.after_chars} 字符"
            f"（省 {self.saved_chars}），摘要调用花 {self.prompt_tokens + self.completion_tokens} tok"
        )


# ============================================================
# 第一步：切成三段（纯计算，不花钱）
# ============================================================

def plan_split(
    messages: list[Any],
    keep_head: int = 2,
    keep_tail_rounds: int = 2,
) -> tuple[list[Any], list[Any], list[Any]] | None:
    """把消息切成「头部 / 要摘要的 / 要原样保留的」三段。

    返回 (head, old, tail)；不需要压缩时返回 None。

    ---------- 为什么边界还是 assistant ----------
    和 Day1 同一个理由：assistant 是一条"轮"的起点，从它切，
    它后面的 tool 自然跟着，不会出现"孤儿 tool"。

    ---------- keep_tail_rounds 可以是 0 ----------
    0 表示"全部摘要、一条都不留"。这看起来激进，但在
    "先总结再回答"的场景里是合理的（比如本文件顶部那个对比任务）。
    ⚠️ 注意 Python 的坑：`boundaries[-0]` 等于 `boundaries[0]`，
       不是"最后一个" —— 所以 0 必须单独判断，不能直接写 [-keep]。
    """
    if len(messages) <= keep_head:
        return None

    head = messages[:keep_head]
    body = messages[keep_head:]

    boundaries = [i for i, m in enumerate(body) if _role(m) == "assistant"]
    if not boundaries:
        return None
    if len(boundaries) <= keep_tail_rounds:
        return None

    if keep_tail_rounds <= 0:
        start = len(body)          # 全压，tail 为空
    else:
        start = boundaries[-keep_tail_rounds]

    return head, body[:start], body[start:]


# ============================================================
# 第二步：渲染成纯文本（喂给摘要模型）
# ============================================================

def render_for_summary(messages: list[Any], per_message_chars: int = 1200) -> str:
    """把消息列表渲染成一段可读文本。

    ---------- 为什么不能直接把 messages 塞给模型 ----------
    因为摘要调用是"另一次调用"，我们只想让它看内容、不想让它看结构。
    而且 messages 里有 tool_calls 这种嵌套结构，原样传过去，
    模型会把注意力花在"解析格式"上，而不是"提炼信息"。

    ---------- 为什么要截断每条 ----------
    摘要调用的输入也要钱。被摘要的内容本来就是"要扔掉的长文本"，
    为了摘要它再完整读一遍，成本会吃掉省下来的钱。
    单条留 1200 字符（约等于一篇文档的开头 + 结尾），够提炼主旨了。
    """
    parts: list[str] = []

    for m in messages:
        role = _role(m)
        content = _get(m, "content") or ""
        content = str(content)
        calls = _get(m, "tool_calls") or []

        if role == "system":
            parts.append(f"[系统设定] {content[:200]}")
            continue

        if role == "user":
            # 这一段里出现的 user 消息，只可能是"上一版摘要"
            if content.startswith(SUMMARY_MARKER):
                parts.append("[上一版摘要]\n" + content[len(SUMMARY_MARKER):].strip())
            else:
                parts.append("[用户] " + content[:per_message_chars])
            continue

        if role == "assistant":
            desc = "[助手]"
            if content.strip():
                desc += " " + content.strip()[:per_message_chars]
            for tc in calls:
                fn = _get(tc, "function") or {}
                name = _get(fn, "name") or "?"
                args = _get(fn, "arguments") or ""
                desc += f"\n  调用工具 {name}({str(args)[:160]})"
            if not content.strip() and not calls:
                desc += "（无内容）"
            parts.append(desc)
            continue

        if role == "tool":
            clipped = content[:per_message_chars]
            suffix = f"…（原文共 {len(content)} 字符）" if len(content) > per_message_chars else ""
            parts.append(f"[工具结果] {clipped}{suffix}")
            continue

        parts.append(f"[{role}] {content[:per_message_chars]}")

    return "\n\n".join(parts)


# ============================================================
# 第三步：把摘要插回去（纯计算，不花钱）
# ============================================================

def apply_summary(
    messages: list[Any],
    summary_text: str,
    keep_head: int = 2,
    keep_tail_rounds: int = 2,
) -> CompactionResult:
    """把一段现成的摘要插进消息列表，替换掉被摘要的轮次。

    ---------- 为什么要单独拆出来 ----------
    因为它是纯函数（不调模型），可以用来做**离线测试**：
    用一句假摘要验证"切得对吗、插完结构还合法吗"，成本为零。
    真要验证效果，才去花在线调用的钱。

    这条分工在 Day1 已经验证过：先离线排除低级错误，再上线看效果。

    ---------- 摘要消息为什么用 role="user" ----------
    可选方案有三个，各有代价：

        role="system"  语义最准（系统笔记），但放在对话中间不符合常见习惯，
                       部分服务商对"中途出现的 system"处理不一致
        role="assistant" 相当于"模型自己记得"，但会污染它的推理链
        role="user"    最兼容，代价是模型可能当成新指令 —— 所以加了标记前缀

    **选了兼容性最好的那个，然后用标记补语义。**
    """
    plan = plan_split(messages, keep_head, keep_tail_rounds)
    if plan is None:
        return CompactionResult(messages, ok=True, note="没有需要摘要的轮次")

    head, old, tail = plan

    # 防御：摘要模型返回空（推理模型把额度耗在思考上时就会这样）
    if not summary_text.strip():
        return CompactionResult(messages, ok=False, note="摘要为空")

    # 防御：这一段里一条 assistant 都没有，说明只有上一版摘要，没有新内容
    if not any(_role(m) == "assistant" for m in old):
        return CompactionResult(messages, ok=True, note="没有新轮次可摘要")

    summary_msg = {
        "role": "user",
        "content": f"{SUMMARY_MARKER}\n{summary_text.strip()}",
    }
    candidate = head + [summary_msg] + tail

    # ---------- 安全网 ----------
    ok, why = validate_structure(candidate)
    if not ok:
        # 宁可多花 token，也不发一个非法请求（Day1 定下的规矩）
        return CompactionResult(messages, ok=False, note=why)

    rounds = sum(1 for m in old if _role(m) == "assistant")
    return CompactionResult(
        messages=candidate,
        summary_text=summary_text.strip(),
        compacted_rounds=rounds,
        before_chars=_total_chars(messages),
        after_chars=_total_chars(candidate),
        ok=True,
    )


# ============================================================
# 第四步：真正调模型做摘要
# ============================================================

async def compact_with_summary(
    client: Any,
    messages: list[Any],
    *,
    keep_head: int = 2,
    keep_tail_rounds: int = 2,
    model: str | None = None,
    max_summary_tokens: int = 600,
    per_message_chars: int = 1200,
) -> CompactionResult:
    """把早期轮次压成一段摘要，返回新的消息列表。

    参数：
        client              AsyncOpenAI 客户端（和 loop 共用，不要新建）
        keep_head           头部保留几条（system + 第一条 user，别动）
        keep_tail_rounds    最近几轮原样保留；0 = 全压
        max_summary_tokens  摘要的输出上限
                            ⚠️ 不能给小：推理模型会把额度耗在"思考"上，
                               给 200 经常拿到空 content（W1 Day5 踩过）
        per_message_chars   渲染时每条最多留多少字符（控成本）

    返回 CompactionResult。**任何一步失败都返回原样 + ok=False**，
    绝不抛异常给上层 —— 摘要只是优化，不该让任务挂掉。
    """
    plan = plan_split(messages, keep_head, keep_tail_rounds)
    if plan is None:
        return CompactionResult(messages, ok=True, note="没有需要摘要的轮次")

    _head, old, _tail = plan

    # 没有新轮次（只有上一版摘要）→ 不用调模型，省一次钱
    if not any(_role(m) == "assistant" for m in old):
        return CompactionResult(messages, ok=True, note="没有新轮次可摘要")

    rendered = render_for_summary(old, per_message_chars=per_message_chars)

    try:
        resp = await client.chat.completions.create(
            model=model or config.MODEL,
            messages=[
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        SUMMARY_INSTRUCTION
                        + "\n\n--- 记录开始 ---\n"
                        + rendered
                        + "\n--- 记录结束 ---"
                    ),
                },
            ],
            temperature=0.0,          # 摘要要稳定，不要创造性
            max_tokens=max_summary_tokens,
        )
        text = (resp.choices[0].message.content or "").strip()
        usage = resp.usage
    except Exception as e:  # noqa: BLE001
        # 摘要失败不是灾难：上下文还是原样，任务照跑，只是继续贵着
        return CompactionResult(
            messages, ok=False, note=f"摘要调用失败：{type(e).__name__}: {e}"
        )

    if not text:
        return CompactionResult(
            messages, ok=False, note="摘要模型返回空（max_tokens 可能给小了）"
        )

    result = apply_summary(messages, text, keep_head, keep_tail_rounds)

    # 把这笔开销如实记上 —— 摘要不是免费的
    if usage is not None:
        result.prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        result.completion_tokens = getattr(usage, "completion_tokens", 0) or 0

    return result
