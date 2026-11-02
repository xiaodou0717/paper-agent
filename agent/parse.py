"""解析兜底 —— 当模型输出的不是"干净的 JSON"时。

前三天的失败都发生在【工具执行】这条线上（超时、权限、重复扣款）。
今天换个方向：**失败发生在模型输出本身** —— 它返回的东西根本不是你要的格式。

模型会给出哪些"垃圾"？实测下来主要是五种：

    ① 包在 markdown 代码块里   ```json {...} ```
    ② 前面带一句解释          "好的，这是结果：{...}"
    ③ 尾随逗号                {"a": 1,}
    ④ 被 max_tokens 截断      {"title": "Attent
    ⑤ 干脆不是 JSON           "我无法完成这个任务"

这五种不能一视同仁 —— **前三种能救，后两种救不了。**

    ①②③ 是"格式脏了但内容完整"  → 剥掉壳/修一下就行，成本几乎为零
    ④   是"内容本身缺了一半"    → 任何修复都是猜，必须重新生成
    ⑤   是"模型拒绝了"        → 重试大概率也一样，要么改提示词要么放弃

所以兜底是有顺序的，不能乱：

    先试着救（宽松解析） → 救不了再重试一次 → 还不行就如实报错

**顺序反了会浪费钱**：直接重试的话，① ② ③ 这三种本来零成本能解决的问题
会白白花掉一轮 token。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

# 匹配 ```json ... ``` 或 ``` ... ```（大小写不敏感，允许语言标记缺失）
_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*\n?(.*?)```", re.S)

# 尾随逗号：,} 或 ,]
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


@dataclass
class ParseResult:
    """解析结果。

    strategy 记录"用了哪一招救回来的"，这是可观测性的关键：
        direct     原样就是合法 JSON（正常情况）
        codeblock  剥掉 markdown 代码块救回来的
        extract    从一段话里抠出 JSON 救回来的
        repaired   修掉尾随逗号救回来的
        failed     救不了

    为什么要记录 strategy？
        如果线上 `extract` 占比很高，说明模型老爱加解释文字 ——
        你该去改 prompt，而不是一直靠兜底。
        **兜底是保险，不是常态。**
    """
    ok: bool
    data: Any | None
    strategy: str
    error: str = ""

    def summary(self) -> str:
        if self.ok:
            return f"✅ {self.strategy}"
        return f"❌ failed: {self.error[:60]}"


def extract_json(text: str | None) -> ParseResult:
    """从任意文本里尽力抠出 JSON 对象。

    四步，从"最省事"到"最费事"：

        1. 直接解析         —— 正常的模型输出走这里
        2. 剥 markdown 代码块 —— 模型最爱加的东西
        3. 找第一个 { 到最后一个 } —— 前面有解释文字的情况
        4. 修尾随逗号        —— 模型第二爱犯的错

    注意每一招只在前一招失败后才试，因为：
        - 第 1 招成功时，说明输出本来就干净，不该再折腾
        - 第 3 招有风险（可能抠错范围），所以放在 2 之后

    返回 ParseResult，**不抛异常** —— 和 execute_tool 的设计保持一致：
    失败也是一种结果，交给调用方决定怎么办。
    """
    if not text or not text.strip():
        return ParseResult(False, None, "failed", "输出为空")

    raw = text.strip()

    # ---------- 第 1 招：直接解析 ----------
    try:
        return ParseResult(True, json.loads(raw), "direct")
    except json.JSONDecodeError as e:
        first_error = str(e)

    # ---------- 第 2 招：剥代码块 ----------
    # 模型很爱把 JSON 包在 ```json ... ``` 里，即使你说了"不要 markdown"
    m = _FENCE_RE.search(raw)
    if m:
        inner = m.group(1).strip()
        try:
            return ParseResult(True, json.loads(inner), "codeblock")
        except json.JSONDecodeError:
            # 代码块里也可能有尾随逗号，顺手再试一次
            fixed = _TRAILING_COMMA_RE.sub(r"\1", inner)
            if fixed != inner:
                try:
                    return ParseResult(True, json.loads(fixed), "codeblock")
                except json.JSONDecodeError:
                    pass

    # ---------- 第 3 招：抠出 {...} 或 [...] ----------
    # 从第一个左括号到最后一个右括号。这个范围可能包含多余的字符，
    # 所以放在第 2 招之后 —— 有代码块时优先用代码块，范围更精确。
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = raw.find(open_ch)
        end = raw.rfind(close_ch)
        if start == -1 or end <= start:
            continue

        candidate = raw[start:end + 1]

        try:
            return ParseResult(True, json.loads(candidate), "extract")
        except json.JSONDecodeError:
            pass

        # ---------- 第 4 招：修尾随逗号 ----------
        fixed = _TRAILING_COMMA_RE.sub(r"\1", candidate)
        if fixed != candidate:
            try:
                return ParseResult(True, json.loads(fixed), "repaired")
            except json.JSONDecodeError:
                pass

    # ---------- 都失败 ----------
    return ParseResult(False, None, "failed", first_error)


def looks_truncated(text: str) -> bool:
    """粗略判断一段 JSON 是不是被截断了。

    为什么要单独判断？因为**截断和"格式脏"的处理方式完全不同**：

        格式脏 → 修复（本地、免费、100% 可控）
        截断   → 必须重新生成（花 token，而且模型可能再截一次）

    判断方法：括号不平衡，或者以逗号/冒号结尾（正说到一半）。

    这是启发式，不是精确判断 —— 但它能帮你区分"该修还是该重来"。
    """
    if not text or not text.strip():
        return False

    s = text.strip()

    # 括号不平衡：数一数，看有没有闭合
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        if s.count(open_ch) > s.count(close_ch):
            return True

    # 结尾是逗号/冒号 → 明显没说完
    if s.endswith((",", ":", "，", "：")):
        return True

    # 结尾在字符串中间（引号数量是奇数）
    if s.count('"') % 2 == 1:
        return True

    return False
