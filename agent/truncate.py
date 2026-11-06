"""工具结果截断 —— 从"粗暴切一刀"升级到"保头尾 + 可回取"。

--------------------------------------------------------------------------
W4 Day3 的截断哪里不够
--------------------------------------------------------------------------
`budget.py` 里的 `_trim_content()` 做的是**只留开头**：

    1500 字符的文档 → 前 800 字符 → 前 400 字符 → 前 200 字符

问题是文档的结构通常是：

    开头：定位 + 目录（信息密度低）
    中间：论证过程（最占篇幅）
    结尾：结论 + 注意事项（**信息密度最高**）

只留开头，等于把最值钱的"结尾结论"扔了，留下最便宜的"引言"。
实测也能看出来：压到 400 以下模型就不够用，又去重新读一遍。

--------------------------------------------------------------------------
今天的三个升级
--------------------------------------------------------------------------
【一】保头尾，砍中间

    头 60%  +  …[中间省略 N 字符]…  +  尾 40%

为什么是 6:4 而不是 5:5？
    开头有标题和"这篇讲什么"，需要多一点；
    结尾往往是结论，必须留。
    （这个比例是拍的，Day5 可以拿数据调。别把拍的值当定理。）

【二】原文落盘，留一个句柄

截断处插一句：

    …[中间省略 812 字符，原文见 artifact:a1b2c3d4e5f6]…

模型要细节就调 `fetch_artifact("a1b2c3d4e5f6", offset=800)`。
**上下文里只放"索引"，正文放磁盘。** 这和数据库不把大字段塞进索引是同一个道理。

【三】在"写入时"截断，而不是"超预算时"

W4 的做法是：预算不够了，回头去压已经写进上下文的结果。
今天改成：**工具结果一回来就先截**。区别很大：

    事后压：全文进过上下文，峰值已经顶上去过了
    写入时截：全文从没进过上下文，峰值根本不会高

Day2 的结论"峰值由最近几轮决定"，就是在提示这件事。

--------------------------------------------------------------------------
一个必须做对的细节：截字段值，不截 JSON 字符串
--------------------------------------------------------------------------
我们的工具返回的是 JSON：

    {"ok": true, "doc_id": "x", "content": "……1500 字……"}

直接在字符串上切一刀，得到的是**非法 JSON**，模型解析不了，
会以为工具坏了 —— 这比截断本身糟得多（W4 Day3 已经踩过一次）。

所以流程是：
    解析 → 找到最长的那个字符串字段 → 只截它 → 重新序列化
非 JSON 的纯文本工具，才走"字符串头尾"这条路。
"""

from __future__ import annotations

import json
import math
from typing import Any

from agent.artifact import artifact_id_for, save_artifact

# 单条工具结果的默认上限（字符）。
# 为什么是 800：约 500 token，够放一条检索结果的全部字段 + 一段正文开头和结尾。
DEFAULT_LIMIT = 800

# 头尾比例：头 60% / 尾 40%。
HEAD_RATIO = 0.6

# 再小就没意义了：200 字符连标题都放不下，不如不截直接报错。
MIN_LIMIT = 200

# JSON 结果里，除了被截的那个字段，还要给其他字段和标记留点地方。
JSON_OVERHEAD = 150

# 最少要省这么多字符才值得截。
#
# 为什么需要这个门槛？（离线实验里撞出来的）
#   一份 995 字符的结果，limit=800 时: 正文砍掉 345 字符，
#   但缺口说明 + _artifact + _hint 又占回去 250 —— 净省不到 100。
#   信息丢了、结构还变复杂了，就为省 100 个字符，不划算。
#
# **优化也有"最小起订量"：收益盖不过开销的事，不如不做。**
MIN_SAVING = 120

# 列表裁剪时至少保留多少比例。
# 0.6 = 最多砍 40%。低于这个数，结果就"不够用"了（见 _shrink_json 第 ③ 步的注释）。
MIN_KEEP_RATIO = 0.6


# ============================================================
# 遍历 JSON，找到"最长的那个字符串"
# ============================================================

def _walk_strings(obj: Any, path: tuple = (), depth: int = 0):
    """递归找出所有字符串值，产出 (路径, 内容)。

    为什么要 depth 限制？
        工具返回的 JSON 是我们自己构造的，最多 3 层。
        不限制的话，万一某个工具返回了畸形深嵌套，递归会炸栈。
        **防御要按"我们知道的最坏情况"来写，不是按"理论上可能"。**
    """
    if isinstance(obj, str):
        yield path, obj
    elif isinstance(obj, dict) and depth < 3:
        for k, v in obj.items():
            yield from _walk_strings(v, path + (k,), depth + 1)
    elif isinstance(obj, list) and depth < 3:
        for i, v in enumerate(obj):
            yield from _walk_strings(v, path + (i,), depth + 1)


def _set_path(obj: Any, path: tuple, value: Any) -> None:
    """按路径写回。path 是 _walk_strings 产出的元组，如 ("result", 0, "snippet")。"""
    cur = obj
    for key in path[:-1]:
        cur = cur[key]
    cur[path[-1]] = value


# ============================================================
# 头尾截断
# ============================================================

def head_tail(
    text: str,
    limit_chars: int,
    head_ratio: float = HEAD_RATIO,
    artifact_id: str = "",
) -> tuple[str, int]:
    """保留头尾、砍掉中间。返回 (新文本, 被砍掉的字符数)。

    ---------- 为什么缺口要写出来 ----------
    截断后如果只留一个"…"，模型会以为这是原文的全部，
    然后基于残缺信息给出自信的错误答案。

    所以缺口必须**说清楚三件事**：
        砍了多少（812 字符）
        去哪取（artifact:a1b2c3d4e5f6）
        怎么取（fetch_artifact）

    **让它知道自己瞎了一只眼，比让它以为看得见更重要。**
    （这句话 W4 Day3 说过，今天依然成立，只是这次还给了它一副备用眼镜。）
    """
    limit_chars = max(MIN_LIMIT, limit_chars)

    if len(text) <= limit_chars:
        return text, 0

    if artifact_id:
        marker = f"\n…[省略 {{n}} 字符，原文 artifact:{artifact_id}]…\n"
    else:
        marker = "\n…[中间省略 {n} 字符]…\n"

    # 缺口说明本身也占长度，所以先按"最坏情况"预留（按 6 位数留）
    reserved = len(marker.format(n=999999))
    budget = max(MIN_LIMIT, limit_chars - reserved)

    head_len = int(budget * head_ratio)
    tail_len = budget - head_len
    omitted = len(text) - head_len - tail_len

    middle = marker.format(n=omitted)
    return text[:head_len] + middle + text[-tail_len:], omitted


# ============================================================
# 对外主函数
# ============================================================

def _mark_truncated(data: Any, original_chars: int, artifact_id: str) -> None:
    """在顶层打三个标记：截过了、原来多长、去哪取。"""
    if not isinstance(data, dict):
        return
    data["_truncated"] = True
    data["_original_chars"] = original_chars
    if artifact_id:
        data["_artifact"] = artifact_id
        data["_hint"] = (
            f"中间内容可用 fetch_artifact({artifact_id}, offset, length) 取回"
        )


def _dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False)


def _is_json_container(s: str) -> bool:
    """这段字符串本身是不是一段 JSON 结构（dict / list）？

    为什么要单独判断？
        工具结果有两层：外层信封 {"ok":true,"result":"<内层 JSON 文本>"}。
        内层那段文本如果按"普通字符串"做头尾截，内层就变成非法 JSON 了。
        （今天踩的坑：模型看到"有 5 个结果"，却读不出是哪 5 个，
         于是疯狂换关键词重搜，8 步烧掉 4.5 万 token。）
    """
    s = s.strip()
    if not s or s[0] not in "{[":
        return False
    try:
        return isinstance(json.loads(s), (dict, list))
    except json.JSONDecodeError:
        return False


def _shrink_json(
    data: Any,
    limit_chars: int,
    head_ratio: float,
    artifact_id: str,
    depth: int = 0,
) -> tuple[Any, bool]:
    """在 JSON 结构内部做截断，保证**每一层都还是合法 JSON**。

    三步，按"损失从小到大"排：

        ① 截最长的普通字符串（头尾保留）—— 只丢中间那段文字
        ② 递归进"本身是 JSON 的字符串"—— 保住两层的结构
        ③ 列表从尾部丢条目 —— 丢的是整条，不是半条

    为什么把"丢条目"放最后？
        丢一条 = 少一个候选；截一半 = 这条彻底废掉但还占着位置。
        **宁可少几条完整的，不要几条残缺的**（Day1 的结论，这里再次成立）。

    ⚠️ 沉默的失败：
        如果三步都动不了（比如结构里全是短字段），返回 changed=False，
        调用方会**原样返回**——结构保不住就不截，绝不把 JSON 弄坏。
    """
    if len(_dumps(data)) <= limit_chars:
        return data, False

    changed = False
    budget = max(MIN_LIMIT, limit_chars - JSON_OVERHEAD)

    # ---------- ① 截最长的"普通字符串" ----------
    plain = [(p, s) for p, s in _walk_strings(data) if not _is_json_container(s)]
    if plain:
        path, longest = max(plain, key=lambda kv: len(kv[1]))
        new_text, omitted = head_tail(longest, budget, head_ratio, artifact_id)
        if omitted > 0:
            _set_path(data, path, new_text)
            _mark_truncated(data, len(longest), artifact_id)
            changed = True

    # ---------- ② 还是超：递归进嵌套的 JSON ----------
    if len(_dumps(data)) > limit_chars and depth < 2:
        for path, s in _walk_strings(data):
            if not _is_json_container(s):
                continue
            inner = json.loads(s)
            new_inner, ch = _shrink_json(inner, budget, head_ratio, artifact_id, depth + 1)
            if ch:
                _set_path(data, path, _dumps(new_inner))
                _mark_truncated(data, len(s), artifact_id)
                changed = True
            break      # 只处理最外层的那个嵌套，避免一次动太多

    # ---------- ③ 还是超：列表丢尾部条目（有下限，且能反悔）----------
    #
    # ⚠️ 这一条必须谨慎，下面是踩过的坑：
    #    搜索结果 5 篇，limit=600 时按"能装多少装多少"砍到了 1 篇。
    #    数据是合法的，但**语义被毁了**——任务是"找出所有文档"，
    #    只给 1 篇等于让它重新搜，省下的 token 又花回去。
    #
    # 所以定了两条规矩：
    #    1. 最多砍 40%（至少保留 60% 的条目）
    #    2. 砍到下限还是装不下 → 全部撤销，宁可整条不截
    # **"部分截断"只有在"还够用"的前提下才成立。**
    if len(_dumps(data)) > limit_chars:
        original_ref = None
        items_ref = None

        if isinstance(data, list) and len(data) > 1:
            items_ref = data
        elif isinstance(data, dict):
            lists = [(k, v) for k, v in data.items() if isinstance(v, list) and len(v) > 1]
            if lists:
                _key, items_ref = max(lists, key=lambda kv: len(_dumps(kv[1])))

        if items_ref is not None:
            original_items = list(items_ref)
            keep_min = max(1, math.ceil(len(original_items) * MIN_KEEP_RATIO))

            while len(items_ref) > keep_min and len(_dumps(data)) > limit_chars:
                items_ref.pop()

            if len(_dumps(data)) > limit_chars:
                # 砍到下限还是装不下：撤销，别把列表砍残
                items_ref[:] = original_items
                if not changed:
                    return data, False
            else:
                dropped = len(original_items) - len(items_ref)
                if dropped:
                    changed = True
                    # 顶层列表没法加标记，字典可以
                    if isinstance(data, dict):
                        data["_truncated"] = True
                        data["_dropped_items"] = dropped
                        if artifact_id:
                            data["_artifact"] = artifact_id
                            data["_hint"] = (
                                f"只保留了前 {len(items_ref)} 条，完整内容可用 "
                                f"fetch_artifact({artifact_id}, offset, length) 取回"
                            )

    return data, changed


def shrink_tool_result(
    raw: str,
    limit_chars: int = DEFAULT_LIMIT,
    head_ratio: float = HEAD_RATIO,
    save: bool = True,
) -> tuple[str, int]:
    """把一条工具结果压到 limit_chars 以内。返回 (新结果, 省下的字符数)。

    三种情况：
        已经够短   → 原样返回，不落盘（内容还在上下文里，取回个啥）
        是 JSON    → 结构化截断，**每层都保持合法**；保不住就不截
        纯文本     → 头尾截断

    ⚠️ 兜底：如果压完反而更长（JSON 转义膨胀、标记太多），返回原样。
       **优化不能把东西变糟**，这个判断比多省几个字符重要。
    """
    raw_len = len(raw)
    if raw_len <= limit_chars:
        return raw, 0

    # 先算 id，不写盘 —— 等确认"这次截断值得做"再落盘
    artifact_id = artifact_id_for(raw) if save else ""

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = None

    candidate: str | None = None

    # ---------- 情况 2：JSON（结构化截断）----------
    if isinstance(data, (dict, list)):
        new_data, changed = _shrink_json(data, limit_chars, head_ratio, artifact_id)
        if not changed:
            # 结构保不住，就不截。**坏的 JSON 比长的 JSON 贵得多。**
            return raw, 0
        out = _dumps(new_data)
        if len(out) < raw_len:
            candidate = out

    # ---------- 情况 3：纯文本 ----------
    if candidate is None:
        new_text, omitted = head_tail(raw, limit_chars, head_ratio, artifact_id)
        if omitted <= 0:
            return raw, 0
        if artifact_id:
            new_text += f"\n[原文 {raw_len} 字符，完整内容见 artifact:{artifact_id}]"
        candidate = new_text

    saved = raw_len - len(candidate)

    # ---------- 门槛：省得不够多就别截 ----------
    if saved < MIN_SAVING:
        return raw, 0

    # 确认要截了，这时候才落盘（避免留下一堆用不上的副本）
    if artifact_id:
        save_artifact(raw)

    return candidate, saved
