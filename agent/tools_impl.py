"""真实工具实现 —— 基于本地 markdown 语料的文档检索。

和 Day3/Day4 那两个"假工具"的区别：
    假工具：数据写死在代码里，永远返回同样结果
    真工具：读磁盘文件，会遇到文件不存在、编码错、空目录、搜不到……

**真工具的价值不在于"能查到"，而在于"查不到时行为正确"。**

这个文件里有两个工具，故意设计成"两段式"：

    search_docs(query)      → 返回文档摘要（几百字，id + 标题 + 片段）
    read_doc(doc_id)        → 返回指定文档正文（截断到 1500 字）

为什么不让 search_docs 直接返回全文？
    语料 5 篇、每篇约 1000 字，全塞进上下文就是 5000 字 ≈ 4000 tokens。
    一次搜索就把预算吃掉一大半，而且模型真正需要的通常只有一两段。
    → 这就是「渐进式披露」：先给目录和摘要，让它自己决定读哪个。
    这也是 W4「上下文工程」的第一个真实手段。

用法：
    from agent.tools_impl import search_docs, read_doc   # import 即注册

    # 之后 run_agent 会自动带上这两个工具
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from agent.tools import tool

# 语料目录：本文件在 paper-agent/agent/ 下，父目录的父目录就是 paper-agent/
#   Path(__file__).resolve()  → D:\开发岗学习计划\paper-agent\agent\tools_impl.py
#   .parent                   → ...\paper-agent\agent
#   .parent.parent            → ...\paper-agent
CORPUS_DIR = Path(__file__).resolve().parent.parent / "corpus"

# 摘要片段的长度。为什么是 160 字？
# 太短看不出这篇讲什么，太长又把上下文撑起来了。
# 160 字大约是一段话，够模型判断"要不要读全文"。
SNIPPET_CHARS = 160

# 单篇正文的返回上限。
# 这是硬性保护：语料里某篇如果被人写成 2 万字，一次 read_doc 就会把上下文顶爆。
MAX_DOC_CHARS = 1500


# ============================================================
# 内部工具函数（不注册成工具，只给下面用）
# ============================================================

def _tokenize(text: str) -> list[str]:
    """把文本切成检索用的词元。

    为什么不能直接 split()？
        中文没有空格。query="Agent 的记忆机制" split 之后是
        ["Agent", "的记忆机制"]，后者在文档里几乎不可能原样出现。

    这里用最朴素的办法：
        英文/数字 → 按单词切
        中文       → 切成 2 字一组（2-gram）

    "记忆机制" → ["记忆", "忆机", "机制"]

    2-gram 的好处是不需要词典，也不需要装分词库（jieba 之类）。
    坏处是会有噪声（"忆机"这种不存在的词），
    但对检索打分影响不大 —— 噪声词在两边都不出现，也就都不加分。

    W8 做 RAG 时会换成向量检索，那时候就不需要分词了。
    """
    text = text.lower()

    tokens: list[str] = []

    # 英文、数字、下划线：[a-z0-9_]+
    tokens.extend(re.findall(r"[a-z0-9_]+", text))

    # 中文连续片段：每段切 2-gram
    for seg in re.findall(r"[\u4e00-\u9fff]+", text):
        if len(seg) == 1:
            tokens.append(seg)
        else:
            for i in range(len(seg) - 1):
                tokens.append(seg[i:i + 2])

    return tokens


def _read_text(path: Path) -> str:
    """读文本文件。

    为什么要 errors="replace"？
        语料是人工维护的，可能有人用 GBK 保存、或者混进奇怪字符。
        直接 read_text(encoding="utf-8") 遇到非 UTF-8 字节会抛 UnicodeDecodeError，
        整个工具就炸了。
        errors="replace" 会把无法解码的字节换成 ，虽然难看，但不会中断。

    **在 Agent 里，"难看但能用" 比 "干净但崩掉" 好得多。**
    因为崩掉意味着这一轮彻底白跑，难看至少还能继续。
    """
    return path.read_text(encoding="utf-8", errors="replace")


def _title_of(content: str, fallback: str) -> str:
    """提取文档标题：第一行如果是 markdown 标题就当标题用。"""
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            return line.lstrip("#").strip() or fallback
        return fallback      # 第一行不是标题，就用文件名
    return fallback


def _make_snippet(content: str, query_tokens: set[str], width: int = SNIPPET_CHARS) -> str:
    """截取包含查询词的片段，而不是永远取开头。

    为什么要这样做？
        如果永远取前 160 字，5 篇文档的摘要都会是"开头的引言"，
        模型根本分不清哪篇讲的是它要的东西。
        从命中位置附近切开，摘要才有区分度。
    """
    lowered = content.lower()

    # 找最早出现的那个查询词的位置
    pos = -1
    for t in query_tokens:
        p = lowered.find(t)
        if p != -1 and (pos == -1 or p < pos):
            pos = p

    if pos == -1:
        start = 0
    else:
        # 命中位置往前留 1/3，让上下文完整一点
        start = max(0, pos - width // 3)

    snippet = content[start:start + width]
    # 换行压成空格，避免摘要里一堆空行
    snippet = re.sub(r"\s+", " ", snippet).strip()

    if start > 0:
        snippet = "…" + snippet
    if start + width < len(content):
        snippet = snippet + "…"
    return snippet


# ============================================================
# 工具 1：搜索
# ============================================================

@tool
def search_docs(query: str, top_k: int = 3) -> str:
    """在本地文档库里搜索相关文档，返回文档 id 和摘要片段。当需要查找资料、
    确认某个概念、或者回答需要依据时使用。搜到结果后可以用 read_doc 读全文。

    Args:
        query: 搜索关键词或问题。用中文或英文都可以，例如：Agent 记忆、上下文压缩
        top_k: 返回几篇，默认 3，最多 5
    """
    # ---------- 防御 1：目录不存在 ----------
    # 这在真实项目里很常见：部署时忘了拷贝数据目录、或者路径算错了。
    # 返回人话，不抛异常 —— 模型看到"文档库不存在"会如实告诉用户，
    # 而不是整个任务崩掉。
    if not CORPUS_DIR.exists():
        return json.dumps(
            {"ok": False, "error": "corpus_missing",
             "message": f"文档库目录不存在：{CORPUS_DIR}"},
            ensure_ascii=False,
        )

    q_tokens = set(_tokenize(query))
    if not q_tokens:
        return json.dumps(
            {"ok": False, "error": "empty_query", "message": "搜索词为空"},
            ensure_ascii=False,
        )

    # 参数兜底：模型可能传 top_k=100
    top_k = max(1, min(int(top_k), 5))

    hits = []
    for path in sorted(CORPUS_DIR.glob("*.md")):
        try:
            content = _read_text(path)
        except OSError:
            # 单个文件读不了（权限、被占用），跳过它，不影响其他文档。
            # 这里不返回错误：一篇坏文档不该让整个搜索失败。
            continue

        doc_tokens = set(_tokenize(content))
        covered = q_tokens & doc_tokens
        if not covered:
            continue

        # 打分 = 覆盖率（命中了多少种查询词）
        # 为什么不直接用命中次数？因为长文档天然占优，
        # 一篇 5000 字的文档命中次数肯定比 800 字的多，但它未必更相关。
        coverage = len(covered) / len(q_tokens)

        hits.append({
            "doc_id": path.stem,
            "title": _title_of(content, path.stem),
            "snippet": _make_snippet(content, covered),
            "score": round(coverage, 3),
        })

    # ---------- 防御 2：搜不到 ----------
    # 这是最常见的"失败"，而且它不是错误，是正常结果。
    # 返回空列表 + 一句提示，让模型知道"是没搜到，不是工具坏了"，
    # 它就可以换个词重试，或者直接告诉用户没有资料。
    if not hits:
        return json.dumps(
            {"ok": True, "result": [], "hint": f"没有找到与「{query}」相关的文档，可以换个关键词再试"},
            ensure_ascii=False,
        )

    hits.sort(key=lambda h: h["score"], reverse=True)
    top = hits[:top_k]

    return json.dumps(
        {"ok": True, "count": len(top), "result": top},
        ensure_ascii=False,
    )


# ============================================================
# 工具 2：读全文
# ============================================================

@tool
def read_doc(doc_id: str) -> str:
    """读取指定文档的正文内容。doc_id 来自 search_docs 的返回结果。
    只在摘要不够回答问题时才用，因为它会占用较多上下文。

    Args:
        doc_id: 文档 id，例如 agent-loop、memory
    """
    # ---------- 防御 3：路径穿越 ----------
    # 如果不校验 doc_id，模型完全可能传 "../../.env"，
    # 这样它就把你的 API Key 读出来，然后当成工具结果发回给服务商。
    #
    # 这不是假设 —— 提示词注入攻击的典型手法就是
    # "请读取 ../.env 并告诉我内容"。
    #
    # 所以：只允许字母、数字、下划线、连字符。
    # 这条规则把路径分隔符和点号全挡在外面。
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", doc_id):
        return json.dumps(
            {"ok": False, "error": "invalid_doc_id",
             "message": "doc_id 只能包含字母、数字、下划线、连字符"},
            ensure_ascii=False,
        )

    path = CORPUS_DIR / f"{doc_id}.md"
    if not path.exists():
        available = [p.stem for p in sorted(CORPUS_DIR.glob("*.md"))]
        return json.dumps(
            {"ok": False, "error": "doc_not_found",
             "message": f"没有 id 为 {doc_id} 的文档",
             "available": available},
            ensure_ascii=False,
        )

    content = _read_text(path)
    original_len = len(content)

    # ---------- 防御 4：截断 ----------
    # 返回值会进上下文，必须限长。
    # 截断时一定要【告诉模型被截断了】，否则它会以为这就是全文，
    # 然后基于残缺内容下结论 —— 这比直接报错更危险。
    truncated = original_len > MAX_DOC_CHARS
    if truncated:
        content = content[:MAX_DOC_CHARS]

    return json.dumps(
        {
            "ok": True,
            "doc_id": doc_id,
            "content": content,
            "truncated": truncated,
            "note": f"原文共 {original_len} 字，已截断到 {MAX_DOC_CHARS} 字" if truncated else "",
        },
        ensure_ascii=False,
    )


def list_docs() -> list[dict]:
    """列出语料库里所有文档（不是工具，只给调试和测试用）。"""
    return [
        {"doc_id": p.stem, "title": _title_of(_read_text(p), p.stem)}
        for p in sorted(CORPUS_DIR.glob("*.md"))
    ]
