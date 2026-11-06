"""落盘存储 —— 把"上下文里放不下的原文"存到磁盘，只留一个句柄。

--------------------------------------------------------------------------
为什么需要它？截断的根本矛盾
--------------------------------------------------------------------------
工具结果太长，必须截断。但截断有个绕不过去的矛盾：

    截得少  → 省不下 token（上下文还是胖）
    截得多  → 信息没了，模型要么瞎猜，要么重新调工具（更贵）

W4 Day3 的实测就是这个矛盾：压到 400 字符以下，模型看不懂，
又去重新搜一遍 —— **省下的 token 原样花回去**。

--------------------------------------------------------------------------
破局思路：上下文不是唯一的存储
--------------------------------------------------------------------------
模型的上下文很小，但**磁盘几乎是无限的**。所以：

    1500 字符的文档全文
      ↓ 写进磁盘（.artifacts/a1b2c3d4e5f6.txt）
    上下文里只留 700 字符的头尾 + 一个句柄 a1b2c3d4e5f6

模型需要细节时，调用 `fetch_artifact("a1b2c3d4e5f6", offset=800)` 取回。
**这就是"按需加载"（progressive disclosure）的第二层**：

    第一层：先给摘要，让它决定读哪篇（W2 Day5 的 search_docs → read_doc）
    第二层：读了也只看头尾，要细节再按段取（今天）

--------------------------------------------------------------------------
为什么用内容哈希当文件名
--------------------------------------------------------------------------
id = sha256(内容) 的前 12 位。带来两个好处，都不用写代码：

    1. 天然去重：同一篇文档读 10 次，只产生 1 个文件
    2. 天然幂等：同一个内容永远对应同一个 id（W3 Day3 那个思路）

如果改用自增序号或时间戳，就得额外写"这段内容是不是存过了"的判断。
**选对主键，功能是免费的。**

--------------------------------------------------------------------------
安全边界
--------------------------------------------------------------------------
- 只存文本，不存二进制（今天是文档语料，够用）
- id 必须是 12 位十六进制：挡住 `../../.env` 这类路径穿越（和 read_doc 同一个防御）
- 目录进 .gitignore：运行时产物不能进仓库
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

# 落盘目录：agent/ 的上一级（paper-agent/）下的 .artifacts/
#   Path(__file__).resolve()  → D:\开发岗学习计划\paper-agent\agent\artifact.py
#   .parent                   → ...\paper-agent\agent
#   .parent.parent            → ...\paper-agent
ARTIFACT_DIR = Path(__file__).resolve().parent.parent / ".artifacts"

# id 的长度。12 位十六进制 = 48 bit，碰撞概率在"几百个文件"的量级下可以忽略。
ID_LEN = 12

# 只允许十六进制，长度必须对 —— 一行正则挡掉所有路径穿越
_ID_PATTERN = re.compile(rf"^[0-9a-f]{{{ID_LEN}}}$")


def artifact_id_for(content: str) -> str:
    """算出一段内容的 id，**但不写盘**。

    为什么要拆出这一步？
        调用方需要先知道 id（好写进结果的句柄里），再决定"到底要不要截"。
        先写盘后判断，会在磁盘上留下一堆"其实没用上"的副本。
        **先算后写，能避免副作用先发生。**
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:ID_LEN]


def save_artifact(content: str) -> str:
    """把一段文本存到磁盘，返回它的 id。

    已存在就不重复写（内容哈希做主键，天然去重）。
    """
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    digest = artifact_id_for(content)
    path = ARTIFACT_DIR / f"{digest}.txt"

    # 为什么用 newline="" 写？
    #   默认写入时 Python 会把 \n 转成 \r\n（Windows）。原文是什么样就存什么样，
    #   否则取回来的内容和工具原本返回的不一致，字符数对不上。
    if not path.exists():
        path.write_text(content, encoding="utf-8", newline="")

    return digest


def artifact_path(artifact_id: str) -> Path | None:
    """把 id 换成路径。id 不合法时返回 None（**不要抛异常**）。

    为什么不抛异常？
        这个函数会被工具调用。工具抛异常 = 整轮白跑；
        返回 None 让调用方给模型一句人话，模型还能自己改。
    """
    if not isinstance(artifact_id, str) or not _ID_PATTERN.match(artifact_id):
        return None
    return ARTIFACT_DIR / f"{artifact_id}.txt"


def load_artifact(artifact_id: str, offset: int = 0, length: int = 1500) -> dict:
    """按片段读取落盘内容。

    返回一个 dict（不是纯字符串），因为调用方需要知道"还有没有更多"：
        {"ok": True, "content": "...", "total_chars": 1500,
         "offset": 800, "returned": 700, "has_more": False}

    为什么按 offset/length 读，而不是整篇返回？
        整篇返回就把"省下来的 token"又塞回去了 —— 那截断等于白做。
        分片读保证**每次取回的量可控**。
    """
    path = artifact_path(artifact_id)
    if path is None:
        return {"ok": False, "error": "invalid_id", "message": "id 必须是 12 位十六进制"}

    if not path.exists():
        return {
            "ok": False,
            "error": "not_found",
            "message": f"没有 id 为 {artifact_id} 的存档",
            "available": list_artifacts()[:10],
        }

    text = path.read_text(encoding="utf-8", errors="replace")
    total = len(text)

    offset = max(0, min(offset, total))
    length = max(1, length)
    piece = text[offset:offset + length]

    return {
        "ok": True,
        "artifact_id": artifact_id,
        "content": piece,
        "total_chars": total,
        "offset": offset,
        "returned": len(piece),
        "has_more": offset + len(piece) < total,
    }


def list_artifacts() -> list[str]:
    """列出已有的 id（新的在前）。只用于调试和报错时给提示。"""
    if not ARTIFACT_DIR.exists():
        return []
    files = sorted(ARTIFACT_DIR.glob("*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [p.stem for p in files]
