"""按需取回工具 —— 给"被截断的原文"配一把钥匙。

--------------------------------------------------------------------------
为什么必须有这个工具
--------------------------------------------------------------------------
`truncate.py` 把长结果压成"头尾 + 句柄"。如果模型拿不到取回的手段，
那截断就退化成 W4 Day3 那个老问题：**信息没了，模型只能重查**。

有了这个工具，链路才闭环：

    工具返回 1500 字符
      ↓ 截断（头尾 700 + 句柄）
    模型看到的是"骨架 + 一把钥匙"
      ↓ 需要细节时
    fetch_artifact("a1b2c3d4e5f6", offset=800, length=600)
      ↓
    拿到中间那 600 字符

这也是**渐进式披露**的完整形态：
    第一层 search_docs  → 只给目录和摘要
    第二层 read_doc     → 给头尾
    第三层 fetch_artifact → 给指定片段

每层都只在"上一层不够用"时才展开。上下文里永远只有当前需要的那点东西。

--------------------------------------------------------------------------
两个防御
--------------------------------------------------------------------------
1. **单次取回上限 3000 字符**：不设上限的话，模型一次 `length=999999`
   就把全文塞回来了 —— 那前面辛苦截断等于白做。
   （模型不会"恶意"这么做，但它确实会为了省事这么干。）

2. **id 白名单**：12 位十六进制。挡住 `../.env` 这类路径穿越，
   和 read_doc 的 doc_id 校验是同一个理由。
"""

from __future__ import annotations

import json

from agent.artifact import load_artifact
from agent.tools import error_result, tool

# 单次最多取回多少字符（约 2000 token）。
MAX_FETCH_CHARS = 3000


@tool
def fetch_artifact(artifact_id: str, offset: int = 0, length: int = 1200) -> str:
    """取回被截断的工具结果原文片段。当工具结果显示 _truncated=true 或
    出现 artifact:xxxx 句柄、而你需要中间被省略的内容时使用。

    Args:
        artifact_id: 句柄 id，来自工具结果里的 _artifact 字段或 artifact:xxxx
        offset: 从第几个字符开始取，默认 0
        length: 取多少字符，默认 1200，最多 3000
    """
    # 上限兜底：模型可能传 999999
    length = max(1, min(int(length), MAX_FETCH_CHARS))

    data = load_artifact(artifact_id, offset=offset, length=length)

    if not data.get("ok"):
        if data.get("error") == "invalid_id":
            # model_fixable：模型抄错了 id 的格式，它自己改一下就好
            return error_result(
                "invalid_artifact_id",
                "artifact_id 必须是 12 位十六进制（例如 a1b2c3d4e5f6）",
                kind="model_fixable",
            )
        # model_fixable：id 格式对但文件不存在（可能记错了），把可用的给它
        return error_result(
            "artifact_not_found",
            f"没有 id 为 {artifact_id} 的存档",
            kind="model_fixable",
            available=data.get("available", []),
        )

    return json.dumps(
        {
            "ok": True,
            "artifact_id": data["artifact_id"],
            "content": data["content"],
            "total_chars": data["total_chars"],
            "offset": data["offset"],
            "returned": data["returned"],
            "has_more": data["has_more"],
            "hint": (
                f"还有 {data['total_chars'] - data['offset'] - data['returned']} 字符未读，"
                f"需要更多请把 offset 设为 {data['offset'] + data['returned']}"
                if data["has_more"] else "已到结尾"
            ),
        },
        ensure_ascii=False,
    )
