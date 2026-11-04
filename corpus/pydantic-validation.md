# 数据校验（Pydantic）

## 一句话定位

在数据进入业务逻辑之前，**把它变成你能信任的形状**。

## 为什么 LLM 场景特别需要它

模型的输出是"看起来像结构化数据的文本"。它会：

- 少字段、多字段
- 类型错（该是数字的给了字符串）
- 语义错（`year: 20` 而不是 `2017`）
- 格式对但内容编造

**这些都不会报错**，除非你校验。校验是唯一能"在错误流下去之前拦住它"的地方。

## v2 的四个常用能力

```python
from pydantic import BaseModel, Field, field_validator

class Paper(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    year: int = Field(ge=1900, le=2100)
    authors: list[str] = Field(default_factory=list)

    @field_validator("title")
    @classmethod
    def strip_title(cls, v: str) -> str:
        return v.strip()
```

- `Field(...)`：约束范围，声明即文档
- `field_validator`：自定义清洗和检查
- `model_json_schema()`：直接生成 function calling 的参数 schema
- `model_validate_json()`：一步完成"解析 + 校验"

最后一个最实用：它把 `json.loads` 和校验合成一步，
失败时给出的错误信息比原生 JSON 错误清晰得多。

## 关键用法：把校验错误回喂给模型

校验失败不是终点，是**一次修正机会**：

```
第 1 次：模型输出 → 校验失败（year 必须是 1900-2100，你给了 20）
第 2 次：把错误信息发回去 → 模型修正 → 校验通过
```

这比"失败就报错给用户"的体验好得多。**前提是错误信息要具体**
（哪个字段、期望什么、实际是什么），而不是笼统的"格式错误"。

## 常见误区

**把校验当装饰**

写了 `BaseModel` 但用 `dict` 直接传下去，等于没校验。
**校验的价值在于"不通过就不往下走"。**

**校验太严**

`title` 要求 `min_length=50`，模型写的短标题全被拒——
这不是模型的问题，是你的约束不合理。

**只校验不记录**

校验失败的字段分布是重要信号：
如果 80% 的失败都在 `year`，说明你的提示词或 schema 描述有问题，该改源头。
