"""工具注册器 —— 把普通 Python 函数变成模型能用的工具。

这个文件解决一个问题：
    模型要调工具，你必须给它两份东西：
      ① 说明（JSON Schema：叫什么、干什么、要什么参数）
      ② 实现（真正执行的函数）

    如果这两份东西分开维护，必然会不同步 —— 改了函数忘了改 schema，
    模型就会按旧的 schema 传参，然后你的函数报错。

    注册器的做法是：只写函数，schema 自动生成。

用法：

    from agent.tools import tool, get_tools_schema, execute_tool

    @tool
    def get_weather(city: str) -> str:
        '''查询指定城市的天气。

        Args:
            city: 城市名，如 北京
        '''
        return f"{city} 晴，18℃"

    get_tools_schema()                      # → 给模型的 tools 参数
    execute_tool("get_weather", '{"city":"北京"}')   # → 执行并返回结果
"""

from __future__ import annotations

import inspect
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, get_type_hints

from pydantic import BaseModel, ValidationError, create_model


# ============================================================
# 数据结构
# ============================================================

@dataclass
class ToolSpec:
    """一个工具的完整描述。

    四样东西缺一不可：
      name        —— 模型用它来指名道姓（必须和注册表键一致）
      description —— 模型靠这句话决定"要不要用这个工具"
      args_model  —— 参数校验用的 Pydantic 模型（也用来生成 schema）
      func        —— 真正执行的函数
    """
    name: str
    description: str
    args_model: type[BaseModel]
    func: Callable[..., Any]

    @property
    def json_schema(self) -> dict:
        """转成 OpenAI tools 格式里的 parameters 部分。

        注意：model_json_schema() 会把类的 docstring / Field 的 description
        自动带上，所以这里不用手写说明。
        """
        return self.args_model.model_json_schema()

    def to_openai_tool(self) -> dict:
        """转成 OpenAI 的 tools 列表项。

        结构固定成这样，模型才认：
            {"type": "function",
             "function": {"name": ..., "description": ..., "parameters": {...}}}
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.json_schema,
            },
        }


# 全局注册表：工具名 → ToolSpec
# 这就是 Day1 流程图里"查注册表"那一步查的东西。
TOOL_REGISTRY: dict[str, ToolSpec] = {}


# ============================================================
# docstring 解析
# ============================================================
# 为什么要解析 docstring？
# 因为模型只看得到 description，看不到你的代码。
# docstring 写得好不好，直接决定模型会不会在该用的时候用它、参数填得对不对。

_DOC_SECTION_RE = re.compile(r"^\s*(Args|Arguments|参数)\s*:\s*$", re.IGNORECASE)
_PARAM_LINE_RE = re.compile(r"^\s*(\w+)\s*[:\-]\s*(.+)$")


def _parse_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    """把 docstring 拆成（总体说明, {参数名: 参数说明}）。

    认这种格式（Google 风格，Python 社区最常见）：

        查询指定城市的天气。

        Args:
            city: 城市名，如 北京
            unit: 温度单位，celsius 或 fahrenheit

    Returns:
        ("查询指定城市的天气。", {"city": "城市名，如 北京", ...})
    """
    if not doc:
        return "", {}

    lines = doc.strip().splitlines()
    summary_lines: list[str] = []
    param_docs: dict[str, str] = {}
    in_args = False

    for line in lines:
        if _DOC_SECTION_RE.match(line):
            in_args = True
            continue
        if in_args:
            m = _PARAM_LINE_RE.match(line)
            if m:
                param_docs[m.group(1)] = m.group(2).strip()
            # Args 段里的续行（缩进更深的说明）忽略，教学版本从简
        else:
            summary_lines.append(line)

    # 总体说明只取第一段（到空行为止），太长反而干扰模型判断
    summary = " ".join(s.strip() for s in summary_lines if s.strip())
    return summary, param_docs


# ============================================================
# 从函数签名生成参数模型
# ============================================================

def _build_args_model(func: Callable, param_docs: dict[str, str]) -> type[BaseModel]:
    """根据函数的参数注解，动态造一个 Pydantic 模型。

    这是整个注册器最"魔法"的一步，但原理很简单：
        def f(a: str, b: int = 3)   →   class f_Args(BaseModel): a: str; b: int = 3

    然后 model_json_schema() 就能生成参数说明，validate 就能校验参数。

    用到两个工具：
        inspect.signature(func)  拿参数列表和默认值
        get_type_hints(func)     拿类型注解（会解析字符串注解）
    """
    hints = get_type_hints(func)
    sig = inspect.signature(func)

    fields: dict[str, tuple[Any, Any]] = {}
    for name, param in sig.parameters.items():
        # 参数的类型；没写注解就默认当字符串（比 Any 安全，schema 更明确）
        annotation = hints.get(name, str)

        # 有默认值就用默认值，没有就是必填（... 表示 required）
        default = ... if param.default is inspect.Parameter.empty else param.default

        # 有参数说明就包一层 Field(description=...)，这样会进 schema
        desc = param_docs.get(name)
        if desc:
            from pydantic import Field
            fields[name] = (annotation, Field(default=default, description=desc))
        else:
            fields[name] = (annotation, default)

    # create_model 是 Pydantic 的"运行时建类"功能
    return create_model(f"{func.__name__}_Args", **fields)   # type: ignore[call-overload]


# ============================================================
# 装饰器
# ============================================================

def tool(func: Callable) -> Callable:
    """把一个函数注册成工具。

    做三件事：
      1. 从签名生成参数模型
      2. 从 docstring 提取说明
      3. 存进全局注册表

    返回原函数，所以被装饰后仍然可以直接调用（测试时很方便）。
    """
    if func.__name__ in TOOL_REGISTRY:
        raise ValueError(f"工具名重复: {func.__name__}")

    summary, param_docs = _parse_docstring(inspect.getdoc(func))
    args_model = _build_args_model(func, param_docs)

    TOOL_REGISTRY[func.__name__] = ToolSpec(
        name=func.__name__,
        description=summary or f"调用 {func.__name__}",
        args_model=args_model,
        func=func,
    )
    return func


# ============================================================
# 给模型用的接口
# ============================================================

def get_tools_schema() -> list[dict]:
    """返回所有已注册工具的 schema，直接塞给 models 的 tools 参数。"""
    return [spec.to_openai_tool() for spec in TOOL_REGISTRY.values()]


def _error(code: str, message: str, **extra) -> str:
    """统一构造错误结果。

    为什么要统一格式：模型要能读懂"哪里错了、该怎么改"。
    一个 JSON 里带上错误类型和说明，比扔一段 Python 堆栈有用得多。

    这个设计对应 Day1 的结论：**工具失败也是结果，要回填，不是中断。**
    """
    payload = {"ok": False, "error": code, "message": message, **extra}
    return json.dumps(payload, ensure_ascii=False)


def execute_tool(name: str, arguments_json: str) -> str:
    """执行一个工具调用，永远返回字符串（成功或失败都返回）。

    为什么"永远返回字符串"而不是抛异常？
      因为调用方是 loop，它没有能力处理各种异常类型。
      把失败也变成一条普通结果回填给模型，让模型自己决定下一步 —— 这是 Agent 的做法。

    四道防线，对应 Day1 流程图里第 3、4 步的分支：
      ① 工具不存在
      ② 参数不是合法 JSON
      ③ 参数不符合 schema
      ④ 工具内部抛异常
    """
    spec = TOOL_REGISTRY.get(name)
    if spec is None:
        # ① 模型编了一个不存在的工具名 —— 这比想象中常见
        return _error(
            "unknown_tool",
            f"没有名为 {name} 的工具。可用工具：{list(TOOL_REGISTRY)}",
        )

    try:
        raw_args = json.loads(arguments_json) if arguments_json else {}
    except json.JSONDecodeError as e:
        # ② 模型给的 JSON 坏了（常见于输出被截断）
        return _error("invalid_json", f"参数不是合法 JSON: {e}")

    try:
        # ③ Pydantic 校验：类型、必填、取值范围全在这一步
        args = spec.args_model(**raw_args)
    except ValidationError as e:
        # 把校验错误翻译成人话（Day3 学过：直接扔 msg 效果差）
        problems = [
            f"{' → '.join(str(x) for x in err['loc'])}: {err['msg']}"
            for err in e.errors()
        ]
        return _error("invalid_args", "参数不符合要求", problems=problems)

    try:
        # ④ 真正执行。注意：这里捕获所有异常，不让它炸穿 loop
        result = spec.func(**args.model_dump())
    except Exception as e:  # noqa: BLE001  —— 故意捕获全部，工具出错是常态
        return _error("tool_error", f"{type(e).__name__}: {e}")

    # 成功也要包一层，让模型能区分"结果"和"错误"
    return json.dumps({"ok": True, "result": result}, ensure_ascii=False)


def list_tools() -> dict[str, str]:
    """列出所有工具及说明，调试用。"""
    return {spec.name: spec.description for spec in TOOL_REGISTRY.values()}
