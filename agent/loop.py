"""Agent Loop —— 把 Day1 那张流程图变成能跑的代码。

这是整个项目的心脏。W1 Day4 实验 4 里写过一个"手写的 for 循环"，
那个版本能用，但有三件事没做：

    1. 不区分为什么停下来（正常结束？步数用完？输出被截断？）
    2. 不记录过程（跑完不知道花了多少 token、调用过哪些工具）
    3. 退出后拿不到结构化结果（只能 print，没法被别的代码用）

这个文件把那三件事补上，循环本身还是那五步。

五步结构（对照 notes/week2.md Day1 你画的那张图）：

    ① 调模型      —— 把 messages + tools 发过去
    ② 要不要工具   —— 看 finish_reason
    ③ 解析意图     —— tool_calls 里可能有多个，遍历
    ④ 执行工具     —— 交给 agent/tools.py 的 execute_tool
    ⑤ 回填结果     —— role="tool"，tool_call_id 必须对上

    然后回到 ①，直到 ② 判断"不要工具了"。

用法（最简单）：

    from agent.loop import run_agent
    result = await run_agent("北京今天天气怎么样？")
    print(result.final_answer)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import httpx2 as _httpx
from openai import AsyncOpenAI

import config
from agent.tools import execute_tool, get_tools_schema

# 默认最大步数。
# "步"= 一次模型调用 + 它要求的那批工具执行。
# 为什么是 6？见 W2 Day4（循环终止条件），今天先用个能跑的值。
DEFAULT_MAX_STEPS = 6

# 单次回复的 token 上限。
# 这个值一定不能太小：W1 Day5 实测过，推理模型会把额度耗在"思考"上，
# 给少了正式输出一个字都出不来（content 为空 + finish_reason=length）。
DEFAULT_MAX_TOKENS = 600


# ============================================================
# 结果：一次运行的完整产出
# ============================================================

@dataclass
class AgentResult:
    """一次 Agent 运行的全部结果。

    为什么要单独定义一个类，而不是直接返回最后的字符串？
    因为"最后那句话"只是结果的一小部分。真实项目里你还需要知道：

      - 它是正常结束的，还是被硬性截断的？（stop_reason）
      - 花了多少 token？（W4 优化上下文时要对比）
      - 调了几次工具、走了几步？（判断任务复杂度）
      - 完整对话历史是什么？（出错时复盘，W4 做压缩实验）

    这些信息如果只是 print 出来，就"没法被别的代码用"——
    比如你以后想跑 100 个问题做评测（W10），总不能靠人眼看 print。
    """
    final_answer: str                  # 最终给用户的回答
    stop_reason: str                   # 为什么停：见下面的取值说明
    steps: int                         # 实际走了几步（模型调了几次）
    tool_calls: int                    # 一共执行了多少次工具
    prompt_tokens: int = 0             # 累计输入 token（多轮相加）
    completion_tokens: int = 0         # 累计输出 token
    messages: list[Any] = field(default_factory=list)   # 完整对话历史

    @property
    def total_tokens(self) -> int:
        """总 token 数。"""
        return self.prompt_tokens + self.completion_tokens

    @property
    def succeeded(self) -> bool:
        """有没有正常完成。

        只有 stop_reason == "stop" 才算成功。
        另外几种都是"没干完"：
          max_steps —— 步数用完（可能是死循环）
          length    —— 单次输出被截断
          error     —— 出异常了
        """
        return self.stop_reason == "stop"

    def summary(self) -> str:
        """一行摘要，调试和写笔记时用。"""
        flag = "✅" if self.succeeded else "⚠️"
        return (
            f"{flag} stop={self.stop_reason} steps={self.steps} "
            f"tool_calls={self.tool_calls} tokens={self.total_tokens}"
        )


# ============================================================
# 客户端
# ============================================================

def make_client() -> AsyncOpenAI:
    """创建一个异步客户端。

    两个关键点（都是 W1 踩过的坑）：

    1. trust_env=False
       不让客户端读 HTTPS_PROXY / HTTP_PROXY 环境变量。
       火山是国内服务，请求被塞进代理会读超时，
       而且报的是 APITimeoutError，很容易误判成"网络不通"。

    2. 地址、模型、密钥全部来自 config.py
       本文件里不出现任何服务商特有的字符串。
    """
    config.check_config()
    return AsyncOpenAI(
        api_key=config.API_KEY,
        base_url=config.BASE_URL,
        http_client=_httpx.AsyncClient(trust_env=False, timeout=60.0),
    )


# ============================================================
# 主循环
# ============================================================

async def run_agent(
    user_message: str,
    *,
    client: AsyncOpenAI | None = None,
    model: str | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    system_prompt: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    verbose: bool = True,
) -> AgentResult:
    """跑一次完整的 Agent 任务。

    参数：
        user_message  用户的问题
        client        传进来就复用（测试时方便替换）；不传就自己建一个
        model         不传就用 config.MODEL
        max_steps     最多走几步（护栏，防无限循环）
        system_prompt 系统提示词；不传就不加 system 消息
        temperature   0 = 尽量稳定。工具调用阶段不要创造性
        max_tokens    单次回复上限
        verbose       是否打印每一步（学习阶段开着，跑评测时关掉）

    返回：
        AgentResult —— 里面有最终答案、停止原因、token 统计、完整历史
    """

    def log(*args):
        """verbose 开着才打印。写成内部函数是为了少写一堆 if。"""
        if verbose:
            print(*args)

    # ---------- 准备 ----------
    own_client = client is None
    if client is None:
        client = make_client()

    # tools 从注册表自动生成（W2 Day2 的产物）
    # 这里不需要知道有哪些工具 —— 用 @tool 装饰过的会自动进来
    tools = get_tools_schema()

    messages: list[Any] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})

    # 统计用的累加器
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tool_calls = 0

    log("=" * 60)
    log(f"问题：{user_message}")
    log(f"可用工具：{_tool_names()}")
    log("=" * 60)

    # 记录当前走到第几步。
    # 单独用变量存，是因为异常处理里也要用到它（for 循环的变量出了循环就没了）。
    current_step = 0

    try:
        # range(1, max_steps + 1) 让 step 从 1 开始，打印出来更符合人的习惯
        for step in range(1, max_steps + 1):
            current_step = step
            log(f"\n--- 第 {step} 步 ---")

            # ==================================================
            # ① 调模型
            # ==================================================
            # 每次都把【完整历史】发过去，不是只发上一轮。
            # 这是理解 Agent 成本的关键：第 5 步的请求里包含前 4 步的全部内容。
            response = await client.chat.completions.create(
                model=model or config.MODEL,
                messages=messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            choice = response.choices[0]
            msg = choice.message

            # token 是逐轮累加的，不是最后一轮的 usage
            # （每轮 response.usage 只是那一轮的）
            if response.usage:
                total_prompt_tokens += response.usage.prompt_tokens
                total_completion_tokens += response.usage.completion_tokens

            # ==================================================
            # ② 要不要工具：看 finish_reason，不要看 content 有没有值
            # ==================================================
            # 这是 W1 Day4 实测过的坑：模型要调工具时 content 是空字符串，
            # 只看 content 会误以为"模型没回答"。
            log(f"finish_reason: {choice.finish_reason}")

            # ---- 出口 2：输出被截断 ----
            # length 表示这一轮的回复没写完（撞到 max_tokens 或被服务端限制）。
            # 这时候拿到的内容是残缺的，不能当最终答案用 —— 必须明确报出来，
            # 否则你会拿到一个半截句子还以为任务完成了。
            if choice.finish_reason == "length":
                log("⚠️ 输出被截断（finish_reason=length），本轮内容不完整")
                return AgentResult(
                    final_answer=(msg.content or "").strip(),
                    stop_reason="length",
                    steps=step,
                    tool_calls=total_tool_calls,
                    prompt_tokens=total_prompt_tokens,
                    completion_tokens=total_completion_tokens,
                    messages=messages,
                )

            # ---- 出口 1：正常结束 ----
            # 不是 tool_calls，说明模型准备直接回答了（绝大多数情况是 stop）。
            if choice.finish_reason != "tool_calls":
                answer = (msg.content or "").strip()
                log(f"最终回答：{answer[:200]}")
                return AgentResult(
                    final_answer=answer,
                    stop_reason="stop",
                    steps=step,
                    tool_calls=total_tool_calls,
                    prompt_tokens=total_prompt_tokens,
                    completion_tokens=total_completion_tokens,
                    messages=messages,
                )

            # ==================================================
            # ③ 解析工具意图
            # ==================================================
            # 先把 assistant 这条消息【原样】放回历史。
            # 必须放，而且必须带 tool_calls —— 否则后面回填 tool 消息时，
            # 服务端找不到对应的调用，会直接拒绝请求。
            messages.append(msg)

            calls = msg.tool_calls or []
            log(f"模型要求调用 {len(calls)} 个工具")

            # ==================================================
            # ④⑤ 执行工具 + 回填（可能有多个，必须遍历）
            # ==================================================
            for tc in calls:
                log(f"  → {tc.function.name}({tc.function.arguments})")

                # 为什么用 asyncio.to_thread？
                # 因为 execute_tool 是同步函数（普通 def），直接在 async 里调用
                # 会阻塞整个事件循环 —— 这是 W1 Day2 实验 2 演示过的现象。
                # to_thread 把它丢到后台线程，事件循环就能继续干别的。
                #
                # 今天的假工具是瞬间返回的，看不出差别；
                # 但 Day5 接真实工具（可能读文件、发网络请求）时，这就很关键。
                result = await asyncio.to_thread(
                    execute_tool, tc.function.name, tc.function.arguments
                )
                total_tool_calls += 1

                log(f"  ← {result[:120]}")

                # 回填。两个必须：
                #   role = "tool"
                #   tool_call_id = 刚才那个调用的 id（一一对应，不能错）
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

            # ==================================================
            # 护栏：这一步做完，还能不能再来一轮？
            # ==================================================
            # 这里用 for 循环天然实现了：走完 max_steps 次就自然退出。
            # 循环正常结束 = 步数用完（出口 3），在下面处理。

        # ---- 出口 3：步数用完 ----
        # 走到这里说明模型一直在要工具，从没说过"我答完了"。
        # 可能是任务真的复杂，也可能是它卡在某个工具上反复重试（W3 会处理）。
        log(f"\n⚠️ 已跑满 {max_steps} 步仍未结束")
        return AgentResult(
            final_answer="",
            stop_reason="max_steps",
            steps=max_steps,
            tool_calls=total_tool_calls,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            messages=messages,
        )

    except Exception as e:  # noqa: BLE001
        # ---- 出口 4：异常中止 ----
        # 网络断了、服务端 500、参数不合法……任何没预料到的错误都到这里。
        # 为什么要捕获而不是让它往上抛？
        # 因为调用方拿到一个异常，无从判断"进行到哪一步了"；
        # 返回一个 stop_reason="error" 的结果，至少保留了 steps 和 messages。
        log(f"\n❌ 异常中止：{type(e).__name__}: {e}")
        return AgentResult(
            final_answer="",
            stop_reason="error",
            steps=current_step,
            tool_calls=total_tool_calls,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            messages=messages,
        )

    finally:
        # 自己建的 client 自己关，避免连接泄漏
        if own_client:
            await client.close()


def _tool_names() -> list[str]:
    """返回当前注册的工具名，只用于打印。"""
    from agent.tools import TOOL_REGISTRY
    return list(TOOL_REGISTRY)
