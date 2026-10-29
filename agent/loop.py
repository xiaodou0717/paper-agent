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

import time
from dataclasses import dataclass, field
from typing import Any

import httpx2 as _httpx
from openai import AsyncOpenAI

import config
from agent.retry import execute_with_retry
from agent.tools import get_tools_schema

# 默认最大步数。
# "步"= 一次模型调用 + 它要求的那批工具执行。
# 为什么是 6？见 W2 Day4（循环终止条件），今天先用个能跑的值。
DEFAULT_MAX_STEPS = 6

# 单次回复的 token 上限。
# 这个值一定不能太小：W1 Day5 实测过，推理模型会把额度耗在"思考"上，
# 给少了正式输出一个字都出不来（content 为空 + finish_reason=length）。
#
# ⚠️ 注意和下面的 DEFAULT_TOKEN_BUDGET 区分：
#     max_tokens    管【一次回复】最多写多少
#     token_budget  管【整个任务】最多花多少
# 两个都要有，防的是完全不同的东西。
DEFAULT_MAX_TOKENS = 600

# 整个任务的 token 预算（输入 + 输出合计）。
# 防的是"不报错但一直在烧钱"——步数没超、每轮都正常，但任务跑得越多花得越多。
DEFAULT_TOKEN_BUDGET = 50_000

# 整个任务的墙钟时间上限（秒）。
# 防的是"卡住"：工具在等网络、服务端在排队、循环空转。
# 步数和 token 都拦不住这种情况——因为根本没有产生新的花费，只是慢。
DEFAULT_MAX_SECONDS = 120.0


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
    retries: int = 0                   # 其中有多少次是重试（W3 Day2 新增）
    prompt_tokens: int = 0             # 累计输入 token（多轮相加）
    completion_tokens: int = 0         # 累计输出 token
    elapsed: float = 0.0               # 墙钟耗时（秒），从开始到退出的总时间
    messages: list[Any] = field(default_factory=list)   # 完整对话历史

    @property
    def total_tokens(self) -> int:
        """总 token 数。"""
        return self.prompt_tokens + self.completion_tokens

    @property
    def succeeded(self) -> bool:
        """有没有正常完成。

        只有 stop_reason == "stop" 才算成功。
        另外几种都是"没干完"，但【原因不同、修法也不同】：

          max_steps —— 步数用完（可能是死循环 / 任务确实复杂）
          budget    —— token 预算耗尽（防烧钱）
          timeout   —— 时间超限（防卡死）
          length    —— 单次输出被截断（max_tokens 给小了）
          error     —— 出异常了
        """
        return self.stop_reason == "stop"

    def summary(self) -> str:
        """一行摘要，调试和写笔记时用。"""
        flag = "✅" if self.succeeded else "⚠️"
        retry_part = f" retries={self.retries}" if self.retries else ""
        return (
            f"{flag} stop={self.stop_reason} steps={self.steps} "
            f"tool_calls={self.tool_calls}{retry_part} tokens={self.total_tokens} "
            f"elapsed={self.elapsed:.2f}s"
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
    token_budget: int | None = DEFAULT_TOKEN_BUDGET,
    max_seconds: float | None = DEFAULT_MAX_SECONDS,
    verbose: bool = True,
) -> AgentResult:
    """跑一次完整的 Agent 任务。

    参数：
        user_message  用户的问题
        client        传进来就复用（测试时方便替换）；不传就自己建一个
        model         不传就用 config.MODEL
        max_steps     最多走几步（护栏①，防无限循环）
        system_prompt 系统提示词；不传就不加 system 消息
        temperature   0 = 尽量稳定。工具调用阶段不要创造性
        max_tokens    单次回复上限（管"一次写多少"）
        token_budget  整个任务的 token 预算（护栏②，管"总共花多少"）
                      传 None 表示不限
        max_seconds   整个任务的时间上限，单位秒（护栏③，防卡死）
                      传 None 表示不限
        verbose       是否打印每一步（学习阶段开着，跑评测时关掉）

    返回：
        AgentResult —— 里面有最终答案、停止原因、token 统计、完整历史

    ⚠️ max_tokens 和 token_budget 是两个东西，别混：
        max_tokens=600, token_budget=50000
        → 每次回复最多写 600 个 token，但整个任务最多花 5 万个
        → 一个任务可能要调 10 次模型，每次都受 600 限制，加起来受 5 万限制
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
    total_retries = 0

    # 记录当前走到第几步。
    # 单独用变量存，是因为异常处理里也要用到它（for 循环的变量出了循环就没了）。
    current_step = 0

    # 开始计时。用 perf_counter 而不是 time.time()：
    # perf_counter 是专门用来测时长的，精度更高，而且不受系统时间被调整的影响
    #（time.time() 如果系统对时了会突然跳一下，测出的耗时可能是负数）。
    t_start = time.perf_counter()

    def finish(reason: str, answer: str = "") -> AgentResult:
        """构造退出结果。

        六个出口（stop / length / max_steps / budget / timeout / error）全走这里。
        为什么要统一：如果每个 return 自己拼一遍 AgentResult，
        很容易漏字段（比如加了 elapsed 之后忘了给某个出口补）。
        **同一个东西只在一个地方构造，改的时候不会漏。**
        """
        return AgentResult(
            final_answer=answer,
            stop_reason=reason,
            steps=current_step,
            tool_calls=total_tool_calls,
            retries=total_retries,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            elapsed=time.perf_counter() - t_start,
            messages=messages,
        )

    log("=" * 60)
    log(f"问题：{user_message}")
    log(f"可用工具：{_tool_names()}")
    log(f"护栏：max_steps={max_steps} token_budget={token_budget} max_seconds={max_seconds}")
    log("=" * 60)

    try:
        # range(1, max_steps + 1) 让 step 从 1 开始，打印出来更符合人的习惯
        for step in range(1, max_steps + 1):
            log(f"\n--- 第 {step} 步 ---")

            # ==================================================
            # 护栏：在【发起下一次请求之前】检查
            # ==================================================
            # 为什么放在这里，而不是收到响应之后？
            # 因为"发起请求"这个动作本身就要花钱、花时间。
            # 放在响应之后检查，等于"先花了再判断要不要花" —— 顺序反了。
            #
            # ---- 护栏①：时间预算 ----
            # 为什么最先检查？因为它最便宜，而且"卡死"是最急的情况：
            # 一旦卡住，后面所有检查都不会有机会执行（因为根本走不到下一轮）。
            elapsed_now = time.perf_counter() - t_start
            if max_seconds is not None and elapsed_now >= max_seconds:
                log(f"⏱️ 时间超限：已用 {elapsed_now:.1f}s ≥ 上限 {max_seconds}s，不再发起新请求")
                return finish("timeout")

            # ---- 护栏②：token 预算 ----
            # 注意这个方法有个【天生的缺陷】：不调用就不知道这次要花多少，
            # 所以只能"花掉之后才知道超了"。结果是最终会超出预算一点点。
            # 这不是 bug，是这类预算的固有性质 —— 想少超就留余量，
            # 比如 budget 设 45000 而心里按 50000 用（W3 会讲更细的做法）。
            spent = total_prompt_tokens + total_completion_tokens
            if token_budget is not None and spent >= token_budget:
                log(f"💰 token 预算耗尽：已用 {spent} ≥ 上限 {token_budget}，不再发起新请求")
                return finish("budget")

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

            # 这一轮模型调用成功了，才算真正走完"一步"。
            # 为什么不在循环开头就赋值？
            # 因为护栏可能在"还没开始第 N 步"时就把它拦住，
            # 那时候 steps 应该还是 N-1 —— 否则你看到的步数会比实际多 1。
            current_step = step

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
                return finish("length", (msg.content or "").strip())

            # ---- 出口 1：正常结束 ----
            # 只有 "stop" 才算真的说完了。
            #
            # ⚠️ 这里原来是 `if choice.finish_reason != "tool_calls":`
            #    那样写有个隐患：finish_reason 还有别的取值，
            #    比如 content_filter（输出被内容审核拦下）、部分服务商的 sensitive。
            #    用"不等于 tool_calls"来判断，会把它们统统当成"正常完成"，
            #    succeeded 变成 True —— 明明没答完，却报告成功。
            #    改成显式判断 "stop"，其余的单独处理。
            if choice.finish_reason == "stop":
                answer = (msg.content or "").strip()
                log(f"最终回答：{answer[:200]}")

                # 任务完成时顺带检查一次预算。
                # 为什么要检查？因为预算护栏只在"下一轮开始前"生效，
                # 如果任务在这一轮就结束了，超支会【静默发生】——
                # 实测见过"预算 2000、实花 2053、结果还是 stop"的情况。
                # 这里不改 stop_reason（任务确实完成了，钱也已经花了），
                # 但要留下痕迹：让你在日志里看得到，而不是月底看账单才发现。
                spent_now = total_prompt_tokens + total_completion_tokens
                if token_budget is not None and spent_now > token_budget:
                    log(
                        f"⚠️ 已完成，但超出预算 {spent_now - token_budget} tokens"
                        f"（实花 {spent_now} / 预算 {token_budget}）"
                    )

                return finish("stop", answer)

            # ---- 出口 1b：其他"说完了但不正常"的原因 ----
            # content_filter / sensitive 等。模型没被允许正常输出，
            # 必须如实反映在 stop_reason 里，不能算成功。
            # 用 `or "unknown"` 兜底：万一某个服务商返回 None，也不至于写出
            # stop_reason=None 这种更难看的东西。
            if choice.finish_reason != "tool_calls":
                reason = choice.finish_reason or "unknown"
                log(f"⚠️ 非正常结束：finish_reason={reason}")
                return finish(reason, (msg.content or "").strip())

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

                # 为什么用 execute_with_retry 而不是 execute_tool？
                # 因为 execute_tool 只执行一次；遇到 transient 失败（超时、限流）
                # 它虽然返回了 retryable=true，但没人去重试 —— 这个信息就白有了。
                #
                # execute_with_retry 做的事：执行 → 判断结果是不是 transient →
                # 是的话退避等待 → 再执行，最多 N 次。
                #
                # 注意：**这是 loop 里唯一为"重试"改动的一行。**
                # 重试的全部逻辑（退避、抖动、上限、总时长预算）
                # 都在 agent/retry.py 里，loop 不需要知道细节。
                #
                # 这也是分层的价值：以后要换重试策略（比如加熔断），只改 retry.py。
                outcome = await execute_with_retry(
                    tc.function.name, tc.function.arguments
                )
                total_tool_calls += 1
                if outcome.retried:
                    total_retries += outcome.attempts - 1
                    log(
                        f"  ↻ 重试 {outcome.attempts - 1} 次，"
                        f"累计退避 {outcome.total_delay:.2f}s"
                    )

                result = outcome.raw

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
        return finish("max_steps")

    except Exception as e:  # noqa: BLE001
        # ---- 出口 4：异常中止 ----
        # 网络断了、服务端 500、参数不合法……任何没预料到的错误都到这里。
        # 为什么要捕获而不是让它往上抛？
        # 因为调用方拿到一个异常，无从判断"进行到哪一步了"；
        # 返回一个 stop_reason="error" 的结果，至少保留了 steps 和 messages。
        log(f"\n❌ 异常中止：{type(e).__name__}: {e}")
        return finish("error")

    finally:
        # 自己建的 client 自己关，避免连接泄漏
        if own_client:
            await client.close()


def _tool_names() -> list[str]:
    """返回当前注册的工具名，只用于打印。"""
    from agent.tools import TOOL_REGISTRY
    return list(TOOL_REGISTRY)
