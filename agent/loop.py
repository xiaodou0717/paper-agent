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
from agent.budget import plan_call, trim_tool_results
from agent.cache import parse_usage
from agent.compactor import compact_with_summary
from agent.tools import get_tools_schema
from agent.truncate import shrink_tool_result
from agent.window import sliding_window

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

def _msg_chars(m: Any) -> int:
    """安全地算一条消息的内容长度。

    为什么需要这个函数？
        messages 列表里混着两种东西：
          - dict：我们自己 append 的 system / user / tool 消息
          - SDK 对象：模型返回的 assistant 消息（W2 Day3 学过"原样放回历史"）

    对 dict 用 `.get()`，对对象用 `getattr`。一刀切会炸。

    ⚠️ 这个函数是踩坑后加的：一开始写了 `m.get("content")`，
       结果第 2 轮遇到 SDK 对象就抛 AttributeError，
       **把整个任务搞挂了**——观测代码的 bug 拖垮了主流程。
    """
    content = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
    return len(str(content)) if content else 0


@dataclass
class StepLog:
    """一轮调用的快照 —— W4 观察上下文用。

    为什么总 token 不够？
        它只告诉你"这个任务花了 8000"，不告诉你"从第几轮开始变贵的"。
        而上下文工程的全部优化，都建立在**看清每一轮的真实成本**上。

    三个关键数字，各有各的用处：

        prompt_tokens     这一轮实际发过去的 token = **当前上下文的大小**
        chars_in_context  消息内容的字符数（和 token 对照，能看出 token/字符比）
        n_messages        消息条数（判断"历史膨胀"还是"单条太长"）

    为什么 token 和字符数都要记？
        因为它们的比值会变。中文约 1.5 字符/token，英文约 4 字符/token。
        当一个工具返回大段代码时，比值会明显变化——
        这是判断"哪种内容最吃 token"的依据（W5 做截断策略时会用到）。
    """
    step: int                 # 第几轮
    prompt_tokens: int        # 这一轮的输入 token（服务商返回的）
    completion_tokens: int    # 这一轮的输出 token
    total_so_far: int         # 到这一轮为止累计花了多少
    n_messages: int           # 发过去时有多少条消息
    chars_in_context: int     # 这些消息的内容字符数
    finish_reason: str        # 这一轮的结束原因
    cached_tokens: int = 0    # 其中命中前缀缓存的部分（W5 Day4 新增）

    @property
    def tokens_per_char(self) -> float:
        """token / 字符 的比值。中文场景大约 0.6~0.7，英文大约 0.25。"""
        return self.prompt_tokens / self.chars_in_context if self.chars_in_context else 0.0


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
    cached_tokens: int = 0             # 其中命中前缀缓存的（W5 Day4 新增，已含在 prompt_tokens 里）
    summary_tokens: int = 0            # 其中摘要调用花掉的（W5 Day2 新增，已含在上面两项里）
    trimmed_chars: int = 0             # 工具结果被截掉多少字符（W5 Day3 新增）
    elapsed: float = 0.0               # 墙钟耗时（秒），从开始到退出的总时间
    messages: list[Any] = field(default_factory=list)   # 完整对话历史
    step_logs: list["StepLog"] = field(default_factory=list)   # 每轮快照（W4 Day1 新增）

    @property
    def total_tokens(self) -> int:
        """总 token 数。"""
        return self.prompt_tokens + self.completion_tokens

    @property
    def cache_hit_rate(self) -> float:
        """输入里有多少比例命中前缀缓存（W5 Day4）。

        为什么要单独算一个比率？
            绝对值没法和别的任务比（任务长短不同），比率可以。
            而且"命中率有没有随轮次上升"是判断"历史是否被改写"的最直接信号。
        """
        return self.cached_tokens / self.prompt_tokens if self.prompt_tokens else 0.0

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
    enable_compaction: bool = True,
    window_keep_rounds: int | None = None,
    summary_keep_rounds: int | None = None,
    tool_result_limit: int | None = None,
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
        window_keep_rounds   滑动窗口保留最近几轮（W5 Day1）。None = 不开
        summary_keep_rounds  摘要压缩保留最近几轮（W5 Day2）。None = 不开
                             和 window_keep_rounds **二选一**：两个都开，
                             滑窗会先把轮次丢掉，摘要就没东西可压了
        tool_result_limit    单条工具结果的字符上限（W5 Day3）。None = 不截
                             注意这是**写入时**截断：全文从没进过上下文。
                             和 budget 里那个"事后压缩"不是一回事（那个还在，
                             作为预算真撑不住时的最后一道保险）
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
    total_summary_tokens = 0           # 摘要调用的开销（W5 Day2）
    total_trimmed_chars = 0            # 工具结果截掉的字符数（W5 Day3）
    total_cached_tokens = 0            # 命中前缀缓存的 token（W5 Day4）
    step_logs: list[StepLog] = []      # 每轮快照，W4 观察上下文用

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
            cached_tokens=total_cached_tokens,
            summary_tokens=total_summary_tokens,
            trimmed_chars=total_trimmed_chars,
            elapsed=time.perf_counter() - t_start,
            messages=messages,
            step_logs=step_logs,
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

            # ---- 护栏②：token 预算（W4 Day3 升级：先压缩，压不动才停）----
            #
            # 原来的做法只有一个动作："花完了就停"。
            # 问题是它只能刹车、不能省油 —— 停下来的任务等于白花钱。
            #
            # 现在多了一步：**调用前先算这次要花多少，装不下就先压缩**。
            #
            #   旧：超预算 → 停
            #   新：超预算 → 压缩 → 再算一遍 → 还不够才停
            #
            # 为什么"再算一遍"很重要？
            #   不重算就是在猜压缩后够不够。可能压过头（白丢信息），
            #   也可能压不够（还是超）。**压缩的效果必须重新度量。**
            #
            # ⚠️ 压缩不是万能的：如果上下文里没有可压缩的东西
            #    （比如只有 system + 一句用户问题），压完还是超，该停就得停。
            spent = total_prompt_tokens + total_completion_tokens

            # ---- W5 Day1：滑动窗口（可选，在预算判断之前）----
            # 顺序很重要：**先丢整轮，再压单条**。
            #   丢整轮 = 扔掉已经用过、不会再看的早期结果（损失小）
            #   压单条 = 把长内容截短（损失大，会丢细节）
            #
            # 反过来做就浪费了：先把每条都压到 400 字符，
            # 再发现"其实整轮都可以不要"，那些压缩就白费了。
            if window_keep_rounds:
                wr = sliding_window(
                    messages, keep_head=2, keep_tail_rounds=window_keep_rounds
                )
                if wr.changed:
                    messages = wr.messages
                    log(f"  {wr.summary()}")
                elif not wr.ok:
                    # 校验没过就保持原样 —— 宁可多花 token，不发非法请求
                    log(f"  {wr.summary()}")

            # ---- W5 Day2：历史摘要（可选，在滑窗之后、预算判断之前）----
            #
            # 和滑窗的区别，一句话：
            #   滑窗 = 把早期轮次**删掉**（信息没了）
            #   摘要 = 把早期轮次**压成一段话**（骨架还在）
            #
            # 为什么放在预算判断之前？
            #   因为摘要会改变上下文大小，而预算判断要用"改变后的大小"来算。
            #   放在后面就成了"用旧尺寸量新衣服"。
            #
            # 为什么不放在滑窗之前？
            #   先丢的应该是"整段都没用的"（零成本、零损失），
            #   再压"有用但太占地方的"（要花钱、会丢细节）。
            #   **损失小的动作排在前面。**
            if summary_keep_rounds is not None:
                comp = await compact_with_summary(
                    client,
                    messages,
                    keep_head=2,
                    keep_tail_rounds=summary_keep_rounds,
                    model=model,
                )
                if comp.changed:
                    messages = comp.messages
                    # ⚠️ 摘要调用本身也花钱，必须记进总账。
                    #   不记的后果很隐蔽：你以为省了 2 万 token，
                    #   实际是"上下文省 2.5 万、摘要调用花 5 千"，净省 2 万 ——
                    #   但账面上只看到"总 token 变少了"，说不出钱去哪了。
                    total_prompt_tokens += comp.prompt_tokens
                    total_completion_tokens += comp.completion_tokens
                    total_summary_tokens += comp.prompt_tokens + comp.completion_tokens
                    log(f"  {comp.summary()}")
                elif not comp.ok:
                    # 失败也照跑：上下文没变，只是继续贵着
                    log(f"  {comp.summary()}")

            if token_budget is not None:
                plan = plan_call(messages, tools, token_budget, spent, max_tokens)

                if not plan.fits:
                    log(f"📊 {plan.summary()}")

                    if enable_compaction:
                        # ---------- 可能要压好几轮 ----------
                        # 为什么不能只压一次？
                        #   因为 trim_chars_needed 是按"刚好够"算的，
                        #   但压缩有粒度（一次压到 800 字符），可能压不够；
                        #   而且估算本身还有 ±10% 误差。
                        #
                        # 所以做法是：压 → 重算 → 还超就压得更狠（上限减半）。
                        # 最多 3 轮（800 → 400 → 200），再压不动就认了。
                        limit = 800
                        for round_no in range(1, 4):
                            messages, saved_chars = trim_tool_results(
                                messages, plan.trim_chars_needed,
                                per_result_limit=limit,
                            )
                            if not saved_chars:
                                log("  ✂️ 没有可压缩的工具结果了")
                                break

                            log(f"  ✂️ 第 {round_no} 轮压缩：省约 {saved_chars} 字符"
                                f"（单条上限 {limit}）")

                            # 压完立刻重算 —— 这一行是"压缩"能成立的关键
                            plan = plan_call(
                                messages, tools, token_budget, spent, max_tokens
                            )
                            if plan.fits:
                                log(f"  ✅ 压缩后装得下：预计 {plan.estimated_total}"
                                    f" ≤ 剩余 {plan.remaining}")
                                break

                            limit = max(100, limit // 2)

                    if not plan.fits:
                        log(
                            f"💰 压缩后仍装不下"
                            f"（预计 {plan.estimated_total} > 剩余 {plan.remaining}），停止"
                        )
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

            # ---------- 每轮快照（W4 Day1 新增）----------
            # 记下这一轮"发过去多大"。这是上下文工程全部优化的度量基础：
            #   不加这一步，你只知道总花费，不知道钱花在哪一轮、为什么贵。
            #
            # 注意 prompt_tokens 记的是【本轮】的，不是累计的 ——
            # 它等于"当前上下文的大小"，这才是我们要看的曲线。
            #
            # ⚠️ 整段用 try 包起来，理由很重要：
            #    观测代码出错，绝不能影响任务本身。
            #    （我第一版没包，结果取 content 时抛异常，整个任务被判 error）
            #    丢一条日志是小事，任务白跑是大事。
            try:
                # 前缀缓存命中（W5 Day4）。
                # 放在 try 里：解析 usage 是观测代码，出问题不能拖垮任务。
                cache_stats = parse_usage(response.usage)
                total_cached_tokens += cache_stats.cached_tokens

                step_logs.append(StepLog(
                    step=step,
                    prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                    completion_tokens=response.usage.completion_tokens if response.usage else 0,
                    total_so_far=total_prompt_tokens + total_completion_tokens,
                    n_messages=len(messages),
                    chars_in_context=sum(_msg_chars(m) for m in messages),
                    finish_reason=choice.finish_reason or "unknown",
                    cached_tokens=cache_stats.cached_tokens,
                ))
            except Exception as e:  # noqa: BLE001
                log(f"  ⚠️ 记录快照失败（不影响任务）：{type(e).__name__}: {e}")

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

                # ---- W5 Day3：写入时就截断 ----
                # 为什么放在这里，而不是等预算报警再回头压？
                #   事后压：全文已经进过上下文，峰值早就顶上去了
                #   写入时截：全文从没进过上下文，峰值根本不会高
                # Day2 的"峰值由最近几轮决定"就是在说这件事。
                #
                # ⚠️ 裁掉的内容没有消失：truncate.py 会先落盘，结果里留 artifact 句柄，
                #    模型需要时用 fetch_artifact 取回（信息不丢，只是不在上下文里）。
                if tool_result_limit:
                    result, cut = shrink_tool_result(result, tool_result_limit)
                    if cut:
                        total_trimmed_chars += cut
                        log(f"  ✂️ {tc.function.name}: {len(outcome.raw)} → {len(result)} 字符（省 {cut}）")

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
