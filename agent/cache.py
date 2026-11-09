"""前缀缓存的可观测层 —— 把 usage 里那几个字段读明白。

--------------------------------------------------------------------------
前缀缓存是什么（一句话版）
--------------------------------------------------------------------------
服务商把"每次请求开头那段完全相同的内容"对应的计算结果缓存起来，
下次请求命中时，这部分 token **不再重新计算**，按折扣价计费。

所以它省的不是"上下文长度"，是"重复计算"。而 Agent 的 loop 恰好是
**教科书级的重复**：

    第 1 轮：system + user
    第 2 轮：system + user + assistant + tool          ← 前两段和第 1 轮一模一样
    第 3 轮：上面全部 + assistant + tool                ← 前三段和第 2 轮一模一样
    ...

只要历史**只追加、不修改**，每一轮都能命中前面全部内容。

--------------------------------------------------------------------------
今天实测到的三件事（都在 W5 Day4 的文档里）
--------------------------------------------------------------------------
1. 首次请求 cached_tokens=0，第二次起才命中（缓存要"先写后读"）
2. 命中量按 **128 token 的块**对齐
   （1084 token 的 prompt 只报 1024 命中 = 8 块）
3. **但这个字段在本服务商上不是"内容精确匹配"的证据**：
   把 prompt 的前一半换成完全不同的内容，命中数依然是 768，一点没变。

第 3 条很重要，它决定了我们应该怎么用这个指标：

    可以看：命中率随时间的变化趋势（同一任务内、只追加时）
    不能看：把它当成"改了内容就一定失效"的证明

**解释不了的指标，只能用来发现异常，不能用来下结论。**

--------------------------------------------------------------------------
为什么还要写这个模块
--------------------------------------------------------------------------
因为"看不见的东西没法优化"。哪怕指标不精确，只要每轮都记下来，
你就能回答三个问题：

    这一轮有多少是重复的？（cached_tokens）
    重复的部分占多少？（hit_rate）
    如果按折扣价算，等效花了多少？（effective_tokens，含假设）

这三个数在 W5 Day5 的组合实验里会用上。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# 实测观察到的缓存块大小（token）。
# ⚠️ 这不是文档承诺，是我们在这家服务商上量出来的：
#     1084 → 1024、875 → 768、427 → 384，都是 128 的整数倍。
#     换服务商要重新量。
CACHE_BLOCK = 128

# 假设的缓存折扣：命中部分按 10% 计价。
# ⚠️ 这是**假设**，不是报价。用户的 Coding Plan 是套餐制，不按 token 扣钱，
#    所以这里算出来的"等效 token"只用于**相对比较**，不能当账单。
DEFAULT_CACHE_DISCOUNT = 0.1


@dataclass
class CacheStats:
    """一次请求的缓存情况。"""
    prompt_tokens: int = 0        # 输入总 token
    cached_tokens: int = 0        # 其中命中缓存的

    @property
    def uncached_tokens(self) -> int:
        return max(0, self.prompt_tokens - self.cached_tokens)

    @property
    def hit_rate(self) -> float:
        """命中率。0.0 ~ 1.0。"""
        return self.cached_tokens / self.prompt_tokens if self.prompt_tokens else 0.0

    @property
    def blocks(self) -> int:
        """命中了几个块（按实测的 128 对齐解释）。"""
        return self.cached_tokens // CACHE_BLOCK

    def effective_tokens(self, discount: float = DEFAULT_CACHE_DISCOUNT) -> float:
        """等效计费 token = 未命中部分 + 命中部分 × 折扣。

        为什么叫"等效"？
            套餐不按 token 扣费，算不出真金白银。
            但用它做 A/B 对比是有效的：两套策略的"等效 token"谁低，谁更省。
        """
        return self.uncached_tokens + self.cached_tokens * discount

    def summary(self) -> str:
        return (
            f"缓存 {self.cached_tokens}/{self.prompt_tokens} tok"
            f"（{self.hit_rate * 100:.0f}%，{self.blocks} 块），"
            f"等效 {self.effective_tokens():.0f} tok"
        )


def parse_usage(usage: Any) -> CacheStats:
    """从 SDK 的 usage 对象里读出缓存字段。

    为什么要兼容两种写法？
        OpenAI 兼容格式：usage.prompt_tokens_details.cached_tokens
        DeepSeek 原生  ：usage.prompt_cache_hit_tokens
        换服务商时字段名会变（见《服务商切换速查表》），
        这里一次兼容，上层代码不用改。

    ⚠️ 任何一步取不到都返回 0，**绝不抛异常**。
       缓存统计是观测代码，观测代码不许拖垮主流程（W4 Day1 的教训）。
    """
    if usage is None:
        return CacheStats()

    prompt = getattr(usage, "prompt_tokens", 0) or 0

    # 路径一：OpenAI 兼容
    details = getattr(usage, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", 0) if details is not None else 0

    # 路径二：DeepSeek 原生字段
    if not cached:
        cached = getattr(usage, "prompt_cache_hit_tokens", 0) or 0

    cached = cached or 0
    # 防御：有些服务商会把 cached 报得比 prompt 还大（中间层统计口径不同）
    cached = min(cached, prompt)

    return CacheStats(prompt_tokens=prompt, cached_tokens=cached)


def explain_block_alignment(prompt_tokens: int, cached_tokens: int) -> str:
    """解释"为什么只命中了一部分"。

    这是最容易被误读的地方：
        看到 1084 的 prompt 只命中 1024，会以为"有 60 个 token 没缓存上"。
        实际是**按块对齐**：最后不满一块的部分不进入缓存。
    """
    remainder = prompt_tokens - cached_tokens
    if cached_tokens == 0:
        return "没有命中（首次请求，或前缀不足一个块）"
    if remainder <= 0:
        return "全部命中"
    return (
        f"命中 {cached_tokens}（{cached_tokens // CACHE_BLOCK} 个 {CACHE_BLOCK} token 块），"
        f"余下 {remainder} 不足一块 —— **正常现象，不是缓存没生效**"
    )
