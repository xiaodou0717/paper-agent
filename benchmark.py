"""W5 总账 —— 六种上下文策略的同条件对比。

运行方式（在 paper-agent 目录、venv 已激活）：
    python benchmark.py                    # 全部 6 组，约 27 万 token
    python benchmark.py --only baseline    # 只跑某一组（逗号分隔）
    python benchmark.py --budget 45000 --max-steps 12

**这个脚本进 git。** 理由和 failure_drill.py 一样：
它是 W5 硬卡点的产出（benchmark 对比表），W14 验收要能重新跑出同样的表。
一次性实验（W5_day1~day4_*.py）不进仓库，这个进。

--------------------------------------------------------------------------
它解决什么问题
--------------------------------------------------------------------------
前四天每天测一个策略，但**条件不一样**：

    Day1 滑窗   预算 60000，跑了两档
    Day2 摘要   预算 60000，跑了三次
    Day3 截断   预算 60000，limit=600
    Day4 缓存   预算 45000

数字摆在一张表里很好看，但**不可比**。而"哪个策略更好"这个问题，
只有在同一任务、同一预算、同一模型、同一 max_tokens 下才有意义。

所以今天全部重跑一遍。**重跑不是浪费，是让结论成立的前提。**

--------------------------------------------------------------------------
两个设计决定
--------------------------------------------------------------------------
【一】完成度不能再用"有没有答案"来判断

原来的判据是 len(final_answer) > 20 —— 只要模型说了句话就算完成。
太粗了：说了 10 篇和说了 1 篇，都是"完成"。

今天改成**标题覆盖率**：答案里出现了几篇文档的标题 / 总共几篇。
它可量化、可复现，而且能区分"敷衍"和"真的干了活"。

【二】结果落盘成 JSON

表格给人看，JSON 给下次的自己看。
W14 收口时要回答"这套东西到底带来了多少改善"，
靠的就是这些可以重新跑出来的数字，而不是笔记里抄来抄去的小数点。
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import config
from agent.loop import run_agent
from agent.tools_artifact import fetch_artifact          # 注册取回工具
from agent.tools_impl import list_docs, read_doc, search_docs   # 注册真实工具

# ============================================================
# 统一的实验条件
# ============================================================
# 为什么是 45000 而不是 60000：
#   前四天发现 45000 已经足够让基线跑到 8 步（42800），能看出差异；
#   再高就是纯烧钱，再低则所有配置都跑不完、没有区分度。
BUDGET = 45_000
MAX_STEPS = 12
MAX_TOKENS = 1500

QUESTION = """请完成一次知识库审计：
1. 先搜索，找出知识库里所有文档的清单
2. 然后逐篇阅读（不要跳过任何一篇），每读完一篇就输出一行：标题 | 一句话主旨
3. 全部读完后，给出汇总：这些文档大致分为哪几类

要求：每篇都必须单独调用 read_doc 阅读，不要凭记忆写。"""


# ============================================================
# 六组配置
# ============================================================
# 排序有讲究：先放"什么都不改"的基线，再放"改写历史"的两组，
# 最后放"不改写历史"的截断和两个组合。
#
# 这样看表的时候能直接读出结论：
#   改写历史的 → 缓存掉、token 省但等效成本不一定省
#   不改写历史的 → 缓存保住、token 也降
CONFIGS: list[dict] = [
    {
        "name": "baseline",
        "note": "基线：什么都不开",
        "kwargs": {},
    },
    {
        "name": "window2",
        "note": "滑窗 keep=2（改写历史）",
        "kwargs": {"window_keep_rounds": 2},
    },
    {
        "name": "summary2",
        "note": "摘要 keep=2（改写历史，有额外调用开销）",
        "kwargs": {"summary_keep_rounds": 2},
    },
    {
        "name": "trunc600",
        "note": "写入时截断 600 + 落盘（不改写历史）",
        "kwargs": {"tool_result_limit": 600},
    },
    {
        "name": "trunc600_window2",
        "note": "截断 + 滑窗（先截后丢）",
        "kwargs": {"tool_result_limit": 600, "window_keep_rounds": 2},
    },
    {
        "name": "trunc600_summary2",
        "note": "截断 + 摘要（先截后压）",
        "kwargs": {"tool_result_limit": 600, "summary_keep_rounds": 2},
    },
]


# ============================================================
# 完成度的尺子：标题覆盖率
# ============================================================

def build_title_index() -> dict[str, str]:
    """把语料库的"标题 → doc_id"建个索引。

    为什么要标题而不是 doc_id？
        模型写答案时用的是标题（如"文档切分策略"），不是文件名。
        用 doc_id 去匹配，永远匹配不上。
    """
    return {d["title"]: d["doc_id"] for d in list_docs()}


def _assistant_text(messages) -> str:
    """把历史里所有 assistant 说的话拼起来。

    ---------- 为什么不能只看 final_answer ----------
    第一版用 `final_answer` 算覆盖率，结果基线跑出 **0/11**。

    原因：预算耗尽时 loop 走的是 `finish("budget")`，**根本不产出 final_answer**。
    但模型其实已经在前面几步里写了"文档切分策略 | 讲了三种切法"这样的行。
    活干了，只是没"交卷"。

    所以完成度必须从**整段对话**里量，而不是只看最后一句。
    **指标要能在"任务失败"时仍然有区分度，否则它只会在成功案例里有意义。**
    """
    parts: list[str] = []
    for m in messages:
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", "")
        if role != "assistant":
            continue
        content = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
        if content:
            parts.append(str(content))
    return "\n".join(parts)


def _docs_read(messages) -> int:
    """数一数它到底读了几篇文档（read_doc 调用，按 doc_id 去重）。

    为什么还要这个指标？
        覆盖率量的是"说了几篇"，它可能吹牛（没读也写）。
        这个量的是"读了几篇"，比较难造假。
        两个一起看：读了 10 篇但只说了 3 篇 = 没时间总结（成本问题）；
        读了 3 篇却说了 10 篇 = 在编（质量问题）。
    """
    seen: set[str] = set()
    for m in messages:
        calls = m.get("tool_calls") if isinstance(m, dict) else getattr(m, "tool_calls", None)
        for tc in (calls or []):
            fn = tc.get("function") if isinstance(tc, dict) else getattr(tc, "function", None)
            name = fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", None)
            args = fn.get("arguments") if isinstance(fn, dict) else getattr(fn, "arguments", "")
            if name != "read_doc":
                continue
            try:
                doc_id = json.loads(args or "{}").get("doc_id")
            except (json.JSONDecodeError, AttributeError):
                doc_id = None
            if doc_id:
                seen.add(str(doc_id))
    return len(seen)


def coverage_of(answer: str, titles: list[str]) -> tuple[int, int]:
    """答案里覆盖了几篇文档的标题。返回 (命中数, 总数)。"""
    hits = sum(1 for t in titles if t and t in answer)
    return hits, len(titles)


@dataclass
class RunResult:
    """一组配置的运行结果。字段就是对比表要用的列。"""
    name: str
    note: str
    stop: str = ""
    steps: int = 0
    tool_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    summary_tokens: int = 0
    trimmed_chars: int = 0
    peak: int = 0
    docs_read: int = 0
    coverage_hits: int = 0
    coverage_total: int = 0
    elapsed: float = 0.0
    error: str = ""

    @property
    def hit_rate(self) -> float:
        return self.cached_tokens / self.prompt_tokens if self.prompt_tokens else 0.0

    @property
    def coverage(self) -> float:
        return self.coverage_hits / self.coverage_total if self.coverage_total else 0.0

    def effective_tokens(self, discount: float = 0.1) -> float:
        """等效计费 token（按假设折扣折算）。

        ⚠️ 套餐制不按 token 计费，这个数只用于**相对比较**。
        """
        return (self.prompt_tokens - self.cached_tokens) + self.cached_tokens * discount


# ============================================================
# 跑一组
# ============================================================

async def run_one(cfg: dict, titles: list[str], budget: int, max_steps: int) -> RunResult:
    r = RunResult(name=cfg["name"], note=cfg["note"])

    print(f"\n{'─' * 74}")
    print(f"▶ {cfg['name']}  —— {cfg['note']}")
    print(f"{'─' * 74}")

    try:
        out = await run_agent(
            QUESTION,
            max_steps=max_steps,
            max_tokens=MAX_TOKENS,
            token_budget=budget,
            verbose=False,
            **cfg["kwargs"],
        )
    except Exception as e:  # noqa: BLE001
        # 一组失败不能拖垮整个 benchmark —— 其余组的数据仍然有价值
        r.error = f"{type(e).__name__}: {e}"
        print(f"  ❌ 失败：{r.error}")
        return r

    transcript = _assistant_text(out.messages) + "\n" + out.final_answer
    hits, total = coverage_of(transcript, titles)

    r.stop = out.stop_reason
    r.steps = out.steps
    r.tool_calls = out.tool_calls
    r.prompt_tokens = out.prompt_tokens
    r.completion_tokens = out.completion_tokens
    r.total_tokens = out.total_tokens
    r.cached_tokens = out.cached_tokens
    r.summary_tokens = out.summary_tokens
    r.trimmed_chars = out.trimmed_chars
    r.peak = max((s.prompt_tokens for s in out.step_logs), default=0)
    r.docs_read = _docs_read(out.messages)
    r.coverage_hits = hits
    r.coverage_total = total
    r.elapsed = out.elapsed

    print(f"  stop={r.stop:<10} 步数={r.steps:<3} 工具={r.tool_calls:<3} "
          f"总tok={r.total_tokens:<7} 缓存={r.hit_rate*100:.0f}%  "
          f"读了={r.docs_read}/{total}  说了={hits}/{total}")

    return r


# ============================================================
# 输出
# ============================================================

def print_table(results: list[RunResult]) -> None:
    print("\n" + "=" * 96)
    print("W5 总账（同一任务 / 同一预算 / 同一模型）")
    print("=" * 96)
    print()
    header = (f"  {'配置':<20}{'stop':<9}{'步':<4}{'工具':<5}"
              f"{'总tok':<8}{'缓存':<7}{'命中率':<7}{'峰值':<7}{'读了':<7}{'说了':<7}"
              f"{'等效@0.1x'}")
    print(header)
    print("  " + "-" * 92)

    for r in results:
        if r.error:
            print(f"  {r.name:<20}❌ {r.error[:60]}")
            continue
        print(
            f"  {r.name:<20}{r.stop:<9}{r.steps:<4}{r.tool_calls:<5}"
            f"{r.total_tokens:<8}{r.cached_tokens:<7}{r.hit_rate*100:>5.0f}%  "
            f"{r.peak:<7}{r.docs_read}/{r.coverage_total:<5}{r.coverage_hits}/{r.coverage_total:<5}"
            f"{r.effective_tokens():.0f}"
        )

    print("\n  列说明：")
    print("    缓存    = 命中前缀缓存的 token 数")
    print("    命中率  = 缓存 / 输入 token")
    print("    读了    = 真正调用 read_doc 读过的篇数（去重）")
    print("    说了    = 助手消息里出现过标题的篇数（从整段对话量，不只看最后一句）")
    print("    等效    = 未命中 + 命中 × 0.1（**假设值**，套餐制不按 token 计费，只看相对大小）")


def save_json(results: list[RunResult], budget: int, max_steps: int) -> Path:
    out = Path(__file__).resolve().parent / "docs" / "benchmark-w5.json"
    out.parent.mkdir(exist_ok=True)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": config.MODEL,
        "budget": budget,
        "max_steps": max_steps,
        "max_tokens": MAX_TOKENS,
        "question": QUESTION,
        "results": [
            {**asdict(r), "hit_rate": r.hit_rate, "coverage": r.coverage,
             "effective_0_1x": r.effective_tokens()}
            for r in results
        ],
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8", newline="")
    return out


def print_conclusions(results: list[RunResult]) -> None:
    ok = [r for r in results if not r.error]
    if not ok:
        return

    base = next((r for r in ok if r.name == "baseline"), None)

    print("\n" + "=" * 96)
    print("结论（自动生成的，不是手写的）")
    print("=" * 96)

    if base:
        print(f"\n  基线：{base.total_tokens} tok，命中率 {base.hit_rate*100:.0f}%，"
              f"读了 {base.docs_read}/{base.coverage_total}，"
              f"说了 {base.coverage_hits}/{base.coverage_total}")
        print("\n  相对基线的变化：")
        print(f"  {'配置':<20}{'token':<10}{'命中率':<10}{'读了':<8}{'说了':<8}{'判断'}")
        print("  " + "-" * 84)
        for r in ok:
            if r is base:
                continue
            dt = (r.total_tokens - base.total_tokens) / base.total_tokens * 100
            dr = (r.hit_rate - base.hit_rate) * 100
            d_read = r.docs_read - base.docs_read
            d_say = r.coverage_hits - base.coverage_hits

            # 判断规则：省 token 且"读的篇数不掉"，才算划算。
            # 为什么用「读了几篇」而不是「说了几篇」？
            #   说的可以吹（没读也能编），读的要真调用工具。
            #   **衡量工作量，要挑那个不好造假的指标。**
            if dt < -5 and d_read >= 0:
                verdict = "✅ 划算（省 token，活没少干）"
            elif dt < -5 and d_read < 0:
                verdict = "⚠️ 省了钱但干少了活"
            elif dt > 5:
                verdict = "❌ 更贵"
            else:
                verdict = "≈ 持平"
            print(f"  {r.name:<20}{dt:>+6.1f}%   {dr:>+6.0f}pp   "
                  f"{d_read:>+4}    {d_say:>+4}    {verdict}")

    print("\n  ⚠️ 三条必须写进笔记的前提：")
    print("     1. 单次运行，模型有随机性 —— 结论是「线索」，不是「定论」")
    print("     2. 缓存是账号级的，组间会互相预热 —— 第 1 组可能略吃亏")
    print("     3. 等效 token 用的是假设折扣 0.1，套餐制不按此计费")


# ============================================================
# 入口
# ============================================================

async def main():
    parser = argparse.ArgumentParser(description="W5 六种策略同条件对比")
    parser.add_argument("--only", default="", help="只跑指定配置，逗号分隔，如 baseline,trunc600")
    parser.add_argument("--budget", type=int, default=BUDGET)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    args = parser.parse_args()

    # 只跑部分配置时，按用户给的顺序
    if args.only:
        wanted = [x.strip() for x in args.only.split(",") if x.strip()]
        cfgs = [c for c in CONFIGS if c["name"] in wanted]
        if not cfgs:
            print(f"没有匹配的配置。可选：{[c['name'] for c in CONFIGS]}")
            return
    else:
        cfgs = CONFIGS

    titles = [d["title"] for d in list_docs()]
    print("=" * 96)
    print(f"W5 总账：{len(cfgs)} 组配置")
    print(f"预算 {args.budget} / 最多 {args.max_steps} 步 / 模型 {config.MODEL}")
    print(f"语料 {len(titles)} 篇：{titles}")
    print("=" * 96)

    results: list[RunResult] = []
    for cfg in cfgs:
        results.append(await run_one(cfg, titles, args.budget, args.max_steps))

    print_table(results)
    print_conclusions(results)

    path = save_json(results, args.budget, args.max_steps)
    print(f"\n  结果已保存：{path}")


if __name__ == "__main__":
    asyncio.run(main())
