"""验收演练：故意制造各种失败，看系统是否"优雅处理"而不是崩溃。

运行方式（在 paper-agent 目录、venv 已激活）：
    python failure_drill.py

**这个脚本进 git，是项目的回归测试。**
每次改 agent/ 里的代码，先跑它确认没破坏已有行为 —— 9 项全过再继续。

（对比：W3_day1~day4_demo.py 是演示脚本，看过就懂，不进仓库。
 区别在于"看一遍就够"还是"每次都要跑"。）

------------------------------
它和前面几天的 demo 有什么不同
------------------------------
前四天的 demo 是"演示一个机制"，跑完你看输出就懂了。
今天是**验收**：每个检查项有明确的通过条件，跑完给一张 ✅/❌ 表。

这是"能演示"和"能验收"的区别：
    演示 → 人眼看
    验收 → 机器判定

W10 做评测时，就是把这种脚本扩展成几百个用例。
"""

import asyncio
import json

from agent.idempotency import IdempotencyStore, execute_idempotent
from agent.parse import extract_json, looks_truncated
from agent.retry import execute_with_retry
from agent.tools import execute_tool, tool


# ============================================================
# 演练用的工具（每个对应一种故障）
# ============================================================

_COUNT = {"drill_flaky": 0, "drill_denied": 0, "drill_ok": 0, "drill_counter": 0}


@tool
def drill_ok(city: str) -> str:
    """永远成功的工具（对照组）。

    Args:
        city: 城市名
    """
    _COUNT["drill_ok"] += 1
    return f"{city} 晴，18℃"


@tool
def drill_flaky(city: str) -> str:
    """前两次超时、第三次成功的工具（验证重试）。

    Args:
        city: 城市名
    """
    _COUNT["drill_flaky"] += 1
    if _COUNT["drill_flaky"] < 3:
        raise TimeoutError(f"第 {_COUNT['drill_flaky']} 次超时")
    return f"{city} 晴，18℃（第 {_COUNT['drill_flaky']} 次才成功）"


@tool
def drill_denied(doc_id: str) -> str:
    """永远权限不足的工具（验证不重试）。

    Args:
        doc_id: 文档 id
    """
    _COUNT["drill_denied"] += 1
    raise PermissionError("无权访问")


@tool
def drill_counter(action: str) -> str:
    """会累加计数的工具（验证幂等）。

    Args:
        action: 动作名
    """
    _COUNT["drill_counter"] += 1
    return f"执行了 {_COUNT['drill_counter']} 次"


def reset():
    for k in _COUNT:
        _COUNT[k] = 0


# ============================================================
# 检查项框架
# ============================================================

CHECKS = []


def check(name: str):
    """注册一个检查项。函数返回 (是否通过, 说明)。"""
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


# ============================================================
# 九个检查项
# ============================================================

@check("1. 正常调用：一次成功，不重试")
async def _():
    reset()
    out = await execute_with_retry("drill_ok", '{"city": "北京"}')
    ok = out.succeeded and out.attempts == 1 and _COUNT["drill_ok"] == 1
    return ok, f"attempts={out.attempts}, 实际执行={_COUNT['drill_ok']}"


@check("2. 瞬时失败：重试后被救回")
async def _():
    reset()
    out = await execute_with_retry(
        "drill_flaky", '{"city": "北京"}', base_delay=0.02
    )
    ok = out.succeeded and out.attempts == 3 and _COUNT["drill_flaky"] == 3
    return ok, f"attempts={out.attempts}, 退避={out.total_delay:.2f}s, 最终={out.kind}"


@check("3. 永久失败：只试 1 次，不浪费时间")
async def _():
    reset()
    out = await execute_with_retry(
        "drill_denied", '{"doc_id": "x"}', base_delay=0.02
    )
    ok = not out.succeeded and out.attempts == 1 and out.kind == "permanent"
    return ok, f"attempts={out.attempts}, kind={out.kind}"


@check("4. 参数非法：报 model_fixable，不进重试")
async def _():
    reset()
    out = await execute_with_retry("drill_ok", "{}", base_delay=0.02)
    ok = (
        not out.succeeded
        and out.attempts == 1
        and json.loads(out.raw).get("kind") == "model_fixable"
    )
    return ok, f"attempts={out.attempts}, error={json.loads(out.raw).get('error')}"


@check("5. 脏 JSON（代码块）：本地救回，不花 token")
async def _():
    dirty = '```json\n{"city": "北京"}\n```'
    out = await execute_with_retry("drill_ok", dirty, base_delay=0.02)
    ok = out.succeeded and out.attempts == 1
    return ok, f"成功={out.succeeded}, 尝试={out.attempts}（1 次说明没走重试）"


@check("6. 截断 JSON：如实报错并标记 truncated")
async def _():
    truncated_args = '{"city": "北'
    out = await execute_with_retry("drill_ok", truncated_args, base_delay=0.02)
    data = json.loads(out.raw)
    ok = (
        not out.succeeded
        and data.get("error") == "invalid_json"
        and data.get("truncated") is True
    )
    return ok, f"error={data.get('error')}, truncated={data.get('truncated')}"


@check("7. 幂等：同一个键重复调用只执行 1 次")
async def _():
    reset()
    store = IdempotencyStore()
    args = '{"action": "pay"}'
    for _ in range(3):
        await execute_idempotent("drill_counter", args, store=store)
    ok = _COUNT["drill_counter"] == 1
    return ok, f"实际执行={_COUNT['drill_counter']} 次（期望 1）"


@check("8. 模型输出脏文本：extract_json 能救回")
async def _():
    dirty = '好的，结果如下：\n```json\n{"title": "Attention", "year": 2017}\n```\n希望有帮助'
    r = extract_json(dirty)
    ok = r.ok and r.data == {"title": "Attention", "year": 2017}
    return ok, f"strategy={r.strategy}"


@check("9. 截断检测：能区分'格式脏'和'内容缺'")
async def _():
    a = looks_truncated('{"title": "A", "year": 20')
    b = looks_truncated('{"title": "A", "year": 2017}')
    ok = a is True and b is False
    return ok, f"截断样本={a}（期望 True）, 完整样本={b}（期望 False）"


# ============================================================
# 跑
# ============================================================

async def main():
    print("=" * 78)
    print("W3 Day5 验收演练")
    print("=" * 78)
    print()

    passed = 0
    failed = 0
    for name, fn in CHECKS:
        try:
            ok, detail = await fn()
        except Exception as e:      # 检查项自己崩了也算失败
            ok, detail = False, f"检查项抛异常: {type(e).__name__}: {e}"

        print(f"  {'✅' if ok else '❌'} {name}")
        print(f"     {detail}")
        if ok:
            passed += 1
        else:
            failed += 1

    print()
    print("=" * 78)
    print(f"结果：{passed} 通过 / {failed} 失败")
    print("=" * 78)

    if failed:
        print("  有检查项失败 —— 说明改动破坏了已有行为，先修再继续。")
    else:
        print("  全部通过。这张表就是 README 里「失败模式清单」的实测依据。")


if __name__ == "__main__":
    asyncio.run(main())
