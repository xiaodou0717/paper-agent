# ============================================================
# config.py —— 全局配置
# 所有"可能变化"的东西都集中在这里，业务代码只从这里读。
# 换服务商时：只改本文件 + .env，其他代码一行不动。
# ============================================================

import os

from dotenv import load_dotenv

# 读 .env，把里面的键值写进环境变量。
# 必须在下面 os.getenv() 之前执行。
load_dotenv()


# ------------------------------------------------------------
# 1. 密钥
# ------------------------------------------------------------
# 只在这里读一次。变量名 ARK_API_KEY 必须和 .env 里写的完全一致（区分大小写）。
# 如果读不到，这里会是 None，调用时会报 401。
API_KEY = os.getenv("ARK_API_KEY")


# ------------------------------------------------------------
# 2. 接口地址（换服务商时改这一行）
# ------------------------------------------------------------
# 火山方舟有两套完全不同的地址，用错就 404 或不扣套餐额度：
#
#   Coding Plan 套餐（你订阅的）→ https://ark.cn-beijing.volces.com/api/coding/v3
#   按量付费 API            → https://ark.cn-beijing.volces.com/api/v3
#
# 官方文档明确警告：不要用 /api/v3 去调 Coding Plan 的 Key，
# 那样不会抵扣套餐额度，还会产生额外费用。
BASE_URL = "https://ark.cn-beijing.volces.com/api/coding/v3"


# ------------------------------------------------------------
# 3. 模型名（换模型时改这一行）
# ------------------------------------------------------------
# ⚠️ 关键坑：Coding Plan 用的是「Model Name」，不是按量 API 的「Model ID」。
#
#    按量 API 的 ID  →  deepseek-v4.1-flash-261017   （带日期后缀）
#    Coding Plan 的 Name → deepseek-v4.1-flash       （不带日期后缀）
#
# 名字对不上就会报「模型匹配不正确」。两者是两套命名体系，不能混用。
#
# Coding Plan 官方支持的 Model Name（全小写，以控制台页面为准）：
#   deepseek-v4.1-flash / deepseek-v4-flash / deepseek-v4-pro
#   glm-5.3(glm-latest) / glm-5.3-flash
#   kimi-k2.8-preview / minimax-m3
#   doubao-seed-evolving / doubao-seed-2.1-pro / doubao-seed-2.1-lite / doubao-seed-2.0-mini
#   另有特殊值 ark-code-latest：填它之后在控制台页面切换模型，3-5 分钟生效。
MODEL = "deepseek-v4.1-flash"


# ------------------------------------------------------------
# 4. 启动自检：配置没配好时立刻报错，而不是等到调用时
# ------------------------------------------------------------
def check_config():
    """检查关键配置是否齐全，缺什么就明确告诉你缺什么。"""
    problems = []

    if not API_KEY:
        problems.append("API_KEY 为空：检查 .env 里有没有 ARK_API_KEY=xxx 这一行")

    if "在此填入" in MODEL:
        problems.append("MODEL 还是占位符：把 config.py 里的 MODEL 换成真实模型 ID")

    if problems:
        raise SystemExit(
            "配置有问题，先修好再运行：\n  - " + "\n  - ".join(problems)
        )


if __name__ == "__main__":
    # 直接运行 `python config.py` 时会走到这里，用于自检
    check_config()
    print("配置检查通过")
    print("  BASE_URL :", BASE_URL)
    print("  MODEL    :", MODEL)
    print("  API_KEY  :", (API_KEY[:6] + "..." + API_KEY[-4:]) if API_KEY else "未设置")
