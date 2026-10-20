# ============================================================
# hello_llm.py —— 第一次真正调用大模型
#
# 目标不是"跑通就行"，而是借这一次调用建立三个习惯：
#   1. 配置从 config 读，不在业务代码里写死
#   2. 每次调用都打印 token 消耗（后面所有优化都要拿它当基线）
#   3. 区分"思考内容"和"正式回答"（reasoning 模型特有）
# ============================================================

# 从 openai 官方 SDK 导入客户端。
# 火山方舟兼容 OpenAI 协议，所以直接用官方 SDK，不需要装火山自己的包。
# openai>=1.0 的写法是 OpenAI(...)，老教程里的 openai.ChatCompletion.create 已废弃。
from openai import OpenAI

# 从我们自己写的 config.py 里导入配置。
# 注意：这里刻意不出现任何 URL、模型名、密钥 —— 那是 config.py 的职责。
import config

# 先自检：配置没配好就直接退出，不发无意义的请求。
# 这一步能省掉大量"为什么报 401"的排查时间。
config.check_config()


# ------------------------------------------------------------
# 1. 创建客户端
# ------------------------------------------------------------
# base_url 决定请求发到哪家；api_key 决定谁付钱。
# 换服务商时这两行不用改，因为值来自 config.py。
client = OpenAI(
    api_key=config.API_KEY,
    base_url=config.BASE_URL,
)


# ------------------------------------------------------------
# 2. 发一条最简单的对话请求
# ------------------------------------------------------------
# messages 是"对话历史"，不是"当前问题"。
# 即使只有一轮，也必须传成列表 —— 这是所有 OpenAI 兼容接口的规定格式。
# 三个 role：
#   system    —— 设定身份和行为准则（优先级最高）
#   user      —— 用户说的话
#   assistant —— 模型之前说过的话（多轮对话时要把历史塞回来）
response = client.chat.completions.create(
    model=config.MODEL,
    messages=[
        {"role": "system", "content": "你是一个简洁的助手，回答不超过两句话。"},
        {"role": "user", "content": "用一句话解释什么是 AI Agent。"},
    ],
    # temperature 控制随机性：0 = 尽量确定，1 = 更有创造性。
    # 做 Agent 时工具调用阶段通常设 0，因为我们要的是稳定，不是惊喜。
    temperature=0,
    # 限制最大输出 token，防止意外烧钱。这是所有生产代码的必备项。
    max_tokens=300,
)


# ------------------------------------------------------------
# 3. 取回答
# ------------------------------------------------------------
# 返回结构是层层嵌套的：response -> choices[0] -> message -> content
# choices 是列表是因为可以一次要多个候选答案（n>1），默认只要 1 个。
message = response.choices[0].message

print("=" * 50)
print("模型回答：")
print(message.content)

# reasoning_content 是推理模型（如 deepseek-v4 系列）独有的字段，
# 装的是模型"想的过程"，和最终回答 content 是分开的。
# 普通模型没有这个字段，用 getattr 兜底避免 AttributeError。
reasoning = getattr(message, "reasoning_content", None)
if reasoning:
    print("-" * 50)
    print("模型思考过程（reasoning_content）：")
    print(reasoning)


# ------------------------------------------------------------
# 4. 打印 token 消耗 —— 这是重点，别跳过
# ------------------------------------------------------------
# 三个数字的含义：
#   prompt_tokens     —— 发出去多少（输入）
#   completion_tokens —— 收回来多少（输出）
#   total_tokens      —— 合计（计费依据）
#
# 为什么现在就要看它：
# W4 做上下文工程时，你要优化的就是 prompt_tokens；
# 优化有没有效果，全靠和今天的基线对比。现在不记，到时候就没参照物。
usage = response.usage
print("=" * 50)
print("Token 消耗：")
print(f"  输入 prompt     : {usage.prompt_tokens}")
print(f"  输出 completion : {usage.completion_tokens}")
print(f"  合计 total      : {usage.total_tokens}")

# 有些服务商会返回更细的拆分，能看出前缀缓存命中了多少。
# 前缀缓存是 W4 的核心优化手段（命中部分便宜很多），先留个印象。
details = getattr(usage, "prompt_tokens_details", None)
if details:
    cached = getattr(details, "cached_tokens", 0) or 0
    print(f"  其中缓存命中     : {cached}")

# 结束原因也很重要：
#   stop      = 正常说完
#   length    = 被 max_tokens 截断（说明你的上限设小了，回答是残缺的）
#   tool_calls = 模型要求调用工具（W3 会重点讲）
print(f"  结束原因        : {response.choices[0].finish_reason}")
print("=" * 50)
