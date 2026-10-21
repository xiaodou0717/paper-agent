# ============================================================
# hello_llm.py —— 第一次真正调用大模型（冒烟测试）
#
# 文件名里的 "hello_llm" 是借 "hello world" 的意思：
# 先用最小代价确认「网络通、密钥对、模型名对」三件事，
# 再去写复杂的 Agent 逻辑。
#
# 这个脚本不只是"跑通就行"，它同时建立三个习惯：
#   1. 配置从 config.py 读，业务代码里不写死 URL / 模型名 / 密钥
#   2. 每次调用都打印 token 消耗（W4 做上下文优化时要拿它当基线）
#   3. 区分"思考内容"和"正式回答"（推理模型特有，很容易被忽略）
#
# 运行前提：同目录下有 .env（放密钥）和 config.py（放地址与模型名）。
# 运行方式：python hello_llm.py
# ============================================================


# ------------------------------------------------------------
# 导入部分
# ------------------------------------------------------------
# 导入 openai 官方 SDK 里的客户端类。
# 火山方舟兼容 OpenAI 协议，所以直接用官方 SDK 就行，
# 不需要额外装火山自己的包 —— 这是"协议兼容"带来的好处。
#
# 注意版本差异：openai 1.0 之后统一是 OpenAI(...) 这种写法。
# 如果你在网上看到 openai.ChatCompletion.create(...)，
# 那是 0.28 时代的老写法，现在会直接报错。
from openai import OpenAI

# 导入我们自己写的 config 模块。
#
# 为什么用 `import config` 而不是 `from config import MODEL`？
# 前者用的时候写 config.MODEL，一眼能看出这个值来自配置文件；
# 后者把名字直接搬进当前文件，读代码时看不出它从哪来。
# 项目变大以后，这个区别很重要。
import config

# 先做配置自检：密钥没读到、模型名还是占位符，就立刻退出。
#
# 为什么要在最前面做这件事？
# 因为不检查的话，你会在发请求后收到一个 401 或"模型不存在"，
# 然后开始怀疑网络、怀疑账号，实际上只是 .env 少了一行。
# 提前报错能省掉大量排查时间。
config.check_config()


# ------------------------------------------------------------
# 1. 创建客户端
# ------------------------------------------------------------
# client 是后面所有请求的入口，创建一次复用即可。
#
# 两个参数：
#   api_key  —— 证明"是谁在调用"，决定谁付钱
#   base_url —— 请求发到哪个服务器
#
# 这两行里没有任何硬编码的值，全部来自 config.py。
# 所以换服务商时，这个文件一个字都不用改。
client = OpenAI(
    api_key=config.API_KEY,
    base_url=config.BASE_URL,
)


# ------------------------------------------------------------
# 2. 发一条最简单的对话请求
# ------------------------------------------------------------
# chat.completions.create 是所有 OpenAI 兼容接口里最核心的方法。
# 拆开看这三段：
#   chat         —— 对话类接口（区别于补全 / 向量 / 图像）
#   completions  —— 生成回复
#   create       —— 创建一个请求
#
# 参数逐个说明：
response = client.chat.completions.create(
    # model：用哪个模型。值来自 config.py，且必须和服务商侧的命名完全一致。
    # Coding Plan 的 Model Name 不带日期后缀，按量 API 的 Model ID 带后缀，
    # 两者混用会报"模型匹配不正确"（Day1 踩过这个坑）。
    model=config.MODEL,

    # messages：对话历史。
    #
    # 关键理解：它叫"历史"不叫"问题"。哪怕只问一句话，也要包成列表，
    # 这是接口的硬性格式要求。
    #
    # 列表里每条消息是一个字典，role 有三种常见取值：
    #   "system"    —— 设定助手的身份、语气、行为准则。优先级最高，
    #                  模型会优先遵守它。Agent 里通常把"你是XX助手，
    #                  只能使用给定工具"这类规则放在这里。
    #   "user"      —— 用户说的话
    #   "assistant" —— 模型之前说过的话。多轮对话时，
    #                  你要把之前的问答按顺序都塞回来，
    #                  模型才知道上下文。这也是"上下文越聊越长、
    #                  token 越来越贵"的直接原因（W4 要处理的问题）。
    messages=[
        {"role": "system", "content": "你是一个简洁的助手，回答不超过两句话。"},
        {"role": "user", "content": "用一句话解释什么是 AI Agent。"},
    ],

    # temperature：随机性。
    #   0    → 尽量选最可能的答案，结果稳定、可复现
    #   0.7  → 有点变化，适合写文案
    #   1+   → 更发散
    #
    # 做 Agent 时，需要模型"稳定地选对工具"，所以工具调用阶段一般设 0。
    # 注意：设 0 也不保证每次完全一样（服务端还有别的随机因素），
    # 这是 W10 做评测时"同一输入允许多个正确答案"的原因之一。
    temperature=0,

    # max_tokens：输出上限，防止意外烧钱，也是生产代码的必备项。
    #
    # ⚠️ 坑：推理模型（如 deepseek-v4 系列）会先花 token "思考"，
    # 这部分算在 completion_tokens 里。max_tokens 设太小（比如 60），
    # 思考就把额度吃光了，正式的 content 会是空字符串，
    # 且 finish_reason 变成 "length"。所以别吝啬这个值。
    max_tokens=300,
)


# ------------------------------------------------------------
# 3. 取回答
# ------------------------------------------------------------
# 返回对象的嵌套结构：
#   response
#     └─ choices        ← 候选回复列表
#          └─ [0]       ← 第一个候选
#               └─ message
#                    └─ content   ← 真正的文字
#
# 为什么 choices 是列表？
# 因为接口支持一次生成多个候选答案（参数 n，默认 1 个）。
# 默认情况下取 [0] 就是唯一那个。
message = response.choices[0].message

print("=" * 50)
print("模型回答：")
print(message.content)

# reasoning_content：推理模型独有的字段，装的是模型"想的过程"。
# 它和 content（正式回答）是两个独立字段。
#
# 为什么用 getattr 而不是 message.reasoning_content？
# 因为普通模型根本没有这个属性，直接点会抛 AttributeError。
# getattr(对象, "属性名", 默认值) 的意思是：
#   有就取出来，没有就返回 None，不报错。
# 这种写法在处理"不同服务商返回字段不一致"时特别常用。
reasoning = getattr(message, "reasoning_content", None)
if reasoning:
    print("-" * 50)
    print("模型思考过程（reasoning_content）：")
    print(reasoning)


# ------------------------------------------------------------
# 4. 打印 token 消耗 —— 这是重点，别跳过
# ------------------------------------------------------------
# usage 里有三个数字：
#   prompt_tokens     —— 发出去多少（输入），包含 system + user + 历史
#   completion_tokens —— 收回来多少（输出），推理模型还含思考部分
#   total_tokens      —— 合计，通常按这个计费
#
# 为什么 Day1 就要看这个？
# W4 做上下文工程时，你要优化的核心指标就是 prompt_tokens。
# 优化有没有效果，全靠和今天这个数字对比。
# 现在不记，到时候就没有参照物，只能凭感觉说"好像快了点"。
usage = response.usage
print("=" * 50)
print("Token 消耗：")
print(f"  输入 prompt     : {usage.prompt_tokens}")
print(f"  输出 completion : {usage.completion_tokens}")
print(f"  合计 total      : {usage.total_tokens}")

# prompt_tokens_details 是更细的拆分，其中 cached_tokens 表示
# 有多少输入命中了「前缀缓存」。
#
# 前缀缓存的意思：如果你的请求开头部分和上次完全一样，
# 服务端可以直接复用上次的计算结果，这部分 token 更便宜、响应更快。
# 这是 W4 的核心优化手段之一 —— 把不变的内容（system prompt、
# 工具定义）固定在前面，把变化的内容放后面，才能提高命中率。
#
# 这里同样用 getattr 兜底，因为不是所有服务商都返回这个明细。
details = getattr(usage, "prompt_tokens_details", None)
if details:
    cached = getattr(details, "cached_tokens", 0) or 0
    print(f"  其中缓存命中     : {cached}")

# finish_reason：结束原因，三种常见取值：
#   "stop"       —— 模型正常说完了
#   "length"     —— 撞到 max_tokens 上限被截断（说明你的上限设小了，
#                   拿到的回答是残缺的，别当成完整结果用）
#   "tool_calls" —— 模型要求调用工具（W2-W3 的重点，那时这里会变成
#                   整个循环的继续条件）
print(f"  结束原因        : {response.choices[0].finish_reason}")
print("=" * 50)

# 跑完之后，把上面这三个 token 数字记到 notes/week1.md 里
# （注意：不是记到 progress.md，那里只记"做到哪了"，不记数据）。
# 它们是后面所有优化实验的起点。
