import os

# 这行代码必须在导入任何 langchain 模块之前执行
os.environ["LANGSMITH_TRACING"] = "false"


# 旧的写法（会报错）
# from langchain.schema import HumanMessage, SystemMessage

# 新的写法（LangChain 1.0.3 正确导入）
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek
from dotenv import load_dotenv
# 加载 .env 文件中的环境变量
load_dotenv()

# 验证 API Key 是否加载成功
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise ValueError("请在 .env 文件中设置 DEEPSEEK_API_KEY")
print(f"✅ API Key 已加载: {api_key[:5]}...{api_key[-5:]}")

# 创建 DeepSeek 模型实例
llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0.7,
    max_tokens=1024,
)

# 构建消息
messages = [
    SystemMessage(content="你是一名精通Python的AI助手，擅长用简洁的代码解释概念。"),
    HumanMessage(content="请用Python写一个递归函数，计算斐波那契数列的第n项，并解释它的工作原理。")
]

# 调用模型
print("\n🤔 正在调用 DeepSeek 模型...")
response = llm.invoke(messages)

# 输出结果
print("\n" + "="*50)
print("📝 模型的回复：")
print("="*50)
print(response.content)

# 查看 token 使用情况
if hasattr(response, 'usage_metadata'):
    print("\n" + "="*50)
    print("📊 Token 使用统计：")
    print(response.usage_metadata)
