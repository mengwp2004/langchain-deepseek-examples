# deepseek_agent_demo.py
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek
from langchain.agents import create_agent
from langchain.tools import tool
import langchain
from langgraph.checkpoint.memory import InMemorySaver

# 加载环境变量
load_dotenv()
#langchain.debug = True  # 开启调试模式
# 验证API Key
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise ValueError("请在 .env 文件中设置 DEEPSEEK_API_KEY")

# ============ 1. 定义工具 ============
@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息。输入应为城市名称，如'北京'、'上海'。"""
    # 这里模拟天气查询，实际应用中可以调用真实天气API
    weather_data = {
        "北京": "晴朗，25度，空气质量良好",
        "上海": "多云，22度，适合出行",
        "广州": "雷阵雨，28度，记得带伞",
        "深圳": "阴天，26度，湿度较大",
        "杭州": "小雨，20度，注意保暖"
    }
    return weather_data.get(city, f"{city}的天气信息暂时无法获取，但据说是好天气！")

@tool
def get_current_time() -> str:
    """获取当前时间。当用户询问时间时使用此工具。"""
    from datetime import datetime
    now = datetime.now()
    return f"当前时间是 {now.strftime('%Y-%m-%d %H:%M:%S')}"

@tool
def search_web(query: str) -> str:
    """搜索网络获取实时信息。当需要最新新闻、实时数据时使用。"""
    # 这里可以集成Tavily、DuckDuckGo等搜索API
    return f"这是关于'{query}'的搜索结果..."

# ============ 2. 配置模型 ============

model = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0.7,
    max_tokens=1024,
    # 注意：api_key 会自动从环境变量读取，这里不需要显式传入
)

# ============ 3. 定义系统提示词 ============
system_prompt = """你是一个乐于助人的AI助手，可以使用工具来回答用户的问题。

你有以下工具可用：
- get_weather：查询天气信息
- get_current_time：获取当前时间

当用户询问天气或时间时，请使用相应的工具获取信息，然后根据工具返回的结果给出友好的回答。
"""

# 创建记忆存储器
checkpointer = InMemorySaver()

# ============ 4. 创建Agent ============

agent = create_agent(
    model=model,
    tools=[get_weather, get_current_time,search_web],
    system_prompt=system_prompt,
    checkpointer=checkpointer  # 添加记忆功能
)

print("✅ Agent创建成功！")
print("="*60)

# ============ 5. 运行测试 ============


# 使用相同的thread_id来维持对话上下文
config = {"configurable": {"thread_id": "user_123"}}
response1 = agent.invoke(
    {"messages": [HumanMessage(content="我叫小明")]},
    config=config
)
response2 = agent.invoke(
    {"messages": [HumanMessage(content="我叫什么名字？")]},
    config=config  # 同一个thread_id，Agent会记得之前的对话
)

messages = response2["messages"]
# 最后一条消息通常是最终的AI回答
final_answer = messages[-1].content
    
print(f"🤖 Agent: {final_answer}")
print("-"*60)