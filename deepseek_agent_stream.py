# deepseek_agent_stream_fixed.py
import os
import sys
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.callbacks import BaseCallbackHandler
from langchain_deepseek import ChatDeepSeek
from langchain.agents import create_agent
from langchain.tools import tool
from typing import Dict, Any, List

# 加载环境变量
load_dotenv()

# ============ 1. 定义工具 ============
@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息。输入应为城市名称，如'北京'、'上海'。"""
    import time
    time.sleep(1)  # 模拟网络延迟
    weather_data = {
        "北京": "晴朗，25度，空气质量良好",
        "上海": "多云，22度，适合出行",
        "广州": "雷阵雨，28度，记得带伞",
        "深圳": "阴天，26度，湿度较大",
        "杭州": "小雨，20度，注意保暖"
    }
    return weather_data.get(city, f"{city}的天气信息暂时无法获取")

@tool
def get_current_time() -> str:
    """获取当前时间。当用户询问时间时使用此工具。"""
    import time
    from datetime import datetime
    time.sleep(0.5)
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# ============ 2. 创建自定义回调处理器 ============
class StreamCallbackHandler(BaseCallbackHandler):
    """自定义回调处理器，处理流式输出"""
    
    def __init__(self):
        super().__init__()
        self.tokens = []
        self.current_tool = None
        
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """当LLM生成新token时调用"""
        self.tokens.append(token)
        # 实时打印token
        print(token, end="", flush=True)
    
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        """工具开始调用时"""
        tool_name = serialized.get("name", "未知工具")
        self.current_tool = tool_name
        print(f"\n\n🔧 [正在调用工具: {tool_name}]", flush=True)
        print(f"📥 工具输入: {input_str}", flush=True)
    
    def on_tool_end(self, output: str, **kwargs) -> None:
        """工具调用完成时"""
        print(f"\n📤 工具输出: {output}", flush=True)
        print("\n🤖 最终回答: ", end="", flush=True)
        self.tokens = []  # 重置token列表
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        """Agent执行完成时"""
        print("\n" + "─"*60, flush=True)


# ============ 3. 使用回调的正确方式 ============
def stream_with_callbacks(user_input: str):
    """
    使用回调机制处理流式输出（修正版）
    """
    # 创建回调处理器实例
    stream_handler = StreamCallbackHandler()
    
    # 正确的方式：使用 callbacks 参数（注意是复数，且是列表）
    streaming_model = ChatDeepSeek(
        model="deepseek-chat",
        temperature=0.7,
        streaming=True,
        callbacks=[stream_handler],  # ✅ 正确的参数名和格式
    )
    
    # 创建Agent
    system_prompt = """你是一个乐于助人的AI助手，可以使用工具来回答用户的问题。
    当用户询问天气或时间时，请使用相应的工具获取信息，然后给出友好的回答。
    回答要简洁生动，可以适当使用表情符号。"""
    
    streaming_agent = create_agent(
        model=streaming_model,
        tools=[get_weather, get_current_time],
        system_prompt=system_prompt,
    )
    
    print(f"\n👤 用户: {user_input}")
    print("🤖 Agent: ", end="", flush=True)
    
    inputs = {"messages": [HumanMessage(content=user_input)]}
    result = streaming_agent.invoke(inputs)
    
    return result


# ============ 4. 更简单的方式：使用继承的回调 ============
class SimpleStreamHandler(BaseCallbackHandler):
    """最简单的流式输出处理器"""
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(token, end="", flush=True)

def simple_stream(user_input: str):
    """
    最简单的流式输出实现
    """
    # 创建带回调的模型
    model = ChatDeepSeek(
        model="deepseek-chat",
        temperature=0.7,
        streaming=True,
        callbacks=[SimpleStreamHandler()],
    )
    
    agent = create_agent(
        model=model,
        tools=[get_weather, get_current_time],
        system_prompt="你是一个有用的AI助手，可以查询天气和时间。",
    )
    
    print(f"\n👤 用户: {user_input}")
    print("🤖 ", end="", flush=True)
    
    inputs = {"messages": [HumanMessage(content=user_input)]}
    return agent.invoke(inputs)


# ============ 5. 方法四：使用 runnable 的 with_listeners ============
def stream_with_listeners(user_input: str):
    """
    使用 with_listeners 方法添加回调
    """
    # 基础模型
    base_model = ChatDeepSeek(
        model="deepseek-chat",
        temperature=0.7,
        streaming=True,
    )
    
    # 创建监听器函数
    def on_token(token: str):
        print(token, end="", flush=True)
    
    def on_tool_start(tool_name: str, input_str: str):
        print(f"\n\n🔧 调用工具 [{tool_name}]...", flush=True)
    
    # 使用 with_listeners 添加回调
    from langchain_core.runnables import RunnableLambda
    
    # 创建Agent
    agent = create_agent(
        model=base_model,
        tools=[get_weather, get_current_time],
        system_prompt="你是一个有用的AI助手。",
    )
    
    print(f"\n👤 用户: {user_input}")
    print("🤖 ", end="", flush=True)
    
    inputs = {"messages": [HumanMessage(content=user_input)]}
    
    # 在invoke时临时添加回调
    result = agent.invoke(
        inputs,
        config={"callbacks": [SimpleStreamHandler()]}  # ✅ 在调用时添加回调
    )
    
    return result


# ============ 6. 测试所有方法 ============
if __name__ == "__main__":
    print("🌟" + "="*58 + "🌟")
    print("           DeepSeek Agent 流式输出演示（修正版）")
    print("🌟" + "="*58 + "🌟\n")
    
    # 测试方法一：使用回调的正确方式
    print("📌 方法一：自定义回调处理器")
    print("="*60)
    stream_with_callbacks("北京今天天气怎么样？")
    
    input("\n\n⏎ 按回车测试下一个方法...")
    
    # 测试方法二：简单流式输出
    print("\n\n📌 方法二：简单流式输出")
    print("="*60)
    simple_stream("现在几点了？")
    
    input("\n\n⏎ 按回车测试下一个方法...")
    
    # 测试方法三：带工具调用的复杂问题
    print("\n\n📌 方法三：复杂问题测试")
    print("="*60)
    stream_with_callbacks("上海天气适合旅游吗？顺便告诉我现在时间")
    
    print("\n\n✨ 所有测试完成！")
