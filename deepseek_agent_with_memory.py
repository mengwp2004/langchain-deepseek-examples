# deepseek_agent_with_memory.py
import os
import sys
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.callbacks import BaseCallbackHandler
from langchain_deepseek import ChatDeepSeek
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

# 加载环境变量
load_dotenv()

# ============ 1. 定义工具 ============
@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息。输入应为城市名称，如'北京'、'上海'。"""
    time.sleep(1)  # 模拟网络延迟
    weather_data = {
        "北京": "晴朗🌞，25度，空气质量良好",
        "上海": "多云☁️，22度，适合出行",
        "广州": "雷阵雨⛈️，28度，记得带伞",
        "深圳": "阴天🌥️，26度，湿度较大",
        "杭州": "小雨🌧️，20度，注意保暖",
        "成都": "阴天☁️，21度，适合吃火锅",
        "西安": "晴天🌞，23度，历史古迹很多"
    }
    return weather_data.get(city, f"{city}的天气：🌤️ 晴转多云，22度")

@tool
def get_current_time() -> str:
    """获取当前时间。当用户询问时间时使用此工具。"""
    from datetime import datetime
    time.sleep(0.5)
    now = datetime.now()
    return f"🕐 {now.strftime('%Y-%m-%d %H:%M:%S')}"

@tool
def calculate(expression: str) -> str:
    """计算数学表达式。输入应为数学表达式，如'23 * 45'或'100/4'。"""
    import re
    time.sleep(0.8)
    try:
        # 安全过滤：只允许数字和基本运算符
        if re.match(r'^[\d\s\+\-\*\/\(\)\.]+$', expression):
            result = eval(expression)
            return f"🧮 计算结果: {expression} = {result}"
        else:
            return "❌ 表达式包含非法字符，只能使用数字和+-*/()"
    except Exception as e:
        return f"❌ 计算错误: {e}"

@tool
def get_user_name(name: str) -> str:
    """记录用户的名字。当用户自我介绍时使用此工具。"""
    return f"👋 你好，{name}！很高兴认识你！"

@tool
def search_web(query: str) -> str:
    """搜索网络获取实时信息。当需要最新新闻、实时数据时使用。"""
    # 这里可以集成Tavily、DuckDuckGo等搜索API
    return f"这是关于'{query}'的搜索结果..."

# ============ 2. 自定义回调处理器（带记忆感知） ============
class MemoryAwareStreamHandler(BaseCallbackHandler):
    """
    支持记忆的流式回调处理器
    可以显示思考过程、工具调用和最终输出
    """
    
    def __init__(self, show_thinking: bool = True):
        super().__init__()
        self.tokens = []
        self.current_tool = None
        self.show_thinking = show_thinking
        self.in_final_response = False
        self.thinking_buffer = []
        
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """LLM开始生成时"""
        if self.show_thinking:
            print("\n🤔 [思考中...]", end="", flush=True)
        self.in_final_response = False
        
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """当LLM生成新token时调用"""
        self.tokens.append(token)
        
        # 如果是最终回答部分，直接输出
        if self.in_final_response or len(self.tokens) > 10:  # 简单启发：超过10个token认为是最终回答
            self.in_final_response = True
            print(token, end="", flush=True)
        elif self.show_thinking:
            # 显示思考过程（但不输出）
            self.thinking_buffer.append(token)
    
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        """工具开始调用时"""
        tool_name = serialized.get("name", "未知工具")
        self.current_tool = tool_name
        
        print(f"\n\n🔧 [正在调用工具: {tool_name}]", flush=True)
        print(f"📥 工具输入: {input_str}", flush=True)
        
    def on_tool_end(self, output: str, **kwargs) -> None:
        """工具调用完成时"""
        print(f"📤 工具输出: {output}", flush=True)
        print("\n🤖 [生成回答中...] ", end="", flush=True)
        self.tokens = []  # 重置token列表
        self.in_final_response = True
        
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        """Agent执行完成时"""
        print("\n" + "─"*60, flush=True)


# ============ 3. 带记忆的对话管理器 ============
class ChatSessionManager:
    """
    对话会话管理器
    管理多个对话会话，每个会话有自己的记忆
    """
    
    def __init__(self):
        # 使用 InMemorySaver 保存对话状态
        self.checkpointer = InMemorySaver()
        self.sessions: Dict[str, Dict] = {}
        self.default_system_prompt = """你是一个友好的AI助手，名叫"小深"。你可以使用工具来帮助用户。

你有以下工具可用：
- get_weather：查询天气
- get_current_time：查询时间  
- calculate：计算数学表达式
- get_user_name：记录用户名字

使用工具时，先思考需要哪个工具，然后调用它。
最后根据工具结果给用户一个生动、友好的回答。

记住用户的偏好和之前说过的话，让对话更自然。"""
        
    def create_session(self, session_id: str, system_prompt: Optional[str] = None) -> Dict:
        """创建新的对话会话"""
        if session_id in self.sessions:
            return self.sessions[session_id]
        
        # 创建流式处理器
        handler = MemoryAwareStreamHandler(show_thinking=True)
        
        # 创建带回调的模型
        model = ChatDeepSeek(
            model="deepseek-chat",
            temperature=0.7,
            streaming=True,
            callbacks=[handler],
            max_tokens=2048,
        )
        
        # 创建 Agent，启用记忆
        agent = create_agent(
            model=model,
            tools=[get_weather, get_current_time, calculate, get_user_name],
            system_prompt=system_prompt or self.default_system_prompt,
            checkpointer=self.checkpointer,  # 启用记忆
        )
        
        # 保存会话信息
        session = {
            "id": session_id,
            "agent": agent,
            "handler": handler,
            "created_at": datetime.now(),
            "message_count": 0,
            "config": {"configurable": {"thread_id": session_id}},  # 用于记忆的配置
        }
        
        self.sessions[session_id] = session
        return session
    
    def chat(self, session_id: str, message: str) -> str:
        """
        发送消息并接收流式响应
        """
        if session_id not in self.sessions:
            self.create_session(session_id)
        
        session = self.sessions[session_id]
        session["message_count"] += 1
        
        print(f"\n👤 [用户]: {message}")
        print("🤖 [小深]: ", end="", flush=True)
        
        # 准备输入
        inputs = {"messages": [HumanMessage(content=message)]}
        
        # 调用 Agent（自动记忆历史）
        result = session["agent"].invoke(
            inputs,
            config=session["config"]  # 使用相同的 thread_id 保持对话连贯
        )
        
        # 获取最后的回答
        final_answer = result["messages"][-1].content
        return final_answer
    
    def get_conversation_history(self, session_id: str) -> List[Dict]:
        """获取对话历史"""
        if session_id not in self.sessions:
            return []
        
        # 从 checkpointer 获取历史状态
        config = self.sessions[session_id]["config"]
        try:
            # 获取检查点列表
            states = []
            for state in self.checkpointer.list(config):
                if "messages" in state:
                    for msg in state["messages"]:
                        if isinstance(msg, (HumanMessage, AIMessage, SystemMessage)):
                            states.append({
                                "role": msg.__class__.__name__.replace("Message", ""),
                                "content": msg.content,
                                "type": type(msg).__name__
                            })
            return states
        except Exception as e:
            print(f"获取历史失败: {e}")
            return []
    
    def clear_session(self, session_id: str):
        """清除会话"""
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    def list_sessions(self) -> List[str]:
        """列出所有会话"""
        return list(self.sessions.keys())


# ============ 4. 交互式命令行界面 ============
class InteractiveChatCLI:
    """
    交互式命令行聊天界面
    """
    
    def __init__(self):
        self.manager = ChatSessionManager()
        self.current_session = None
        self.username = None
        
    def print_header(self):
        """打印标题"""
        os.system('clear' if os.name == 'posix' else 'cls')
        print("🌟" + "="*58 + "🌟")
        print("           DeepSeek Agent 智能对话系统（带记忆功能）")
        print("🌟" + "="*58 + "🌟")
        print("\n📝 命令说明：")
        print("  /new    - 开始新对话")
        print("  /list   - 列出所有会话")
        print("  /switch - 切换会话")
        print("  /clear  - 清除当前会话")
        print("  /history- 查看当前会话历史")
        print("  /help   - 显示帮助")
        print("  /quit   - 退出程序")
        print("-"*60)
    
    def print_help(self):
        """打印帮助信息"""
        print("\n📚 可用命令：")
        print("  /new    - 开始新对话")
        print("  /list   - 列出所有会话")
        print("  /switch - 切换会话")
        print("  /clear  - 清除当前会话")
        print("  /history- 查看当前会话历史")
        print("  /help   - 显示本帮助")
        print("  /quit   - 退出程序")
        print("\n💡 提示：")
        print("  - 直接输入问题开始对话")
        print("  - Agent 会自动调用工具回答问题")
        print("  - 所有对话都会被记忆")
        print("  - 不同会话之间记忆独立")
    
    def handle_command(self, cmd: str) -> bool:
        """处理命令，返回是否继续运行"""
        cmd = cmd.lower().strip()
        
        if cmd == "/quit":
            print("👋 再见！期待再次与你对话！")
            return False
        
        elif cmd == "/new":
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.current_session = session_id
            self.manager.create_session(session_id)
            print(f"✅ 创建新会话: {session_id}")
            if self.username:
                # 如果有用户名，自动打招呼
                self.manager.chat(session_id, f"我叫{self.username}")
        
        elif cmd == "/list":
            sessions = self.manager.list_sessions()
            if sessions:
                print("\n📋 现有会话：")
                for i, sid in enumerate(sessions, 1):
                    marker = "👉 " if sid == self.current_session else "   "
                    print(f"  {marker}{i}. {sid}")
            else:
                print("📭 暂无会话")
        
        elif cmd == "/switch":
            sessions = self.manager.list_sessions()
            if not sessions:
                print("❌ 没有可切换的会话")
                return True
            
            print("\n📋 选择要切换的会话：")
            for i, sid in enumerate(sessions, 1):
                print(f"  {i}. {sid}")
            
            try:
                choice = input("\n请输入序号: ").strip()
                idx = int(choice) - 1
                if 0 <= idx < len(sessions):
                    self.current_session = sessions[idx]
                    print(f"✅ 已切换到会话: {self.current_session}")
                else:
                    print("❌ 无效序号")
            except ValueError:
                print("❌ 请输入数字")
        
        elif cmd == "/clear":
            if self.current_session:
                self.manager.clear_session(self.current_session)
                self.current_session = None
                print("✅ 当前会话已清除")
            else:
                print("❌ 没有活动的会话")
        
        elif cmd == "/history":
            if self.current_session:
                history = self.manager.get_conversation_history(self.current_session)
                if history:
                    print("\n📜 对话历史：")
                    print("="*60)
                    for msg in history:
                        role = msg["role"]
                        content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
                        print(f"[{role}]: {content}")
                    print("="*60)
                else:
                    print("📭 暂无对话历史")
            else:
                print("❌ 没有活动的会话")
        
        elif cmd == "/help":
            self.print_help()
        
        else:
            print(f"❓ 未知命令: {cmd}")
            print("输入 /help 查看可用命令")
        
        return True
    
    def run(self):
        """运行交互式界面"""
        self.print_header()
        
        # 创建默认会话
        self.current_session = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.manager.create_session(self.current_session)
        print(f"\n✅ 已创建默认会话: {self.current_session}")
        
        while True:
            try:
                # 获取用户输入
                if self.current_session:
                    prompt_prefix = f"[{self.current_session[:8]}...] "
                else:
                    prompt_prefix = "[无会话] "
                
                user_input = input(f"\n{prompt_prefix}💬 输入消息 (/help 查看命令): ").strip()
                
                if not user_input:
                    continue
                
                # 处理命令
                if user_input.startswith("/"):
                    if not self.handle_command(user_input):
                        break
                    continue
                
                # 处理普通消息
                if not self.current_session:
                    print("❌ 没有活动的会话，请先用 /new 创建会话")
                    continue
                
                # 检查是否是自我介绍
                if "我叫" in user_input or "我是" in user_input:
                    # 提取名字（简单处理）
                    for prefix in ["我叫", "我是"]:
                        if prefix in user_input:
                            name = user_input.split(prefix)[-1].strip()
                            self.username = name
                            break
                
                # 发送消息
                self.manager.chat(self.current_session, user_input)
                
            except KeyboardInterrupt:
                print("\n\n👋 检测到中断，正在退出...")
                break
            except Exception as e:
                print(f"\n❌ 发生错误: {e}")
                import traceback
                traceback.print_exc()


# ============ 5. 演示模式 ============
def demo_mode():
    """
    演示模式：展示记忆功能的效果
    """
    print("🌟" + "="*58 + "🌟")
    print("           DeepSeek Agent 记忆功能演示")
    print("🌟" + "="*58 + "🌟")
    
    # 创建会话管理器
    manager = ChatSessionManager()
    session_id = "demo_session"
    
    # 创建会话
    print("\n📝 创建演示会话...")
    manager.create_session(session_id)
    
    # 演示多轮对话
    test_conversations = [
        "你好，我叫小明",
        "你还记得我叫什么名字吗？",
        "北京天气怎么样？",
        "现在几点了？",
        "计算 123 * 456",
        "我刚才问了什么？",
        "我名字是什么？"
    ]
    
    for i, question in enumerate(test_conversations, 1):
        print(f"\n--- 第{i}轮对话 ---")
        manager.chat(session_id, question)
        time.sleep(1)  # 暂停一下，让用户看清
    
    # 显示完整历史
    print("\n📜" + "="*58)
    print("完整对话历史：")
    print("="*60)
    history = manager.get_conversation_history(session_id)
    for msg in history:
        role = msg["role"]
        content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
        print(f"[{role}]: {content}")


# ============ 6. 主程序 ============
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DeepSeek Agent 带记忆的对话系统")
    parser.add_argument("--demo", action="store_true", help="运行演示模式")
    parser.add_argument("--no-color", action="store_true", help="禁用颜色输出")
    
    args = parser.parse_args()
    
    if args.demo:
        demo_mode()
    else:
        # 运行交互式CLI
        cli = InteractiveChatCLI()
        try:
            cli.run()
        except Exception as e:
            print(f"\n❌ 程序出错: {e}")
            import traceback
            traceback.print_exc()
