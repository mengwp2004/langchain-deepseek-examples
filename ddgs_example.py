# ddgs_example.py - 修正版
"""
DuckDuckGo Search (duckduckgo-search) 完整使用示例
"""

import time
from duckduckgo_search import DDGS  # ✅ 正确的导入方式

def simple_search():
    """最简单的搜索示例"""
    print("🔍 简单搜索示例")
    print("=" * 40)
    
    # 使用 DDGS 类
    with DDGS() as ddgs:
        # 文本搜索
        results = list(ddgs.text(
            keywords="人工智能最新进展",  # 注意参数名是 keywords
            max_results=3
        ))
        
        for i, result in enumerate(results, 1):
            print(f"\n【{i}】{result.get('title', '无标题')}")
            print(f"📄 {result.get('body', '无内容')[:100]}...")
            print(f"🔗 {result.get('href', '无链接')}")
            time.sleep(0.5)

def search_with_params():
    """带参数的搜索示例"""
    print("\n🔍 参数化搜索示例")
    print("=" * 40)
    
    with DDGS() as ddgs:
        # 带参数的文本搜索
        results = list(ddgs.text(
            keywords="量子计算",
            region="cn-zh",        # 中文结果
            safesearch="moderate",  # 安全搜索级别
            timelimit="m",          # 最近一个月
            max_results=5
        ))
        
        print(f"找到 {len(results)} 条结果：")
        for i, r in enumerate(results, 1):
            print(f"{i}. {r.get('title', '')[:50]}")

def search_news():
    """新闻搜索示例"""
    print("\n🔍 新闻搜索示例")
    print("=" * 40)
    
    with DDGS() as ddgs:
        results = list(ddgs.news(
            keywords="科技热点",
            region="cn-zh",
            timelimit="d",  # 最近一天
            max_results=3
        ))
        
        for i, r in enumerate(results, 1):
            print(f"\n【{i}】{r.get('title', '无标题')}")
            if 'date' in r:
                print(f"📅 {r['date']}")
            print(f"📄 {r.get('body', '')[:100]}...")

def search_images():
    """图片搜索示例"""
    print("\n🔍 图片搜索示例")
    print("=" * 40)
    
    with DDGS() as ddgs:
        results = list(ddgs.images(
            keywords="风景",
            max_results=2,
            size="Wallpaper",  # 壁纸尺寸
            color="Green"      # 绿色调
        ))
        
        for i, r in enumerate(results, 1):
            print(f"\n【{i}】{r.get('title', '无标题')}")
            print(f"🖼️ {r.get('image', '')}")

def search_videos():
    """视频搜索示例"""
    print("\n🔍 视频搜索示例")
    print("=" * 40)
    
    with DDGS() as ddgs:
        results = list(ddgs.videos(
            keywords="Python教程",
            max_results=2,
            duration="medium"  # 中等时长
        ))
        
        for i, r in enumerate(results, 1):
            print(f"\n【{i}】{r.get('title', '无标题')}")
            print(f"⏱️ 时长: {r.get('duration', '未知')}")
            print(f"🔗 {r.get('content', '')}")

def search_with_proxy():
    """使用代理的示例（如果需要）"""
    print("\n🔍 代理搜索示例")
    print("=" * 40)
    
    # 如果需要使用代理
    # proxy = "http://user:pass@proxy.example.com:8080"
    # 或者使用 Tor
    # proxy = "socks5://127.0.0.1:9150"
    
    with DDGS(proxy=None) as ddgs:  # 不需要代理时设为 None
        results = list(ddgs.text(
            keywords="Python programming",
            region="us-en",
            max_results=2
        ))
        
        for r in results:
            print(f"标题: {r.get('title', '')}")

def handle_errors():
    """错误处理示例"""
    print("\n🔍 错误处理示例")
    print("=" * 40)
    
    try:
        from duckduckgo_search import DDGS
        from duckduckgo_search.exceptions import DuckDuckGoSearchException
        
        with DDGS() as ddgs:
            results = list(ddgs.text(
                keywords="test",
                max_results=3
            ))
            print(f"成功获取 {len(results)} 条结果")
            
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请安装: pip install -U duckduckgo-search")
    except DuckDuckGoSearchException as e:
        print(f"❌ 搜索错误: {e}")
    except Exception as e:
        print(f"❌ 其他错误: {e}")

def langchain_integration():
    """LangChain 集成示例"""
    print("\n🔍 LangChain 集成示例")
    print("=" * 40)
    
    try:
        from langchain.tools import tool
        from duckduckgo_search import DDGS
        
        @tool
        def search_web(query: str) -> str:
            """搜索网络获取实时信息"""
            try:
                with DDGS() as ddgs:
                    results = list(ddgs.text(
                        keywords=query,
                        max_results=5,
                        region="cn-zh"
                    ))
                    
                    if not results:
                        return "未找到相关结果"
                    
                    output = ["找到以下信息："]
                    for i, r in enumerate(results, 1):
                        output.append(f"\n{i}. {r.get('title', '')}")
                        output.append(f"   {r.get('body', '')[:150]}...")
                    
                    return "\n".join(output)
            except Exception as e:
                return f"搜索失败: {e}"
        
        # 测试工具
        result = search_web.invoke("人工智能")
        print(result)
        
    except ImportError as e:
        print(f"请安装依赖: pip install langchain duckduckgo-search")

# ============ 主程序 ============
if __name__ == "__main__":
    print("🌟" + "="*58 + "🌟")
    print("           DuckDuckGo Search 完整示例 (修正版)")
    print("🌟" + "="*58 + "🌟")
    
    # 运行示例
    simple_search()
    search_with_params()
    search_news()
    search_images()
    search_videos()
    handle_errors()
    
    # 如果需要 LangChain 集成
    # langchain_integration()
    
    print("\n✨ 所有示例运行完成！")