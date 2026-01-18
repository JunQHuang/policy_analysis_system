"""
火山引擎API客户端
"""
from volcenginesdkarkruntime import Ark
from typing import List, Dict
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import VOLCENGINE_API_KEY, VOLCENGINE_BASE_URL, VOLCENGINE_MODEL


class VolcEngineClient:
    """火山引擎API客户端"""
    
    def __init__(self):
        self.api_key = VOLCENGINE_API_KEY
        self.base_url = VOLCENGINE_BASE_URL
        self.model = VOLCENGINE_MODEL
        
        self.client = Ark(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def chat_completion(self, messages: List[Dict[str, str]], 
                       temperature: float = 0.3,
                       max_tokens: int = 32768,  # ⭐ API最大限制32768
                       retry_count: int = 3) -> str:
        """调用聊天完成API"""
        import time
        
        for attempt in range(retry_count + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                

                # ⭐ 检查是否被截断
                finish_reason = response.choices[0].finish_reason
                if finish_reason == 'length':
                    print(f"[VolcEngine] ⚠️ 输出被截断（达到max_tokens={max_tokens}限制）")
                
                return response.choices[0].message.content
                
            except Exception as e:
                error_str = str(e)
                
                if 'rate limit' in error_str.lower() or '429' in error_str:
                    if attempt < retry_count:
                        wait_time = (2 ** attempt) * 5
                        print(f"[VolcEngine] 速率限制，等待 {wait_time}秒后重试...")
                        time.sleep(wait_time)
                        continue
                    else:
                        return f"错误: 速率限制"
                else:
                    if attempt < retry_count:
                        time.sleep(1)
                        continue
                    else:
                        return f"错误: {str(e)}"
        
        return "错误: 未知错误"


def get_volcengine_client() -> VolcEngineClient:
    """获取火山引擎客户端实例"""
    return VolcEngineClient()


def test_volcengine_client():
    """测试火山引擎客户端"""
    print("=" * 80)
    print("火山引擎客户端测试")
    print("=" * 80)
    
    # 1. 检查配置
    print("\n[1] 检查配置...")
    print(f"  API Key: {VOLCENGINE_API_KEY[:20]}..." if len(VOLCENGINE_API_KEY) > 20 else f"  API Key: {VOLCENGINE_API_KEY}")
    print(f"  Base URL: {VOLCENGINE_BASE_URL}")
    print(f"  Model: {VOLCENGINE_MODEL}")
    
    # 2. 初始化客户端
    print("\n[2] 初始化客户端...")
    try:
        client = get_volcengine_client()
        print("  ✅ 客户端初始化成功")
    except Exception as e:
        print(f"  ❌ 客户端初始化失败: {e}")
        return
    
    # 3. 测试简单对话
    print("\n[3] 测试简单对话...")
    test_messages = [
        {"role": "user", "content": "请用一句话介绍人工智能"}
    ]
    
    try:
        response = client.chat_completion(
            messages=test_messages,
            temperature=0.3,
            max_tokens=100
        )
        
        if response.startswith("错误:"):
            print(f"  ❌ API调用失败: {response}")
        else:
            print(f"  ✅ API调用成功")
            print(f"  响应内容: {response[:100]}..." if len(response) > 100 else f"  响应内容: {response}")
            
    except Exception as e:
        print(f"  ❌ 测试失败: {e}")
        import traceback
        print(f"  错误详情:\n{traceback.format_exc()}")
    
    # 4. 测试参数
    print("\n[4] 测试不同参数...")
    test_cases = [
        {"temperature": 0.1, "max_tokens": 50, "desc": "低温度、短输出"},
        {"temperature": 0.7, "max_tokens": 200, "desc": "高温度、长输出"},
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n  测试 {i}: {test_case['desc']}")
        try:
            response = client.chat_completion(
                messages=[{"role": "user", "content": "用10个字总结机器学习"}],
                temperature=test_case["temperature"],
                max_tokens=test_case["max_tokens"]
            )
            if not response.startswith("错误:"):
                print(f"    ✅ 成功: {response[:50]}...")
            else:
                print(f"    ❌ 失败: {response}")
        except Exception as e:
            print(f"    ❌ 异常: {e}")
    
    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)


if __name__ == "__main__":
    test_volcengine_client()
