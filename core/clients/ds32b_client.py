"""
DS32B模型客户端
"""
import hashlib
import json
import time
import requests
from typing import Optional


class DS32BClient:
    """DS32B模型客户端"""
    
    def __init__(self):
        self.url = "https://gpt.gjzq.cn/prod-api/guojin/openapi/stream/deepseek/r1-distill-32b"
        self.key = "s8porj9mn0f9210y"
        self.xClientId = "ixyg1867a81zxgw"
    
    
    def chat_completion(self, messages: list, temperature: float = 0.3, max_tokens: int = 100) -> Optional[str]:
        """
        调用DS32B模型
        
        Args:
            messages: 消息列表
            temperature: 温度参数（暂不使用，保持接口兼容）
            max_tokens: 最大token数（暂不使用，保持接口兼容）
            
        Returns:
            模型返回的文本内容
        """
        # 按照API要求，payload只包含messages, stream, stream_options
        payload = {
            "messages": messages,
            "stream": True,
            "stream_options": {
                "include_usage": True
            }
        }
        
        # 生成MD5签名
        timestamp = time.time()
        timestamp_ms = int(timestamp * 1000)
        
        # 对字典按键排序
        sorted_payload = {k: payload[k] for k in sorted(payload)}
        # 将字典转换为JSON字符串
        json_str = str(json.dumps(sorted_payload, ensure_ascii=False))
        # 拼接：JSON字符串（去掉空格）+ 时间戳 + 密钥
        concatenated_str = json_str.replace(" ", "") + timestamp_ms.__str__() + self.key
        # 计算MD5
        md5_hash = hashlib.md5()
        md5_hash.update(concatenated_str.encode('utf-8'))
        md5_digest = md5_hash.hexdigest()
        
        # 构建请求头
        headers = {
            "X-Client-Id": self.xClientId,
            "X-Timestamp": timestamp_ms.__str__(),
            "X-Sign": md5_digest,
            "content-type": "application/json"
        }
        
        try:
            response = requests.post(
                self.url, 
                json=payload, 
                headers=headers, 
                stream=True, 
                timeout=(10, 300)
            )
            response.raise_for_status()
            
            # 处理流式响应（SSE格式）
            content_parts = []
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    # 处理SSE格式：data: {...}
                    if line.startswith('data: '):
                        data_str = line[6:]  # 去掉 "data: " 前缀
                        if data_str.strip() == '[DONE]':
                            break
                        try:
                            data = json.loads(data_str)
                            # 提取并输出内容
                            if 'choices' in data and len(data['choices']) > 0:
                                delta = data['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                if content:
                                    content_parts.append(content)
                        except json.JSONDecodeError:
                            continue
            
            return ''.join(content_parts) if content_parts else None
            
        except requests.exceptions.RequestException as e:
            print(f"[DS32B] 请求错误: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"[DS32B] 响应内容: {e.response.text}")
            return None
        except Exception as e:
            print(f"[DS32B] 发生错误: {e}")
            import traceback
            traceback.print_exc()
            return None


def get_ds32b_client() -> DS32BClient:
    """获取DS32B客户端实例"""
    return DS32BClient()


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("测试DS32B客户端")
    print("=" * 60)
    
    client = get_ds32b_client()
    
    test_messages = [
        {
            "role": "system",
            "content": "你好"
        },
        {
            "role": "user",
            "content": "今日天气"
        }
    ]
    
    print("\n发送测试请求...")
    response = client.chat_completion(test_messages)
    
    if response:
        print(f"\n✅ 响应成功: {response}")
    else:
        print("\n❌ 响应失败")
    
    print("\n" + "=" * 60)

