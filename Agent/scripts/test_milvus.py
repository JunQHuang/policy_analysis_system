"""
测试Milvus连接
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pymilvus import connections, utility

print("正在连接Milvus (localhost:19530)...")
try:
    connections.connect('default', host='localhost', port='19530', timeout=5)
    print("✅ SUCCESS: Milvus连接成功！")
    
    collections = utility.list_collections()
    print(f"当前有 {len(collections)} 个集合: {collections}")
    sys.exit(0)
except Exception as e:
    print(f"❌ FAILED: {type(e).__name__}: {e}")
    sys.exit(1)

