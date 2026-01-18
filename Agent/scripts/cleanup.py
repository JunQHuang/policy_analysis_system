"""
数据库清理工具
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pymilvus import utility, connections

def cleanup_database():
    """清理Milvus数据库中的所有相关集合"""
    print("="*80)
    print("数据库清理工具")
    print("="*80)
    
    try:
        print("1. 连接Milvus...")
        connections.connect(alias='default', host='localhost', port='19530')
        print("   ✅ 已连接到Milvus")
        
        print("\n2. 检查现有集合...")
        collection_names = ["policy_documents", "policy_documents_chunks"]
        
        cleaned_any = False
        for name in collection_names:
            if utility.has_collection(name):
                print(f"   🗑️ 正在删除集合: {name}...")
                utility.drop_collection(name)
                print(f"   ✅ 集合 '{name}' 已删除")
                cleaned_any = True
            else:
                print(f"   ⏭️ 集合 '{name}' 不存在，跳过")
        
        if not cleaned_any:
            print("   ℹ️ 没有找到需要清理的集合")
        
        print("\n✅ 数据库清理完成！")
        return True
        
    except Exception as e:
        print(f"\n❌ 清理失败: {e}")
        return False

if __name__ == "__main__":
    success = cleanup_database()
    if success:
        print("\n🎉 清理成功！现在可以重新构建数据库了")
        print("下一步: python main/build_knowledge_base.py")
    else:
        print("\n💥 清理失败！请检查Milvus服务状态")

