"""
向量数据库模块 - Milvus + GPU 实现
使用GPU加速的向量检索，性能远超ChromaDB

架构：
- Milvus: 分布式向量数据库，支持大规模向量检索
- GPU加速的嵌入模型（sentence-transformers on GPU）
- 双层索引：文档级（粗排）+ Chunk级（精排）
- 时间感知：支持时间过滤和时间加权
"""
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import torch
import os
from sentence_transformers import SentenceTransformer
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility
)

from models import PolicySegment
from config import OUTPUT_DIR
from utils.chunking import PolicyDocumentChunker, DocumentChunk


class MilvusVectorDatabase:
    """
    Milvus向量数据库 - GPU加速版
    
    特性：
    1. 使用Milvus进行分布式向量存储和检索
    2. 使用cuvs进行GPU加速的向量搜索
    3. 支持批量插入和批量检索
    4. 支持多种距离度量（L2, IP, Cosine等）
    """
    
    def __init__(self, collection_name: str = "policy_documents", 
                 embedding_model: str = "./models/xiaobu-embedding-v2",
                 dim: int = 1792,
                 enable_chunking: bool = True,
                 chunk_only: bool = True):  # 修改默认值为True，只使用chunk级别
        """
        初始化Milvus向量数据库
        
        Args:
            collection_name: 集合名称
            embedding_model: 嵌入模型名称
            dim: 向量维度
            enable_chunking: 是否启用chunk级索引
            chunk_only: 是否只使用chunk级别（简化版RAG）
        """
        self.collection_name = collection_name
        self.chunk_collection_name = f"{collection_name}_chunks"
        self.embedding_dim = dim
        self.enable_chunking = enable_chunking
        self.chunk_only = chunk_only
        
        print(f"[MilvusVectorDB] 正在初始化...")
        print(f"[MilvusVectorDB] 简化版RAG: {'只使用chunk级别' if chunk_only else '双层索引' if enable_chunking else '禁用'}")
        
        # 1. 连接到Milvus（自动检测运行环境和IP）
        connection_configs = []
        
        # 检测运行环境：是否在WSL中运行
        import platform
        is_wsl = False
        is_windows = platform.system() == 'Windows'
        
        try:
            # 检测是否在WSL中运行（检查/proc/version或WSL相关环境变量）
            if os.path.exists('/proc/version'):
                with open('/proc/version', 'r') as f:
                    version_info = f.read().lower()
                    if 'microsoft' in version_info or 'wsl' in version_info:
                        is_wsl = True
        except:
            pass
        
        # 如果明确设置了WSL_DISTRO_NAME，说明在WSL中
        if os.environ.get('WSL_DISTRO_NAME'):
            is_wsl = True
        
        print(f"[MilvusVectorDB] 🔍 运行环境检测: Windows={is_windows}, WSL={is_wsl}")
        
        # 优先检查环境变量（允许手动指定主机）
        milvus_host = os.environ.get('MILVUS_HOST')
        if not milvus_host and is_windows:
            # 如果当前进程中没有，尝试从注册表读取用户级环境变量（Windows）
            try:
                import winreg
                key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r'Environment')
                try:
                    milvus_host, _ = winreg.QueryValueEx(key, 'MILVUS_HOST')
                    winreg.CloseKey(key)
                except FileNotFoundError:
                    winreg.CloseKey(key)
                    milvus_host = None
            except:
                milvus_host = None
        
        if milvus_host:
            connection_configs.append({'host': milvus_host, 'port': '19530'})
            print(f"[MilvusVectorDB] ✅ 使用环境变量指定的Milvus主机: {milvus_host}")
        
        # 根据运行环境选择连接方式
        try:
            import subprocess
            import re
            
            if is_wsl:
                # 在WSL中运行：直接使用localhost（最简单的方式）
                print(f"[MilvusVectorDB] 🔍 检测到WSL环境，使用localhost连接")
                connection_configs = [
                    {'host': 'localhost', 'port': '19530'},
                    {'host': '127.0.0.1', 'port': '19530'},
                ]
            elif is_windows:
                # 在Windows中运行：需要尝试WSL网关IP或配置端口转发
                wsl_gateway_ip = None
                # 方法1: 获取WSL2的默认网关IP（Windows主机在WSL网络中的IP）- 这是从Windows访问WSL服务的正确IP
                try:
                    # 尝试多种方式提取IP
                    result = subprocess.run(
                        ['wsl', 'bash', '-c', "ip route show default | head -1"],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if result.returncode == 0:
                        route_output = result.stdout.strip()
                        print(f"[MilvusVectorDB] 🔍 WSL路由命令输出: '{route_output}'")
                        
                        # 从路由输出中提取IP（格式：default via 172.28.48.1 dev eth0...）
                        # 使用正则表达式提取IP
                        ip_match = re.search(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})', route_output)
                        if ip_match:
                            wsl_gateway_ip = ip_match.group(1)
                            if wsl_gateway_ip not in [c['host'] for c in connection_configs]:
                                connection_configs.append({'host': wsl_gateway_ip, 'port': '19530'})
                                print(f"[MilvusVectorDB] ✅ 检测到WSL网关IP（Windows在WSL中的IP）: {wsl_gateway_ip} ⭐ 这是从Windows访问WSL的正确IP")
                        else:
                            print(f"[MilvusVectorDB] ⚠️ 无法从路由输出中提取IP: '{route_output}'")
                    else:
                        print(f"[MilvusVectorDB] ⚠️ WSL路由命令失败，返回码: {result.returncode}")
                        if result.stderr:
                            print(f"[MilvusVectorDB] ⚠️ 错误输出: {result.stderr[:200]}")
                except Exception as e:
                    print(f"[MilvusVectorDB] ⚠️ 无法获取WSL网关IP，异常: {type(e).__name__}: {e}")
                
                # 如果网关IP检测失败，尝试其他方法
                if not wsl_gateway_ip:
                    # 方法2: 从WSL hostname -I获取第一个IP（可能是网关IP）
                    try:
                        result = subprocess.run(
                            ['wsl', 'hostname', '-I'],
                            capture_output=True,
                            text=True,
                            timeout=3
                        )
                        if result.returncode == 0:
                            wsl_ips = result.stdout.strip().split()
                            # 通常第一个IP是主IP，可能是172.x.x.x格式（WSL2常用）
                            for wsl_ip in wsl_ips[:2]:  # 只取前2个IP尝试
                                if wsl_ip and re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', wsl_ip):
                                    # WSL2通常使用172.x.x.x网段
                                    if wsl_ip.startswith('172.') or wsl_ip.startswith('192.168.'):
                                        if wsl_ip not in [c['host'] for c in connection_configs]:
                                            connection_configs.append({'host': wsl_ip, 'port': '19530'})
                                            print(f"[MilvusVectorDB] 🔍 检测到WSL IP: {wsl_ip}")
                                        break
                    except Exception as e:
                        print(f"[MilvusVectorDB] ⚠️ 无法获取WSL IP: {e}")
                
                # 最后尝试localhost（需要WSL端口转发配置）
                connection_configs.append({'host': 'localhost', 'port': '19530'})
                connection_configs.append({'host': '127.0.0.1', 'port': '19530'})
            else:
                # 纯Linux/Mac环境（非WSL），直接使用localhost
                print(f"[MilvusVectorDB] 🔍 检测到Linux/Mac环境，使用localhost连接")
                if not connection_configs:
                    connection_configs = [
                        {'host': 'localhost', 'port': '19530'},
                        {'host': '127.0.0.1', 'port': '19530'},
                    ]
        except Exception as e:
            print(f"[MilvusVectorDB] ⚠️ 环境检测出错: {e}")
            # 如果检测失败，使用默认配置
            if not connection_configs:
                connection_configs = [
                    {'host': 'localhost', 'port': '19530'},
                    {'host': '127.0.0.1', 'port': '19530'},
                ]
        
        # 尝试连接
        connected = False
        last_error = None
        print(f"[MilvusVectorDB] 🔍 尝试连接Milvus，共 {len(connection_configs)} 个配置...")
        for i, config in enumerate(connection_configs):
            try:
                print(f"[MilvusVectorDB]   尝试 {i+1}/{len(connection_configs)}: {config['host']}:{config['port']}")
                connections.connect(
                    alias='default',
                    host=config['host'],
                    port=config['port'],
                    timeout=5  # 5秒超时
                )
                print(f"[MilvusVectorDB] ✅ 已连接到Milvus ({config['host']}:{config['port']})")
                connected = True
                break
            except Exception as e:
                print(f"[MilvusVectorDB]   ❌ 连接失败: {str(e)[:100]}")
                last_error = e
                continue
        
        if not connected:
            print(f"\n[MilvusVectorDB] ❌ 所有连接尝试均失败！")
            print(f"[MilvusVectorDB] 最后错误: {last_error}")
            
            if is_wsl:
                print(f"\n[MilvusVectorDB] 💡 排查步骤（WSL环境中）：")
                print(f"   1. 检查Milvus容器是否运行:")
                print(f"      docker ps | grep milvus")
                print(f"      docker logs milvus-standalone")
                print(f"   2. 检查端口是否监听:")
                print(f"      ss -tuln | grep 19530")
                print(f"   3. 测试连接:")
                print(f"      python3 -c \"from pymilvus import connections; connections.connect('default', host='localhost', port='19530')\"")
            elif is_windows:
                print(f"\n[MilvusVectorDB] 💡 排查步骤（Windows访问WSL中的Milvus）：")
                print(f"\n   ⚠️ 重要：WSL2网络隔离，Windows无法直接访问WSL服务")
                print(f"\n   方案1：配置端口转发（推荐，以管理员身份运行PowerShell）:")
                print("      $wslIP = (wsl bash -c \"ip route show default | awk '{print \\$3}'\").Trim()")
                print("      netsh interface portproxy add v4tov4 listenport=19530 listenaddress=0.0.0.0 connectport=19530 connectaddress=$wslIP")
                print("      然后设置环境变量: $env:MILVUS_HOST = 'localhost'")
                print(f"\n   方案2：在WSL中运行（最简单）:")
                print("      wsl")
                print("      cd /mnt/c/Users/qq100/Desktop/国金证券/Agent")
                print("      conda activate quant")
                print("      python run_full_pipeline.py")
                print(f"\n   方案3：检查Milvus是否运行:")
                print(f"      wsl docker ps | grep milvus")
            else:
                print(f"\n[MilvusVectorDB] 💡 排查步骤:")
                print(f"   1. 检查Milvus服务是否运行")
                print(f"   2. 检查端口19530是否监听")
            
            raise last_error
        
        # 2. 加载GPU嵌入模型
        print(f"[MilvusVectorDB] 加载嵌入模型: {embedding_model}")
        self.model = SentenceTransformer(embedding_model)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            print(f"[MilvusVectorDB] ✅ 模型已加载到GPU: {torch.cuda.get_device_name(0)}")
        else:
            print(f"[MilvusVectorDB] ⚠️ GPU不可用，使用CPU")
        
        # 3. 初始化chunker（如果启用）
        if self.enable_chunking:
            # ⭐ chunk配置：与Milvus现有数据一致（约450字）
            # 新政策检索时LLM也会生成400-450字的片段，确保粒度匹配
            self.chunker = PolicyDocumentChunker(
                chunk_size_target=400,
                chunk_size_max=450,
                overlap=50,
                absolute_max=450
            )
        
        # 4. 创建或获取集合（只使用chunk级别）
        self._init_chunk_collection()
        
        chunk_count = self.chunk_collection.num_entities
        print(f"[MilvusVectorDB] ✅ 初始化完成（简化版：只使用chunk级别）")
        print(f"[MilvusVectorDB]   - Chunk级: {chunk_count} 个chunks")
    
    # 删除文档级集合初始化方法，只使用chunk级别
    
    def _init_chunk_collection(self):
        """初始化或获取Chunk级Milvus集合"""
        # 定义Chunk Schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=150),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=5000),  # 超保守设置，规避pymilvus bug
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(name="chunk_type", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),  # 文档标题
            FieldSchema(name="timestamp", dtype=DataType.VARCHAR, max_length=150),  # 发布时间
            FieldSchema(name="industries", dtype=DataType.VARCHAR, max_length=500),  # ⭐ 中信一级行业（逗号分隔，经过DS32B过滤）
            FieldSchema(name="investment_relevance", dtype=DataType.VARCHAR, max_length=10),  # ⭐ 投资相关性：高/低
            FieldSchema(name="report_series", dtype=DataType.VARCHAR, max_length=50),  # ⭐ 报告系列：晨会纪要/晚间速递/策略研究等
            FieldSchema(name="industry_policy_segments", dtype=DataType.VARCHAR, max_length=20000),  # ⭐ 行业及对应政策片段（JSON格式，增加到20000以支持大型JSON）
        ]
        
        schema = CollectionSchema(fields, description="政策文档Chunk向量库")
        
        # 创建或获取集合
        if utility.has_collection(self.chunk_collection_name):
            print(f"[MilvusVectorDB] Chunk集合已存在，检查schema...")
            existing_collection = Collection(self.chunk_collection_name)
            existing_fields = [field.name for field in existing_collection.schema.fields]
            
            # 检查是否包含title字段
            if "title" not in existing_fields:
                print(f"[MilvusVectorDB] ⚠️ 现有collection缺少title字段，重新创建...")
                utility.drop_collection(self.chunk_collection_name)
                print(f"[MilvusVectorDB] 创建新的Chunk集合...")
                self.chunk_collection = Collection(self.chunk_collection_name, schema)
                
                # 创建索引
                index_params = {
                    "metric_type": "L2",
                    "index_type": "HNSW",
                    "params": {"M": 16, "efConstruction": 200}
                }
                self.chunk_collection.create_index("embedding", index_params)
                print(f"[MilvusVectorDB] ✅ Chunk索引创建完成 (HNSW)")
            else:
                print(f"[MilvusVectorDB] ✅ Chunk集合schema正确，加载中...")
                self.chunk_collection = existing_collection
        else:
            print(f"[MilvusVectorDB] 创建新Chunk集合...")
            self.chunk_collection = Collection(self.chunk_collection_name, schema)
            
            # 创建索引
            index_params = {
                "metric_type": "L2",
                "index_type": "HNSW",
                "params": {"M": 16, "efConstruction": 200}
            }
            self.chunk_collection.create_index("embedding", index_params)
            print(f"[MilvusVectorDB] ✅ Chunk索引创建完成 (HNSW)")
        
        # 加载集合到内存（优化：检查状态后再加载）
        print(f"[MilvusVectorDB] 🔄 正在加载Chunk集合到内存...")
        try:
            # 检查集合是否为空（带超时保护）
            import threading
            
            entity_count = [None]
            count_error = [None]
            
            def get_count():
                try:
                    entity_count[0] = self.chunk_collection.num_entities
                except Exception as e:
                    count_error[0] = e
            
            count_thread = threading.Thread(target=get_count, daemon=True)
            count_thread.start()
            count_thread.join(timeout=10)  # 10秒超时
            
            if count_thread.is_alive():
                print(f"[MilvusVectorDB] ⚠️ 检查集合大小超时，假设为空集合（新集合）")
                print(f"[MilvusVectorDB] ✅ 跳过load操作（新集合不需要load，插入数据时会自动加载）")
                return  # 直接返回，不执行后续load
            
            if count_error[0]:
                print(f"[MilvusVectorDB] ⚠️ 检查集合大小失败: {count_error[0]}，假设为空集合")
                print(f"[MilvusVectorDB] ✅ 跳过load操作（新集合不需要load）")
                return
            
            print(f"[MilvusVectorDB] 🔍 当前集合实体数: {entity_count[0]}")
            
            if entity_count[0] == 0:
                # 空集合，跳过load操作（空集合不需要load）
                print(f"[MilvusVectorDB] ✅ 空集合，跳过load操作（插入数据时会自动加载）")
            else:
                # 非空集合，必须加载
                # 检查是否已经加载
                try:
                    # 尝试查询，如果能查询说明已加载
                    self.chunk_collection.query(expr="id >= 0", limit=1, output_fields=["id"])
                    print(f"[MilvusVectorDB] ✅ 集合已在内存中")
                except:
                    # 未加载，执行加载（带超时保护）
                    print(f"[MilvusVectorDB] 🔄 正在加载 {entity_count[0]} 个实体到内存...")
                    print(f"[MilvusVectorDB] ⚠️ 如果长时间卡在此处，可能是MinIO未运行")
                    
                    # 使用线程+超时机制，避免无限等待
                    import threading
                    import time
                    
                    load_success = [False]
                    load_error = [None]
                    
                    def load_in_thread():
                        try:
                            self.chunk_collection.load()
                            load_success[0] = True
                        except Exception as e:
                            load_error[0] = e
                    
                    load_thread = threading.Thread(target=load_in_thread, daemon=True)
                    load_thread.start()
                    load_thread.join(timeout=30)  # 30秒超时
                    
                    if load_thread.is_alive():
                        print(f"[MilvusVectorDB] ❌ 加载超时（30秒）")
                        print(f"[MilvusVectorDB] 💡 可能原因：MinIO服务未正常运行")
                        print(f"[MilvusVectorDB] 💡 检查命令: docker ps | grep minio")
                        print(f"[MilvusVectorDB] 💡 修复MinIO:")
                        print(f"   cd ~/milvus && docker compose down")
                        print(f"   sudo rm -rf volumes/minio/.minio.sys")
                        print(f"   docker compose up -d")
                        print(f"[MilvusVectorDB] ⚠️ 跳过load操作，继续初始化（如果是新集合可能不需要load）")
                        # 不抛出异常，允许继续（对于新集合可以跳过load）
                    elif load_error[0]:
                        print(f"[MilvusVectorDB] ❌ 加载失败: {load_error[0]}")
                        print(f"[MilvusVectorDB] 💡 可能原因：MinIO服务未正常运行")
                        print(f"[MilvusVectorDB] 💡 检查命令: docker ps | grep minio")
                        print(f"[MilvusVectorDB] 💡 修复步骤:")
                        print(f"   1. cd ~/milvus")
                        print(f"   2. docker compose down")
                        print(f"   3. sudo rm -rf volumes/minio/.minio.sys")
                        print(f"   4. docker compose up -d")
                        print(f"   5. 等待30秒后重试")
                        print(f"[MilvusVectorDB] ⚠️ 跳过load操作，继续初始化（新集合可能不需要load）")
                        # 不抛出异常，允许继续
                    elif load_success[0]:
                        print(f"[MilvusVectorDB] ✅ Chunk集合已加载到内存")
                
        except Exception as e:
            print(f"[MilvusVectorDB] ❌ 加载集合时出错: {e}")
            raise
    
    def get_max_doc_id_number(self) -> int:
        """
        获取最大的doc_id编号（用于继续编号）
        
        从Milvus和缓存文件中查找最大编号，取两者中的最大值
        
        Returns:
            最大的doc_id编号（例如：如果最大是doc_1144，返回1144），如果没有找到则返回0
        """
        import re
        import json
        from pathlib import Path
        
        max_milvus = 0
        max_cache = 0
        
        # 方法1：从Milvus查询
        try:
            if hasattr(self, 'chunk_collection') and self.chunk_collection is not None:
                existing_doc_ids = self._get_existing_doc_ids()
                for doc_id in existing_doc_ids:
                    match = re.match(r'^doc_(\d+)$', str(doc_id))
                    if match:
                        number = int(match.group(1))
                        if number > max_milvus:
                            max_milvus = number
        except Exception as e:
            print(f"[MilvusVectorDB] ⚠️ 从Milvus查询最大编号失败: {e}")
        
        # 方法2：从缓存文件查询
        try:
            cache_file = Path("cache/industry_agent_cache.json")
            if cache_file.exists():
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                for doc_id in cache_data.keys():
                    match = re.match(r'^doc_(\d+)$', str(doc_id))
                    if match:
                        number = int(match.group(1))
                        if number > max_cache:
                            max_cache = number
        except Exception as e:
            print(f"[MilvusVectorDB] ⚠️ 从缓存文件查询最大编号失败: {e}")
        
        # 取两者中的最大值
        max_number = max(max_milvus, max_cache)
        
        if max_number > 0:
            print(f"[MilvusVectorDB] ✅ 最大doc_id编号: {max_number} (Milvus: {max_milvus}, 缓存: {max_cache})")
        else:
            print(f"[MilvusVectorDB] ⚠️ 未找到doc_格式的doc_id，返回0")
        
        return max_number
    
    def _get_existing_doc_ids(self) -> set:
        """
        获取Milvus中已存在的所有doc_id（用于断点续传）
        
        Returns:
            已存在的doc_id集合
        """
        try:
            # 检查集合是否存在且有数据
            if not utility.has_collection(self.chunk_collection_name):
                print(f"[MilvusVectorDB] ⚠️ 集合 {self.chunk_collection_name} 不存在")
                return set()
            
            # 尝试加载集合（如果未加载）
            try:
                self.chunk_collection.load()
                print(f"[MilvusVectorDB] ✅ 集合已加载")
            except Exception as load_error:
                print(f"[MilvusVectorDB] ⚠️ 集合加载失败: {load_error}，尝试继续查询")
                # 如果加载失败，尝试继续查询（新插入的数据可能在内存中）
            
            entity_count = self.chunk_collection.num_entities
            print(f"[MilvusVectorDB] 📊 集合 {self.chunk_collection_name} 共有 {entity_count} 个entities")
            if entity_count == 0:
                print(f"[MilvusVectorDB] ⚠️ 集合为空，无已存在的doc_id")
                return set()
            
            # 查询所有唯一的doc_id
            existing_doc_ids = set()
            
            try:
                total = self.chunk_collection.num_entities
                print(f"[MilvusVectorDB] 🔍 检查已存在的文档（共 {total} 个chunks）...")
                
                # 使用迭代器查询所有doc_id（分批查询避免内存溢出）
                batch_size = 10000
                for offset in range(0, total, batch_size):
                    limit = min(batch_size, total - offset)
                    try:
                        results = self.chunk_collection.query(
                            expr=f"id >= {offset} && id < {offset + limit}",
                            output_fields=["doc_id"],
                            limit=limit
                        )
                        for result in results:
                            doc_id = result.get('doc_id')
                            if doc_id:
                                existing_doc_ids.add(doc_id)
                    except Exception as batch_error:
                        # 如果批量查询失败，尝试简单查询
                        print(f"[MilvusVectorDB] ⚠️ 批量查询失败，尝试简单查询: {batch_error}")
                        break
                    
                    if (offset + batch_size) % 50000 == 0:
                        print(f"[MilvusVectorDB]   已检查 {min(offset + batch_size, total)}/{total} 个chunks...")
                
                # 如果批量查询失败，尝试简单查询所有数据
                if len(existing_doc_ids) == 0 and total > 0:
                    try:
                        results = self.chunk_collection.query(
                            expr="id >= 0",
                            output_fields=["doc_id"],
                            limit=min(100000, total)  # 最多查询10万条
                        )
                        for result in results:
                            doc_id = result.get('doc_id')
                            if doc_id:
                                existing_doc_ids.add(doc_id)
                    except Exception as e2:
                        print(f"[MilvusVectorDB] ⚠️ 查询失败，假设无已存在文档: {e2}")
                        return set()
                
                print(f"[MilvusVectorDB] ✅ 检查完成，发现 {len(existing_doc_ids)} 个唯一的doc_id")
            except Exception as e:
                print(f"[MilvusVectorDB] ⚠️ 检查已存在文档失败: {e}，假设无已存在文档")
                return set()
            
            return existing_doc_ids
        except Exception as e:
            print(f"[MilvusVectorDB] ⚠️ 检查已存在文档失败: {e}，假设无已存在文档")
            return set()
    
    def get_existing_title_timestamp_pairs(self) -> set:
        """
        获取Milvus中已存在的所有 (标题, 发布时间) 组合（用于入库前去重）
        
        去重逻辑：标题和发布时间同时一样才算重复
        
        Returns:
            已存在的 (title, timestamp) 元组集合
        """
        try:
            if not utility.has_collection(self.chunk_collection_name):
                return set()
            
            # 尝试加载集合
            try:
                self.chunk_collection.load()
            except:
                pass
            
            entity_count = self.chunk_collection.num_entities
            if entity_count == 0:
                return set()
            
            existing_pairs = set()
            
            # 分批查询所有 (title, timestamp) 组合
            # ⭐ Milvus限制：offset + limit 不能超过 16384
            print(f"[MilvusVectorDB] 🔍 获取已存在的 (标题, 时间) 组合（共 {entity_count} 条记录）...")
            
            batch_size = 10000  # 每批查询10000条
            offset = 0
            
            while offset < entity_count:
                try:
                    # 计算本批次查询数量（不超过16384限制）
                    current_limit = min(batch_size, entity_count - offset, 16384 - offset % 16384)
                    if current_limit <= 0:
                        break
                    
                    results = self.chunk_collection.query(
                        expr="id >= 0",
                        output_fields=["title", "timestamp"],
                        offset=offset,
                        limit=current_limit
                    )
                    
                    if not results:
                        break
                    
                    for result in results:
                        title = result.get('title', '')
                        timestamp = result.get('timestamp', '')
                        if title:
                            # 使用 (title, timestamp) 元组作为唯一标识
                            existing_pairs.add((title, timestamp))
                    
                    offset += len(results)
                    
                    # 如果返回数量少于请求数量，说明已到末尾
                    if len(results) < current_limit:
                        break
                        
                except Exception as e:
                    print(f"[MilvusVectorDB] ⚠️ 分批查询失败 (offset={offset}): {e}")
                    break
            
            print(f"[MilvusVectorDB] ✅ 已存在 {len(existing_pairs)} 个唯一的 (标题, 时间) 组合")
            return existing_pairs
            
        except Exception as e:
            print(f"[MilvusVectorDB] ⚠️ 获取已存在组合失败: {e}")
            return set()
    
    def add_documents(self, segments: List[PolicySegment], batch_size: int = 100, skip_existing: bool = True):
        """
        批量添加文档到Milvus向量库（只使用chunk级别）
        
        Args:
            segments: PolicySegment列表
            batch_size: 批处理大小
            skip_existing: 是否跳过已存在的文档（断点续传功能，默认True）
        """
        print(f"[MilvusVectorDB] 开始添加 {len(segments)} 个文档...")
        print(f"[MilvusVectorDB] 简化版模式：只添加chunk级向量")
        
        # ⭐ 断点续传：检查已存在的文档
        if skip_existing:
            existing_doc_ids = self._get_existing_doc_ids()
            if existing_doc_ids:
                original_count = len(segments)
                segments = [seg for seg in segments if seg.doc_id not in existing_doc_ids]
                skipped_count = original_count - len(segments)
                if skipped_count > 0:
                    print(f"[MilvusVectorDB] ✅ 跳过 {skipped_count} 个已存在的文档，剩余 {len(segments)} 个待处理")
                else:
                    print(f"[MilvusVectorDB] ✅ 所有文档都是新的，无需跳过")
            else:
                print(f"[MilvusVectorDB] ✅ 未发现已存在的文档，将处理所有 {len(segments)} 个文档")
        
        if not segments:
            print(f"[MilvusVectorDB] ✅ 所有文档都已存在，无需插入")
            return
        
        # 打印入库前的文档详细信息
        print("\n" + "="*80)
        print("【入库前数据检查】")
        print("="*80)
        for i, seg in enumerate(segments, 1):
            print(f"\n文档 {i}/{len(segments)}:")
            print(f"  doc_id: {seg.doc_id}")
            print(f"  title: {seg.title[:80]}..." if len(seg.title) > 80 else f"  title: {seg.title}")
            print(f"  timestamp: {seg.timestamp}")
            print(f"  industries: {seg.industries}")  # ⭐ 经过DS32B过滤后的行业
            print(f"  investment_relevance: {seg.metadata.get('investment_relevance', 'N/A')}")
            print(f"  report_series: {seg.metadata.get('report_series', 'N/A')}")  # ⭐ 报告系列
            print(f"  content_length: {len(seg.content)} 字符")
            print(f"  content_preview: {seg.content[:200]}..." if len(seg.content) > 200 else f"  content: {seg.content}")
            
            # 显示行业政策片段
            industry_segments = seg.metadata.get('industry_policy_segments', {})
            if industry_segments:
                print(f"  industry_policy_segments:")
                for industry, segments_list in industry_segments.items():
                    print(f"    - {industry}: {len(segments_list)} 个片段")
                    for j, segment_text in enumerate(segments_list[:2], 1):  # 只显示前2个
                        preview = segment_text[:100] + "..." if len(segment_text) > 100 else segment_text
                        print(f"      片段{j}: {preview}")
        print("="*80 + "\n")
        
        try:
            # 只添加chunk级向量
            self._add_chunk_level(segments, batch_size)
            
            print(f"[MilvusVectorDB] ✅ 全部插入完成")
            print(f"[MilvusVectorDB]   - Chunk级: {self.chunk_collection.num_entities} 个")
        except Exception as e:
            print(f"[MilvusVectorDB] ❌ 数据插入失败: {e}")
            print(f"[MilvusVectorDB] 错误类型: {type(e).__name__}")
            import traceback
            print(f"[MilvusVectorDB] 错误详情: {traceback.format_exc()}")
            print(f"[MilvusVectorDB] 💡 提示: 已插入的数据已保存，可以重新运行程序继续插入剩余数据（断点续传）")
            raise e
    
    # 删除文档级向量添加方法，只使用chunk级别
    
    def _add_chunk_level(self, segments: List[PolicySegment], batch_size: int = 100):
        """添加Chunk级别数据到Milvus"""
        if not segments:
            print(f"[MilvusVectorDB] [Chunk级] ⚠️ 没有文档需要切分")
            return
        
        # 检查集合是否需要加载（空集合不需要load，插入时会自动加载）
        try:
            # 先检查集合是否为空
            entity_count = self.chunk_collection.num_entities
            print(f"[MilvusVectorDB] [Chunk级] 🔍 当前集合实体数: {entity_count}")
            
            if entity_count == 0:
                # 空集合，不需要load，插入数据时会自动加载
                print(f"[MilvusVectorDB] [Chunk级] ✅ 空集合，跳过load（插入数据时会自动加载）")
            else:
                # 非空集合，需要先加载才能插入新数据
                try:
                    # 尝试查询，如果能查询说明已加载
                    self.chunk_collection.query(expr="id >= 0", limit=1, output_fields=["id"])
                    print(f"[MilvusVectorDB] [Chunk级] ✅ 集合已在内存中")
                except:
                    # 未加载，执行加载（带超时保护）
                    print(f"[MilvusVectorDB] [Chunk级] 🔄 集合有数据但未加载，正在加载...")
                    
                    import threading
                    
                    load_success = [False]
                    load_error = [None]
                    
                    def load_in_thread():
                        try:
                            self.chunk_collection.load()
                            load_success[0] = True
                        except Exception as e:
                            load_error[0] = e
                    
                    load_thread = threading.Thread(target=load_in_thread, daemon=True)
                    load_thread.start()
                    load_thread.join(timeout=30)  # 30秒超时
                    
                    if load_thread.is_alive():
                        print(f"[MilvusVectorDB] [Chunk级] ⚠️ 加载超时（30秒），尝试继续插入")
                        print(f"[MilvusVectorDB] [Chunk级] 💡 如果后续插入失败，检查MinIO: docker ps | grep minio")
                    elif load_error[0]:
                        error_msg = str(load_error[0])
                        if "collection not loaded" in error_msg.lower():
                            print(f"[MilvusVectorDB] [Chunk级] ⚠️ 集合未加载，尝试继续插入（Milvus可能会自动加载）")
                        else:
                            print(f"[MilvusVectorDB] [Chunk级] ⚠️ 加载失败，但尝试继续插入: {load_error[0]}")
                    elif load_success[0]:
                        print(f"[MilvusVectorDB] [Chunk级] ✅ 集合已加载")
        except Exception as check_error:
            # 检查失败，假设是空集合，继续插入
            print(f"[MilvusVectorDB] [Chunk级] ⚠️ 检查集合状态失败: {check_error}，假设为空集合，继续插入")
        
        print(f"[MilvusVectorDB] [Chunk级] 开始切分和向量化...")
        
        all_chunks: List[DocumentChunk] = []
        chunk_texts = []
        
        # 对每个文档进行切分（传入完整元数据）
        for seg in segments:
            # ⭐ 统一timestamp处理，与generate_insights.py保持一致
            timestamp_str = seg.timestamp.isoformat() if seg.timestamp else ""
            # 从metadata中提取投资相关性标签、报告系列和行业政策片段
            investment_relevance = seg.metadata.get('investment_relevance', '')
            report_series = seg.metadata.get('report_series', 'N/A')  # ⭐ 报告系列
            industry_policy_segments_dict = seg.metadata.get('industry_policy_segments', {})
            # 序列化为JSON字符串
            import json
            industry_policy_segments_json = json.dumps(industry_policy_segments_dict, ensure_ascii=False) if industry_policy_segments_dict else ""
            chunks = self.chunker.chunk_document(
                doc_id=seg.doc_id,
                title=seg.title,
                content=seg.content,
                timestamp=timestamp_str,
                industries=','.join(seg.industries) if seg.industries else '',
                investment_relevance=investment_relevance,
                report_series=report_series,
                industry_policy_segments=industry_policy_segments_json
            )
            
            # 收集chunk和文本（元数据已在chunk对象中）
            for chunk in chunks:
                all_chunks.append(chunk)
                chunk_texts.append(chunk.content)
        
        print(f"[MilvusVectorDB] [Chunk级] 共切分为 {len(all_chunks)} 个chunks")
        
        # 打印切分后的chunk详细信息
        print("\n" + "="*80)
        print("【Chunk切分后数据检查】")
        print("="*80)
        for i, chunk in enumerate(all_chunks[:5], 1):  # 显示前5个chunk
            print(f"\nChunk {i}/{min(5, len(all_chunks))}:")
            print(f"  chunk_id: {chunk.chunk_id}")
            print(f"  doc_id: {chunk.doc_id}")
            print(f"  chunk_index: {chunk.chunk_index}")
            print(f"  chunk_type: {chunk.chunk_type}")
            print(f"  title: {chunk.title[:50]}..." if len(chunk.title) > 50 else f"  title: {chunk.title}")
            print(f"  timestamp: {chunk.timestamp}")
            print(f"  industries: {chunk.industries}")  # ⭐ 过滤后的行业
            print(f"  investment_relevance: {chunk.investment_relevance}")
            print(f"  report_series: {chunk.report_series}")  # ⭐ 报告系列
            print(f"  content_length: {len(chunk.content)} 字符")
            print(f"  content: {chunk.content[:150]}..." if len(chunk.content) > 150 else f"  content: {chunk.content}")
            
            # 显示行业政策片段（JSON格式）
            if chunk.industry_policy_segments:
                import json
                try:
                    segments_dict = json.loads(chunk.industry_policy_segments)
                    print(f"  industry_policy_segments: {list(segments_dict.keys())}")
                except:
                    print(f"  industry_policy_segments: (JSON解析失败)")
        
        if len(all_chunks) > 5:
            print(f"\n... 还有 {len(all_chunks) - 5} 个chunks未显示")
        print("="*80 + "\n")
        
        # 🔍 步骤1：检查原始all_chunks数据（技术调试用）
        print(f"\n[调试1] 检查原始all_chunks对象:")
        for i, chunk in enumerate(all_chunks[:3]):  # 只检查前3个
            print(f"  Chunk {i}:")
            print(f"    chunk_id类型={type(chunk.chunk_id)}, 长度={len(chunk.chunk_id)}")
            print(f"    doc_id类型={type(chunk.doc_id)}, 长度={len(chunk.doc_id)}")
            print(f"    content类型={type(chunk.content)}, 长度={len(chunk.content)}")
            print(f"    chunk_type类型={type(chunk.chunk_type)}, 长度={len(chunk.chunk_type)}")
        
        # ⭐ 优化：限制500字符（embedding max_tokens=512，中文约500字符安全）
        MAX_CHUNK_LEN = 450  # 与Milvus现有数据一致
        before_max = max(len(chunk.content) for chunk in all_chunks) if all_chunks else 0
        
        truncated_count = 0
        for i, chunk in enumerate(all_chunks):
            # 检查并同步chunk_texts
            if len(chunk.content) > MAX_CHUNK_LEN:
                print(f"⚠️ chunk {chunk.chunk_id} 超长({len(chunk.content)})，这不应该发生！")
                chunk.content = chunk.content[:MAX_CHUNK_LEN]
                truncated_count += 1
            chunk_texts[i] = chunk.content[:MAX_CHUNK_LEN]  # 同步
        
        after_max = max(len(chunk.content) for chunk in all_chunks) if all_chunks else 0
        
        print(f"[MilvusVectorDB] [Chunk级] 长度检查 (限制{MAX_CHUNK_LEN}字符，embedding max_tokens=512):")
        print(f"  - 最大长度: {before_max} 字符")
        if truncated_count > 0:
            print(f"  - ⚠️ 发现{truncated_count}个超长chunk（已截断）")
        
        # 使用GPU批量生成向量（归一化）
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        embeddings = self.model.encode(
            chunk_texts,
            device=device,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # ⭐ 归一化：方便计算相似度
        )
        
        print(f"[MilvusVectorDB] [Chunk级] ✅ 向量生成完成，shape: {embeddings.shape}")
        
        # ⭐⭐⭐ 最终检查：确保不超过450
        final_contents = [str(txt)[:450] for txt in chunk_texts]
        final_max = max(len(c) for c in final_contents) if final_contents else 0
        print(f"[MilvusVectorDB] [Chunk级] ✅ 最终content长度: max={final_max} (限制:450)")
        
        # 🔍 步骤2：检查chunk_texts列表
        print(f"\n[调试2] 检查辅助列表:")
        print(f"  chunk_texts长度: {len(chunk_texts)}")
        if chunk_texts:
            print(f"  chunk_texts[0]类型={type(chunk_texts[0])}, 长度={len(chunk_texts[0])}")
        if all_chunks:
            print(f"  all_chunks[0]元数据:")
            print(f"    - title: {all_chunks[0].title[:50]}...")
            print(f"    - timestamp: {all_chunks[0].timestamp}")
            print(f"    - industries: {all_chunks[0].industries[:50]}")
        
        # 准备插入数据（直接从all_chunks提取，包含完整元数据）
        print(f"\n[调试3] 构建entities数组...")
        
        # 🔥 关键：检查embeddings结构
        print(f"  embeddings类型: {type(embeddings)}")
        print(f"  embeddings shape: {embeddings.shape}")
        print(f"  embeddings[0]类型: {type(embeddings[0])}")
        print(f"  embeddings[0] shape/len: {embeddings[0].shape if hasattr(embeddings[0], 'shape') else len(embeddings[0])}")
        
        # 转换embeddings
        embeddings_list = embeddings.tolist()
        print(f"  embeddings_list类型: {type(embeddings_list)}")
        print(f"  embeddings_list[0]类型: {type(embeddings_list[0])}")
        print(f"  embeddings_list[0]长度: {len(embeddings_list[0])}")
        
        # ⭐ Milvus 的 VARCHAR(max_length=...) 是按“字节长度”限制（UTF-8），不是按字符数
        # 中文字符通常3字节：即使len(title)=185，也可能>500字节而插入失败。
        def _truncate_utf8(value: Any, max_bytes: int) -> str:
            s = "" if value is None else str(value)
            b = s.encode("utf-8")
            if len(b) <= max_bytes:
                return s
            # 逐步缩短直到字节长度<=max_bytes（保证不截断在UTF-8中间）
            cut = max_bytes
            while cut > 0:
                try:
                    return b[:cut].decode("utf-8")
                except UnicodeDecodeError:
                    cut -= 1
            return ""  # 极端情况

        entities = [
            [_truncate_utf8(chunk.chunk_id, 150) for chunk in all_chunks],  # chunk_id
            [_truncate_utf8(chunk.doc_id, 100) for chunk in all_chunks],  # doc_id
            embeddings_list,  # embedding
            [_truncate_utf8(chunk.content, 450) for chunk in all_chunks],  # content（限制450字符/字节；这里按字节截断更安全）
            [chunk.chunk_index for chunk in all_chunks],  # chunk_index
            [_truncate_utf8(chunk.chunk_type, 20) for chunk in all_chunks],  # chunk_type
            [_truncate_utf8(chunk.title, 500) for chunk in all_chunks],  # title（按字节）
            [_truncate_utf8(chunk.timestamp, 150) for chunk in all_chunks],  # timestamp
            [_truncate_utf8(chunk.industries, 500) for chunk in all_chunks],  # industries（按字节）
            [_truncate_utf8(chunk.investment_relevance, 10) for chunk in all_chunks],  # investment_relevance
            [_truncate_utf8(chunk.report_series, 50) for chunk in all_chunks],  # report_series
            [_truncate_utf8(chunk.industry_policy_segments, 20000) for chunk in all_chunks],  # industry_policy_segments
        ]
        
        # 🔍 步骤3：立即检查构建后的entities
        print(f"[调试3] entities构建完成，检查前3个元素:")
        for idx in range(10):
            if idx < len(entities) and entities[idx]:
                item = entities[idx][0] if entities[idx] else None
                if isinstance(item, str):
                    print(f"  entities[{idx}][0]: type=str, len={len(item)}")
                elif isinstance(item, list):
                    print(f"  entities[{idx}][0]: type=list, len={len(item)}")
                else:
                    print(f"  entities[{idx}][0]: type={type(item)}")
        
        # 🔍 终极修复：直接在entities数组中强制截断所有字段
        print(f"[MilvusVectorDB] [Chunk级] ===== 开始终极字段长度检查 =====")
        
        # 字段配置（索引，名称，最大长度）
        # entities数组顺序: 0=chunk_id, 1=doc_id, 2=embedding, 3=content, 4=chunk_index, 
        #                   5=chunk_type, 6=title, 7=timestamp, 8=industries, 
        #                   9=investment_relevance, 10=report_series, 11=industry_policy_segments
        string_fields_config = [
            (0, 'chunk_id', 150),
            (1, 'doc_id', 100),
            (3, 'content', 450),  # ⭐ 关键限制：embedding max_tokens=512，中文约450字符
            (5, 'chunk_type', 20),
            (6, 'title', 500),  # ⭐ 修复：title是索引6，不是7
            (7, 'timestamp', 150),
            (8, 'industries', 500),  # 中信一级行业
            (9, 'investment_relevance', 10),  # 投资相关性
            (10, 'report_series', 50),  # 报告系列
            (11, 'industry_policy_segments', 20000),  # 行业及对应政策片段
        ]
        
        total_truncated = 0
        for idx, name, max_len in string_fields_config:
            # 检查当前最大长度
            field = entities[idx]
            current_max = max(len(str(field[i])) for i in range(len(field))) if field else 0
            print(f"  [{idx}] {name:15s}: 最大 {current_max:6d} / 限制 {max_len:6d}", end="")
            
            # 如果超长，立即截断
            if current_max > max_len:
                print(f"  ⚠️ 超长！")
                truncated_count = 0
                for i in range(len(field)):
                    item_len = len(str(field[i]))
                    if item_len > max_len:
                        old_val = str(field[i])
                        # 打印前50字符的样本，找出真凶
                        if truncated_count == 0:
                            print(f"      🔍 第1个超长元素[{i}]:")
                            print(f"         类型: {type(field[i])}")
                            print(f"         长度: {item_len}")
                            print(f"         前100字符: {old_val[:100]}")
                        
                        field[i] = old_val[:max_len]  # 直接修改
                        truncated_count += 1
                
                print(f"      ✅ 截断了 {truncated_count} 个元素")
                total_truncated += truncated_count
                
                # 验证截断结果
                new_max = max(len(str(field[i])) for i in range(len(field)))
                print(f"      ✅ 新最大长度: {new_max}")
            else:
                print(f"  ✅")
        
        print(f"[MilvusVectorDB] [Chunk级] ===== 检查完成，共截断 {total_truncated} 个元素 =====")
        
        # 🔥🔥🔥 最终验证：插入前1秒再检查一次
        print(f"\n[MilvusVectorDB] [Chunk级] 🔥 插入前最终验证 🔥")
        print(f"  entities数组长度: {len(entities)}")
        print(f"  entities[2] (embedding) 类型: {type(entities[2])}")
        print(f"  entities[2] 元素数: {len(entities[2])}")
        if entities[2]:
            print(f"  entities[2][0] 类型: {type(entities[2][0])}")
            print(f"  entities[2][0] 长度: {len(entities[2][0]) if isinstance(entities[2][0], list) else 'N/A'}")
        
        for idx in [0, 1, 3, 5, 6, 7, 8, 9, 10, 11]:  # 所有VARCHAR字段
            if idx < len(entities):
                max_len_now = max(len(str(entities[idx][i])) for i in range(len(entities[idx])))
                print(f"  entities[{idx}] 当前最大长度: {max_len_now}")
        
        print(f"[MilvusVectorDB] [Chunk级] ✅ 最终验证完成\n")
        
        # 🔍 打印Milvus collection的实际schema
        print(f"\n[调试5] 检查Milvus schema:")
        schema = self.chunk_collection.schema
        for field in schema.fields:
            if field.dtype == DataType.VARCHAR:
                max_len = getattr(field, 'max_length', 'N/A')
                print(f"  {field.name}: VARCHAR(max_length={max_len})")
        
        # 🔍 步骤6：插入前终极检查 - 扫描所有元素找超长项
        print(f"\n[调试6] 插入前扫描所有元素（查找超长项）:")
        print(f"  ⚠️ 注意：Milvus报错'1th string'是第2个VARCHAR字段 = doc_id")
        print(f"  ⚠️ 如果doc_id收到content的值，说明entities顺序错了！")
        
        field_names_for_debug = ['chunk_id', 'doc_id', 'embedding', 'content', 'chunk_index', 
                                  'chunk_type', 'title', 'timestamp', 'industries', 'investment_relevance', 'report_series', 'industry_policy_segments']
        
        # 字段长度限制映射（Milvus按UTF-8字节计数）
        field_max_lengths = {
            0: 150,   # chunk_id
            1: 100,   # doc_id
            3: 450,   # content
            5: 20,    # chunk_type
            6: 500,   # title ⭐ 重要
            7: 150,   # timestamp
            8: 500,   # industries
            9: 10,    # investment_relevance
            10: 50,   # report_series
            11: 20000 # industry_policy_segments
        }
        
        # 完整检查并截断所有VARCHAR字段
        print(f"\n[调试6] 检查并按UTF-8字节截断所有VARCHAR字段:")
        for idx in [0, 1, 3, 5, 6, 7, 8, 9, 10, 11]:
            if idx >= len(entities):
                continue
            field_name = field_names_for_debug[idx] if idx < len(field_names_for_debug) else f'field_{idx}'
            field_data = entities[idx]
            max_allowed = field_max_lengths.get(idx, 500)
            
            # 找出所有长度（按UTF-8字节）
            lengths = [len(str(item).encode("utf-8")) for item in field_data]
            current_max = max(lengths) if lengths else 0
            
            # 找出并截断超长项
            truncated = 0
            for i, length in enumerate(lengths):
                if length > max_allowed:
                    field_data[i] = _truncate_utf8(field_data[i], max_allowed)
                    truncated += 1
            
            if truncated > 0:
                print(f"  [{idx}] {field_name}: 截断了 {truncated} 个元素 (限制{max_allowed})")
            else:
                print(f"  [{idx}] {field_name}: ✅ 无超长 (max={current_max}, 限制{max_allowed})")
        
        # 最终验证
        print(f"\n[调试7] 截断后验证:")
        for idx in [0, 1, 3, 5, 6, 7, 8, 9, 10, 11]:
            if idx >= len(entities):
                continue
            max_len = max(len(str(item).encode("utf-8")) for item in entities[idx])
            max_allowed = field_max_lengths.get(idx, 500)
            status = "✅" if max_len <= max_allowed else "❌"
            print(f"  entities[{idx}]: max={max_len} / 限制{max_allowed} {status}")
        
        # ⭐ 分批插入（避免gRPC消息大小限制：64MB）
        print(f"\n[MilvusVectorDB] [Chunk级] 🚀 开始分批插入数据...")
        print(f"  总共 {len(entities[0])} 个chunks")
        print(f"  entities数组结构: {len(entities)} 个字段")
        
        CHUNK_INSERT_BATCH = 500  # 每批500个chunks（约4MB）
        total_chunks = len(entities[0])
        total_inserted = 0
        
        # ⭐ 检查并截断 industry_policy_segments 字段（索引11）
        print(f"\n[MilvusVectorDB] [Chunk级] 检查 industry_policy_segments 字段长度...")
        max_segments_length = 20000  # 与Schema中的max_length一致
        truncated_count = 0
        for i in range(len(entities[11])):
            seg_str = str(entities[11][i])
            if len(seg_str) > max_segments_length:
                # 尝试智能截断：保留JSON结构
                try:
                    import json
                    seg_dict = json.loads(seg_str)
                    # 如果JSON太大，截断每个行业的政策片段列表
                    for industry, segments_list in seg_dict.items():
                        if isinstance(segments_list, list):
                            # 限制每个行业的片段数量，并截断每个片段长度
                            max_segments_per_industry = 20
                            max_segment_length = 200
                            seg_dict[industry] = [
                                seg[:max_segment_length] if len(seg) > max_segment_length else seg
                                for seg in segments_list[:max_segments_per_industry]
                            ]
                    # 重新序列化
                    new_seg_str = json.dumps(seg_dict, ensure_ascii=False)
                    if len(new_seg_str) > max_segments_length:
                        # 如果还是太长，直接截断
                        new_seg_str = new_seg_str[:max_segments_length-3] + "..."
                    entities[11][i] = new_seg_str
                    truncated_count += 1
                except:
                    # JSON解析失败，直接截断
                    entities[11][i] = seg_str[:max_segments_length-3] + "..."
                    truncated_count += 1
        
        if truncated_count > 0:
            print(f"  ⚠️ 已截断 {truncated_count} 个超长的 industry_policy_segments 字段")
        else:
            print(f"  ✅ 所有 industry_policy_segments 字段长度正常")
        
        for i in range(0, total_chunks, CHUNK_INSERT_BATCH):
            end_idx = min(i + CHUNK_INSERT_BATCH, total_chunks)
            
            # 准备批次数据 - 修复字段顺序匹配Schema
            batch_entities = [
                entities[0][i:end_idx],  # chunk_id
                entities[1][i:end_idx],  # doc_id
                entities[2][i:end_idx],  # embedding
                entities[3][i:end_idx],  # content
                entities[4][i:end_idx],  # chunk_index
                entities[5][i:end_idx],  # chunk_type
                entities[6][i:end_idx],  # title
                entities[7][i:end_idx],  # timestamp
                entities[8][i:end_idx],  # industries
                entities[9][i:end_idx],  # investment_relevance
                entities[10][i:end_idx],  # report_series
                entities[11][i:end_idx],  # industry_policy_segments
            ]
            
            # 插入批次（添加错误处理和断点续传提示）
            batch_num = i//CHUNK_INSERT_BATCH + 1
            total_batches = (total_chunks-1)//CHUNK_INSERT_BATCH + 1
            print(f"[MilvusVectorDB] [Chunk级] 插入批次 {batch_num}/{total_batches} ({end_idx-i} chunks)...")
            try:
                self.chunk_collection.insert(batch_entities)
                total_inserted += (end_idx - i)
            except Exception as e:
                print(f"[MilvusVectorDB] ❌ 批次 {batch_num} 插入失败: {e}")
                print(f"[MilvusVectorDB] 💡 提示: 已成功插入前 {total_inserted} 个chunks")
                print(f"[MilvusVectorDB] 💡 提示: 可以重新运行程序，已插入的数据不会重复（Milvus会自动去重）")
                raise e
        
        # 执行flush（数据持久化）- 添加超时保护
        # 注意：插入数据后不需要load集合，直接flush即可
        print(f"[MilvusVectorDB] [Chunk级] 🔄 正在flush数据到存储...")
        print(f"[MilvusVectorDB] [Chunk级] ⚠️ 如果长时间卡在此处，可能是Milvus rootcoord服务异常")
        
        import threading
        
        flush_success = [False]
        flush_error = [None]
        
        def flush_in_thread():
            try:
                # ⭐ flush操作：刷新数据到磁盘（可能失败，但不影响数据插入）
                try:
                    self.chunk_collection.flush()
                    print(f"[MilvusVectorDB] ✅ 数据已刷新到磁盘")
                except Exception as flush_error:
                    # flush失败不影响数据插入，数据已经在Milvus中
                    error_msg = str(flush_error)
                    if "channel not found" in error_msg or "rootcoord" in error_msg:
                        print(f"[MilvusVectorDB] ⚠️ flush操作失败（Milvus服务内部错误），但数据已成功插入")
                        print(f"[MilvusVectorDB] 💡 建议：如果频繁出现此错误，请重启Milvus服务")
                    else:
                        print(f"[MilvusVectorDB] ⚠️ flush操作失败: {flush_error}")
                    # 不抛出异常，因为数据已经插入成功
                flush_success[0] = True
            except Exception as e:
                flush_error[0] = e
        
        flush_thread = threading.Thread(target=flush_in_thread, daemon=True)
        flush_thread.start()
        flush_thread.join(timeout=60)  # 60秒超时
        
        if flush_thread.is_alive():
            print(f"[MilvusVectorDB] [Chunk级] ⚠️ Flush超时（60秒），跳过flush继续执行")
            print(f"[MilvusVectorDB] [Chunk级] 💡 可能原因：Milvus rootcoord服务异常")
            print(f"[MilvusVectorDB] [Chunk级] 💡 排查步骤：")
            print(f"   1. 检查Milvus容器状态: docker ps | grep milvus")
            print(f"   2. 查看Milvus日志: docker logs milvus-standalone --tail 50")
            print(f"   3. 重启Milvus服务: docker restart milvus-standalone")
            print(f"[MilvusVectorDB] [Chunk级] ⚠️ 数据已插入但未flush，Milvus会在后台自动flush")
            # 确保集合被加载到内存（带重试机制）
            print(f"[MilvusVectorDB] [Chunk级] 🔄 尝试加载集合到内存...")
            max_retries = 3
            for retry in range(max_retries):
                try:
                    # 使用线程+超时避免卡死
                    load_done = [False]
                    load_err = [None]
                    
                    def do_load():
                        try:
                            self.chunk_collection.load()
                            load_done[0] = True
                        except Exception as e:
                            load_err[0] = e
                    
                    load_t = threading.Thread(target=do_load, daemon=True)
                    load_t.start()
                    load_t.join(timeout=30)  # 30秒超时
                    
                    if load_done[0]:
                        print(f"[MilvusVectorDB] [Chunk级] ✅ 集合已加载到内存")
                        break
                    elif load_t.is_alive():
                        print(f"[MilvusVectorDB] [Chunk级] ⚠️ 加载超时（尝试 {retry+1}/{max_retries}）")
                        if retry < max_retries - 1:
                            print(f"[MilvusVectorDB] [Chunk级] 🔄 等待5秒后重试...")
                            import time
                            time.sleep(5)
                    else:
                        print(f"[MilvusVectorDB] [Chunk级] ⚠️ 加载失败: {load_err[0]} （尝试 {retry+1}/{max_retries}）")
                        if retry < max_retries - 1:
                            print(f"[MilvusVectorDB] [Chunk级] 🔄 等待5秒后重试...")
                            import time
                            time.sleep(5)
                except Exception as e:
                    print(f"[MilvusVectorDB] [Chunk级] ⚠️ 加载异常: {e} （尝试 {retry+1}/{max_retries}）")
            # 不抛出异常，允许程序继续
        elif flush_error[0]:
            error_msg = str(flush_error[0])
            if "channel not found" in error_msg.lower():
                print(f"[MilvusVectorDB] [Chunk级] ⚠️ Flush失败：Milvus rootcoord服务异常")
                print(f"[MilvusVectorDB] [Chunk级] 💡 可能原因：")
                print(f"   1. Milvus rootcoord服务未正常运行")
                print(f"   2. Milvus服务状态异常")
                print(f"[MilvusVectorDB] [Chunk级] 💡 排查步骤：")
                print(f"   1. 检查Milvus容器状态: docker ps | grep milvus")
                print(f"   2. 查看Milvus日志: docker logs milvus-standalone --tail 50")
                print(f"   3. 重启Milvus服务: docker restart milvus-standalone")
                print(f"[MilvusVectorDB] [Chunk级] ⚠️ 数据已插入但未flush，Milvus会在后台自动flush")
                # 确保集合被加载到内存（带重试机制）
                print(f"[MilvusVectorDB] [Chunk级] 🔄 尝试加载集合到内存...")
                max_retries = 3
                for retry in range(max_retries):
                    try:
                        load_done = [False]
                        load_err = [None]
                        def do_load():
                            try:
                                self.chunk_collection.load()
                                load_done[0] = True
                            except Exception as e:
                                load_err[0] = e
                        load_t = threading.Thread(target=do_load, daemon=True)
                        load_t.start()
                        load_t.join(timeout=30)
                        if load_done[0]:
                            print(f"[MilvusVectorDB] [Chunk级] ✅ 集合已加载到内存")
                            break
                        elif load_t.is_alive():
                            print(f"[MilvusVectorDB] [Chunk级] ⚠️ 加载超时（尝试 {retry+1}/{max_retries}）")
                            if retry < max_retries - 1:
                                import time
                                time.sleep(5)
                        else:
                            print(f"[MilvusVectorDB] [Chunk级] ⚠️ 加载失败: {load_err[0]} （尝试 {retry+1}/{max_retries}）")
                            if retry < max_retries - 1:
                                import time
                                time.sleep(5)
                    except Exception as e:
                        print(f"[MilvusVectorDB] [Chunk级] ⚠️ 加载异常: {e} （尝试 {retry+1}/{max_retries}）")
                # 不抛出异常，允许程序继续（数据可能已经部分持久化）
            else:
                print(f"[MilvusVectorDB] [Chunk级] ⚠️ Flush失败: {flush_error[0]}")
                print(f"[MilvusVectorDB] [Chunk级] ⚠️ 数据已插入但未flush，继续执行")
                # 确保集合被加载到内存（带重试机制）
                print(f"[MilvusVectorDB] [Chunk级] 🔄 尝试加载集合到内存...")
                max_retries = 3
                for retry in range(max_retries):
                    try:
                        load_done = [False]
                        load_err = [None]
                        def do_load():
                            try:
                                self.chunk_collection.load()
                                load_done[0] = True
                            except Exception as e:
                                load_err[0] = e
                        load_t = threading.Thread(target=do_load, daemon=True)
                        load_t.start()
                        load_t.join(timeout=30)
                        if load_done[0]:
                            print(f"[MilvusVectorDB] [Chunk级] ✅ 集合已加载到内存")
                            break
                        elif load_t.is_alive():
                            print(f"[MilvusVectorDB] [Chunk级] ⚠️ 加载超时（尝试 {retry+1}/{max_retries}）")
                            if retry < max_retries - 1:
                                import time
                                time.sleep(5)
                        else:
                            print(f"[MilvusVectorDB] [Chunk级] ⚠️ 加载失败: {load_err[0]} （尝试 {retry+1}/{max_retries}）")
                            if retry < max_retries - 1:
                                import time
                                time.sleep(5)
                    except Exception as e:
                        print(f"[MilvusVectorDB] [Chunk级] ⚠️ 加载异常: {e} （尝试 {retry+1}/{max_retries}）")
                # 不抛出异常，允许程序继续
        elif flush_success[0]:
            print(f"[MilvusVectorDB] [Chunk级] ✅ 数据已flush到存储")
        
        print(f"[MilvusVectorDB] [Chunk级] ✅ 插入完成，共 {total_inserted} 个chunks")
    
    def search_similar(self, query_text: str = None, query_segment: PolicySegment = None,
                      top_k: int = 20, where_filter: Dict = None) -> List[Dict[str, Any]]:
        """
        向量相似度搜索 (GPU加速)
        
        Args:
            query_text: 查询文本
            query_segment: 查询文档
            top_k: 返回结果数量
            where_filter: 过滤条件
            
        Returns:
            相似文档列表
        """
        # 准备查询文本（优化：跳过格式内容，提取实质部分）
        if query_segment:
            # 标题权重：重复3次
            title_part = f"{query_segment.title}\n{query_segment.title}\n{query_segment.title}"
            
            # 提取实质内容（跳过前面的格式部分）
            content = query_segment.content
            start_pos = 200  # 默认跳过前200字（发文单位、文号等）
            for marker in ['第一条', '一、', '（一）', '1.', '总则', '第一章']:
                pos = content.find(marker)
                if pos > 0 and pos < 500:
                    start_pos = pos
                    break
            
            content_part = content[start_pos:start_pos+1000]
            query_text = f"{title_part}\n\n{content_part}"
        
        if not query_text:
            return []
        
        # 生成查询向量
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        query_embedding = self.model.encode(
            [query_text],
            device=device,
            convert_to_numpy=True,
            normalize_embeddings=True  # ⭐ 归一化
        )[0]
        
        # 搜索参数（ef必须>=top_k，设置为512保证足够大）
        search_params = {"metric_type": "L2", "params": {"ef": 512}}
        
        # 构建过滤表达式（只用时间过滤，避免漏检）
        expr = None
        if where_filter:
            filter_parts = []
            
            # ⭐ 核心：时间范围过滤
            if 'timestamp_after' in where_filter:
                filter_parts.append(f'timestamp >= "{where_filter["timestamp_after"]}"')
            
            if 'timestamp_before' in where_filter:
                filter_parts.append(f'timestamp <= "{where_filter["timestamp_before"]}"')
            
            if filter_parts:
                expr = ' && '.join(filter_parts)
        
        # 注意：不使用行业等字段进行过滤
        # 原因：避免因分类错误导致漏检，提高召回率
        
        # 执行搜索
        results = self.collection.search(
            data=[query_embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr,  # 时间过滤表达式
            output_fields=["doc_id", "title", "timestamp", "industries"]
            )
            
            # 格式化结果
        similar_docs = []
        for hit in results[0]:
            # 计算相似度得分（归一化向量的L2距离转换为余弦相似度）
            # 对于归一化向量: cosine_similarity = 1 - (L2_distance^2 / 2)
            # L2距离范围: [0, 2]，其中0=完全相同，2=完全相反
            l2_distance = hit.distance
            cosine_similarity = 1.0 - (l2_distance ** 2) / 2.0
            # 确保在[0, 1]范围内
            similarity = max(0.0, min(1.0, cosine_similarity))
            
            similar_docs.append({
                'doc_id': hit.entity.get('doc_id'),
                'title': hit.entity.get('title'),
                'timestamp': hit.entity.get('timestamp'),         # ⭐ 核心：用于时间跨度计算
                'industries': hit.entity.get('industries'),       # ⭐ 核心：用于行业过滤
                'distance': l2_distance,                          # L2距离（越小越相似，范围0-2）
                'similarity': similarity,                         # 余弦相似度（0-1，越大越相似）
                'score': similarity,                              # ⭐ 统一字段名：score
                    })
            
            return similar_docs
    
    def search_by_doc(self, doc_id: str, top_k: int = 5, exclude_self: bool = True) -> List[Dict[str, Any]]:
        """
        根据文档ID搜索相似文档
        
        Args:
            doc_id: 文档ID
            top_k: 返回结果数量
            exclude_self: 是否排除自己
            
        Returns:
            相似文档列表
        """
        # 查询文档
        query_result = self.collection.query(
            expr=f'doc_id == "{doc_id}"',
            output_fields=["embedding"]
        )
        
        if not query_result:
                return []
            
        # 获取向量
        query_embedding = query_result[0]['embedding']
        
        # 搜索（ef设置为512保证足够大）
        search_params = {"metric_type": "L2", "params": {"ef": 512}}
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k + (1 if exclude_self else 0),
            output_fields=["doc_id", "title", "timestamp", "industries"]
        )
            
            # 格式化结果
        similar_docs = []
        for hit in results[0]:
                    # 排除自己
            if exclude_self and hit.entity.get('doc_id') == doc_id:
                        continue
                    
            similar_docs.append({
                'doc_id': hit.entity.get('doc_id'),
                'title': hit.entity.get('title'),
                'timestamp': hit.entity.get('timestamp'),         # ⭐ 核心
                'industries': hit.entity.get('industries'),       # ⭐ 核心
                'distance': hit.distance,
                'similarity': 1 / (1 + hit.distance),
                    })
            
            return similar_docs[:top_k]
            
    def search_with_dual_layer(
        self,
        query_text: str = None,
        query_segment: PolicySegment = None,
        query_timestamp: str = None,
        top_k_docs: int = 200,  # ⭐ 增加到200，提高召回率
        top_k_chunks: int = 50,  # ⭐ 增加到50个chunk
        enable_time_filter: bool = True,
        enable_time_weighting: bool = True,
        enable_industry_boost: bool = True  # ⭐ 启用行业匹配加权
    ) -> Dict[str, Any]:
        """
        双层检索：文档级粗排 + Chunk级精排
        
        Args:
            query_text: 查询文本
            query_segment: 查询文档
            query_timestamp: 查询时间戳（用于时间过滤），格式：YYYY-MM-DD
            top_k_docs: 文档级返回数量
            top_k_chunks: Chunk级返回数量
            enable_time_filter: 是否启用时间过滤（只检索历史文档）
            enable_time_weighting: 是否启用时间加权
            
        Returns:
            {'documents': [...], 'chunks': [...], 'query_info': {...}}
        """
        if not self.enable_chunking:
            # 如果未启用chunking，只返回文档级检索
            docs = self.search_similar(
                query_text=query_text,
                query_segment=query_segment,
                top_k=top_k_docs
            )
            return {'documents': docs, 'chunks': [], 'query_info': {'chunking_enabled': False}}
        
        # 准备查询文本（优化：提取实质内容）
        if query_segment:
            # 标题权重：重复3次
            title_part = f"{query_segment.title}\n{query_segment.title}\n{query_segment.title}"
            
            # 提取实质内容（跳过格式部分）
            content = query_segment.content
            start_pos = 200
            for marker in ['第一条', '一、', '（一）', '1.', '总则', '第一章']:
                pos = content.find(marker)
                if pos > 0 and pos < 500:
                    start_pos = pos
                    break
            
            content_part = content[start_pos:start_pos+1000]
            query_text = f"{title_part}\n\n{content_part}"
            
            if not query_timestamp:
                query_timestamp = query_segment.timestamp.isoformat()
        
        if not query_text:
            return {'documents': [], 'chunks': [], 'query_info': {'error': 'No query text'}}
        
        print(f"[MilvusVectorDB] [双层检索] 开始...")
        print(f"[MilvusVectorDB]   - 查询: {query_text[:50]}...")
        if query_timestamp:
            print(f"[MilvusVectorDB]   - 时间: {query_timestamp}")
        
        # 生成查询向量
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        query_embedding = self.model.encode(
            [query_text],
            device=device,
            convert_to_numpy=True,
            normalize_embeddings=True  # ⭐ 归一化
        )[0]
        
        # === 步骤1: 文档级检索（粗排） ===
        print(f"[MilvusVectorDB] [步骤1] 文档级检索（粗排）...")
        
        # 构建时间过滤表达式
        expr = None
        if enable_time_filter and query_timestamp:
            expr = f'timestamp < "{query_timestamp}"'
            print(f"[MilvusVectorDB]   - 时间过滤: {expr}")
        
        # ⭐ ef必须>=top_k_docs（200），设置为512保证足够
        search_params = {"metric_type": "L2", "params": {"ef": 512}}
        doc_results = self.collection.search(
            data=[query_embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=top_k_docs,
            expr=expr,  # 时间过滤表达式
            output_fields=["doc_id", "title", "timestamp", "industries"]
        )
        
        # 格式化文档级结果
        documents = []
        doc_ids = []
        for hit in doc_results[0]:
            doc_id = hit.entity.get('doc_id')
            doc_ids.append(doc_id)
            doc_timestamp = hit.entity.get('timestamp')
            
            # 计算时间权重
            time_weight = 1.0
            time_diff_days = None
            if enable_time_weighting and query_timestamp and doc_timestamp:
                try:
                    query_date = datetime.fromisoformat(query_timestamp)
                    doc_date = datetime.fromisoformat(doc_timestamp)
                    time_diff_days = (query_date - doc_date).days
                    
                    if time_diff_days <= 90:
                        time_weight = 1.2
                    elif time_diff_days <= 180:
                        time_weight = 1.0
                    elif time_diff_days <= 365:
                        time_weight = 0.8
                    else:
                        time_weight = 0.6
                except:
                    pass
            
            similarity = 1 / (1 + hit.distance)
            weighted_score = similarity * time_weight
            
            documents.append({
                'doc_id': doc_id,
                'title': hit.entity.get('title'),
                'timestamp': doc_timestamp,                       # ⭐ 核心：时间跨度计算
                'industries': hit.entity.get('industries'),       # ⭐ 核心：行业过滤
                'distance': hit.distance,
                'similarity': similarity,
                'time_weight': time_weight,
                'weighted_score': weighted_score,
                'time_diff_days': time_diff_days,
            })
        
        # 按加权得分重新排序
        if enable_time_weighting:
            documents.sort(key=lambda x: x['weighted_score'], reverse=True)
        
        print(f"[MilvusVectorDB] [步骤1] ✅ 找到 {len(documents)} 个相关文档")
        
        # === 步骤2: Chunk级检索（精排） ===
        print(f"[MilvusVectorDB] [步骤2] Chunk级检索（精排）...")
        
        if not doc_ids:
            return {
                'documents': documents,
                'chunks': [],
                'query_info': {
                    'query_text': query_text[:100],
                    'query_timestamp': query_timestamp,
                    'time_filter_enabled': enable_time_filter,
                    'time_weighting_enabled': enable_time_weighting,
                }
            }
        
        # 构建Chunk过滤表达式
        doc_ids_str = '", "'.join(doc_ids)
        chunk_expr = f'doc_id in ["{doc_ids_str}"]'
        
        if enable_time_filter and query_timestamp:
            chunk_expr += f' && timestamp < "{query_timestamp}"'
        
        print(f"[MilvusVectorDB]   - 过滤条件: doc_id in top-{len(doc_ids)}")
        
        # 执行Chunk级检索
        chunk_results = self.chunk_collection.search(
            data=[query_embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=top_k_chunks,
            expr=chunk_expr,
            output_fields=["chunk_id", "doc_id", "content", "chunk_index", "chunk_type", "timestamp", "industries", "investment_relevance", "report_series", "industry_policy_segments"]
        )
        
        # 格式化Chunk级结果
        chunks = []
        for hit in chunk_results[0]:
            chunks.append({
                'chunk_id': hit.entity.get('chunk_id'),
                'doc_id': hit.entity.get('doc_id'),
                'content': hit.entity.get('content'),
                'chunk_index': hit.entity.get('chunk_index'),
                'chunk_type': hit.entity.get('chunk_type'),
                'timestamp': hit.entity.get('timestamp'),         # ⭐ 核心
                'industries': hit.entity.get('industries'),       # ⭐ 核心
                'investment_relevance': hit.entity.get('investment_relevance'),
                'report_series': hit.entity.get('report_series'),  # ⭐ 报告系列
                'industry_policy_segments': hit.entity.get('industry_policy_segments'),
                'distance': hit.distance,
                'similarity': 1 / (1 + hit.distance),
            })
        
        print(f"[MilvusVectorDB] [步骤2] ✅ 找到 {len(chunks)} 个相关段落")
        print(f"[MilvusVectorDB] [双层检索] ✅ 完成")
        
        return {
            'documents': documents[:top_k_docs],
            'chunks': chunks[:top_k_chunks],
            'query_info': {
                'query_text': query_text[:100],
                'query_timestamp': query_timestamp,
                'time_filter_enabled': enable_time_filter,
                'time_weighting_enabled': enable_time_weighting,
                'total_docs_found': len(documents),
                'total_chunks_found': len(chunks),
            }
        }
    
    def clear(self):
        """清空Milvus向量库（包括文档级和Chunk级）"""
        try:
            # 清空文档级集合
            utility.drop_collection(self.collection_name)
            print(f"[MilvusVectorDB] ✅ 文档级集合 {self.collection_name} 已删除")
            
            # 清空Chunk级集合
            if self.enable_chunking and utility.has_collection(self.chunk_collection_name):
                utility.drop_collection(self.chunk_collection_name)
                print(f"[MilvusVectorDB] ✅ Chunk级集合 {self.chunk_collection_name} 已删除")
            
            # 重新创建
            self._init_collection()
            if self.enable_chunking:
                self._init_chunk_collection()
                
            print(f"[MilvusVectorDB] ✅ 集合已重新创建")
        except Exception as e:
            print(f"[MilvusVectorDB] ❌ 清空失败: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取向量库统计信息（只使用chunk级别）"""
        stats = {
            'total_documents': 0,  # 简化版：没有文档级数据
            'collection_name': self.chunk_collection_name,
            'embedding_dim': self.embedding_dim,
            'gpu_enabled': torch.cuda.is_available(),
            'gpu_device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
            'chunking_enabled': self.enable_chunking,
            'status': '运行中（简化版：只使用chunk级别）'
        }
        
        if self.enable_chunking:
            stats['total_chunks'] = self.chunk_collection.num_entities
            stats['chunk_collection_name'] = self.chunk_collection_name
            stats['avg_chunks_per_doc'] = 0  # 简化版：无法计算平均值
        
        return stats
    
    def search_chunks(self, query_text: str, top_k: int = 500, rerank_top_k: int = None, exclude_doc_id: str = None, exclude_title: str = None, exclude_timestamp = None, before_timestamp = None, after_timestamp = None, allow_same_day: bool = False, use_reranker: bool = True) -> List[Dict[str, Any]]:
        """
        搜索chunk级别向量（简化版RAG）+ 可选Reranking精排
        
        Args:
            query_text: 查询文本
            top_k: 粗排召回数量（默认500）
            rerank_top_k: 精排返回数量（如果为None，则使用top_k）
            exclude_doc_id: 要排除的doc_id（避免匹配到自己）
            exclude_title: 要排除的标题（与exclude_timestamp配合使用，过滤标题+时间都相同的文档）
            exclude_timestamp: 要排除的发文时间（与exclude_title配合使用）
            before_timestamp: 时间约束，只检索早于此时间的文档（粗排阶段过滤）
            after_timestamp: 时间约束，只检索晚于此时间的文档（用于限制时间窗口，如只检索2年内的政策）
            allow_same_day: 是否允许同一天的文档（默认False，严格早于）
            use_reranker: 是否使用Reranker二阶段精排（默认True，提升精度）
            
        Returns:
            chunk搜索结果列表（如果启用reranker，会添加'rerank_score'字段）
        """
        if not self.enable_chunking:
            print(f"[MilvusVectorDB] ⚠️ Chunk搜索未启用")
            return []
        
        # ⭐ 粗排和精排数量设置
        if rerank_top_k is None:
            rerank_top_k = top_k
        
        retrieval_top_k = top_k  # 粗排召回数量
        final_top_k = rerank_top_k if use_reranker else top_k  # 精排后返回数量
        
        print(f"[MilvusVectorDB] 搜索chunks: '{query_text[:50]}...'")
        if use_reranker:
            print(f"[MilvusVectorDB]   - 粗排: 召回top-{retrieval_top_k} 候选")
            print(f"[MilvusVectorDB]   - 精排: Reranker筛选top-{final_top_k}")
        
        # 生成查询向量
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        query_embedding = self.model.encode(
            [query_text],
            device=device,
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]
        
        # 确保集合已加载（搜索前必须加载）
        try:
            # 尝试查询，如果能查询说明已加载
            self.chunk_collection.query(expr="id >= 0", limit=1, output_fields=["id"])
        except:
            # 未加载，执行加载
            print(f"[MilvusVectorDB] [搜索] 🔄 集合未加载，正在加载...")
            try:
                import threading
                
                load_success = [False]
                load_error = [None]
                
                def load_in_thread():
                    try:
                        self.chunk_collection.load()
                        load_success[0] = True
                    except Exception as e:
                        load_error[0] = e
                
                load_thread = threading.Thread(target=load_in_thread, daemon=True)
                load_thread.start()
                load_thread.join(timeout=30)  # 30秒超时
                
                if load_thread.is_alive():
                    print(f"[MilvusVectorDB] [搜索] ⚠️ 加载超时（30秒），尝试继续搜索（新插入的数据可能在内存中）")
                    print(f"[MilvusVectorDB] [搜索] 💡 如果搜索失败，可能是Milvus服务异常")
                    # 不抛出异常，尝试继续搜索（新插入的数据可能在内存中）
                elif load_error[0]:
                    error_msg = str(load_error[0])
                    if "not loaded" in error_msg.lower() or "collection not loaded" in error_msg.lower():
                        print(f"[MilvusVectorDB] [搜索] ⚠️ 集合加载失败，尝试继续搜索（新插入的数据可能在内存中）")
                        # 不抛出异常，尝试继续搜索
                    else:
                        print(f"[MilvusVectorDB] [搜索] ⚠️ 集合加载失败: {load_error[0]}，尝试继续搜索")
                        # 不抛出异常，尝试继续搜索
                elif load_success[0]:
                    print(f"[MilvusVectorDB] [搜索] ✅ 集合已加载")
            except Exception as load_error:
                print(f"[MilvusVectorDB] [搜索] ⚠️ 集合加载过程出错: {load_error}")
                print(f"[MilvusVectorDB] [搜索] 💡 尝试继续搜索（新插入的数据可能在内存中）")
                # 不抛出异常，尝试继续搜索
        
        # 搜索参数 - 优化：增加nprobe提高搜索精度
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 20}  # 从10增加到20，提高搜索精度
        }
        
        # ⭐ 构建Milvus过滤表达式（时间过滤在最前面！）
        # 这样Milvus只返回符合时间条件的结果，避免浪费计算
        filter_expr_parts = []
        
        # before_timestamp: 只检索早于此时间的文档
        if before_timestamp:
            try:
                from datetime import datetime
                if hasattr(before_timestamp, 'isoformat'):
                    ts_str = before_timestamp.isoformat()
                elif isinstance(before_timestamp, str):
                    ts_str = before_timestamp
                else:
                    ts_str = str(before_timestamp)
                
                # Milvus字符串比较：timestamp < "2025-01-01"
                if allow_same_day:
                    # 允许同一天：timestamp <= "2025-01-01"
                    filter_expr_parts.append(f'timestamp <= "{ts_str[:10]}"')
                else:
                    # 严格早于：timestamp < "2025-01-01"
                    filter_expr_parts.append(f'timestamp < "{ts_str[:10]}"')
            except Exception as e:
                print(f"[MilvusVectorDB] ⚠️ 构建before_timestamp过滤表达式失败: {e}")
        
        # after_timestamp: 只检索晚于此时间的文档（用于限制时间窗口）
        if after_timestamp:
            try:
                from datetime import datetime
                if hasattr(after_timestamp, 'isoformat'):
                    after_ts_str = after_timestamp.isoformat()
                elif isinstance(after_timestamp, str):
                    after_ts_str = after_timestamp
                else:
                    after_ts_str = str(after_timestamp)
                
                # Milvus字符串比较：timestamp >= "2023-01-01"
                filter_expr_parts.append(f'timestamp >= "{after_ts_str[:10]}"')
            except Exception as e:
                print(f"[MilvusVectorDB] ⚠️ 构建after_timestamp过滤表达式失败: {e}")
        
        # 合并过滤表达式
        filter_expr = ' && '.join(filter_expr_parts) if filter_expr_parts else None
        
        if filter_expr:
            print(f"[MilvusVectorDB] ⭐ Milvus层面时间过滤: {filter_expr}")
        
        # 执行搜索（带时间过滤）
        results = self.chunk_collection.search(
            data=[query_embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=retrieval_top_k,  # ⭐ 使用调整后的召回数量
            expr=filter_expr,  # ⭐ 时间过滤在Milvus层面进行！
            output_fields=["chunk_id", "doc_id", "content", "chunk_index", "chunk_type", "title", "timestamp", "industries", "investment_relevance", "report_series", "industry_policy_segments"]
        )
        
        # ⭐ 预处理排除条件：标题+时间（用于排除新政策自身）
        exclude_timestamp_date = None
        if exclude_title and exclude_timestamp:
            try:
                from datetime import datetime
                if hasattr(exclude_timestamp, 'date'):
                    exclude_timestamp_date = exclude_timestamp.date()
                elif isinstance(exclude_timestamp, str):
                    if 'T' in exclude_timestamp:
                        exclude_timestamp_date = datetime.fromisoformat(exclude_timestamp.replace('Z', '+00:00')).date()
                    else:
                        exclude_timestamp_date = datetime.fromisoformat(exclude_timestamp).date()
            except:
                pass
        
        # 格式化结果
        # ⭐ 注意：时间过滤已在Milvus层面完成（filter_expr），这里只需要排除自身
        formatted_results = []
        excluded_by_title_time = 0
        for hits in results:
            for hit in hits:
                doc_id = hit.entity.get('doc_id')
                hit_timestamp = hit.entity.get('timestamp', '')
                
                # 过滤掉exclude_doc_id
                if exclude_doc_id and doc_id == exclude_doc_id:
                    continue
                
                # ⭐ 过滤掉标题+发文时间都相同的文档（排除新政策自身）
                if exclude_title and exclude_timestamp_date:
                    hit_title = hit.entity.get('title', '')
                    if hit_title == exclude_title:
                        try:
                            from datetime import datetime
                            if hit_timestamp:
                                if 'T' in str(hit_timestamp):
                                    hit_date = datetime.fromisoformat(str(hit_timestamp).replace('Z', '+00:00')).date()
                                else:
                                    hit_date = datetime.fromisoformat(str(hit_timestamp)).date()
                                if hit_date == exclude_timestamp_date:
                                    excluded_by_title_time += 1
                                    continue  # 标题+时间都相同，跳过（是新政策自身）
                        except:
                            pass
                
                # ⭐ 对于归一化向量：L2距离范围[0, 2]，转换为余弦相似度[0, 1]
                l2_distance = hit.distance
                cosine_similarity = 1.0 - (l2_distance ** 2) / 2.0
                similarity = max(0.0, min(1.0, cosine_similarity))  # 确保在[0, 1]范围内
                
                formatted_results.append({
                    'chunk_id': hit.entity.get('chunk_id'),
                    'doc_id': doc_id,
                    'content': hit.entity.get('content'),
                    'similarity': similarity,  # 余弦相似度（0-1，越大越相似）
                    'chunk_index': hit.entity.get('chunk_index'),
                    'chunk_type': hit.entity.get('chunk_type'),
                    'title': hit.entity.get('title'),
                    'timestamp': hit_timestamp,
                    'industries': hit.entity.get('industries')
                })
        
        exclude_msg = f"（已排除doc_id={exclude_doc_id}）" if exclude_doc_id else ""
        if excluded_by_title_time > 0:
            exclude_msg += f"（已排除{excluded_by_title_time}个标题+时间相同的chunks）"
        time_filter_msg = f"（时间过滤已在Milvus层面完成）" if filter_expr else ""
        print(f"[MilvusVectorDB] ✅ 粗排完成: {len(formatted_results)} 个相关chunks{exclude_msg}{time_filter_msg}")
        
        # ⭐ 二阶段精排（Reranking）
        if use_reranker and len(formatted_results) > final_top_k:
            try:
                from utils.reranker import get_reranker
                reranker = get_reranker()
                formatted_results = reranker.rerank(query_text, formatted_results, top_k=final_top_k)
            except Exception as e:
                print(f"[MilvusVectorDB] ⚠️ Reranking失败: {e}，返回向量检索结果")
                formatted_results = formatted_results[:final_top_k]
        else:
            formatted_results = formatted_results[:final_top_k]
        
        return formatted_results
    
    def search_chunks_multi_query(self, query_chunks: List[str], top_k_per_query: int = 10, exclude_doc_id: str = None, use_reranker: bool = True, final_top_k: int = None) -> List[Dict[str, Any]]:
        """
        精细化搜索：对多个query chunks分别进行向量搜索，然后合并结果 + 可选全局Reranking
        
        这种方法确保：
        1. 每个query chunk都能找到最匹配的数据库chunks
        2. 匹配更精确，因为粒度一致（chunk vs chunk）
        3. 避免了长文档查询时的语义稀释问题
        4. 全局Reranking：合并后对所有候选统一精排
        
        Args:
            query_chunks: 查询chunks列表（已切分的查询文本）
            top_k_per_query: 每个query chunk返回的最相似chunk数量
            exclude_doc_id: 要排除的doc_id（避免匹配到自己）
            use_reranker: 是否对合并后的结果进行全局Reranking
            final_top_k: 最终返回的结果数量（如果为None，则返回所有合并后的结果）
            
        Returns:
            合并后的chunk搜索结果列表（已去重，按相似度或rerank_score排序）
        """
        if not self.enable_chunking:
            print(f"[MilvusVectorDB] ⚠️ Chunk搜索未启用")
            return []
        
        if not query_chunks:
            return []
        
        print(f"[MilvusVectorDB] 精细化搜索: {len(query_chunks)} 个query chunks，每个返回top_{top_k_per_query}")
        
        # 批量生成查询向量（GPU加速）
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        query_embeddings = self.model.encode(
            query_chunks,
            device=device,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False
        )
        
        # 搜索参数
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 20}
        }
        
        # 对每个query chunk执行搜索
        all_results = []
        chunk_id_set = set()  # 用于去重
        
        for i, query_embedding in enumerate(query_embeddings):
            results = self.chunk_collection.search(
                data=[query_embedding.tolist()],
                anns_field="embedding",
                param=search_params,
                limit=top_k_per_query,
                output_fields=["chunk_id", "doc_id", "content", "chunk_index", "chunk_type", "title", "timestamp", "industries", "investment_relevance", "report_series", "industry_policy_segments"]
            )
            
            for hits in results:
                for hit in hits:
                    chunk_id = hit.entity.get('chunk_id')
                    doc_id = hit.entity.get('doc_id')
                    
                    # 过滤掉exclude_doc_id
                    if exclude_doc_id and doc_id == exclude_doc_id:
                        continue
                    
                    # 去重：如果同一个chunk被多个query chunk匹配到，保留相似度更高的
                    # ⭐ 对于归一化向量：L2距离范围[0, 2]，转换为余弦相似度[0, 1]
                    l2_distance = hit.distance
                    cosine_similarity = 1.0 - (l2_distance ** 2) / 2.0
                    similarity = max(0.0, min(1.0, cosine_similarity))  # 确保在[0, 1]范围内
                    
                    if chunk_id not in chunk_id_set:
                        chunk_id_set.add(chunk_id)
                        all_results.append({
                            'chunk_id': chunk_id,
                            'doc_id': doc_id,
                            'content': hit.entity.get('content'),
                            'similarity': similarity,
                            'chunk_index': hit.entity.get('chunk_index'),
                            'chunk_type': hit.entity.get('chunk_type'),
                            'title': hit.entity.get('title'),
                            'timestamp': hit.entity.get('timestamp'),
                            'industries': hit.entity.get('industries'),
                            'investment_relevance': hit.entity.get('investment_relevance'),
                            'report_series': hit.entity.get('report_series'),  # ⭐ 报告系列
                            'industry_policy_segments': hit.entity.get('industry_policy_segments'),
                            'matched_by_query_chunk': i  # 记录是哪个query chunk匹配到的
                        })
                    else:
                        # 如果已存在，检查是否需要更新相似度（保留更高的）
                        for existing in all_results:
                            if existing['chunk_id'] == chunk_id and similarity > existing['similarity']:
                                existing['similarity'] = similarity
                                existing['matched_by_query_chunk'] = i
                                break
        
        # 按相似度排序
        all_results.sort(key=lambda x: x['similarity'], reverse=True)
        
        exclude_msg = f"（已排除doc_id={exclude_doc_id}）" if exclude_doc_id else ""
        print(f"[MilvusVectorDB] ✅ 精细化搜索完成: 合并后找到 {len(all_results)} 个相关chunks（已去重）{exclude_msg}")
        
        # ⭐ 全局Reranking：用完整query对合并后的结果进行统一精排
        if use_reranker and final_top_k and len(all_results) > final_top_k:
            try:
                # 将多个query chunks合并为一个完整query
                full_query = "\n".join(query_chunks)
                
                from utils.reranker import get_reranker
                reranker = get_reranker()
                print(f"[MilvusVectorDB] 🔄 对合并后的 {len(all_results)} 个候选进行全局Reranking...")
                all_results = reranker.rerank(full_query, all_results, top_k=final_top_k)
            except Exception as e:
                print(f"[MilvusVectorDB] ⚠️ 全局Reranking失败: {e}，返回向量检索结果")
                all_results = all_results[:final_top_k] if final_top_k else all_results
        elif final_top_k:
            all_results = all_results[:final_top_k]
        
        return all_results
    
    def query_by_report_series(self, report_series: str, exclude_doc_id: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        按报告系列查询历史政策（不使用向量相似度，直接按标签查询）
        
        Args:
            report_series: 报告系列标签（如"五年规划-建议"、"中央经济工作会议"等）
            exclude_doc_id: 要排除的doc_id（避免匹配到自己）
            limit: 返回结果数量
            
        Returns:
            同系列历史政策列表（按时间排序）
        """
        if not report_series or report_series == 'N/A':
            print(f"[MilvusVectorDB] ⚠️ 报告系列为空或为'N/A'，跳过查询")
            return []
        
        print(f"[MilvusVectorDB] 🔍 按报告系列查询: {report_series}")
        
        # 构建查询表达式
        expr = f'report_series == "{report_series}"'
        if exclude_doc_id:
            expr += f' && doc_id != "{exclude_doc_id}"'
        
        # 查询字段
        output_fields = [
            "chunk_id", "doc_id", "content", "chunk_index", "chunk_type",
            "title", "timestamp", "industries", "investment_relevance", 
            "report_series", "industry_policy_segments"
        ]
        
        try:
            # 执行查询（不使用向量搜索，直接按字段查询）
            results = self.chunk_collection.query(
                expr=expr,
                output_fields=output_fields,
                limit=limit * 10  # 查询更多chunks，然后按doc_id合并
            )
            
            if not results:
                print(f"[MilvusVectorDB] ⚠️ 未找到报告系列为'{report_series}'的历史政策")
                return []
            
            print(f"[MilvusVectorDB] ✅ 找到 {len(results)} 个chunks（报告系列: {report_series}）")
            
            # 按doc_id合并chunks
            docs_by_id = {}
            for chunk in results:
                doc_id = chunk.get('doc_id')
                if not doc_id:
                    continue
                
                if doc_id not in docs_by_id:
                    docs_by_id[doc_id] = {
                        'doc_id': doc_id,
                        'title': chunk.get('title', ''),
                        'timestamp': chunk.get('timestamp', ''),
                        'industries': chunk.get('industries', ''),
                        'investment_relevance': chunk.get('investment_relevance', ''),
                        'report_series': chunk.get('report_series', report_series),
                        'chunks': [],
                        'content': ''
                    }
                
                # 添加chunk内容
                content = chunk.get('content', '')
                if content:
                    docs_by_id[doc_id]['chunks'].append({
                        'chunk_id': chunk.get('chunk_id'),
                        'content': content,
                        'chunk_index': chunk.get('chunk_index', 0)
                    })
                    docs_by_id[doc_id]['content'] += f"{content}\n\n"
            
            # 转换为列表并按时间排序
            doc_list = []
            for doc_id, doc_data in docs_by_id.items():
                # 清理内容
                doc_data['content'] = doc_data['content'].strip()
                # 按chunk_index排序chunks
                doc_data['chunks'].sort(key=lambda x: x.get('chunk_index', 0))
                doc_list.append(doc_data)
            
            # 按时间排序（从旧到新，方便时间对比分析）
            from datetime import datetime as dt_class
            def parse_timestamp(ts):
                if not ts:
                    return None
                try:
                    # 尝试解析ISO格式
                    if 'T' in str(ts):
                        return dt_class.fromisoformat(str(ts).replace('Z', '+00:00'))
                    else:
                        return dt_class.strptime(str(ts), '%Y-%m-%d')
                except:
                    return None
            
            doc_list.sort(key=lambda x: parse_timestamp(x.get('timestamp')) or dt_class.min, reverse=False)  # 从旧到新
            
            # 限制返回数量
            result_list = doc_list[:limit]
            
            print(f"[MilvusVectorDB] ✅ 按报告系列查询完成: 找到 {len(result_list)} 个同系列历史政策（已按时间排序）")
            
            return result_list
            
        except Exception as e:
            print(f"[MilvusVectorDB] ❌ 按报告系列查询失败: {e}")
            return []


    def get_full_document_content(self, doc_id: str = None, title: str = None, timestamp: str = None) -> str:
        """
        根据doc_id或(title, timestamp)获取该文档的完整内容（合并所有chunks）
        
        ⭐ 用途：RAG检索后，回库拉取完整文档内容，解决"检索到的内容太短"问题
        
        Args:
            doc_id: 文档ID（优先使用）
            title: 文档标题（与timestamp配合使用）
            timestamp: 文档时间戳（与title配合使用）
            
        Returns:
            合并后的完整文档内容
        """
        if not doc_id and not (title and timestamp):
            return ""
        
        try:
            # 构建查询表达式
            if doc_id:
                expr = f'doc_id == "{doc_id}"'
            else:
                # 使用 title + timestamp 查询
                expr = f'title == "{title}" && timestamp == "{timestamp}"'
            
            # 查询该文档的所有chunks
            results = self.chunk_collection.query(
                expr=expr,
                output_fields=["chunk_id", "content", "chunk_index"],
                limit=100  # 一个文档最多100个chunks
            )
            
            if not results:
                return ""
            
            # 按chunk_index排序
            results.sort(key=lambda x: x.get('chunk_index', 0))
            
            # 合并所有chunks的内容
            contents = [r.get('content', '') for r in results if r.get('content')]
            full_content = '\n\n'.join(contents)
            
            return full_content
            
        except Exception as e:
            print(f"[MilvusVectorDB] ⚠️ 获取完整文档内容失败: {e}")
            return ""
    
    def get_documents_full_content(self, doc_ids: list = None, title_timestamp_pairs: list = None) -> Dict[str, str]:
        """
        批量获取多个文档的完整内容
        
        Args:
            doc_ids: 文档ID列表
            title_timestamp_pairs: (title, timestamp) 元组列表
            
        Returns:
            {doc_id或"title|timestamp": 完整内容} 字典
        """
        result = {}
        
        if doc_ids:
            for doc_id in doc_ids:
                content = self.get_full_document_content(doc_id=doc_id)
                if content:
                    result[doc_id] = content
        
        if title_timestamp_pairs:
            for title, timestamp in title_timestamp_pairs:
                content = self.get_full_document_content(title=title, timestamp=timestamp)
                if content:
                    result[f"{title}|{timestamp}"] = content
        
        return result


def get_vector_db() -> MilvusVectorDatabase:
    """获取Milvus向量数据库实例"""
    return MilvusVectorDatabase(chunk_only=True)  # 确保只使用chunk级别



