"""
系统配置文件
"""
import os
from pathlib import Path

# 项目路径
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data_processing"
OUTPUT_DIR = PROJECT_ROOT / "output"

# API配置 - 使用火山引擎
# 火山引擎API配置（使用官方SDK）
VOLCENGINE_API_KEY = "your-api-key-here"  # ⚠️ 请替换为你的API密钥
VOLCENGINE_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
# ⚠️ 模型名称格式：通常是 endpoint ID 格式，如 "ep-20250120-xxxxx" 或直接使用模型名称
# 请在火山引擎控制台查看正确的 endpoint ID 或模型名称
VOLCENGINE_MODEL = "your-model-name"  # ⚠️ 请替换为正确的 endpoint ID 或模型名称
VOLCENGINE_EMBEDDING_MODEL = "text-embedding-ada-002"

# RAG配置
RAG_CONFIG = {
    "chunk_size": 500,  # ⭐ 优化：从450增加到500，更好利用512 tokens，保持语义完整
    "chunk_overlap": 50,
    "top_k_docs": 20,  # 文档级检索数量
    "top_k_chunks": 20,  # ⭐ Chunk级检索数量（增加到20）
    "similarity_threshold": 0.75,  # ⭐ 相似度阈值（过滤低质量结果）
    "max_keywords": 30,
    "enable_llm_enhancement": False,  # 不使用LLM做关键词提取
    "enable_llm_novelty": False,  # 不使用LLM计算增量度（用RAG匹配）
    "llm_max_retries": 5,
    "llm_batch_delay": 3,
    "enable_industry_filter": True,  # ⭐ 启用行业过滤
    "industry_match_boost": 1.5,  # 同行业政策的相似度加权
}

# ⚠️ 注意：行业分类配置已迁移到citic_industries.py
# 使用中信一级、二级、三级行业分类标准

# 确保输出目录存在
OUTPUT_DIR.mkdir(exist_ok=True)