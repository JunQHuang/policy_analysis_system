"""
核心Agent系统

主要Agent：
- NoveltyAgent: 政策增量分析（两阶段LLM：主题提取+分主题RAG+增量分析）
- IndustryAgent: 行业分类和投资相关性判断
- InvestmentAgent: 投资建议生成（集中所有LLM prompt调用）
- SimplifiedRAGAgent: 简化版RAG检索（chunk级别向量相似度+reranker精排）
"""
from .novelty_agent import NoveltyAgent
from .industry_agent import IndustryAgent
from .enhanced_rag_agent import SimplifiedRAGAgent
from .investment_agent import InvestmentAgent

__all__ = [
    'NoveltyAgent',
    'IndustryAgent',
    'SimplifiedRAGAgent',
    'InvestmentAgent',
]

