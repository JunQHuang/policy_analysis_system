"""
统一数据模型定义
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


# ==========================================
# 已废弃的模型（保留用于向后兼容，但不再使用）
# ==========================================

# class PolicySource(BaseModel):
#     """政策来源信息（已废弃，使用PolicySegment代替）"""
#     doc_id: str = Field(description="文档唯一ID")
#     title: str = Field(description="政策标题")
#     publish_date: datetime = Field(description="发布时间")
#     full_text: str = Field(description="政策全文")


# ==========================================
# 核心模型（正在使用）
# ==========================================

class PolicySegment(BaseModel):
    """
    政策文档（简化版，不切分）
    
    metadata 字段包含的键：
    - citic_industries: Dict[str, List[str]] - 中信行业标签（level1: 一级行业列表）
    - investment_relevance: str - 投资相关性：高/低
    - report_series: str - 报告系列：两会-政府工作报告/两会-计划报告/两会-预算报告/五年规划-建议/五年规划-纲要/国务院常务会议/国务院常务会议-解读/中央政治局会议/中央经济工作会议/全国财政工作会议/中央委员会全体会议公报/NA
    - industry_policy_segments: Dict[str, List[str]] - 行业及对应政策片段（JSON格式，用于存入Milvus）
    - category: str - 政策分类（可选）
    - chunks: List - 文档切分结果（可选）
    """
    doc_id: str = Field(description="文档ID")
    content: str = Field(description="完整文档内容")
    timestamp: datetime = Field(description="发布时间")
    
    # 分析结果
    novelty_score: Optional[float] = Field(default=0.5, description="增量度得分 0-1")
    industries: List[str] = Field(default_factory=list, description="相关行业（经过DS32B过滤后的最终行业标签）")
    
    # 元数据
    title: str = Field(description="文档标题")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="额外元数据（包含citic_industries、investment_relevance、report_series、industry_policy_segments等）")


# ==========================================
# 已废弃的模型（保留用于向后兼容，但不再使用）
# ==========================================

# class IncrementAnalysis(BaseModel):
#     """增量分析结果（已废弃）"""
#     segment_id: str
#     compared_segments: List[str] = Field(description="对比的历史片段ID")
#     novelty_score: float = Field(description="增量得分 0-1", default=0.5)
#     timestamp: datetime = Field(description="分析时间")
#
#
# class SectorMapping(BaseModel):
#     """行业映射结果（已废弃）"""
#     segment_id: str
#     industries: List[str] = Field(description="相关行业列表")
#     confidence: float = Field(description="映射置信度 0-1", default=0.8)
#
#
# class PolicyScore(BaseModel):
#     """政策评分（已废弃）"""
#     segment_id: str
#     importance_score: float = Field(description="重要性得分 0-1", default=0.5)
#     composite_score: float = Field(description="综合得分 0-1", default=0.5)
#
#
# class RotationRecommendation(BaseModel):
#     """行业轮动建议（已废弃）"""
#     industry: str = Field(description="行业名称")
#     direction: str = Field(description="建议方向: 增持/减持/中性")
#     reasoning: str = Field(description="建议理由")
#     supporting_policies: List[str] = Field(description="支持政策事件ID列表")
#     time_factor: str = Field(description="时序因素说明")
#     momentum_adjustment: float = Field(description="动量调整系数 -1到1")
#     confidence: float = Field(description="建议置信度 0-1")
#     timestamp: datetime = Field(description="建议时间")
#
#
# class AnalysisReport(BaseModel):
#     """分析报告（已废弃）"""
#     report_id: str
#     title: str = Field(description="报告标题")
#     generate_time: datetime = Field(description="生成时间")
#     
#     # 主要内容（简化）
#     key_segments: List[PolicySegment] = Field(description="关键政策片段", default_factory=list)
#     rotation_recommendations: List[RotationRecommendation] = Field(description="轮动建议", default_factory=list)
#     
#     # 统计信息
#     total_segments: int = Field(description="总片段数", default=0)
#     macro_count: int = Field(description="宏观政策数", default=0)
#     
#     # 元数据
#     metadata: Dict[str, Any] = Field(default_factory=dict)
#
#
# class VectorSearchResult(BaseModel):
#     """向量检索结果（已废弃，实际使用Dict返回）"""
#     segment_id: str
#     doc_id: str
#     content: str
#     similarity: float
#     metadata: Dict[str, Any]

