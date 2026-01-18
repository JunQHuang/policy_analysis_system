"""
Novelty Agent - 增量分析（政策新旧对比）
分主题RAG检索 + LLM对比分析
"""
from typing import List, Dict, Any
from datetime import datetime, timedelta
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from .base import BaseAgent
from models import PolicySegment
from core.clients.volcengine_client import get_volcengine_client


class NoveltyAgent(BaseAgent):
    """增量分析Agent - 分主题RAG + LLM相关性评分 + Reranker精排"""
    
    MAX_DOCS = 50  # 最大返回文档数
    
    def __init__(self, vector_db=None):
        super().__init__("NoveltyAgent")
        self.vector_db = vector_db
        self.llm_client = get_volcengine_client()
        self.log("✅ NoveltyAgent初始化完成")
    
    def process(self, input_data: Any) -> Dict[str, Any]:
        """
        处理数据的主方法
        
        Args:
            input_data: PolicySegment对象
            
        Returns:
            包含analysis、topics、topic_rag_results的字典
        """
        if isinstance(input_data, PolicySegment):
            return self.analyze_with_topics(input_data)
        return {
            'analysis': '## 政策增量分析\n\n（生成失败）',
            'topics': [],
            'topic_rag_results': {}
        }
    
    def analyze_with_topics(self, segment: PolicySegment, topics: List[str] = None,
                             use_dimension_analysis: bool = True) -> Dict[str, Any]:
        """
        使用外部提供的主题词进行分主题RAG增量分析
        
        新流程（use_dimension_analysis=True）：
        1. 使用meeting对比提取的主题词（或自动提取）
        2. 每个主题拆分为3个细分维度
        3. 每个维度独立RAG检索 + LLM对比分析
        4. 汇总各维度分析生成主题报告
        
        旧流程（use_dimension_analysis=False）：
        1. 每个主题独立RAG检索历史政策
        2. 直接生成一对多对比分析
        
        Args:
            segment: 当前政策文档
            topics: 外部提供的主题词列表（来自meeting对比）
            use_dimension_analysis: 是否使用分维度精细化分析
            
        Returns:
            {
                'analysis': markdown文本,
                'topic_rag_results': {topic: [历史政策列表], ...},
                'topics': 使用的主题词列表
            }
        """
        # 计算2年时间窗口
        if segment.timestamp:
            after_timestamp = segment.timestamp - timedelta(days=730)
            self.log(f"⏰ 时间窗口: {after_timestamp.strftime('%Y-%m-%d')} ~ {segment.timestamp.strftime('%Y-%m-%d')} (2年内)")
        else:
            after_timestamp = None
        
        # 如果没有提供主题词，使用默认方法提取
        if not topics:
            self.log("未提供主题词，使用默认方法提取...")
            investment_summary = self._extract_investment_content(segment)
            topics = self._extract_investment_topics(investment_summary)
        
        self.log(f"使用 {len(topics)} 个主题词进行分主题分析: {topics}")
        
        if use_dimension_analysis:
            # 新流程：分维度精细化RAG
            self.log("🚀 使用分维度精细化RAG流程...")
            
            # 直接在_generate_topic_analysis_report中处理分维度逻辑
            analysis = self._generate_topic_analysis_report(
                segment=segment,
                topics=topics,
                topic_rag_results={},  # 新流程不需要预先RAG
                use_dimension_analysis=True
            )
            
            return {
                'analysis': analysis,
                'topic_rag_results': {},  # 新流程的RAG结果在内部处理
                'topics': topics
            }
        else:
            # 旧流程：每个主题独立RAG
            self.log("📚 使用传统RAG流程...")
            topic_rag_results = {}
            
            if self.vector_db and topics:
                for topic in topics:
                    self.log(f"  检索主题: {topic}")
                    results, query_text = self._search_by_topic_with_time(
                        segment, topic, top_k=50, after_timestamp=after_timestamp
                    )
                    if results:
                        # LLM相关性过滤
                        results = self._llm_relevance_rerank(
                            new_policy_title=f"{segment.title} - {topic}主题",
                            new_policy_content=query_text,
                            candidates=results,
                            top_k=30
                        )
                        if results:
                            topic_rag_results[topic] = results
                            self.log(f"    主题 '{topic}': 最终保留 {len(results)} 篇")
            
            # 生成分主题增量分析报告
            self.log("生成分主题增量分析报告...")
            analysis = self._generate_topic_analysis_report(
                segment=segment,
                topics=topics,
                topic_rag_results=topic_rag_results,
                use_dimension_analysis=False
            )
            
            return {
                'analysis': analysis,
                'topic_rag_results': topic_rag_results,
                'topics': topics
            }
    
    def _generate_topic_analysis_report(self, segment: PolicySegment, 
                                         topics: List[str],
                                         topic_rag_results: Dict[str, List[Dict]],
                                         use_dimension_analysis: bool = True) -> str:
        """
        生成分主题增量分析报告
        
        Args:
            segment: 当前政策
            topics: 主题词列表
            topic_rag_results: 主题RAG结果（旧流程用）
            use_dimension_analysis: 是否使用分维度精细化分析（新流程）
        """
        all_parts = []
        
        # 标题
        all_parts.append("## 分主题深度分析\n\n")
        
        if not topics:
            all_parts.append("（未提取到投资主题）\n")
            return "".join(all_parts)
        
        # 概述
        all_parts.append(f"本次分析聚焦以下 **{len(topics)}** 个核心投资主题：{', '.join(topics)}\n\n")
        all_parts.append("---\n\n")
        
        # 计算2年时间窗口
        if segment.timestamp:
            after_timestamp = segment.timestamp - timedelta(days=730)
        else:
            after_timestamp = None
        
        # 每个主题的对比分析
        topic_idx = 1
        total_docs = 0
        
        for topic in topics:
            self.log(f"📊 处理主题 '{topic}'...")
            
            if use_dimension_analysis and self.vector_db:
                # 新流程：分维度精细化RAG
                topic_analysis, doc_count = self._analyze_topic_by_dimensions(
                    segment=segment,
                    topic=topic,
                    topic_idx=topic_idx,
                    after_timestamp=after_timestamp
                )
                total_docs += doc_count
            else:
                # 旧流程：直接使用已有的RAG结果
                topic_docs = topic_rag_results.get(topic, [])
                if not topic_docs:
                    continue
                topic_analysis = self._generate_topic_comparison(
                    segment=segment,
                    topic=topic,
                    topic_idx=topic_idx,
                    topic_docs=topic_docs
                )
                total_docs += len(topic_docs)
            
            all_parts.append(topic_analysis)
            topic_idx += 1
        
        # 总结
        topics_str = '、'.join(topics)
        summary = f"""
---

### 主题分析小结

本次分主题分析共涉及 **{len(topics)}** 个核心投资主题，对比了 **{total_docs}** 篇相关历史政策。

**核心主题**：{topics_str}

**分析要点**：
- 通过分维度精细化检索，提高了政策对比的精准度
- 每个主题拆分为3个细分维度，分别进行RAG检索和对比分析
- 通过表格对比，清晰展示新旧政策的边际变化

**后续跟踪**：建议持续关注上述主题的政策落地进展、产业订单和产能变化。
"""
        all_parts.append(summary)
        
        return "".join(all_parts)

    def _analyze_topic_by_dimensions(self, segment: PolicySegment, topic: str,
                                      topic_idx: int, after_timestamp=None) -> tuple:
        """
        分维度精细化分析单个主题
        
        流程：
        1. 用LLM将主题拆分为3个细分维度
        2. 每个维度：提取新政策内容 → RAG检索 → LLM生成对比分析
        3. 汇总3个维度的分析结果
        
        Args:
            segment: 当前政策
            topic: 主题词
            topic_idx: 主题序号
            after_timestamp: 时间窗口下限
            
        Returns:
            (主题分析文本, 检索到的历史政策数量)
        """
        self.log(f"  🔍 拆分主题'{topic}'为细分维度...")
        
        # Step 1: 拆分维度
        dimensions = self._split_topic_to_dimensions(segment, topic)
        
        if not dimensions:
            self.log(f"  ⚠️ 主题'{topic}'拆分维度失败，使用旧流程")
            return f"### 3.{topic_idx} {topic}\n\n（维度拆分失败）\n\n", 0
        
        # Step 2: 每个维度独立RAG + 分析
        dimension_analyses = []
        all_history_docs = []
        
        for dim in dimensions:
            dim_name = dim.get('dimension', '')
            self.log(f"  📌 处理维度'{dim_name}'...")
            
            # RAG检索
            history_docs = self._search_by_dimension(
                segment=segment,
                topic=topic,
                dimension=dim,
                top_k=15,
                after_timestamp=after_timestamp
            )
            
            # LLM相关性过滤
            if history_docs:
                history_docs = self._llm_relevance_rerank(
                    new_policy_title=f"{segment.title} - {topic}/{dim_name}",
                    new_policy_content=dim.get('content', ''),
                    candidates=history_docs,
                    top_k=10
                )
            
            all_history_docs.extend(history_docs)
            
            # 生成该维度的对比分析
            dim_analysis = self._generate_dimension_comparison(
                segment=segment,
                topic=topic,
                dimension=dim,
                history_docs=history_docs
            )
            dimension_analyses.append(dim_analysis)
        
        # Step 3: 汇总各维度分析
        # 去重统计历史政策数量
        unique_docs = {}
        for doc in all_history_docs:
            key = (doc.get('title', ''), doc.get('timestamp', ''))
            if key not in unique_docs:
                unique_docs[key] = doc
        
        topic_analysis = self._generate_topic_comparison(
            segment=segment,
            topic=topic,
            topic_idx=topic_idx,
            topic_docs=list(unique_docs.values()),
            dimension_analyses=dimension_analyses
        )
        
        return topic_analysis, len(unique_docs)
    
    def _extract_investment_content(self, segment: PolicySegment) -> str:
        """
        提取政策中具有投资相关性的核心内容
        
        Args:
            segment: 政策文档
            
        Returns:
            投资相关的核心内容
        """
        prompt = f"""请从以下政策文档中提取**具有投资相关性**的核心内容。

政策标题：{segment.title}

政策原文：
{segment.content}

---

## 任务说明

你需要提取对**投资分析**有价值的内容，包括：

### 必须保留的内容：
1. **产业政策**：支持/限制哪些行业、产业升级方向
2. **量化目标**：具体数字、百分比、金额、产能目标
3. **时间节点**：2025年、2030年等关键时间点的目标
4. **财政/金融支持**：补贴、税收优惠、专项资金、信贷支持
5. **重点项目**：基础设施、重大工程、试点示范
6. **技术方向**：新能源、人工智能、半导体等具体技术
7. **区域布局**：哪些地区重点发展什么产业

### 必须过滤掉的内容：
1. 政治宣示语、原则性表述
2. 空洞表态、重复内容
3. 与投资无关的行政管理内容

### 输出要求：
- 直接输出提炼后的核心内容
- 保留原文的关键数据和措施
- 可以整理语句，但不改变原意
- 不要加标题或格式"""

        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.llm_client.chat_completion(
                messages=messages,
                temperature=0.2,
                max_tokens=32768
            )
            return response.strip()
        except Exception as e:
            self.log(f"提取投资内容失败: {e}", level="warning")
            return segment.content
    
    def _extract_investment_topics(self, investment_content: str) -> List[str]:
        """
        从投资核心内容中提取关键投资主题（使用中信一级行业分类）
        
        Args:
            investment_content: 提取后的投资相关核心内容
            
        Returns:
            主题词列表（5-10个，按重要性排序）
        """
        prompt = f"""请从以下投资相关政策内容中提取**最核心的投资主题**。

政策内容：
{investment_content}

---

## 中信一级行业分类（必须从以下行业中选择）：
金融、电子、计算机、通信、传媒、医药生物、机械设备、电力设备、国防军工、汽车、家用电器、轻工制造、商贸零售、社会服务、食品饮料、农林牧渔、钢铁、有色金属、基础化工、石油石化、煤炭、建筑材料、建筑装饰、房地产、交通运输、公用事业、纺织服饰、美容护理、环保、综合

## 要求
1. 必须从上述中信一级行业分类中选择
2. 按政策中的重要性排序（政策中先提到的、篇幅更大的排在前面）
3. 提取5-10个最相关的行业
4. 只输出行业名称，用逗号分隔

## 输出格式
行业1,行业2,行业3,..."""

        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.llm_client.chat_completion(
                messages=messages,
                temperature=0.1,
                max_tokens=200
            )
            topics = [t.strip() for t in response.split(',') if t.strip()]
            return topics[:10]  # 最多返回10个
        except Exception as e:
            self.log(f"提取投资主题失败: {e}", level="warning")
            return []

    def _search_by_topic_with_time(self, segment: PolicySegment, topic: str, 
                                    top_k: int = 10, after_timestamp=None) -> tuple:
        """
        按主题检索（带时间约束）
        
        Args:
            segment: 当前政策
            topic: 主题词
            top_k: 返回数量
            after_timestamp: 时间窗口下限（只检索此时间之后的政策，用于2年限制）
            
        Returns:
            (检索结果列表, 提取的主题内容query_text)
        """
        if not self.vector_db:
            return [], ""
        
        try:
            # 用LLM生成该主题的检索片段
            query_text = self._extract_topic_content(segment, topic)
            self.log(f"  主题 '{topic}' 检索片段: {len(query_text)}字")
            
            # 检索（带2年时间窗口）
            chunk_results = self.vector_db.search_chunks(
                query_text=query_text,
                top_k=500,  # 粗排500
                rerank_top_k=100,  # Reranker精排100
                exclude_doc_id=segment.doc_id,
                exclude_title=segment.title,
                exclude_timestamp=segment.timestamp,
                before_timestamp=segment.timestamp,
                after_timestamp=after_timestamp,  # 2年时间窗口下限
                allow_same_day=True,
                use_reranker=True
            )
            
            self.log(f"  主题 '{topic}' RAG召回: {len(chunk_results)} 个chunks")
            
            # 去重 + 时间加权
            deduplicated = self._deduplicate_chunks(
                chunk_results=chunk_results,
                policy_timestamp=segment.timestamp,
                top_k=top_k
            )
            
            self.log(f"  主题 '{topic}' 去重后: {len(deduplicated)} 篇")
            
            return deduplicated, query_text
            
        except Exception as e:
            self.log(f"主题检索失败 '{topic}': {e}", level="warning")
            return [], ""
    
    def _extract_topic_content(self, segment: PolicySegment, topic: str) -> str:
        """用LLM提取该主题相关的政策内容"""
        prompt = f"""请从以下政策中提取与"{topic}"相关的内容。

政策标题：{segment.title}

政策内容：
{segment.content}

要求：
1. 直接摘录原文，不要改写
2. 完整提取相关内容，不截断
3. 只输出摘录内容"""

        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.llm_client.chat_completion(
                messages=messages,
                temperature=0.2,
                max_tokens=32768
            )
            return response.strip()
        except Exception as e:
            self.log(f"提取主题内容失败: {e}", level="warning")
            return segment.content

    def _split_topic_to_dimensions(self, segment: PolicySegment, topic: str) -> List[Dict[str, str]]:
        """
        将主题词拆分为3个细分维度
        
        Args:
            segment: 当前政策文档
            topic: 主题词（如"国防军工"）
            
        Returns:
            [
                {"dimension": "维度名称", "description": "维度描述", "content": "新政策该维度的内容"},
                ...
            ]
        """
        prompt = f"""你是一名资深的{topic}行业分析师。请基于新政策内容，将"{topic}"主题拆分为3个最具投资价值的细分板块/子领域。

=== 新政策 ===
《{segment.title}》

{segment.content}

=== 任务 ===

请将"{topic}"主题拆分为3个最重要且具有投资价值的细分板块，每个板块需要：
1. 板块名称：具体的子行业或细分领域（如"新能源汽车"→"整车制造"、"动力电池"、"充电基础设施"）
2. 板块描述：说明这个细分板块包含什么
3. 新政策内容：直接摘录新政策中与该板块相关的原文

=== 输出格式（JSON） ===

```json
{{
  "dimensions": [
    {{
      "dimension": "细分板块1名称",
      "description": "这个板块包含什么",
      "content": "新政策中与该板块相关的原文摘录"
    }},
    {{
      "dimension": "细分板块2名称", 
      "description": "这个板块包含什么",
      "content": "新政策中与该板块相关的原文摘录"
    }},
    {{
      "dimension": "细分板块3名称",
      "description": "这个板块包含什么", 
      "content": "新政策中与该板块相关的原文摘录"
    }}
  ]
}}
```

=== 要求 ===
1. 拆分为具体的子行业/细分板块，不要拆分为分析角度（如"政策力度"、"技术路线"）
2. 优先选择政策着墨较多、投资价值较高的细分方向
3. 板块要有区分度，不要重叠
4. content必须是新政策原文摘录，不要改写
5. 只输出JSON"""

        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.llm_client.chat_completion(
                messages=messages,
                temperature=0.2,
                max_tokens=32768
            )
            
            # 解析JSON
            import json
            import re
            response = response.strip()
            if response.startswith('```'):
                response = re.sub(r'^```\w*\n?', '', response)
                response = re.sub(r'\n?```$', '', response)
            
            result = json.loads(response)
            dimensions = result.get('dimensions', [])
            
            self.log(f"  主题'{topic}'拆分为{len(dimensions)}个维度: {[d['dimension'] for d in dimensions]}")
            
            return dimensions
            
        except Exception as e:
            self.log(f"  拆分维度失败: {e}", level="warning")
            # 降级：返回单一维度
            return [{
                "dimension": topic,
                "description": f"{topic}相关政策",
                "content": segment.content
            }]

    def _search_by_dimension(self, segment: PolicySegment, topic: str, 
                              dimension: Dict[str, str], top_k: int = 20,
                              after_timestamp=None) -> List[Dict]:
        """
        按维度检索历史政策
        
        Args:
            segment: 当前政策
            topic: 主题词
            dimension: 维度信息 {"dimension": "维度名", "description": "描述", "content": "新政策内容"}
            top_k: 返回数量
            after_timestamp: 时间窗口下限
            
        Returns:
            检索结果列表
        """
        if not self.vector_db:
            return []
        
        try:
            dim_name = dimension.get('dimension', '')
            dim_content = dimension.get('content', '')
            
            # 用维度内容作为检索query
            query_text = f"{topic} {dim_name}: {dim_content}"
            self.log(f"    维度'{dim_name}'检索: {len(query_text)}字")
            
            # 检索
            chunk_results = self.vector_db.search_chunks(
                query_text=query_text,
                top_k=300,  # 粗排300
                rerank_top_k=50,  # Reranker精排50
                exclude_doc_id=segment.doc_id,
                exclude_title=segment.title,
                exclude_timestamp=segment.timestamp,
                before_timestamp=segment.timestamp,
                after_timestamp=after_timestamp,
                allow_same_day=True,
                use_reranker=True
            )
            
            self.log(f"    维度'{dim_name}'RAG召回: {len(chunk_results)}个chunks")
            
            # 去重 + 时间加权
            deduplicated = self._deduplicate_chunks(
                chunk_results=chunk_results,
                policy_timestamp=segment.timestamp,
                top_k=top_k
            )
            
            self.log(f"    维度'{dim_name}'去重后: {len(deduplicated)}篇")
            
            return deduplicated
            
        except Exception as e:
            self.log(f"    维度检索失败: {e}", level="warning")
            return []

    def _generate_dimension_comparison(self, segment: PolicySegment, topic: str,
                                        dimension: Dict[str, str], 
                                        history_docs: List[Dict]) -> str:
        """
        生成单个维度的对比分析
        
        Args:
            segment: 当前政策
            topic: 主题词
            dimension: 维度信息
            history_docs: 该维度检索到的历史政策
            
        Returns:
            该维度的对比分析文本
        """
        dim_name = dimension.get('dimension', '')
        dim_desc = dimension.get('description', '')
        dim_content = dimension.get('content', '')
        
        if not history_docs:
            return f"""**{dim_name}**

新政策表述：{dim_content}

历史对比：未检索到相关历史政策

"""
        
        # 构建历史政策列表
        history_for_llm = ""
        for i, doc in enumerate(history_docs, 1):
            title = doc.get('title', '未知标题')
            timestamp = doc.get('timestamp', 'N/A')
            content = doc.get('content', '')
            history_for_llm += f"""
【历史政策{i}】《{title}》（{timestamp}）
{content}
"""
        
        prompt = f"""你是{topic}行业分析师，请针对"{dim_name}"这个维度，对比新政策与历史政策的边际变化。

=== 维度说明 ===
维度名称：{dim_name}
维度描述：{dim_desc}

=== 新政策该维度内容 ===
《{segment.title}》

{dim_content}

=== 历史政策（{len(history_docs)}篇） ===
{history_for_llm}

=== 输出要求 ===

请用表格对比新政策与历史政策在"{dim_name}"维度的边际变化：

**新政策表述**：
完整引用新政策原文

**与历史政策对比**：

| 历史政策表述 | 边际变化 |
|-------------|---------|
| 《政策名》（YYYY年MM月）完整引用历史政策原文 | 具体说明变化内容（如：新增XX表述/从XX升级为XX/删除XX要求） |

要求：
- 只选取与该维度高度相关的历史政策进行对比，相关性弱的不要放入表格
- 每行对比一个不同的历史政策要点，不要重复
- 历史政策表述要带上政策名称和时间，格式：《政策名》（YYYY年MM月）原文内容
- 边际变化要具体说明与新政策相比的变化内容，不要只写"强化"、"延续"等笼统词汇
- 列出3-5个不同的对比要点

然后用1-2句话总结该维度的核心变化。

=== 要求 ===
1. 新政策表述单独列在表格前面，不要放在表格里
2. 表格只有两列：历史政策表述、边际变化
3. 只展示与该维度高度相关的对比，相关性弱的历史政策不要放入表格
4. 历史政策表述必须带上《政策名》（YYYY年MM月），完整引用原文
5. 边际变化要具体描述变化内容，不要只写"强化/延续"
6. 不要用省略号(...)省略内容，完整输出所有分析
7. 只输出新政策表述、表格和总结，不要其他内容"""

        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.llm_client.chat_completion(
                messages=messages,
                temperature=0.2,
                max_tokens=32768
            )
            
            return f"""**{dim_name}**（{dim_desc}）

{response.strip()}

"""
        except Exception as e:
            self.log(f"    维度'{dim_name}'分析生成失败: {e}", level="warning")
            return f"""**{dim_name}**

新政策表述：{dim_content}

（分析生成失败）

"""
    
    def _deduplicate_chunks(self, chunk_results: List[Dict], 
                            policy_timestamp: datetime,
                            top_k: int) -> List[Dict[str, Any]]:
        """
        去重 + 时间加权
        按 (title, timestamp) 去重，保留rerank_score最高的chunk
        """
        # 按 (title, timestamp) 去重，保留最高分的
        seen = {}
        for chunk in chunk_results:
            title = chunk.get('title', '')
            timestamp = chunk.get('timestamp', '')
            key = (title, timestamp)
            
            rerank_score = chunk.get('rerank_score', 0.0)
            
            if key not in seen or rerank_score > seen[key].get('rerank_score', 0):
                seen[key] = chunk
        
        # 转为列表，计算时间加权
        results = []
        for chunk in seen.values():
            # 计算时间加权
            time_bonus = 0.0
            timestamp = chunk.get('timestamp', '')
            if timestamp and policy_timestamp:
                try:
                    if isinstance(timestamp, str):
                        if 'T' in timestamp:
                            doc_dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        else:
                            doc_dt = datetime.fromisoformat(timestamp)
                    elif isinstance(timestamp, datetime):
                        doc_dt = timestamp
                    else:
                        doc_dt = None
                    
                    if doc_dt:
                        days_diff = (policy_timestamp.date() - doc_dt.date()).days
                        if days_diff <= 365:
                            time_bonus = 0.1 * (1 - days_diff / 365)
                        elif days_diff <= 1095:
                            time_bonus = 0.03 * (1 - (days_diff - 365) / 730)
                except:
                    pass
            
            chunk['time_bonus'] = time_bonus
            chunk['final_score'] = chunk.get('rerank_score', 0.0) + time_bonus
            
            results.append(chunk)
        
        # 按final_score排序
        results.sort(key=lambda x: x.get('final_score', 0.0), reverse=True)
        
        return results[:top_k]

    def _llm_relevance_rerank(self, new_policy_title: str, new_policy_content: str, 
                               candidates: List[Dict], top_k: int = 10) -> List[Dict]:
        """
        使用LLM对RAG检索到的chunk进行多维度相关性评分
        
        评分维度：
        - 主题相关度：是否讨论同一细分领域
        - 政策延续性：是否是同一政策的迭代/修订
        - 对比价值：对比能否得出有意义的增量分析结论
        
        Args:
            new_policy_title: 新政策标题
            new_policy_content: 新政策中该主题的具体内容（LLM提取的）
            candidates: RAG检索到的chunk列表
            top_k: 返回数量
            
        Returns:
            按总分排序后的chunk列表（带评分）
        """
        if not candidates:
            return []
        
        # 构建所有chunk的内容列表
        chunk_list_text = ""
        for i, chunk in enumerate(candidates, 1):
            chunk_title = chunk.get('title', '')
            chunk_content = chunk.get('content', '')
            if chunk_content:
                chunk_list_text += f"""
---
【{i}】《{chunk_title}》
{chunk_content}
"""
        
        # 多维度评分prompt
        prompt = f"""对以下历史政策片段与新政策的相关性进行**多维度评分**。

【新政策】{new_policy_title}

【新政策该主题的具体内容】
{new_policy_content}

【历史政策片段列表】
{chunk_list_text}

---

## 评分维度（每项1-5分）

1. **主题相关度**：历史政策是否讨论与新政策相同的细分领域？
   - 1分：完全无关（如新政策讲国防军工，历史政策讲土地管理）
   - 3分：大方向相关但细分领域不同
   - 5分：高度相关，讨论同一细分领域的同类内容

2. **政策延续性**：是否是同一政策链条上的文件？
   - 1分：无关联（如立法计划、茶话会讲话等）
   - 3分：相关但非直接延续
   - 5分：明确的修订/实施细则/配套政策

3. **对比价值**：对比能否得出有意义的增量分析？
   - 1分：无对比价值（内容太泛或不相关）
   - 3分：有一定参考价值
   - 5分：高对比价值，能看出明确的政策变化

## 输出要求

只输出JSON，格式如下：
{{
  "scores": [
    {{"id": 1, "topic": 4, "continuity": 3, "value": 5, "total": 12}},
    {{"id": 2, "topic": 2, "continuity": 1, "value": 2, "total": 5}},
    ...
  ]
}}

注意：
- 评分所有片段，total >= 9分的都值得对比
- 明显无关的（如立法计划、土地管理条例、茶话会讲话等）给低分
- id对应片段编号
- 尽量多保留相关政策用于一对多对比"""

        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.llm_client.chat_completion(
                messages=messages,
                temperature=0.1,
                max_tokens=32768
            )
            
            # 解析JSON
            import json
            import re
            response = response.strip()
            if response.startswith('```'):
                response = re.sub(r'^```\w*\n?', '', response)
                response = re.sub(r'\n?```$', '', response)
            
            result = json.loads(response)
            scores_list = result.get('scores', [])
            
            # 构建id -> 评分的映射
            score_map = {}
            for score_item in scores_list:
                idx = score_item.get('id')
                if idx:
                    score_map[idx] = {
                        'topic': score_item.get('topic', 0),
                        'continuity': score_item.get('continuity', 0),
                        'value': score_item.get('value', 0),
                        'total': score_item.get('total', 0)
                    }
            
            # 给每个candidate添加评分，并过滤低分的
            scored_candidates = []
            min_score = 9  # 最低总分阈值（降低到9分，保留更多相关政策用于一对多对比）
            
            for i, chunk in enumerate(candidates, 1):
                if i in score_map:
                    scores = score_map[i]
                    if scores['total'] >= min_score:
                        chunk['llm_scores'] = scores
                        chunk['llm_total_score'] = scores['total']
                        scored_candidates.append(chunk)
            
            # 按总分排序
            scored_candidates.sort(key=lambda x: x.get('llm_total_score', 0), reverse=True)
            
            self.log(f"  LLM多维度评分: {len(candidates)}个chunk → {len(scored_candidates)}个相关(≥{min_score}分)")
            
            # 打印top3的评分详情
            for i, chunk in enumerate(scored_candidates[:3]):
                scores = chunk.get('llm_scores', {})
                title = chunk.get('title', '')[:30]
                self.log(f"    [{i+1}] {title}... | 主题:{scores.get('topic',0)} 延续:{scores.get('continuity',0)} 价值:{scores.get('value',0)} 总分:{scores.get('total',0)}")
            
            return scored_candidates[:top_k]
            
        except Exception as e:
            self.log(f"  LLM多维度评分失败: {e}", level="warning")
            # 降级：返回原始candidates的前top_k个
            return candidates[:top_k]

    def _generate_topic_comparison(self, segment: PolicySegment, 
                                    topic: str,
                                    topic_idx: int,
                                    topic_docs: List[Dict],
                                    dimension_analyses: List[str] = None) -> str:
        """
        生成主题深度分析（汇总各维度的分析结果）
        
        Args:
            segment: 当前政策
            topic: 主题词
            topic_idx: 主题序号
            topic_docs: 该主题检索到的历史政策（用于统计）
            dimension_analyses: 各维度的分析结果列表
        """
        if dimension_analyses:
            # 新流程：汇总各维度分析
            return self._generate_topic_summary_from_dimensions(
                segment, topic, topic_idx, topic_docs, dimension_analyses
            )
        else:
            # 旧流程：直接一对多对比（兼容）
            return self._generate_topic_comparison_legacy(
                segment, topic, topic_idx, topic_docs
            )

    def _generate_topic_summary_from_dimensions(self, segment: PolicySegment,
                                                  topic: str, topic_idx: int,
                                                  topic_docs: List[Dict],
                                                  dimension_analyses: List[str]) -> str:
        """
        汇总各维度分析，生成主题总结
        """
        # 合并各维度分析
        dimensions_content = "\n".join(dimension_analyses)
        
        # 用LLM生成核心观点（不含投资建议）
        prompt = f"""你是{topic}行业首席分析师。以下是"{topic}"主题各维度的政策对比分析，请撰写核心观点。

=== 各维度分析 ===

{dimensions_content}

=== 输出要求 ===

#### 核心观点

**投资评级**：看多/看平/看空
**核心逻辑**：用2-3句话说清楚政策信号和最大边际变化

=== 要求 ===
1. 核心观点要有明确判断，不要模棱两可
2. 基于上述维度分析得出结论
3. 不要用省略号(...)省略内容，完整输出所有分析
4. 总字数100-200字"""

        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.llm_client.chat_completion(
                messages=messages,
                temperature=0.2,
                max_tokens=32768
            )
            
            final_output = f"""### 3.{topic_idx} {topic}

本主题拆分为多个维度进行精细化分析，共检索到 **{len(topic_docs)}** 篇相关历史政策。

---

{response.strip()}

---

#### 分维度政策对比

{dimensions_content}

"""
            return final_output
        except Exception as e:
            self.log(f"  主题'{topic}'汇总生成失败: {e}", level="warning")
            # 降级：直接输出各维度分析
            return f"""### 3.{topic_idx} {topic}

本主题共检索到 **{len(topic_docs)}** 篇相关历史政策。

---

#### 分维度政策对比

{dimensions_content}

"""

    def _generate_topic_comparison_legacy(self, segment: PolicySegment, 
                                           topic: str, topic_idx: int,
                                           topic_docs: List[Dict]) -> str:
        """
        旧流程：直接一对多对比（兼容保留）
        """
        # 构建给LLM的历史政策列表
        history_for_llm = ""
        for i, doc in enumerate(topic_docs, 1):
            title = doc.get('title', '未知标题')
            timestamp = doc.get('timestamp', 'N/A')
            content = doc.get('content', '')
            history_for_llm += f"""
【历史政策{i}】《{title}》（{timestamp}）
{content}
"""
        
        num_history = len(topic_docs)
        
        prompt = f"""你是一名顶级券商的{topic}行业首席分析师，请基于新政策和{num_history}篇历史政策，撰写一份深度政策点评。

=== 新政策 ===
《{segment.title}》（{segment.timestamp.strftime('%Y年%m月%d日') if segment.timestamp else 'N/A'}）

{segment.content}

=== 历史政策（{num_history}篇） ===
{history_for_llm}

=== 分析任务 ===

请对比新政策与上述{num_history}篇历史政策，撰写"{topic}"主题的深度分析。

重点回答：
1. 新政策释放了什么信号？政策方向是加码还是收缩？
2. 相比历史政策，有哪些边际变化（新增/强化/弱化）？

=== 输出格式 ===

#### 核心观点

**投资评级**：看多/看平/看空
**核心逻辑**：说清楚政策信号和最大边际变化

#### 政策对比与边际变化

**新政策核心表述**：
直接引用新政策中关于{topic}的重要表述

**与历史政策对比**：

| 历史政策表述 | 边际变化 |
|-------------|---------|
| 《政策名》（YYYY年MM月）完整引用历史政策原文 | 具体说明变化内容 |

要求：
- 新政策表述单独列在表格前面，不要放在表格里
- 表格只有两列：历史政策表述、边际变化
- 只选取与该主题高度相关的历史政策进行对比，相关性弱的不要放入表格
- 历史政策表述要带上政策名称和时间，格式：《政策名》（YYYY年MM月）原文内容
- 边际变化要具体说明与新政策相比的变化内容，不要只写"强化"、"延续"
- 列出3-5个不同的对比要点，不要重复


=== 要求 ===

1. 新政策表述单独列在表格前面，表格只有两列（历史政策表述、边际变化）
2. 只展示与该主题高度相关的对比，相关性弱的历史政策不要放入表格
3. 历史政策表述必须带上《政策名》（YYYY年MM月），完整引用原文
4. 引用原文时直接写出来，不要用特殊引号格式
5. 边际变化要具体描述变化内容，不要只写"强化/延续"等笼统词汇
6. 不要用省略号(...)省略内容，完整输出所有分析
7. 总字数1500-2500字"""

        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.llm_client.chat_completion(
                messages=messages,
                temperature=0.2,
                max_tokens=32768
            )
            
            final_output = f"""### 3.{topic_idx} {topic}

本主题共检索到 **{len(topic_docs)}** 篇相关历史政策。

---

{response.strip()}

"""
            return final_output
        except Exception as e:
            self.log(f"  主题'{topic}'深度分析生成失败: {e}", level="warning")
            return f"### 3.{topic_idx} 主题：{topic}\n\n（生成失败）\n\n"
