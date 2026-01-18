"""
智能文档Chunk切分工具
支持按段落/条款切分政策文档
严格控制chunk长度，确保不超过Milvus限制
"""
import re
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class DocumentChunk:
    """文档块（包含完整元数据供RAG使用）"""
    # Chunk标识字段
    chunk_id: str
    doc_id: str
    chunk_index: int
    chunk_type: str  # 'clause'（条款）或 'paragraph'（段落）
    
    # Chunk内容
    content: str  # chunk文本内容（最大450字符，受embedding模型限制）
    
    # 文档元数据（从PolicySegment继承）
    title: str = ""
    timestamp: str = ""
    
    # 行业标签元数据（经过DS32B过滤后的结果）⭐ 核心字段
    industries: str = ""  # 中信一级行业（逗号分隔）
    investment_relevance: str = ""  # 投资相关性：高/低
    report_series: str = ""  # ⭐ 报告系列：两会-政府工作报告/两会-计划报告/两会-预算报告/五年规划-建议/五年规划-纲要/国务院常务会议/国务院常务会议-解读/中央政治局会议/中央经济工作会议/全国财政工作会议/中央委员会全体会议公报/N/A
    industry_policy_segments: str = ""  # 行业及对应政策片段（JSON格式）
    

class PolicyDocumentChunker:
    """政策文档智能切分器（优化版：更大chunk + 语义感知分割）"""
    
    def __init__(
        self,
        chunk_size_target: int = 800,  # ⭐ 大幅增加：800字符，保持更完整的语义上下文
        chunk_size_max: int = 1000,    # ⭐ 大幅增加：1000字符，允许更大的chunk
        overlap: int = 150,            # ⭐ 增加重叠：150字符，确保上下文连贯
        absolute_max: int = 1200       # ⭐ 大幅增加：1200字符（约600-800 tokens）
    ):
        """
        Args:
            chunk_size_target: 目标chunk大小（字符数）- 推荐800，保持完整语义
            chunk_size_max: 最大chunk大小（触发切分的阈值）- 推荐1000
            overlap: chunk间重叠字符数 - 推荐150，保持上下文连贯
            absolute_max: 绝对最大长度，基于以下限制：
                         1. Embedding模型: xiaobu-embedding-v2 支持较长输入
                         2. 中文: 1字符 ≈ 1-1.5 tokens
                         3. 安全值: 1200字符 ≈ 1200-1800 tokens
                         4. Milvus: VARCHAR(5000) - 远大于此，不是瓶颈
                         
        优化说明（2024-12更新）：
        - 更大的chunk（800-1200字符）可以保持更完整的政策语义
        - 政策文档通常按条款组织，每个条款200-500字，需要2-3个条款才能表达完整政策点
        - 小chunk（400-500字符）容易导致语义碎片化，检索时匹配到不相关内容
        - 更大的overlap（150字符）确保跨chunk的政策点不丢失
        """
        self.chunk_size_target = chunk_size_target
        self.chunk_size_max = chunk_size_max
        self.overlap = overlap
        self.absolute_max = absolute_max
        
        # ⭐ 优化：优先在段落/条款边界截断，保持语义完整
        # 优先级：段落分隔符 > 条款标记 > 句号 > 其他标点
        self.paragraph_delimiters = ['\n\n', '\n\n\n', '\r\n\r\n']  # 段落分隔符（最高优先级）
        self.sentence_delimiters = ['。', '！', '？', '；']  # 句子分隔符（次优先级）
        self.all_delimiters = self.paragraph_delimiters + self.sentence_delimiters
        
        # 政策文档常见的条款标记模式
        self.clause_patterns = [
            r'^[一二三四五六七八九十]+[、\.]',  # 一、 二、
            r'^\（[一二三四五六七八九十]+\）',  # （一）（二）
            r'^\d+[、\.]',  # 1. 2.
            r'^\(\d+\)',  # (1) (2)
            r'^第[一二三四五六七八九十\d]+[条章节]',  # 第一条 第二章
            r'^【.*?】',  # 【重要】【通知】
        ]
    
    def _smart_truncate(self, text: str, max_length: int) -> str:
        """
        智能截断：优先在段落/条款边界截断，保持语义完整
        
        Args:
            text: 待截断文本
            max_length: 最大长度
            
        Returns:
            截断后的文本
        """
        if len(text) <= max_length:
            return text
        
        # 在max_length范围内查找最佳截断点
        truncated = text[:max_length]
        
        # ⭐ 优化：优先在段落分隔符处截断（保持完整段落）
        best_pos = -1
        best_priority = -1
        
        # 优先级1：段落分隔符（最高优先级）
        for delimiter in self.paragraph_delimiters:
            pos = truncated.rfind(delimiter)
            if pos > best_pos and pos > max_length * 0.5:  # 至少保留50%内容
                best_pos = pos + len(delimiter)
                best_priority = 1
        
        # 优先级2：句子分隔符（如果没找到段落分隔符）
        if best_pos < 0:
            for delimiter in self.sentence_delimiters:
                pos = truncated.rfind(delimiter)
                if pos > best_pos and pos > max_length * 0.6:  # 至少保留60%内容
                    best_pos = pos + len(delimiter)
                    best_priority = 2
        
        # 如果找到合适的截断点，在那里截断
        if best_pos > 0:
            return text[:best_pos].strip()
        
        # 优先级3：在空格处截断（避免截断单词）
        space_pos = truncated.rfind(' ')
        if space_pos > max_length * 0.8:
            return text[:space_pos].strip()
        
        # 最后手段：硬截断
        return truncated.strip()
    
    def chunk_document(
        self,
        doc_id: str,
        title: str,
        content: str,
        timestamp: str = "",
        industries: str = "",
        investment_relevance: str = "",
        report_series: str = "",  # ⭐ 报告系列
        industry_policy_segments: str = "",  # JSON格式：{"行业名": ["政策片段1", "政策片段2"]}
        create_summary: bool = False  # 默认不创建summary
    ) -> List[DocumentChunk]:
        """
        对文档进行智能切分（严格长度控制+完整元数据）
        
        Args:
            doc_id: 文档ID
            title: 文档标题
            content: 文档内容
            timestamp: 时间戳（ISO格式）
            industries: 中信一级行业（逗号分隔）
            create_summary: 是否创建摘要chunk
            
        Returns:
            DocumentChunk列表（每个chunk保证≤450字符，包含完整元数据）
        """
        # ⭐ 预先截断所有元数据，确保不超过Milvus限制
        title = str(title)[:500]
        timestamp = str(timestamp)[:150]
        industries = str(industries)[:500]
        
        chunks = []
        
        # 不再创建summary chunk，直接从内容开始切分
        
        # 按段落切分
        paragraphs = self._split_into_paragraphs(content)
        
        # 合并段落为chunks
        current_chunk_parts = []
        current_chunk_size = 0
        chunk_idx = 0  # 从0开始，不再有summary chunk
        
        for para in paragraphs:
            para_size = len(para)
            
            # 单个段落太长，智能切分
            if para_size > self.absolute_max:
                # 先保存当前积累的
                if current_chunk_parts:
                    chunk_content = '\n\n'.join(current_chunk_parts)
                    chunk_content = self._smart_truncate(chunk_content, self.absolute_max)
                    chunks.append(DocumentChunk(
                        chunk_id=f"{doc_id}_chunk_{chunk_idx}",
                        doc_id=doc_id,
                        chunk_index=chunk_idx,
                        chunk_type='paragraph',
                        content=chunk_content,
                        title=title,
                        timestamp=timestamp,
                        industries=industries,
                        investment_relevance=investment_relevance,
                        report_series=report_series,
                        industry_policy_segments=industry_policy_segments
                    ))
                    chunk_idx += 1
                    current_chunk_parts = []
                    current_chunk_size = 0
                
                # 超长段落智能切分（在句号处截断）
                remaining = para
                while len(remaining) > 0:
                    if len(remaining) <= self.absolute_max:
                        chunk_part = remaining
                        remaining = ""
                    else:
                        chunk_part = self._smart_truncate(remaining, self.absolute_max)
                        remaining = remaining[len(chunk_part):].lstrip()
                    
                    if chunk_part:
                        chunks.append(DocumentChunk(
                            chunk_id=f"{doc_id}_chunk_{chunk_idx}",
                            doc_id=doc_id,
                            chunk_index=chunk_idx,
                            chunk_type='paragraph',
                            content=chunk_part,
                            title=title,
                            timestamp=timestamp,
                            industries=industries,
                            investment_relevance=investment_relevance,
                            report_series=report_series,
                            industry_policy_segments=industry_policy_segments
                        ))
                        chunk_idx += 1
                continue
            
            # 检测是否是新条款
            is_clause = self._is_clause_start(para)
            
            # ⭐ 优化：决定是否开始新chunk（更智能的策略）
            # 1. 遇到新条款：强制开始新chunk（保持条款完整性）
            # 2. 超过目标大小：开始新chunk
            # 3. 如果当前段落很大且会超过max，也提前切分
            should_start_new = (
                is_clause or  # 遇到新条款（最高优先级）
                current_chunk_size + para_size > self.chunk_size_target or  # 超过目标大小
                (current_chunk_size > 0 and current_chunk_size + para_size > self.chunk_size_max)  # 会超过最大限制
            )
            
            if should_start_new and current_chunk_parts:
                # 保存当前chunk（智能截断）
                chunk_content = '\n\n'.join(current_chunk_parts)
                chunk_content = self._smart_truncate(chunk_content, self.absolute_max)
                
                chunks.append(DocumentChunk(
                    chunk_id=f"{doc_id}_chunk_{chunk_idx}",
                    doc_id=doc_id,
                    chunk_index=chunk_idx,
                    chunk_type='clause' if any(self._is_clause_start(p) for p in current_chunk_parts) else 'paragraph',
                    content=chunk_content,
                    title=title,
                    timestamp=timestamp,
                    industries=industries,
                    investment_relevance=investment_relevance,
                    report_series=report_series,
                    industry_policy_segments=industry_policy_segments
                ))
                chunk_idx += 1
                
                # ⭐ 优化：添加重叠（使用更大的重叠窗口）
                if self.overlap > 0 and current_chunk_parts:
                    # 从最后一个段落提取重叠文本
                    last_part = current_chunk_parts[-1]
                    if len(last_part) > self.overlap:
                        # 尝试在句子边界处截取重叠部分
                        overlap_text = self._extract_overlap_text(last_part, self.overlap)
                    else:
                        overlap_text = last_part
                    
                    current_chunk_parts = [overlap_text, para] if overlap_text else [para]
                    current_chunk_size = len(overlap_text) + para_size if overlap_text else para_size
                else:
                    current_chunk_parts = [para]
                    current_chunk_size = para_size
            else:
                # 继续添加到当前chunk
                current_chunk_parts.append(para)
                current_chunk_size += para_size
            
            # 如果累积太大，智能切分
            if current_chunk_size > self.chunk_size_max:
                chunk_content = '\n\n'.join(current_chunk_parts)
                chunk_content = self._smart_truncate(chunk_content, self.absolute_max)
                
                chunks.append(DocumentChunk(
                    chunk_id=f"{doc_id}_chunk_{chunk_idx}",
                    doc_id=doc_id,
                    chunk_index=chunk_idx,
                    chunk_type='paragraph',
                    content=chunk_content,
                    title=title,
                    timestamp=timestamp,
                    industries=industries,
                    investment_relevance=investment_relevance,
                    report_series=report_series,
                    industry_policy_segments=industry_policy_segments
                ))
                chunk_idx += 1
                current_chunk_parts = []
                current_chunk_size = 0
        
        # 保存最后一个chunk（智能截断）
        if current_chunk_parts:
            chunk_content = '\n\n'.join(current_chunk_parts)
            chunk_content = self._smart_truncate(chunk_content, self.absolute_max)
            
            chunks.append(DocumentChunk(
                chunk_id=f"{doc_id}_chunk_{chunk_idx}",
                doc_id=doc_id,
                chunk_index=chunk_idx,
                chunk_type='paragraph',
                content=chunk_content,
                title=title,
                timestamp=timestamp,
                industries=industries,
                investment_relevance=investment_relevance,
                report_series=report_series,
                industry_policy_segments=industry_policy_segments
            ))
        
        # 最终验证：确保所有chunk都不超过absolute_max（智能截断）
        for chunk in chunks:
            if len(chunk.content) > self.absolute_max:
                print(f"⚠️ 警告: chunk {chunk.chunk_id} 超长({len(chunk.content)})，智能截断到{self.absolute_max}")
                chunk.content = self._smart_truncate(chunk.content, self.absolute_max)
        
        max_len = max(len(c.content) for c in chunks) if chunks else 0
        print(f"✅ 文档切分完成: {len(chunks)} 个chunks，最大长度={max_len}字符 (限制:{self.absolute_max})")
        return chunks
    
    def _split_into_paragraphs(self, content: str) -> List[str]:
        """
        将文档切分为段落（优化版：更智能的段落识别）
        
        Args:
            content: 文档内容
            
        Returns:
            段落列表
        """
        # ⭐ 优化：先按双换行符切分（主要段落分隔符）
        paragraphs = re.split(r'\n\s*\n+', content)
        
        # 如果段落太少，尝试按单换行符+条款标记切分
        if len(paragraphs) < 3:
            # 尝试按条款标记切分（如：一、二、三、）
            clause_split = re.split(r'(\n[一二三四五六七八九十]+[、\.])', content)
            if len(clause_split) > len(paragraphs):
                # 合并条款标记和内容
                merged = []
                for i in range(0, len(clause_split) - 1, 2):
                    if i + 1 < len(clause_split):
                        merged.append(clause_split[i] + clause_split[i + 1])
                if merged:
                    paragraphs = merged
        
        # 过滤空段落并清理
        paragraphs = [p.strip() for p in paragraphs if p.strip() and len(p.strip()) > 10]  # 过滤太短的段落
        
        return paragraphs
    
    def _is_clause_start(self, paragraph: str) -> bool:
        """
        判断段落是否是条款开头
        
        Args:
            paragraph: 段落文本
            
        Returns:
            是否是条款开头
        """
        for pattern in self.clause_patterns:
            if re.match(pattern, paragraph.strip()):
                return True
        return False
    
    def _extract_overlap_text(self, text: str, target_length: int) -> str:
        """
        从文本末尾提取重叠文本（在句子边界处截取）
        
        Args:
            text: 源文本
            target_length: 目标长度
            
        Returns:
            重叠文本
        """
        if len(text) <= target_length:
            return text
        
        # 从末尾开始，查找句子边界
        start_pos = len(text) - target_length
        truncated = text[start_pos:]
        
        # 查找第一个句子分隔符
        for delimiter in self.sentence_delimiters:
            pos = truncated.find(delimiter)
            if pos > 0 and pos < target_length * 0.3:  # 在开头30%内找到
                return truncated[pos + len(delimiter):].strip()
        
        # 如果没找到，返回末尾文本
        return truncated.strip()


def chunk_documents_batch(
    documents: List[Dict[str, Any]],
    chunker: PolicyDocumentChunker = None
) -> Dict[str, List[DocumentChunk]]:
    """
    批量切分文档
    
    Args:
        documents: 文档列表，每个文档包含 {'doc_id', 'title', 'content'}
        chunker: 切分器实例，None则使用默认配置
        
    Returns:
        {doc_id: [chunks]} 的字典
    """
    if chunker is None:
        chunker = PolicyDocumentChunker()
    
    result = {}
    for doc in documents:
        chunks = chunker.chunk_document(
            doc_id=doc['doc_id'],
            title=doc['title'],
            content=doc['content']
        )
        result[doc['doc_id']] = chunks
    
    return result


if __name__ == "__main__":
    # 测试chunking
    test_doc = {
        'doc_id': 'test_001',
        'title': '关于促进新能源汽车产业发展的税收优惠政策',
        'content': '''
为深入贯彻落实国家新能源发展战略，促进新能源汽车产业健康发展，现就有关税收优惠政策通知如下：

一、购置税减免政策
对购置新能源汽车的消费者，免征车辆购置税。该政策执行期限延长至2025年12月31日。新能源汽车包括纯电动汽车、插电式混合动力汽车和燃料电池汽车。

二、企业所得税优惠
对符合条件的新能源汽车生产企业，其研发费用可按照实际发生额的100%加计扣除。对新能源汽车关键零部件生产企业，减按15%的税率征收企业所得税。

三、增值税优惠
销售新能源汽车及关键零部件，增值税税率由13%降至9%。对充电基础设施建设和运营企业，提供增值税即征即退政策。

四、地方税收支持
鼓励地方政府根据实际情况，对新能源汽车产业给予地方税收减免或财政补贴支持。

本通知自2024年1月1日起执行。
        '''
    }
    
    chunker = PolicyDocumentChunker()
    chunks = chunker.chunk_document(
        doc_id=test_doc['doc_id'],
        title=test_doc['title'],
        content=test_doc['content']
    )
    
    print(f"\n总共切分为 {len(chunks)} 个chunks:\n")
    for chunk in chunks:
        print(f"Chunk {chunk.chunk_index} [{chunk.chunk_type}]:")
        print(f"  内容长度: {len(chunk.content)} 字符")
        print(f"  内容预览: {chunk.content[:100]}...")
        print()
    
    # 验证长度
    max_len = max(len(c.content) for c in chunks)
    print(f"最大chunk长度: {max_len} 字符")
    if max_len <= 1200:
        print(f"✅ 所有chunk长度符合要求（≤1200字符）！")
    else:
        print(f"❌ 有chunk超长（限制1200字符）！")
