"""
简化版RAG Agent - 纯chunk级别向量相似度匹配
"""
from typing import List, Dict, Any
from collections import defaultdict
from .base import BaseAgent


class SimplifiedRAGAgent(BaseAgent):
    """
    简化版RAG Agent，支持：
    1. 纯chunk级别向量相似度检索
    2. 按doc_id合并chunks
    """
    
    def __init__(self, vector_db, **kwargs):
        super().__init__("SimplifiedRAGAgent", kwargs.get('config'))
        self.vector_db = vector_db
    
    def process(self, input_data: Any) -> Any:
        """
        处理数据的主方法（实现抽象方法）
        
        Args:
            input_data: 可以是查询文本或查询chunks列表
            
        Returns:
            搜索结果
        """
        if isinstance(input_data, str):
            return self.search_enhanced(query_text=input_data)
        elif isinstance(input_data, list):
            return self.search_enhanced(query_chunks=input_data)
        else:
            return self.search_enhanced(query_text=str(input_data))
    
    def search_chunks_by_similarity(self, query_text: str = None, query_chunks: List[str] = None, top_k: int = 10, exclude_doc_id: str = None, use_reranker: bool = True) -> List[Dict[str, Any]]:
        """
        纯chunk级别向量相似度搜索 + Reranking精排
        
        Args:
            query_text: 查询文本（传统模式）
            query_chunks: 查询chunks列表（精细化模式，优先级更高）
            top_k: 返回结果数量
            exclude_doc_id: 要排除的doc_id（避免匹配到自己）
            use_reranker: 是否使用Reranker精排（默认True）
            
        Returns:
            chunk级别的搜索结果（如果启用reranker，会添加'rerank_score'字段）
        """
        if query_chunks:
            # 精细化模式：多query并行，然后全局reranking
            results = self.vector_db.search_chunks_multi_query(
                query_chunks, 
                top_k_per_query=top_k, 
                exclude_doc_id=exclude_doc_id,
                use_reranker=use_reranker,
                final_top_k=top_k * 2  # 召回2倍数量，供后续合并
            )
        else:
            # 传统模式：单query + reranking
            results = self.vector_db.search_chunks(
                query_text or "", 
                top_k=top_k, 
                exclude_doc_id=exclude_doc_id,
                use_reranker=use_reranker
            )
        
        formatted_results = []
        for result in results:
            # 再次过滤，确保不包含exclude_doc_id
            if exclude_doc_id and result.get('doc_id') == exclude_doc_id:
                continue
            formatted_results.append({
                'chunk_id': result.get('chunk_id'),
                'doc_id': result.get('doc_id'),
                'content': result.get('content', ''),
                'chunk_index': result.get('chunk_index', 0),
                'title': result.get('title', ''),
                'timestamp': result.get('timestamp', ''),
                'industries': result.get('industries', ''),
                'investment_relevance': result.get('investment_relevance', ''),
                'report_series': result.get('report_series', 'N/A'),  # ⭐ 报告系列
                'similarity': result.get('similarity', 0.0),
                'source': 'chunk_similarity_fine_grained' if query_chunks else 'chunk_similarity'
            })
        
        return formatted_results
    
    def merge_chunks_by_doc_id(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        按doc_id合并chunks
        
        Args:
            chunks: chunk列表
            
        Returns:
            按doc_id合并后的文档列表（最多5个）
        """
        if not chunks:
            return []
        
        # 按doc_id分组
        docs_by_id = defaultdict(list)
        chunks_without_doc_id = []
        
        for chunk in chunks:
            doc_id = chunk.get('doc_id')
            if doc_id and str(doc_id).strip():
                docs_by_id[doc_id].append(chunk)
            else:
                chunks_without_doc_id.append(chunk)
        
        # 合并每个文档的chunks
        merged_docs = []
        for doc_id, doc_chunks in docs_by_id.items():
            # 去重：按chunk_id去重
            unique_chunks = {}
            for chunk in doc_chunks:
                chunk_id = chunk.get('chunk_id')
                if chunk_id not in unique_chunks:
                    unique_chunks[chunk_id] = chunk
                else:
                    if chunk.get('similarity', 0) > unique_chunks[chunk_id].get('similarity', 0):
                        unique_chunks[chunk_id] = chunk
            
            doc_chunks = list(unique_chunks.values())
            doc_chunks.sort(key=lambda x: x.get('chunk_index', 0))
            
            # 合并内容
            merged_content = ""
            for chunk in doc_chunks:
                content = chunk.get('content', '')
                if content:
                    merged_content += f"{content}\n\n"
            
            # 计算平均相似度
            similarities = [chunk.get('similarity', 0) for chunk in doc_chunks]
            avg_similarity = sum(similarities) / len(similarities) if similarities else 0
            
            first_chunk = doc_chunks[0]
            merged_docs.append({
                'doc_id': doc_id,
                'title': first_chunk.get('title', ''),
                'timestamp': first_chunk.get('timestamp', ''),
                'industries': first_chunk.get('industries', ''),
                'investment_relevance': first_chunk.get('investment_relevance', ''),
                'report_series': first_chunk.get('report_series', 'N/A'),  # ⭐ 报告系列
                'content': merged_content.strip(),
                'chunk_count': len(doc_chunks),
                'avg_similarity': avg_similarity,
                'similarity': avg_similarity,  # ⭐ 添加similarity字段，方便其他代码使用
                'chunks': doc_chunks
            })
        
        # 处理没有doc_id的chunks
        for chunk in chunks_without_doc_id:
            chunk_similarity = chunk.get('similarity', 0)
            merged_docs.append({
                'doc_id': chunk.get('chunk_id', ''),
                'title': chunk.get('title', ''),
                'timestamp': chunk.get('timestamp', ''),
                'industries': chunk.get('industries', ''),
                'investment_relevance': chunk.get('investment_relevance', ''),
                'report_series': chunk.get('report_series', 'N/A'),  # ⭐ 报告系列
                'content': chunk.get('content', ''),
                'chunk_count': 1,
                'avg_similarity': chunk_similarity,
                'similarity': chunk_similarity,  # ⭐ 添加similarity字段
                'chunks': [chunk]
            })
        
        # 按平均相似度排序
        merged_docs.sort(key=lambda x: x.get('avg_similarity', 0), reverse=True)
        
        # 按doc_id去重
        unique_docs = {}
        for doc in merged_docs:
            doc_id = doc.get('doc_id', '').strip()
            if doc_id:
                if doc_id not in unique_docs:
                    unique_docs[doc_id] = doc
                else:
                    # 合并内容
                    existing = unique_docs[doc_id]
                    existing_chunks = {c.get('chunk_id'): c for c in existing.get('chunks', [])}
                    new_chunks = {c.get('chunk_id'): c for c in doc.get('chunks', [])}
                    all_chunks = {**existing_chunks, **new_chunks}
                    
                    merged_content = ""
                    for chunk in sorted(all_chunks.values(), key=lambda x: x.get('chunk_index', 0)):
                        content = chunk.get('content', '')
                        if content:
                            merged_content += f"{content}\n\n"
                    
                    similarities = [c.get('similarity', 0) for c in all_chunks.values()]
                    new_avg = sum(similarities) / len(similarities) if similarities else 0
                    
                    unique_docs[doc_id] = {
                        'doc_id': doc_id,
                        'title': existing.get('title', doc.get('title', '')),
                        'timestamp': existing.get('timestamp', doc.get('timestamp', '')),
                        'industries': existing.get('industries', doc.get('industries', '')),
                        'investment_relevance': existing.get('investment_relevance', doc.get('investment_relevance', '')),
                        'report_series': existing.get('report_series', doc.get('report_series', 'N/A')),  # ⭐ 报告系列
                        'content': merged_content.strip(),
                        'chunk_count': len(all_chunks),
                        'avg_similarity': new_avg,
                        'similarity': new_avg,  # ⭐ 添加similarity字段
                        'chunks': list(all_chunks.values())
                    }
        
        final_docs = list(unique_docs.values())
        final_docs.sort(key=lambda x: x.get('avg_similarity', 0), reverse=True)
        
        # ⭐ 移除数量限制，返回所有合并后的文档（由调用方控制数量）
        return final_docs
    
    def search_enhanced(self, query_text: str = None, query_chunks: List[str] = None, top_k: int = 10, exclude_doc_id: str = None, use_reranker: bool = True) -> List[Dict[str, Any]]:
        """
        简化版搜索接口（支持Reranking精排）
        
        Args:
            query_text: 查询文本（传统模式）
            query_chunks: 查询chunks列表（精细化模式，优先级更高）
            top_k: 返回结果数量
            exclude_doc_id: 要排除的doc_id（避免匹配到自己）
            use_reranker: 是否使用Reranker精排（默认True）
            
        Returns:
            搜索结果（按doc_id合并）
        """
        rag_results = self.search_chunks_by_similarity(
            query_text=query_text, 
            query_chunks=query_chunks,
            top_k=top_k*2,
            exclude_doc_id=exclude_doc_id,
            use_reranker=use_reranker
        )
        
        merged_docs = self.merge_chunks_by_doc_id(rag_results)
        
        # 再次过滤，确保不包含exclude_doc_id
        if exclude_doc_id:
            merged_docs = [doc for doc in merged_docs if doc.get('doc_id') != exclude_doc_id]
        
        return merged_docs[:top_k]
