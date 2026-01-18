"""
Reranker模块 - 二阶段精排（提升检索精度）

使用方式：
    from utils.reranker import get_reranker
    
    reranker = get_reranker()
    results = vector_db.search_chunks(query, top_k=50)  # 粗排：召回50个
    results = reranker.rerank(query, results, top_k=10)  # 精排：选出最好的10个
"""
from typing import List, Dict, Any
import torch


class BCEReranker:
    """BCE Reranker - 基于Cross-Encoder的精排模型"""
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        """
        初始化Reranker
        
        Args:
            model_name: Hugging Face模型名称
        """
        print(f"[Reranker] 正在加载模型: {model_name}")
        
        try:
            from sentence_transformers import CrossEncoder
            
            # 检查safetensors是否安装（避免torch版本问题）
            try:
                import safetensors
                print(f"[Reranker] ✅ safetensors已安装")
            except ImportError:
                print(f"[Reranker] ⚠️ safetensors未安装，可能导致加载失败")
                print(f"[Reranker] 💡 建议安装: pip install safetensors")
            
            # 强制使用safetensors格式加载（避免torch版本问题）
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"[Reranker] 🔄 强制使用safetensors格式加载模型...")
            
            # 设置环境变量，强制transformers使用safetensors
            import os
            os.environ['TRANSFORMERS_OFFLINE'] = '0'  # 确保可以下载
            
            self.model = CrossEncoder(
                model_name, 
                device=device,
                model_kwargs={
                    'use_safetensors': True,  # 强制使用safetensors
                    'ignore_mismatched_sizes': False
                }
            )
            
            print(f"[Reranker] ✅ 模型已加载")
            if torch.cuda.is_available():
                print(f"[Reranker] ✅ 使用GPU: {torch.cuda.get_device_name(0)}")
            else:
                print(f"[Reranker] ⚠️ 使用CPU（速度较慢）")
            
            self.enabled = True
            
        except Exception as e:
            error_msg = str(e)
            print(f"[Reranker] ❌ 模型加载失败: {error_msg}")
            
            # 根据错误类型给出具体建议
            if "torch.load" in error_msg or "CVE-2025" in error_msg:
                print(f"[Reranker] 💡 解决方案1（推荐）：安装safetensors")
                print(f"   pip install safetensors")
                print(f"[Reranker] 💡 解决方案2：升级PyTorch到2.6+")
                print(f"   pip install torch>=2.6.0 --upgrade")
            else:
                print(f"[Reranker] 💡 请安装依赖: pip install sentence-transformers safetensors")
            
            self.enabled = False
    
    def rerank(
        self, 
        query: str, 
        results: List[Dict[str, Any]], 
        top_k: int = 10,
        query_max_length: int = 512,
        passage_max_length: int = 512
    ) -> List[Dict[str, Any]]:
        """
        对检索结果进行重排序
        
        Args:
            query: 查询文本
            results: 检索结果列表（每个结果必须有'content'字段）
            top_k: 返回top-K结果
            query_max_length: query最大长度
            passage_max_length: passage最大长度
            
        Returns:
            重排序后的结果列表（添加了'rerank_score'字段）
        """
        if not self.enabled:
            print(f"[Reranker] ⚠️ Reranker未启用，返回原始结果")
            # 添加默认的rerank_score（方便调试）
            for i, result in enumerate(results):
                result['rerank_score'] = 0.0  # 未启用时设为0
                result['original_rank'] = i + 1
            return results[:top_k]
        
        if not results:
            return []
        
        print(f"[Reranker] 🔄 正在精排 {len(results)} 个候选文档...")
        
        # 截断query和passage
        query = query[:query_max_length]
        
        # 构建query-passage对
        pairs = []
        for r in results:
            passage = r.get('content', '')[:passage_max_length]
            pairs.append([query, passage])
        
        # 批量计算精排分数
        try:
            scores = self.model.predict(pairs, show_progress_bar=False)
            
            # 将分数添加到结果中
            for i, result in enumerate(results):
                result['rerank_score'] = float(scores[i])
                result['original_rank'] = i + 1  # 记录原始排名
            
            # 按精排分数排序
            results.sort(key=lambda x: x['rerank_score'], reverse=True)
            
            print(f"[Reranker] ✅ 精排完成，返回top-{min(top_k, len(results))} 结果")
            
            # 打印前3个结果的分数对比
            for i, r in enumerate(results[:3]):
                original_rank = r.get('original_rank', '?')
                rerank_score = r.get('rerank_score', 0)
                original_score = r.get('similarity', 0)
                print(f"  [{i+1}] 原排名:{original_rank}, 向量分:{original_score:.4f}, 精排分:{rerank_score:.4f}")
            
            return results[:top_k]
            
        except Exception as e:
            print(f"[Reranker] ❌ 精排失败: {e}，返回原始结果")
            import traceback
            print(f"[Reranker] 错误详情:")
            traceback.print_exc()
            # 添加默认的rerank_score（方便调试）
            for i, result in enumerate(results):
                if 'rerank_score' not in result:
                    result['rerank_score'] = 0.0
                if 'original_rank' not in result:
                    result['original_rank'] = i + 1
            return results[:top_k]


# 全局单例
_reranker_instance = None


def get_reranker():
    """
    获取Reranker单例（避免重复加载模型）
    
    优先尝试手动加载方式（绕过torch版本检查），失败后回退到标准加载
    """
    global _reranker_instance
    if _reranker_instance is None:
        # 优先尝试手动加载（避免torch版本问题）
        try:
            from .reranker_manual import get_manual_reranker
            print("[Reranker] 尝试使用手动加载方式...")
            manual_reranker = get_manual_reranker()
            if manual_reranker.enabled:
                print("[Reranker] ✅ 手动加载成功！")
                _reranker_instance = manual_reranker
                return _reranker_instance
            else:
                print("[Reranker] ⚠️ 手动加载失败，尝试标准加载...")
        except Exception as e:
            print(f"[Reranker] ⚠️ 手动加载出错: {e}，尝试标准加载...")
        
        # 回退到标准加载
        _reranker_instance = BCEReranker()
    
    return _reranker_instance

