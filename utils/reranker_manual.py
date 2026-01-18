"""
Reranker手动加载版 - 绕过torch版本检查
直接加载safetensors文件，不依赖transformers的自动加载机制
"""
from typing import List, Dict, Any
import torch
import os
from pathlib import Path


class ManualBCEReranker:
    """手动加载BCE Reranker - 绕过torch版本检查"""
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        """
        初始化Reranker（手动加载方式）
        
        Args:
            model_name: Hugging Face模型名称
        """
        print(f"[Reranker-手动] 正在初始化模型: {model_name}")
        
        self.enabled = False
        
        try:
            # 1. 检查是否已下载模型
            cache_dir = self._get_model_cache_dir(model_name)
            if not cache_dir:
                print(f"[Reranker-手动] 模型未下载，正在下载...")
                cache_dir = self._download_model(model_name)
            
            print(f"[Reranker-手动] 模型缓存位置: {cache_dir}")
            
            # 2. 手动加载tokenizer
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                cache_dir,
                local_files_only=True  # 只使用本地文件
            )
            print(f"[Reranker-手动] ✅ Tokenizer加载完成")
            
            # 3. 手动加载模型（使用safetensors，绕过torch.load检查）
            from transformers import AutoModelForSequenceClassification
            from safetensors.torch import load_file
            
            # 查找safetensors文件（支持多种缓存结构）
            safetensors_files = list(Path(cache_dir).glob("*.safetensors"))
            
            if not safetensors_files:
                # 方法1: 尝试从当前目录的子目录查找
                for sub_dir in Path(cache_dir).rglob("*.safetensors"):
                    safetensors_files.append(sub_dir)
                    if safetensors_files:
                        cache_dir = str(sub_dir.parent)
                        break
            
            if not safetensors_files:
                # 方法2: 尝试从snapshots目录查找
                cache_path = Path(cache_dir)
                if "snapshots" in cache_path.parts:
                    # 已经在snapshots目录中
                    pass
                else:
                    # 尝试查找snapshots目录
                    snapshots_dir = cache_path.parent / "snapshots"
                    if snapshots_dir.exists():
                        for snapshot_dir in snapshots_dir.iterdir():
                            if snapshot_dir.is_dir():
                                safetensors_files = list(snapshot_dir.glob("*.safetensors"))
                                if safetensors_files:
                                    cache_dir = str(snapshot_dir)
                                    break
            
            # 如果没有safetensors文件，自动下载
            if not safetensors_files:
                print(f"[Reranker-手动] ⚠️ 未找到safetensors文件，正在下载...")
                print(f"[Reranker-手动] 💡 这需要约400MB，首次下载约需1-2分钟")
                cache_dir = self._download_model(model_name)
                safetensors_files = list(Path(cache_dir).glob("*.safetensors"))
                
                if not safetensors_files:
                    raise FileNotFoundError(f"下载后仍未找到safetensors文件")
            
            print(f"[Reranker-手动] 找到safetensors文件: {safetensors_files[0].name}")
            
            # 加载config
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(cache_dir, local_files_only=True)
            
            # 创建模型（不加载权重）
            self.model = AutoModelForSequenceClassification.from_config(config)
            
            # 手动加载safetensors权重
            print(f"[Reranker-手动] 🔄 手动加载safetensors权重...")
            state_dict = load_file(str(safetensors_files[0]))
            
            # 使用strict=False，允许忽略不匹配的key（如position_ids）
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            if unexpected_keys:
                print(f"[Reranker-手动] ℹ️ 忽略的键: {unexpected_keys[:3]}...")  # 只显示前3个
            print(f"[Reranker-手动] ✅ 模型权重加载完成")
            
            # 4. 移动到GPU
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                self.device = 'cuda'
                print(f"[Reranker-手动] ✅ 模型已加载到GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = 'cpu'
                print(f"[Reranker-手动] ⚠️ 使用CPU（速度较慢）")
            
            self.model.eval()
            self.enabled = True
            print(f"[Reranker-手动] ✅ 手动加载完成！")
            
        except Exception as e:
            print(f"[Reranker-手动] ❌ 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            self.enabled = False
    
    def _get_model_cache_dir(self, model_name: str) -> str:
        """获取模型缓存目录"""
        from huggingface_hub import try_to_load_from_cache, _CACHED_NO_EXIST
        
        # 尝试从缓存加载
        try:
            # 检查config.json是否存在
            cached_file = try_to_load_from_cache(
                repo_id=model_name,
                filename="config.json"
            )
            if cached_file and cached_file != _CACHED_NO_EXIST:
                return str(Path(cached_file).parent)
        except:
            pass
        
        return None
    
    def _download_model(self, model_name: str) -> str:
        """下载模型（只下载safetensors文件）"""
        from huggingface_hub import snapshot_download
        
        print(f"[Reranker-手动] 正在下载模型（只下载safetensors和配置文件）...")
        print(f"[Reranker-手动] 提示: 约400MB，可能需要1-2分钟")
        
        try:
            cache_dir = snapshot_download(
                repo_id=model_name,
                allow_patterns=["*.safetensors", "*.json", "tokenizer*", "vocab.txt", "special_tokens_map.json"],
                ignore_patterns=["*.bin", "*.msgpack", "*.h5", "*.onnx"],  # 明确忽略.bin文件
                resume_download=True  # 支持断点续传
            )
            print(f"[Reranker-手动] ✅ 模型下载完成: {cache_dir}")
            return cache_dir
        except Exception as e:
            print(f"[Reranker-手动] ❌ 下载失败: {e}")
            print(f"[Reranker-手动] 💡 手动下载命令:")
            print(f"   huggingface-cli download {model_name} --include '*.safetensors' --include '*.json' --include 'tokenizer*'")
            raise
    
    def rerank(
        self, 
        query: str, 
        results: List[Dict[str, Any]], 
        top_k: int = 10,
        query_max_length: int = 512,
        passage_max_length: int = 512,
        batch_size: int = 32
    ) -> List[Dict[str, Any]]:
        """
        对检索结果进行重排序（分批处理，避免显存溢出）
        
        Args:
            query: 查询文本
            results: 检索结果列表（每个结果必须有'content'字段）
            top_k: 返回top-K结果
            query_max_length: query最大长度
            passage_max_length: passage最大长度
            batch_size: 每批处理的文档数量（默认32，8GB显存建议16-32）
            
        Returns:
            重排序后的结果列表（添加了'rerank_score'字段）
        """
        if not self.enabled:
            print(f"[Reranker-手动] ⚠️ Reranker未启用，返回原始结果")
            for i, result in enumerate(results):
                result['rerank_score'] = 0.0
                result['original_rank'] = i + 1
            return results[:top_k]
        
        if not results:
            return []
        
        num_batches = (len(results) + batch_size - 1) // batch_size
        print(f"[Reranker-手动] 🔄 正在精排 {len(results)} 个候选文档（分{num_batches}批，每批{batch_size}个）...")
        
        try:
            all_scores = []
            
            # ⭐ 分批处理，避免显存溢出
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(results))
                batch_results = results[start_idx:end_idx]
                
                # 构建query-passage对
                pairs = []
                for r in batch_results:
                    passage = r.get('content', '')[:passage_max_length]
                    pairs.append((query[:query_max_length], passage))
                
                # 批量tokenize
                encoded = self.tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                )
                
                # 移动到GPU
                if self.device == 'cuda':
                    encoded = {k: v.cuda() for k, v in encoded.items()}
                
                # 推理
                with torch.no_grad():
                    outputs = self.model(**encoded)
                    batch_scores = outputs.logits.squeeze(-1).cpu().numpy()
                
                # 收集分数
                if batch_scores.ndim == 0:
                    all_scores.append(float(batch_scores))
                else:
                    all_scores.extend(batch_scores.tolist())
                
                # ⭐ 清理GPU缓存，防止内存累积
                if self.device == 'cuda':
                    del encoded, outputs
                    torch.cuda.empty_cache()
            
            # 将分数添加到结果中
            for i, result in enumerate(results):
                result['rerank_score'] = float(all_scores[i])
                result['original_rank'] = i + 1
            
            # 按精排分数排序
            results.sort(key=lambda x: x['rerank_score'], reverse=True)
            
            print(f"[Reranker-手动] ✅ 精排完成，返回top-{min(top_k, len(results))} 结果")
            
            # 打印前5个结果的分数对比（增加调试信息）
            for i, r in enumerate(results[:5]):
                original_rank = r.get('original_rank', '?')
                rerank_score = r.get('rerank_score', 0)
                original_score = r.get('similarity', 0)
                doc_id = r.get('doc_id', 'N/A')
                title = r.get('title', 'N/A')[:30] if r.get('title') else 'N/A'
                print(f"  [{i+1}] 原排名:{original_rank}, 向量分:{original_score:.4f}, 精排分:{rerank_score:.4f}, doc_id:{doc_id}, title:{title}...")
            
            return results[:top_k]
            
        except Exception as e:
            print(f"[Reranker-手动] ❌ 精排失败: {e}")
            import traceback
            traceback.print_exc()
            for i, result in enumerate(results):
                if 'rerank_score' not in result:
                    result['rerank_score'] = 0.0
                if 'original_rank' not in result:
                    result['original_rank'] = i + 1
            return results[:top_k]


# 全局单例
_manual_reranker_instance = None


def get_manual_reranker() -> ManualBCEReranker:
    """获取手动加载的Reranker单例"""
    global _manual_reranker_instance
    if _manual_reranker_instance is None:
        _manual_reranker_instance = ManualBCEReranker()
    return _manual_reranker_instance

