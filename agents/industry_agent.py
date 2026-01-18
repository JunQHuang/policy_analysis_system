"""
Industry Agent - 行业标签分类（关键词匹配 + DS32B过滤）
"""
from typing import List, Dict, Any
import sys
import json
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 处理相对导入（支持直接运行和模块导入）
if __name__ == "__main__":
    # 直接运行时，直接导入base.py文件（避免触发agents.__init__）
    import importlib.util
    base_path = project_root / "agents" / "base.py"
    spec = importlib.util.spec_from_file_location("agents_base", base_path)
    base_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base_module)
    BaseAgent = base_module.BaseAgent
else:
    # 作为模块导入时，使用相对导入
    from .base import BaseAgent

from models import PolicySegment
from utils.investment_relevance import judge_investment_and_industries


class IndustryAgent(BaseAgent):
    """行业标签分类Agent - 关键词匹配 + DS32B过滤"""
    
    def __init__(self, cache_file: str = None):
        """
        初始化IndustryAgent
        
        Args:
            cache_file: 缓存文件路径（用于断点续传），如果为None则使用默认路径
        """
        super().__init__("IndustryAgent")
        
        # 直接使用CITIC_INDUSTRY_TAG_GROUPS（一级行业为索引，值为所有关键词列表）
        from citic_industries import CITIC_INDUSTRY_TAG_GROUPS
        self.industry_tag_groups = CITIC_INDUSTRY_TAG_GROUPS
        self.log(f"✅ 行业标签组加载完成: {len(self.industry_tag_groups)} 个一级行业")
        
        # 设置缓存文件路径
        if cache_file is None:
            cache_file = project_root / "cache" / "industry_agent_cache.json"
        self.cache_file = Path(cache_file)
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache = self._load_cache()
        self.log(f"✅ 缓存文件: {self.cache_file}，已加载 {len(self.cache)} 条记录")
    
    def _load_cache(self) -> Dict[str, Dict]:
        """加载缓存文件"""
        if not self.cache_file.exists():
            return {}
        
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                cache = json.load(f)
                self.log(f"✅ 从缓存文件加载了 {len(cache)} 条记录")
                return cache
        except Exception as e:
            self.log(f"⚠️ 加载缓存失败: {e}，将创建新缓存")
            return {}
    
    def _save_cache(self):
        """保存缓存到文件"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
            self.log(f"✅ 缓存已保存到 {self.cache_file}")
        except Exception as e:
            self.log(f"⚠️ 保存缓存失败: {e}")
    
    def _get_cache_key(self, doc: PolicySegment) -> str:
        """
        生成缓存key（基于文档的唯一标识）
        
        使用 doc_id 作为key，如果doc_id相同则认为文档相同
        """
        return doc.doc_id
    
    def _get_cached_result(self, doc: PolicySegment) -> Dict[str, Any]:
        """
        从缓存中获取DS32B判断结果
        
        Returns:
            如果缓存存在，返回缓存的结果；否则返回None
        """
        cache_key = self._get_cache_key(doc)
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            self.log(f"  ✅ 文档 {doc.doc_id} 使用缓存结果")
            return cached
        return None
    
    def _save_result_to_cache(self, doc: PolicySegment, result: Dict[str, Any]):
        """保存DS32B判断结果到缓存"""
        cache_key = self._get_cache_key(doc)
        self.cache[cache_key] = {
            'industries': result.get('filtered_industries', []),
            'investment_relevance': result.get('investment_relevance', '低'),
            'report_series': result.get('report_series', 'NA'),
            'industry_policy_segments': {
                item['industry']: item['policy_segments']
                for item in result.get('filtered_industries', [])
            },
            'llm_industries': result.get('llm_industries', []),  # LLM打标结果
            'keyword_industries': result.get('keyword_industries', [])  # 关键词匹配结果
        }
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """将文本分句（按中文标点符号）"""
        import re
        # 按句号、问号、感叹号、分号分句
        sentences = re.split(r'[。！？；\n]', text)
        # 只过滤空句，不过滤短句（短句也可能包含关键词）
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def _match_industries_with_keywords(self, text: str, return_matched_segments: bool = False):
        """
        使用关键词匹配行业
        
        逻辑：
        1. 先在整个文本中匹配（确保不遗漏）
        2. 按标点符号分句（用于记录证据句）
        3. 遍历CITIC_INDUSTRY_TAG_GROUPS，对每个一级行业的所有关键词进行匹配
        4. 如果文本中包含该行业的任何关键词，记录该一级行业
        5. 返回去重后的一级行业标签列表（无关强度，仅罗列提及的板块）
        
        Args:
            text: 待匹配文本（整段doc内容）
            return_matched_segments: 是否返回匹配到的文段（证据句）
            
        Returns:
            如果return_matched_segments=False: 一级行业标签列表
            如果return_matched_segments=True: {
                'industries': List[str],
                'matched_segments': Dict[str, List[Dict]]  # {标签: [{'sentence': 证据句, 'matched_word': 匹配的词}]}
            }
        """
        # 步骤1：按标点符号分句（用于记录证据句）
        sentences = self._split_into_sentences(text)
        if not sentences:
            sentences = [text]
        
        # 步骤2：遍历所有一级行业，对每个行业的所有关键词进行匹配
        matched_industries = set()  # 使用set去重
        industry_evidence = {}  # {一级行业名: [{'sentence': 证据句, 'matched_word': 匹配的词}]}
        
        for level1_name, keywords in self.industry_tag_groups.items():
            # 遍历该一级行业的所有关键词，收集所有匹配的句子
            industry_matched = False
            # ⭐ matched_sentences_set 必须在行业循环外部初始化，这样每个行业的所有关键词共享同一个set去重
            matched_sentences_set = set()  # 用于去重，避免同一句子被重复添加
            
            for keyword in keywords:
                if not keyword or len(keyword.strip()) < 2:  # 只匹配长度>=2的词
                    continue
                keyword = keyword.strip()
                
                # 先在整个文本中检查（确保不遗漏，包括标题等短文本）
                if keyword in text:
                    matched_industries.add(level1_name)
                    industry_matched = True
                    
                    # 记录匹配证据（找到所有包含该关键词的句子）
                    if return_matched_segments:
                        if level1_name not in industry_evidence:
                            industry_evidence[level1_name] = []
                        # 找到所有包含关键词的句子（不限制数量）
                        for sentence in sentences:
                            if keyword in sentence:
                                # 使用句子内容作为key去重，避免重复添加相同句子
                                sentence_key = sentence.strip()
                                if sentence_key not in matched_sentences_set:
                                    matched_sentences_set.add(sentence_key)
                                    industry_evidence[level1_name].append({
                                        'sentence': sentence,
                                        'matched_word': keyword
                                    })
            
            # 如果该行业匹配到了，但return_matched_segments为False，也需要初始化
            if industry_matched and not return_matched_segments:
                if level1_name not in industry_evidence:
                    industry_evidence[level1_name] = []
        
        # 转换为列表（无需排序，无关强度）
        result = list(matched_industries)
        
        # 构建返回结果
        if return_matched_segments:
            return {
                'industries': result,
                'matched_segments': {industry: industry_evidence.get(industry, []) for industry in result}
            }
        
        return result
    
    def process(self, documents: List[PolicySegment]) -> List[PolicySegment]:
        """
        为政策文档分类行业标签（关键词匹配 + 合并DS32B判断）
        
        逻辑：
        1. 检查缓存，如果存在则直接使用
        2. 关键词匹配所有行业名称与原文，得到候选行业和匹配片段
        3. 使用合并的DS32B判断（投资相关性 + 行业过滤），一次调用完成
        4. 保存结果到缓存和文档metadata
        """
        self.log(f"开始为 {len(documents)} 个政策文档进行行业分类（关键词匹配 + 合并DS32B判断）")
        
        cached_count = 0
        new_count = 0
        
        for i, doc in enumerate(documents, 1):
            # ⭐ 步骤1：检查缓存
            cached_result = self._get_cached_result(doc)
            if cached_result:
                # 使用缓存结果
                industries = [item['industry'] for item in cached_result['industries']]
                doc.industries = industries
                doc.metadata['citic_industries'] = {'level1': industries}
                doc.metadata['investment_relevance'] = cached_result['investment_relevance']
                # ⭐ 报告系列：完全由parquet决定，不做任何修改
                doc.metadata['industry_policy_segments'] = cached_result['industry_policy_segments']
                cached_count += 1
                final_rs = doc.metadata.get('report_series', '')
                self.log(f"  [{i}/{len(documents)}] 文档 {doc.doc_id}: 使用缓存（行业 {len(industries)} 个，投资相关性: {cached_result['investment_relevance']}, 报告系列: {final_rs or '无'}）")
                continue
            
            # 步骤2：关键词匹配，获取候选行业和匹配片段
            match_result = self._match_industries_with_keywords(
                doc.title + " " + doc.content, 
                return_matched_segments=True
            )
            candidate_industries = match_result.get('industries', [])
            matched_segments = match_result.get('matched_segments', {})
            
            # ⭐ 调试信息：打印每个行业匹配到的片段数量
            for industry, segments_list in matched_segments.items():
                self.log(f"  [关键词匹配] 行业 {industry}: 匹配到 {len(segments_list)} 个片段")
            
            # 步骤3：合并判断（投资相关性 + 行业过滤）- 调用DS32B
            self.log(f"  [{i}/{len(documents)}] 文档 {doc.doc_id}: 调用DS32B进行判断...")
            combined_result = judge_investment_and_industries(
                doc, candidate_industries, matched_segments
            )
            
            # 步骤4：保存结果到缓存
            self._save_result_to_cache(doc, combined_result)
            
            # 步骤5：保存结果到文档metadata
            industries = [item['industry'] for item in combined_result['filtered_industries']]
            doc.industries = industries  # 只保存过滤后的行业作为最终标签
            doc.metadata['citic_industries'] = {'level1': industries}
            doc.metadata['investment_relevance'] = combined_result['investment_relevance']  # 投资相关性标签
            # ⭐ 报告系列：完全由parquet决定，不做任何修改
            industry_policy_segments_dict = {
                item['industry']: item['policy_segments'] 
                for item in combined_result['filtered_industries']
            }  # 行业及对应政策片段（存入Milvus）
            doc.metadata['industry_policy_segments'] = industry_policy_segments_dict
            # ⭐ 调试信息：打印保存到metadata的每个行业的片段数量
            for industry, segments_list in industry_policy_segments_dict.items():
                self.log(f"  [保存metadata] 行业 {industry}: {len(segments_list)} 个片段")
            
            new_count += 1
            final_rs = doc.metadata.get('report_series', '')
            self.log(f"  [{i}/{len(documents)}] 文档 {doc.doc_id}: 过滤后行业 {len(industries)} 个，投资相关性: {combined_result['investment_relevance']}, 报告系列: {final_rs or '无'}")
        
        # 保存缓存到文件
        if new_count > 0:
            self._save_cache()
        
        self.log(f"行业分类完成: 使用缓存 {cached_count} 个，新处理 {new_count} 个")
        return documents
    
    def classify_single(self, document: PolicySegment) -> PolicySegment:
        """为单个政策文档分类行业标签（关键词匹配 + 合并DS32B判断）"""
        # ⭐ 步骤1：检查缓存
        cached_result = self._get_cached_result(document)
        if cached_result:
            industries = [item['industry'] for item in cached_result['industries']]
            document.industries = industries
            document.metadata['citic_industries'] = {'level1': industries}
            document.metadata['investment_relevance'] = cached_result['investment_relevance']
            # ⭐ 报告系列：完全由parquet决定，不做任何修改
            document.metadata['industry_policy_segments'] = cached_result['industry_policy_segments']
            return document
        
        # 步骤2：关键词匹配
        match_result = self._match_industries_with_keywords(
            document.title + " " + document.content,
            return_matched_segments=True
        )
        candidate_industries = match_result.get('industries', [])
        matched_segments = match_result.get('matched_segments', {})
        
        # 步骤3：合并判断（投资相关性 + 行业过滤）
        combined_result = judge_investment_and_industries(
            document, candidate_industries, matched_segments
        )
        
        # 步骤4：保存结果到缓存
        self._save_result_to_cache(document, combined_result)
        self._save_cache()
        
        # 步骤5：保存结果到文档metadata
        industries = [item['industry'] for item in combined_result['filtered_industries']]
        document.industries = industries  # 只保存过滤后的行业作为最终标签
        document.metadata['citic_industries'] = {'level1': industries}
        document.metadata['investment_relevance'] = combined_result['investment_relevance']  # 投资相关性标签
        # ⭐ 报告系列：完全由parquet决定，不做任何修改
        document.metadata['industry_policy_segments'] = {
            item['industry']: item['policy_segments']
            for item in combined_result['filtered_industries']
        }  # 行业及对应政策片段（存入Milvus）
        
        return document


if __name__ == "__main__":
    """单独测试IndustryAgent"""
    from datetime import datetime
    
    print("=" * 80)
    print("IndustryAgent 单独测试（关键词匹配）")
    print("=" * 80)
    
    # 初始化Agent
    print("\n[1] 初始化IndustryAgent...")
    agent = IndustryAgent()
    
    # 测试用例1：医药相关
    print("\n[2] 测试用例1：医药相关政策")
    print("-" * 80)
    test_doc1 = PolicySegment(
        doc_id="test_001",
        title="关于支持生物医药产业高质量发展的通知",
        content="为深入贯彻落实国家创新驱动发展战略，加快推进生物医药产业高质量发展，现就有关事项通知如下：一、加强创新药物研发。支持企业开展化学原料药、化学制剂、生物制品等创新药物研发。二、促进中药产业发展。支持中药饮片、中成药等传统医药产业发展。三、完善医疗器械监管。加强医疗器械质量监管，提升医疗服务水平。",
        timestamp=datetime(2024, 5, 15),
        industries=[]
    )
    
    result1 = agent.classify_single(test_doc1)
    print(f"标题: {result1.title}")
    print(f"最终标记的一级行业: {result1.industries}")
    print(f"  匹配到 {len(result1.industries)} 个一级行业")
    
    # 打印匹配文段（用于判断匹配是否正确）
    match_result = agent._match_industries_with_keywords(test_doc1.title + " " + test_doc1.content, return_matched_segments=True)
    if match_result.get('matched_segments'):
        print(f"  匹配详情:")
        for industry in result1.industries[:5]:  # 只显示前5个
            segments = match_result.get('matched_segments', {}).get(industry, [])
            if segments:
                print(f"    {industry}:")
                for seg in segments[:3]:  # 显示前3个证据
                    matched_word = seg.get('matched_word', '未知')
                    print(f"      证据句: {seg['sentence'][:80]}... (匹配词: {matched_word})")
    
    # 测试用例2：电子相关
    print("\n[3] 测试用例2：电子半导体相关政策")
    print("-" * 80)
    test_doc2 = PolicySegment(
        doc_id="test_002",
        title="关于促进集成电路产业发展的指导意见",
        content="为加快集成电路产业发展，提升产业链供应链现代化水平，现提出以下意见：一、加强集成电路设计。支持企业开展芯片设计、集成电路设计等关键技术研发。二、完善半导体材料产业链。支持半导体材料、半导体设备等关键环节发展。三、推进PCB产业发展。支持PCB、被动元件等电子元器件产业发展。",
        timestamp=datetime(2024, 6, 1),
        industries=[]
    )
    
    result2 = agent.classify_single(test_doc2)
    print(f"标题: {result2.title}")
    print(f"最终标记的一级行业: {result2.industries}")
    print(f"  匹配到 {len(result2.industries)} 个一级行业")
    
    # 打印匹配文段（用于判断匹配是否正确）
    match_result = agent._match_industries_with_keywords(test_doc2.title + " " + test_doc2.content, return_matched_segments=True)
    if match_result.get('matched_segments'):
        print(f"  匹配详情:")
        for industry in result2.industries[:5]:  # 只显示前5个
            segments = match_result.get('matched_segments', {}).get(industry, [])
            if segments:
                print(f"    {industry}:")
                for seg in segments[:3]:  # 显示前3个证据
                    matched_word = seg.get('matched_word', '未知')
                    print(f"      证据句: {seg['sentence'][:80]}... (匹配词: {matched_word})")
    
    # 测试用例3：汽车相关
    print("\n[4] 测试用例3：新能源汽车相关政策")
    print("-" * 80)
    test_doc3 = PolicySegment(
        doc_id="test_003",
        title="关于加快新能源汽车产业发展的通知",
        content="为促进新能源汽车产业高质量发展，现就有关事项通知如下：一、支持整车制造。鼓励乘用车、商用车等新能源汽车整车制造企业发展。二、完善零部件产业链。支持汽车零部件、锂电池、燃料电池等关键零部件产业发展。三、推进充电基础设施建设。加快充电桩、换电站等基础设施建设。",
        timestamp=datetime(2024, 7, 1),
        industries=[]
    )
    
    result3 = agent.classify_single(test_doc3)
    print(f"标题: {result3.title}")
    print(f"最终标记的一级行业: {result3.industries}")
    print(f"  匹配到 {len(result3.industries)} 个一级行业")
    
    # 打印匹配文段（用于判断匹配是否正确）
    match_result = agent._match_industries_with_keywords(test_doc3.title + " " + test_doc3.content, return_matched_segments=True)
    if match_result.get('matched_segments'):
        print(f"  匹配详情:")
        for industry in result3.industries[:5]:  # 只显示前5个
            segments = match_result.get('matched_segments', {}).get(industry, [])
            if segments:
                print(f"    {industry}:")
                for seg in segments[:3]:  # 显示前3个证据
                    matched_word = seg.get('matched_word', '未知')
                    print(f"      证据句: {seg['sentence'][:80]}... (匹配词: {matched_word})")
    
    # 批量测试
    print("\n[5] 批量测试")
    print("-" * 80)
    test_docs = [test_doc1, test_doc2, test_doc3]
    results = agent.process(test_docs)
    print(f"批量处理 {len(test_docs)} 个文档")
    for i, doc in enumerate(results, 1):
        print(f"  文档{i}: {doc.industries}")
    
    print("\n" + "=" * 80)
    print("✅ 测试完成！")
    print("=" * 80)
