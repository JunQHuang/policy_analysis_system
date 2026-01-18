"""
投资相关性判断模块（合并投资相关性、LLM行业打标、行业过滤）

流程：
1. 关键词匹配得到候选行业（由IndustryAgent完成）
2. LLM打标：从所有中信一级行业中识别相关行业
3. 合并关键词匹配和LLM打标结果
4. LLM过滤：判断每个行业是否有实质性影响

注意：报告系列（会议系列）标签已改为纯人工标注，不再由LLM自动判断
"""
from typing import List, Dict, Any
from models import PolicySegment
from core.clients.ds32b_client import get_ds32b_client

# 中信一级行业列表（用于LLM打标）
CITIC_LEVEL1_INDUSTRIES = [
    "石油石化", "煤炭", "有色金属", "钢铁", "基础化工", "建筑材料", "建筑", "建材",
    "轻工制造", "机械", "电力设备及新能源", "国防军工", "汽车", "商贸零售", "消费者服务",
    "家电", "纺织服装", "医药", "食品饮料", "农林牧渔", "银行", "非银行金融", "房地产",
    "交通运输", "电力及公用事业", "电子", "通信", "计算机", "传媒", "综合", "综合金融"
]


def judge_investment_and_industries(segment: PolicySegment, 
                                    candidate_industries: List[str],
                                    matched_segments: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """
    合并判断：LLM行业打标 + 投资相关性 + 行业过滤（一次DS32B调用）
    
    流程：
    1. LLM从所有中信一级行业中识别相关行业（打标）
    2. 合并关键词匹配结果和LLM打标结果
    3. 判断投资相关性
    4. 过滤每个行业是否有实质性影响
    
    Args:
        segment: 政策文档
        candidate_industries: 关键词匹配得到的候选行业列表
        matched_segments: 每个行业匹配到的政策片段
        
    Returns:
        {
            'investment_relevance': '高'/'低',
            'report_series': 'N/A',  # 固定返回N/A，需人工标注
            'filtered_industries': [{'industry': 行业名, 'policy_segments': [政策片段列表]}]
        }
    """
    client = get_ds32b_client()
    
    # 提取政策内容
    content_preview = segment.content[:2000] if len(segment.content) > 2000 else segment.content
    title = segment.title
    
    # 构建关键词匹配的行业信息（板块+对应政策片段）
    industry_sections = []
    for industry in candidate_industries:
        segments = matched_segments.get(industry, [])
        if not segments:
            continue
        policy_sentences = [seg.get('sentence', '').strip()[:200] for seg in segments[:3] if seg.get('sentence')]
        if policy_sentences:
            policy_text = "\n".join([f"{i}. {sent}" for i, sent in enumerate(policy_sentences, 1)])
            industry_sections.append(f"板块：{industry}\n匹配到的政策片段：\n{policy_text}")
    
    keyword_industry_text = "\n\n".join(industry_sections) if industry_sections else "无"
    
    # 先输出给用户看
    print(f"\n  [合并判断：LLM行业打标 + 投资相关性 + 行业过滤]")
    print(f"    标题: {title}")
    print(f"    内容预览: {content_preview[:200]}...")
    if candidate_industries:
        print(f"    关键词匹配行业: {', '.join(candidate_industries)}")
    
    # 构建候选行业列表字符串
    keyword_candidate_list = ", ".join(candidate_industries) if candidate_industries else "无"
    all_industries_list = ", ".join(CITIC_LEVEL1_INDUSTRIES)
    
    # 构建prompt（三合一：LLM打标 + 投资相关性 + 行业过滤）
    prompt = f"""请完成三个任务，严格按照JSON格式回答：

任务1：【LLM行业打标】从中信一级行业列表中，识别该政策涉及的所有相关行业
任务2：【投资相关性判断】判断政策投资相关性（只回答"高"或"低"）
任务3：【行业过滤】对合并后的所有行业，判断是否有实质性影响（只回答"是"或"否"）

政策标题：{title}

政策内容：
{content_preview}

---

【中信一级行业列表】（共{len(CITIC_LEVEL1_INDUSTRIES)}个）：
{all_industries_list}

【关键词匹配结果】（共{len(candidate_industries)}个）：
{keyword_candidate_list}

【关键词匹配的政策片段】：
{keyword_industry_text}

---

判断标准：

1. LLM行业打标：
   - 从上述中信一级行业列表中，选择该政策实际涉及的行业
   - 不要只看关键词，要理解政策内容的实际影响范围
   - 可以识别关键词匹配遗漏的行业

2. 投资相关性：
   - "高"：政策包含具体的投资机会、资金支持、税收优惠、产业扶持、市场准入、监管变化等实质性内容
   - "低"：只是原则性表述、一般性通知、会议纪要、工作总结等，没有具体的投资相关措施

3. 行业实质性影响：
   - "是"：政策对该行业有实质性影响（有具体措施、资金支持、时间目标等）
   - "否"：只是简单提及行业名称但没有实质性政策内容

请严格按照以下JSON格式回答（不要添加任何其他文字）：
{{
  "llm_industries": ["行业1", "行业2", ...],
  "investment_relevance": "高"或"低",
  "industry_filter": {{
    "行业1": "是"或"否",
    "行业2": "是"或"否",
    ...
  }}
}}

注意：
- llm_industries 只能包含中信一级行业列表中的行业名称
- industry_filter 必须对【关键词匹配结果 + LLM打标结果】的所有行业都给出判断

回答："""
    
    messages = [
        {"role": "system", "content": "你是一位资深的投资策略专家，擅长分析政策对投资市场的实际影响。你需要准确识别政策涉及的行业，并严格区分实质性政策内容和一般性表述。请严格按照JSON格式回答。"},
        {"role": "user", "content": prompt}
    ]
    
    try:
        response = client.chat_completion(messages)
        if response:
            # 尝试解析JSON
            import json
            
            # 提取JSON部分（支持嵌套JSON）
            start_idx = response.find('{')
            if start_idx != -1:
                brace_count = 0
                end_idx = start_idx
                for i in range(start_idx, len(response)):
                    if response[i] == '{':
                        brace_count += 1
                    elif response[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i + 1
                            break
                
                if end_idx > start_idx:
                    json_str = response[start_idx:end_idx]
                else:
                    json_str = None
            else:
                json_str = None
            
            if json_str:
                try:
                    result = json.loads(json_str)
                    
                    # 1. 提取LLM打标的行业
                    llm_industries = result.get('llm_industries', [])
                    # 过滤：只保留有效的中信一级行业
                    llm_industries = [ind for ind in llm_industries if ind in CITIC_LEVEL1_INDUSTRIES]
                    print(f"    [LLM打标] 识别行业: {llm_industries}")
                    
                    # 2. 合并关键词匹配和LLM打标结果（去重）
                    all_industries = list(set(candidate_industries + llm_industries))
                    print(f"    [合并结果] 关键词({len(candidate_industries)}) + LLM({len(llm_industries)}) = {len(all_industries)} 个行业")
                    
                    # 3. 提取投资相关性
                    investment_relevance = result.get('investment_relevance', '低')
                    if investment_relevance not in ['高', '低']:
                        investment_relevance = '低'
                    
                    # 4. 提取行业过滤结果
                    industry_filter = result.get('industry_filter', {})
                    print(f"    [DS32B返回] industry_filter: {industry_filter}")
                    
                    filtered_industries = []
                    
                    for industry in all_industries:
                        decision = industry_filter.get(industry, None)
                        # 获取匹配片段（关键词匹配的行业有片段，LLM打标的行业可能没有）
                        segments = matched_segments.get(industry, [])
                        segment_count = len(segments)
                        
                        # 判断逻辑：
                        # 1. 如果DS32B明确返回"是"，保留
                        # 2. 如果DS32B没有返回该行业（None）：
                        #    - 如果是LLM打标的行业，保留（LLM认为相关）
                        #    - 如果是关键词匹配的行业且匹配片段>=2，保留
                        # 3. 如果DS32B明确返回"否"，不保留
                        should_keep = False
                        reason = ""
                        is_from_llm = industry in llm_industries
                        is_from_keyword = industry in candidate_industries
                        
                        if decision is None:
                            # DS32B没有返回该行业
                            if is_from_llm:
                                should_keep = True
                                reason = "LLM打标识别，保留"
                            elif segment_count >= 2:
                                should_keep = True
                                reason = f"关键词匹配，有{segment_count}个片段，保留"
                            else:
                                should_keep = False
                                reason = f"关键词匹配，片段少({segment_count}个)，不保留"
                        elif decision == '是' or (decision and '是' in str(decision)):
                            should_keep = True
                            reason = "DS32B判断'是'"
                        else:
                            should_keep = False
                            reason = f"DS32B判断'{decision}'"
                        
                        if should_keep:
                            policy_segments_list = [seg.get('sentence', '') for seg in segments]
                            source = []
                            if is_from_keyword:
                                source.append("关键词")
                            if is_from_llm:
                                source.append("LLM")
                            source_str = "+".join(source)
                            print(f"    [保留] {industry} ({source_str}): {len(policy_segments_list)} 个片段 ({reason})")
                            filtered_industries.append({
                                'industry': industry,
                                'policy_segments': policy_segments_list,
                                'source': source_str
                            })
                        else:
                            print(f"    [过滤] {industry}: ({reason})")
                    
                    print(f"    ✅ 投资相关性: {investment_relevance}")
                    print(f"    ✅ 最终行业: {len(filtered_industries)}/{len(all_industries)} 个")
                    print(f"    ⚠️ 报告系列: 需人工标注（使用manual_label_editor.ipynb）")
                    
                    return {
                        'investment_relevance': investment_relevance,
                        'report_series': 'N/A',  # 固定返回N/A，需人工标注
                        'filtered_industries': filtered_industries,
                        'llm_industries': llm_industries,  # 额外返回LLM打标结果
                        'keyword_industries': candidate_industries  # 额外返回关键词匹配结果
                    }
                except json.JSONDecodeError:
                    print(f"    ⚠️ JSON解析失败，使用保守策略")
            
            # 如果JSON解析失败，尝试简单文本解析
            result_text = response.strip()
            investment_relevance = '低'
            if '高' in result_text and '低' not in result_text:
                investment_relevance = '高'
            
            # 保守策略：如果无法解析，保留所有关键词匹配的行业
            filtered_industries = []
            for industry in candidate_industries:
                segments = matched_segments.get(industry, [])
                filtered_industries.append({
                    'industry': industry,
                    'policy_segments': [seg.get('sentence', '') for seg in segments],
                    'source': '关键词'
                })
            
            print(f"    ⚠️ 使用保守策略（投资相关性: {investment_relevance}，保留所有关键词匹配行业）")
            return {
                'investment_relevance': investment_relevance,
                'report_series': 'N/A',
                'filtered_industries': filtered_industries,
                'llm_industries': [],
                'keyword_industries': candidate_industries
            }
            
    except Exception as e:
        print(f"    ❌ 判断失败: {e}")
        print(f"[合并判断] 失败: {e}")
    
    # 默认返回（保守策略）
    filtered_industries = []
    for industry in candidate_industries:
        segments = matched_segments.get(industry, [])
        filtered_industries.append({
            'industry': industry,
            'policy_segments': [seg.get('sentence', '') for seg in segments],
            'source': '关键词'
        })
    
    return {
        'investment_relevance': '低',
        'report_series': 'N/A',
        'filtered_industries': filtered_industries,
        'llm_industries': [],
        'keyword_industries': candidate_industries
    }
