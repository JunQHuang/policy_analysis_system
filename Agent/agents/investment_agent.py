"""
Investment Agent - 投资分析生成（集中所有prompt调用）
"""
from typing import List, Dict, Any
import sys
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from .base import BaseAgent
from models import PolicySegment
from core.clients.volcengine_client import get_volcengine_client


class InvestmentAgent(BaseAgent):
    """投资分析Agent - 集中所有prompt调用，生成完整的分析报告"""
    
    def __init__(self, vector_db=None):
        super().__init__("InvestmentAgent")
        self.llm_client = get_volcengine_client()
        self.vector_db = vector_db
        self.log("✅ 使用Volcengine LLM")
    
    def process(self, input_data: Any) -> Any:
        """处理数据的主方法"""
        if isinstance(input_data, dict):
            section_type = input_data.get('type')
            if section_type == 'industry':
                return self.generate_industry_section(input_data.get('segment'))
            elif section_type == 'meeting':
                return self.generate_meeting_section(
                    input_data.get('segment'),
                    input_data.get('detected_meetings'),
                    input_data.get('meeting_docs')
                )
        return None
    
    def generate_industry_section(self, segment: PolicySegment) -> str:
        """生成行业分析部分"""
        import json
        
        industries = segment.industries if segment.industries else []
        
        if not industries:
            return "## 行业分类分析\n\n相关行业: 未分类"
        
        industry_segments = segment.metadata.get('industry_policy_segments', {})
        self._save_industry_segments_to_json(segment, industries, industry_segments)
        
        industries_text = "、".join(industries)
        return f"## 行业分类分析\n\n相关行业: {industries_text}"
    
    def _save_industry_segments_to_json(self, segment: PolicySegment, 
                                         industries: List[str], 
                                         industry_segments: Dict[str, List[str]]):
        """将行业标签及对应政策片段保存到JSON文件"""
        import json
        
        industry_dir = Path("industry")
        industry_dir.mkdir(exist_ok=True)
        
        json_data = {
            "doc_id": segment.doc_id,
            "title": segment.title,
            "timestamp": segment.timestamp.strftime('%Y-%m-%d') if segment.timestamp else 'N/A',
            "industries": industries,
            "industry_policy_segments": {}
        }
        
        for industry in industries:
            segments = industry_segments.get(industry, [])
            if segments:
                self.log(f"  行业 {industry}: 保存 {len(segments)} 个政策片段到JSON")
            json_data["industry_policy_segments"][industry] = segments
        
        json_file = industry_dir / f"{segment.doc_id}_industry.json"
        try:
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            self.log(f"  ✅ 行业政策片段已保存: {json_file}")
        except Exception as e:
            self.log(f"  ⚠️ 保存行业JSON失败: {e}", level="warning")

    def generate_meeting_section(self, segment: PolicySegment, 
                                 vector_db = None) -> Dict[str, Any]:
        """
        生成报告系列时间对比分析部分
        
        新流程（分步LLM处理）：
        1. LLM提取：从当前政策和历史政策中分别提取货币/财政相关段落
        2. LLM分析：基于提取的段落做对比分析
        
        Returns:
            {
                'analysis': Markdown格式的报告系列时间对比分析文本,
                'topics': 提取的主题词列表（用于后续分主题RAG）,
                'history_policies': 同系列历史政策列表
            }
        """
        current_series = segment.metadata.get('report_series', 'N/A')
        current_title = segment.title
        current_time = segment.timestamp
        current_time_str = current_time.strftime('%Y年%m月%d日') if current_time else 'N/A'
        current_content = segment.content
        
        # 从数据库按报告系列查询历史政策
        same_series_policies = []
        if vector_db and current_series and current_series != 'N/A':
            same_series_policies = vector_db.query_by_report_series(
                report_series=current_series,
                exclude_doc_id=segment.doc_id,
                limit=20
            )
            
            # 过滤掉标题和时间都相同的政策
            filtered_policies = []
            for doc in same_series_policies:
                doc_title = doc.get('title', '')
                doc_timestamp = doc.get('timestamp', '')
                if doc_title == current_title:
                    try:
                        if doc_timestamp and current_time:
                            if 'T' in str(doc_timestamp):
                                doc_dt = datetime.fromisoformat(str(doc_timestamp).replace('Z', '+00:00'))
                            else:
                                doc_dt = datetime.fromisoformat(str(doc_timestamp))
                            if doc_dt.date() == current_time.date():
                                continue
                    except:
                        continue
                filtered_policies.append(doc)
            same_series_policies = filtered_policies
        
        # 构建同系列历史政策信息
        history_policies_full = []
        for doc in same_series_policies[:3]:
            title = doc.get('title', '未知标题')
            timestamp = doc.get('timestamp', 'N/A')
            if timestamp:
                try:
                    if 'T' in str(timestamp):
                        dt = datetime.fromisoformat(str(timestamp).replace('Z', '+00:00'))
                        timestamp = dt.strftime('%Y年%m月%d日')
                except:
                    pass
            content = doc.get('content', '')
            history_policies_full.append({
                'title': title,
                'timestamp': timestamp,
                'content': content
            })
        
        # ========== 新流程：分步LLM处理 ==========
        
        # Step 1: LLM提取货币/财政相关段落
        self.log("  📌 Step 1: LLM提取宏观政策相关段落...")
        
        # 提取当前政策的宏观政策段落
        current_macro = self._extract_macro_policy_content(
            title=current_title,
            content=current_content
        )
        self.log(f"    当前政策提取: 货币{len(current_macro.get('monetary', ''))}字, 财政{len(current_macro.get('fiscal', ''))}字")
        
        # 提取历史政策的宏观政策段落
        history_macro_list = []
        for doc in history_policies_full:
            history_macro = self._extract_macro_policy_content(
                title=doc['title'],
                content=doc['content']
            )
            history_macro['title'] = doc['title']
            history_macro['timestamp'] = doc['timestamp']
            history_macro_list.append(history_macro)
            self.log(f"    历史政策《{doc['title'][:20]}...》提取: 货币{len(history_macro.get('monetary', ''))}字, 财政{len(history_macro.get('fiscal', ''))}字")
        
        # Step 2: LLM分析对比
        self.log("  📌 Step 2: LLM生成对比分析...")
        
        # 构建历史政策文本（完整内容，用于产业政策分析）
        history_text = ""
        if history_policies_full:
            for i, doc in enumerate(history_policies_full, 1):
                history_text += f"\n### 历史政策 {i}：{doc['title']}\n**发布时间**：{doc['timestamp']}\n\n**政策全文**：\n{doc['content']}\n\n---\n"
        else:
            history_text = "无同系列历史政策"
        
        # 构建prompt（使用提取的宏观政策段落）
        prompt = self._build_meeting_prompt_v2(
            current_title=current_title,
            current_time_str=current_time_str,
            current_series=current_series,
            current_content=current_content,
            current_macro=current_macro,
            history_policies_full=history_policies_full,
            history_macro_list=history_macro_list,
            history_text=history_text
        )
        
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.llm_client.chat_completion(
                messages=messages,
                temperature=0.3
            )
            
            import json
            import re
            response = response.strip()
            
            self.log(f"  LLM响应长度: {len(response)} 字符")
            
            # 提取topics（从TOPICS_JSON行）
            topics = []
            
            # 打印响应末尾用于调试
            self.log(f"  响应末尾200字: {response[-200:] if len(response) > 200 else response}")
            
            # 尝试多种格式匹配TOPICS_JSON
            # 格式1: TOPICS_JSON:{"topics":["板块1","板块2"]}
            topics_match = re.search(r'TOPICS_JSON[:\s]*\{["\']?topics["\']?\s*:\s*\[([^\]]+)\]', response)
            if topics_match:
                try:
                    topics_str = topics_match.group(1)
                    topics = [t.strip().strip('"\'') for t in topics_str.split(',')]
                    topics = [t for t in topics if t and t not in ['板块1', '板块2', '板块3']]
                    self.log(f"  从TOPICS_JSON格式1提取: {topics[:5]}...")
                    # 清除整个TOPICS_JSON行（包含嵌套的数组）
                    response = re.sub(r'\n*TOPICS_JSON[:\s]*\{[^}]*\[[^\]]*\][^}]*\}\s*', '', response)
                except Exception as e:
                    self.log(f"  解析TOPICS_JSON格式1失败: {e}", level="warning")
            
            # 格式2: TOPICS_JSON: {"topics": ["板块1", "板块2"]} (带空格)
            if not topics:
                topics_match2 = re.search(r'TOPICS_JSON\s*[:\s]*(\{[^}]*\[[^\]]*\][^}]*\})', response)
                if topics_match2:
                    try:
                        json_str = topics_match2.group(1)
                        self.log(f"  找到TOPICS_JSON: {json_str[:100]}...")
                        data = json.loads(json_str)
                        topics = data.get('topics', [])
                        topics = [t for t in topics if t and t not in ['板块1', '板块2', '板块3']]
                        self.log(f"  从TOPICS_JSON格式2提取: {topics[:5]}...")
                        response = re.sub(r'\n*TOPICS_JSON\s*[:\s]*\{[^}]*\[[^\]]*\][^}]*\}\s*', '', response)
                    except Exception as e:
                        self.log(f"  解析TOPICS_JSON格式2失败: {e}", level="warning")
            
            # 最后兜底：清除任何残留的TOPICS_JSON
            response = re.sub(r'\n*TOPICS_JSON[^\n]*\n*', '', response)
            
            # 如果没找到TOPICS_JSON，尝试从"重点关注板块"提取
            if not topics:
                sectors_match = re.search(r'重点关注板块[：:]\s*([^\n]+)', response)
                if sectors_match:
                    sectors_text = sectors_match.group(1)
                    self.log(f"  从重点关注板块提取: {sectors_text[:60]}...")
                    citic_industries = ["金融", "电子", "计算机", "通信", "传媒", "医药生物", "机械设备", 
                                       "电力设备", "国防军工", "汽车", "家用电器", "轻工制造", "商贸零售",
                                       "社会服务", "食品饮料", "农林牧渔", "钢铁", "有色金属", "基础化工",
                                       "石油石化", "煤炭", "建筑材料", "建筑装饰", "房地产", "交通运输",
                                       "公用事业", "纺织服饰", "美容护理", "环保", "综合"]
                    for ind in citic_industries:
                        if ind in sectors_text:
                            topics.append(ind)
            
            # 如果还是没有，从表格中提取
            if not topics:
                self.log("  尝试从表格提取主题词...")
                citic_industries = ["金融", "电子", "计算机", "通信", "传媒", "医药生物", "机械设备", 
                                   "电力设备", "国防军工", "汽车", "家用电器", "轻工制造", "商贸零售",
                                   "社会服务", "食品饮料", "农林牧渔", "钢铁", "有色金属", "基础化工",
                                   "石油石化", "煤炭", "建筑材料", "建筑装饰", "房地产", "交通运输",
                                   "公用事业", "纺织服饰", "美容护理", "环保", "综合"]
                # 从整个响应中查找中信行业
                for ind in citic_industries:
                    if ind in response and ind not in topics:
                        topics.append(ind)
                if topics:
                    self.log(f"  从全文提取到行业: {topics[:10]}...")
            
            self.log(f"  ✅ 提取到 {len(topics)} 个主题词: {topics[:5]}...")
            
            # 构建最终的Markdown报告
            history_list = ""
            if history_policies_full:
                for i, doc in enumerate(history_policies_full, 1):
                    history_list += f"{i}. {doc['title']}（{doc['timestamp']}）\n"
            else:
                history_list = "无同系列历史政策"
            
            analysis = f"""## 报告系列时间对比分析

**报告系列**：{current_series}
**当前政策发布时间**：{current_time_str}

**同系列历史政策**：
{history_list}

---

{response}
"""
            
            return {
                'analysis': analysis,
                'topics': topics,
                'history_policies': history_policies_full,
                'raw_result': {}
            }
            
        except Exception as e:
            self.log(f"LLM调用失败: {e}", level="error")
            return {
                'analysis': f"## 报告系列时间对比分析\n\n报告系列：{current_series}\n\n（分析生成失败：{e}）",
                'topics': [],
                'history_policies': history_policies_full,
                'raw_result': {}
            }

    def _build_meeting_prompt(self, current_title: str, current_time_str: str, 
                               current_series: str, current_content: str, history_text: str) -> str:
        """构建会议对比分析的prompt - 直接输出Markdown报告，分点一对多对比"""
        return f"""你是一名资深宏观策略分析师，请对比分析当前政策与历史政策的边际变化。

=== 当前政策 ===
标题：{current_title}
时间：{current_time_str}
系列：{current_series}

{current_content}

=== 历史政策 ===
{history_text}

=== 分析要求 ===

请输出以下格式的Markdown分析报告：

### 一、宏观政策基调对比

#### 1. 经济形势判断

**当前政策表述**：直接引用原文

**历史政策表述**：
- **《政策标题》**：直接引用原文

**边际变化**：说明

#### 2. 货币政策

从当前政策和历史政策中提取货币政策相关内容。

货币政策类似表述举例：
- 取向类：稳健的货币政策、适度宽松的货币政策、灵活适度、松紧适度
- 流动性类：流动性合理充裕、保持流动性、精准滴灌、总量适度
- 利率类：利率市场化、LPR改革、贷款市场报价利率、存贷款利率、降低融资成本、实际利率
- 准备金类：存款准备金率、降准、差别化准备金、定向降准
- 信贷类：信贷投放、信贷结构优化、结构性货币政策工具、再贷款、再贴现、支农支小、制造业中长期贷款
- 汇率类：人民币汇率、汇率稳定、跨境资金流动、外汇储备、汇率弹性
- 货币供应类：M2增速、社会融资规模、货币供应量、信贷增速与GDP匹配
- 央行制度类：现代中央银行制度、货币政策传导机制、宏观审慎管理、金融稳定

注意：金融强国、科技金融、绿色金融、普惠金融、养老金融、数字金融、资本市场改革、股票发行注册制、数字人民币等不属于货币政策，放到"金融改革"部分。

**当前政策表述**：
直接引用原文

**历史政策表述**：
- **《政策标题》（时间）**：直接引用原文

**边际变化**：分析

**投资含义**：说明

#### 3. 财政政策

从当前政策和历史政策中提取财政政策相关内容。

财政政策类似表述举例：
- 取向类：积极的财政政策、更加积极、加力提效、适度加力、提质增效
- 赤字类：赤字率、财政赤字、赤字规模、拟按X%安排
- 债务类：政府债务、债务规模、债务率、负债率、债务限额
- 债券类：国债、地方政府债券、专项债、一般债、超长期特别国债、增发国债
- 税费类：减税降费、税制改革、增值税、所得税、宏观税负、涉企收费
- 支出类：财政支出、支出结构、民生支出、重点领域保障、财政资金直达
- 转移支付类：中央对地方转移支付、一般性转移支付、专项转移支付、财力均衡
- 债务风险类：地方债务风险、隐性债务、债务化解、防范化解风险、存量债务
- 体制类：央地财政关系、财政体制改革、财政可持续性、财权事权匹配

注意：产业补贴、科技投入、基建投资具体项目等不属于财政政策，放到对应产业部分。

**当前政策表述**：
直接引用原文

**历史政策表述**：
- **《政策标题》（时间）**：直接引用原文

**边际变化**：分析

**投资含义**：说明

### 二、重点产业政策对比

根据政策涉及的领域分析，常见领域如：科技创新、制造业、绿色能源、金融改革、消费内需、房地产、基础设施等。

每个领域格式：

#### [领域名称]

**当前政策表述**：直接引用原文

**历史政策表述**：
- **《历史政策标题》**：直接引用原文

**边际变化**：分析

### 三、关键表述变化汇总

| 领域 | 当前政策 | 历史政策 | 变化 | 投资含义 |
|------|---------|---------|------|---------|
| 货币政策 | 概括 | 概括 | 变化 | 含义 |
| 财政政策 | 概括 | 概括 | 变化 | 含义 |

TOPICS_JSON:{{"topics":["板块1","板块2","板块3"]}}

=== 注意事项 ===    

1. 货币政策部分不要包含金融强国、科技金融、绿色金融等内容（放到"金融改革"）
2. 财政政策部分不要包含产业补贴、科技投入等内容（放到对应产业部分）
3. 如果政策确实未提及，写"当前政策未明确提及"
4. TOPICS_JSON从中信一级行业选择5-10个最相关板块
5. 引用原文时直接写出来，不要用特殊引号格式
6. 不要用省略号(...)省略内容，完整输出所有分析"""

    def _extract_macro_policy_content(self, title: str, content: str) -> Dict[str, str]:
        """
        用LLM从政策原文中分别提取货币政策、财政政策、经济形势判断
        货币政策使用关键词预筛选+LLM精提取，提高召回率
        
        Args:
            title: 政策标题
            content: 政策原文
            
        Returns:
            {
                'monetary': 货币政策相关段落,
                'fiscal': 财政政策相关段落,
                'economic': 经济形势判断相关段落
            }
        """
        result = {
            'monetary': '',
            'fiscal': '',
            'economic': ''
        }
        
        # === 第1次调用：提取货币政策（关键词预筛选+LLM精提取） ===
        
        # Step 1: 关键词预筛选相关段落
        monetary_keywords = [
            '货币政策', '央行', '中央银行', '人民银行',
            '利率', 'LPR', '贷款利率', '存款利率', '融资成本',
            '降准', '存款准备金', '准备金率',
            '流动性', '精准滴灌', '合理充裕',
            '信贷', '贷款', '再贷款', '再贴现', 'MLF', 'SLF',
            '汇率', '人民币', '跨境资金',
            'M2', '社会融资', '社融', '货币供应',
            '宏观审慎', '货币政策传导', '稳健', '适度宽松'
        ]
        
        # 按段落分割（按换行或句号分割）
        import re
        paragraphs = re.split(r'\n+|。', content)
        
        # 筛选包含关键词的段落
        relevant_paragraphs = []
        for p in paragraphs:
            p = p.strip()
            if len(p) < 10:  # 跳过太短的段落
                continue
            if any(kw in p for kw in monetary_keywords):
                relevant_paragraphs.append(p + '。')
        
        # 记录预筛选结果
        if relevant_paragraphs:
            pre_filtered_content = '\n'.join(relevant_paragraphs)
            self.log(f"    货币政策关键词预筛选: 找到{len(relevant_paragraphs)}个相关段落")
        else:
            pre_filtered_content = ""
            self.log(f"    货币政策关键词预筛选: 未找到相关段落")
        
        # Step 2: LLM精提取（预筛选段落+原文都给，重点看预筛选的）
        monetary_prompt = f"""请从以下内容中提取【货币政策】相关表述。

=== 重点关注内容（关键词预筛选结果） ===
{pre_filtered_content if pre_filtered_content else "（未找到明显相关段落）"}

=== 完整原文（用于补充遗漏） ===
{content}

=== 货币政策定义 ===

请提取以下内容（宁多勿漏）：
- 货币政策取向（稳健、适度宽松、灵活适度、松紧适度等）
- 央行制度（中央银行制度、现代央行制度等）
- 货币政策工具（利率、LPR、存款准备金率、降准、再贷款、MLF等）
- 流动性管理（流动性合理充裕、精准滴灌等）
- 信贷政策（信贷投放、结构性货币政策工具等）
- 汇率政策（人民币汇率、跨境资金流动等）
- 货币供应（M2、社会融资规模等）
- 货币政策传导机制、宏观审慎管理

不要提取：金融强国、科技金融、绿色金融、普惠金融、数字金融、资本市场、金融机构、数字人民币、金融监管等内容。

=== 提取策略 ===
1. 优先从"重点关注内容"中提取
2. 同时检查"完整原文"，补充可能遗漏的货币政策相关表述
3. 确保不遗漏任何货币政策相关内容

=== 输出要求 ===
直接输出摘录的原文内容，每条表述单独一行。不要分析，不要加标题。如果未提及写"未提及"。"""

        try:
            messages = [{"role": "user", "content": monetary_prompt}]
            response = self.llm_client.chat_completion(messages=messages, temperature=0.1)
            result['monetary'] = response.strip()
        except Exception as e:
            self.log(f"  提取货币政策失败: {e}", level="warning")
        
        # === 第2次调用：提取财政政策 ===
        fiscal_prompt = f"""请从以下政策文件中提取【财政政策】相关内容。

=== 政策文件 ===
标题：{title}

{content}

=== 财政政策定义 ===

请提取以下内容（宁多勿漏）：
- 财政政策取向（积极的财政政策、加力提效、适度加力等）
- 赤字率、财政赤字规模
- 政府债务（债务限额、债务规模、债务管理等）
- 政府债券（国债、专项债、特别国债、地方债等）
- 税费政策（减税降费、税制改革、宏观税负等）
- 财政支出、预算管理
- 转移支付制度
- 地方债务风险、隐性债务、债务化解
- 央地财政关系、财政体制、财政可持续性

不要提取：产业补贴、科技投入、基建投资具体项目等内容。

=== 输出要求 ===
直接输出摘录的原文内容，不要分析，不要加标题。如果未提及写"未提及"。"""

        try:
            messages = [{"role": "user", "content": fiscal_prompt}]
            response = self.llm_client.chat_completion(messages=messages, temperature=0.1)
            result['fiscal'] = response.strip()
        except Exception as e:
            self.log(f"  提取财政政策失败: {e}", level="warning")
        
        # === 第3次调用：提取经济形势判断 ===
        economic_prompt = f"""请从以下政策文件中提取【经济形势判断】相关内容。

=== 政策文件 ===
标题：{title}

{content}

=== 经济形势判断定义 ===

请提取以下内容（宁多勿漏）：
- 对当前经济形势的总体判断和定性描述
- GDP增速目标、经济增长目标
- 就业目标、失业率目标
- 物价目标、通胀目标、CPI目标
- 国际经济形势判断、外部环境判断
- 经济发展面临的挑战和机遇

=== 输出要求 ===
直接输出摘录的原文内容，不要分析，不要加标题。如果未提及写"未提及"。"""

        try:
            messages = [{"role": "user", "content": economic_prompt}]
            response = self.llm_client.chat_completion(messages=messages, temperature=0.1)
            result['economic'] = response.strip()
        except Exception as e:
            self.log(f"  提取经济形势判断失败: {e}", level="warning")
        
        return result

    def _build_meeting_prompt_v2(self, current_title: str, current_time_str: str,
                                  current_series: str, current_content: str,
                                  current_macro: Dict[str, str],
                                  history_policies_full: List[Dict],
                                  history_macro_list: List[Dict],
                                  history_text: str) -> str:
        """
        构建会议对比分析的prompt v2 - 使用预提取的宏观政策段落
        """
        # 构建货币政策对比材料
        monetary_material = f"""**当前政策《{current_title}》（{current_time_str}）货币政策相关内容：**
{current_macro.get('monetary', '未提及')}

"""
        for h in history_macro_list:
            monetary_material += f"""**历史政策《{h['title']}》（{h['timestamp']}）货币政策相关内容：**
{h.get('monetary', '未提及')}

"""
        
        # 构建财政政策对比材料
        fiscal_material = f"""**当前政策《{current_title}》（{current_time_str}）财政政策相关内容：**
{current_macro.get('fiscal', '未提及')}

"""
        for h in history_macro_list:
            fiscal_material += f"""**历史政策《{h['title']}》（{h['timestamp']}）财政政策相关内容：**
{h.get('fiscal', '未提及')}

"""
        
        # 构建经济形势对比材料
        economic_material = f"""**当前政策《{current_title}》（{current_time_str}）经济形势判断：**
{current_macro.get('economic', '未提及')}

"""
        for h in history_macro_list:
            economic_material += f"""**历史政策《{h['title']}》（{h['timestamp']}）经济形势判断：**
{h.get('economic', '未提及')}

"""
        
        return f"""你是一名资深宏观策略分析师，请对比分析当前政策与历史政策的边际变化。

=== 宏观政策对比材料（已预提取） ===

### 货币政策相关内容
{monetary_material}

### 财政政策相关内容
{fiscal_material}

### 经济形势判断
{economic_material}

=== 完整政策原文（用于产业政策分析） ===

**当前政策**：{current_title}（{current_time_str}）
{current_content}

**历史政策**：
{history_text}

=== 分析要求 ===

请输出以下格式的Markdown分析报告：

### 一、宏观政策基调对比

#### 1. 经济形势判断

**当前政策表述**：直接引用预提取内容

**历史政策表述**：
- **《政策标题》**：直接引用预提取内容

**边际变化**：说明对经济形势判断的变化

#### 2. 货币政策

基于上面预提取的货币政策内容进行对比分析。

**当前政策表述**：
直接引用预提取内容中的关键表述

**历史政策表述**：
- **《政策标题》（时间）**：直接引用预提取内容中的关键表述

**边际变化**：分析政策取向、力度、工具等方面的变化

**投资含义**：说明对金融市场和实体经济的影响

#### 3. 财政政策

基于上面预提取的财政政策内容进行对比分析。

**当前政策表述**：
直接引用预提取内容中的关键表述

**历史政策表述**：
- **《政策标题》（时间）**：直接引用预提取内容中的关键表述

**边际变化**：分析政策取向、赤字率、债务、税费等方面的变化

**投资含义**：说明对基建、民生等领域的影响

### 二、重点产业政策对比

根据完整政策原文分析产业政策，常见领域如：科技创新、制造业、绿色能源、金融改革、消费内需、房地产、基础设施等。

每个领域格式：

#### [领域名称]

**当前政策表述**：直接引用原文

**历史政策表述**：
- **《历史政策标题》**：直接引用原文

**边际变化**：分析

### 三、关键表述变化汇总

| 领域 | 当前政策 | 历史政策 | 变化 | 投资含义 |
|------|---------|---------|------|---------|
| 货币政策 | 概括 | 概括 | 变化 | 含义 |
| 财政政策 | 概括 | 概括 | 变化 | 含义 |

TOPICS_JSON:{{"topics":["板块1","板块2","板块3"]}}

=== 注意事项 ===

1. 宏观政策部分（经济形势、货币、财政）必须基于预提取的内容分析，确保完整性
2. 产业政策部分基于完整原文分析
3. 引用原文时直接写出来，不要用特殊引号格式
4. TOPICS_JSON从中信一级行业选择5-10个最相关板块
5. 不要用省略号(...)省略内容，完整输出所有分析"""

    def get_full_text_by_title(self, title: str) -> str:
        """根据标题获取完整文档内容"""
        if not self.vector_db:
            self.log("⚠️ vector_db未初始化，无法查询全文", level="warning")
            return ""
        
        try:
            chunk_collection = self.vector_db.chunk_collection
            
            try:
                chunk_collection.query(expr="id >= 0", limit=1, output_fields=["id"])
            except:
                chunk_collection.load()
            
            escaped_title = title.replace('"', '\\"').replace("'", "\\'")
            expr = f'title == "{escaped_title}"'
            results = chunk_collection.query(
                expr=expr,
                output_fields=["content", "chunk_index", "doc_id", "title"]
            )
            
            if not results:
                self.log(f"未找到标题为'{title}'的文档", level="warning")
                return ""
            
            results.sort(key=lambda x: x.get('chunk_index', 0))
            full_text = "\n\n".join([chunk.get('content', '') for chunk in results if chunk.get('content')])
            
            self.log(f"✅ 找到标题为'{title}'的文档，共{len(results)}个chunks")
            return full_text
            
        except Exception as e:
            self.log(f"查询全文失败: {e}", level="error")
            return ""

    def generate_final_investment_summary(self, report_content: str, policy_title: str) -> str:
        """
        基于完整报告内容生成总的投资建议
        
        Args:
            report_content: 完整的Markdown报告内容
            policy_title: 政策标题
            
        Returns:
            投资建议的Markdown文本
        """
        prompt = f"""你是一名资深投资策略分析师，请基于以下政策分析报告，撰写一份详细的投资建议总结。

=== 政策分析报告 ===

{report_content}

=== 任务 ===

请基于上述分析报告（包括会议对比和分主题增量分析），撰写一份详细、深入的投资建议总结。

=== 输出格式 ===

## 投资建议总结

### 核心观点

用5-8句话深入分析本次政策的核心信号和投资方向：

1. **政策定位与基调**：本次政策在宏观政策周期中的定位，整体基调是积极/稳健/收紧，与前期政策相比有何变化
2. **最大边际变化**：相比历史政策，最显著的增量信息是什么，为什么这个变化重要
3. **政策优先级判断**：从政策表述的篇幅、措辞强度判断，哪些领域是政策重点发力方向
4. **市场影响路径**：政策如何传导到市场，对流动性、风险偏好、行业景气度的影响
5. **投资主线提炼**：基于以上分析，提炼出1-2条核心投资主线

### 各板块政策增量分析

基于报告中的分主题增量分析，逐一总结每个板块的政策增量（必须覆盖报告中分析的所有板块）：

**板块名称1**
- 投资评级：看多/看平/看空
- 政策增量：相比历史政策新增/强化了什么（具体说明边际变化，引用关键表述）
- 核心逻辑：为什么给出这个评级，政策变化如何影响行业基本面
- 受益方向：哪类公司/细分领域最受益

**板块名称2**
- 投资评级：看多/看平/看空
- 政策增量：相比历史政策新增/强化了什么（具体说明边际变化，引用关键表述）
- 核心逻辑：为什么给出这个评级，政策变化如何影响行业基本面
- 受益方向：哪类公司/细分领域最受益

（以此类推，列出报告中分析的所有板块）

### 投资策略建议

**短期（0-3个月）**：
- 事件催化：近期可能的政策落地节点（如重要会议、政策细则出台、项目审批等）
- 市场节奏判断：政策发布后市场可能的反应路径
- 操作建议：具体的配置方向，哪些板块可以积极参与，哪些需要等待
- 仓位建议：整体仓位水平建议

**中期（3-12个月）**：
- 主线逻辑：中期配置的核心逻辑，为什么这些方向具有持续性
- 景气度判断：政策如何影响相关行业的盈利周期
- 配置方向：具体的板块和细分方向，按优先级排序
- 估值考量：当前估值水平是否支持配置

### 风险提示

列出4-5个主要风险点，每个风险说明：
- 风险描述：具体是什么风险
- 影响程度：对投资逻辑的影响有多大
- 应对建议：如何规避或应对

=== 要求 ===

1. 内容必须基于报告中的分析，不要凭空编造
2. 各板块政策增量分析必须覆盖报告中分析的所有板块，不要遗漏
3. 核心观点要有深度，体现对政策的深层理解和投资逻辑推演
4. 投资策略要具体可操作，不要泛泛而谈
5. 重点突出政策的边际变化和增量信息
6. 观点要明确，不要模棱两可
7. 不要用省略号(...)省略内容，完整输出所有分析
8. 总字数2000-3000字"""

        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.llm_client.chat_completion(
                messages=messages,
                temperature=0.3
            )
            
            self.log(f"✅ 投资建议总结生成完成")
            return response.strip()
            
        except Exception as e:
            self.log(f"投资建议总结生成失败: {e}", level="error")
            return "## 投资建议总结\n\n（生成失败）"
