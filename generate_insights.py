"""
生成投资建议 - 并行化生成流程（各Agent使用独立prompt并行生成Word部分）
"""
import pandas as pd
from datetime import datetime
from vector_db import MilvusVectorDatabase
from agents import IndustryAgent, NoveltyAgent, SimplifiedRAGAgent, InvestmentAgent
from models import PolicySegment
from report_generator import ReportGenerator
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

print("=" * 80)
print("政策分析系统 - 分块化生成投资建议")
print("=" * 80)

# ==========================================
# 步骤1: 加载测试数据
# ==========================================
print(f"\n[步骤1] 加载测试数据...")
print("-" * 80)

test_file = './test.parquet'
df_test = pd.read_parquet(test_file)
print(f"✅ 测试数据加载完成: {len(df_test)} 个新政策")

# ==========================================
# 步骤2: 初始化系统
# ==========================================
print(f"\n[步骤2] 初始化系统...")
print("-" * 80)

# 连接Milvus
db = MilvusVectorDatabase(
    collection_name="policy_documents",
    chunk_only=True
)
print(f"✅ RAG系统初始化完成")

# 初始化Agents
industry_agent = IndustryAgent()
rag_agent = SimplifiedRAGAgent(db)
novelty_agent = NoveltyAgent(vector_db=db)  # 新版：两阶段LLM处理
investment_agent = InvestmentAgent(vector_db=db)  # 集中所有prompt调用
report_generator = ReportGenerator()

# ==========================================
# 步骤3: 合并附件内容（与run_full_pipeline.py保持一致）
# ==========================================
print(f"\n[步骤3] 合并附件内容...")
print("-" * 80)

# 查找各个列（按列名匹配）
attachment_col = None
content_col = None
report_series_col = None

for col in df_test.columns:
    col_str = str(col)
    if '附件' in col_str and '内容' in col_str:
        attachment_col = col
    elif '报告系列' in col_str:
        report_series_col = col
    elif content_col is None:
        if '政策全文' in col_str:
            content_col = col
        elif '内容' in col_str and '附件' not in col_str:
            content_col = col

# 如果通过列名找不到政策全文，使用索引
if content_col is None and len(df_test.columns) > 7:
    content_col = df_test.columns[7]

# ⭐ 显示识别到的列
print(f"   识别到的列:")
print(f"   - 政策全文列: {content_col}")
print(f"   - 附件内容列: {attachment_col or '无'}")
print(f"   - 报告系列列: {report_series_col or '无'}")

# 合并附件内容
if attachment_col is not None and content_col is not None:
    merged_count = 0
    for idx in df_test.index:
        policy_content = str(df_test.at[idx, content_col]) if pd.notna(df_test.at[idx, content_col]) else ""
        attachment_content = str(df_test.at[idx, attachment_col]) if pd.notna(df_test.at[idx, attachment_col]) else ""
        
        if attachment_content and attachment_content.strip() not in ['None', 'nan', 'NaN', '']:
            if policy_content.strip():
                merged_content = f"{policy_content}\n\n---附件内容---\n{attachment_content}"
            else:
                merged_content = attachment_content
            df_test.at[idx, content_col] = merged_content
            merged_count += 1
    
    if merged_count > 0:
        print(f"✅ 合并完成: {merged_count} 个文档")
else:
    print(f"✅ 无需合并附件")

# ==========================================
# 步骤4: 转换为PolicySegment并打行业标签（与run_full_pipeline.py保持一致）
# ==========================================
print(f"\n[步骤4] 转换为PolicySegment并打行业标签...")
print("-" * 80)

# ⭐ 获取Milvus中最大的doc_id编号，继续编号
max_doc_id_number = db.get_max_doc_id_number()
print(f"✅ 当前最大doc_id编号: {max_doc_id_number}，新政策将从 doc_{max_doc_id_number+1:04d} 开始编号")

# ⭐ 获取Milvus中已存在的 (标题, 时间) 组合（用于入库前去重）
print(f"\n[步骤3.5] 检查Milvus中已存在的文档...")
existing_pairs_in_milvus = db.get_existing_title_timestamp_pairs()
print(f"✅ Milvus中已存在 {len(existing_pairs_in_milvus)} 个唯一文档")

output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

segments = []  # 需要入库的新文档
segments_existing = []  # 已存在于Milvus的文档（不需要入库，但需要生成报告）
seen_titles = set()
seen_contents = set()
seen_pairs = set()  # 本批次内的 (title, timestamp) 去重

for i, row in df_test.iterrows():
    print(f"\n{'='*60}")
    title = str(row.iloc[0]) if len(row) > 0 else "未命名文档"
    print(f"政策 {i+1}/{len(df_test)}: {title[:50]}...")
    print(f"{'='*60}")
    
    try:
        # 先解析时间戳（用于去重判断）
        timestamp_value = None
        timestamp_str_for_check = ""
        if len(row) > 2:
            timestamp_str = str(row.iloc[2]).strip()
            if timestamp_str and timestamp_str.lower() not in ['', 'nan', 'none', 'nat']:
                try:
                    timestamp_value = datetime.fromisoformat(timestamp_str)
                    timestamp_str_for_check = timestamp_value.isoformat()
                except:
                    try:
                        timestamp_value = pd.to_datetime(timestamp_str)
                        timestamp_str_for_check = timestamp_value.isoformat()
                    except:
                        pass
        
        if timestamp_value is None:
            timestamp_value = datetime(2024, 1, 1)
            timestamp_str_for_check = timestamp_value.isoformat()
        
        # 获取内容
        if content_col is not None:
            content = str(row[content_col]) if pd.notna(row[content_col]) else ""
        else:
            content = str(row.iloc[7]) if len(row) > 7 else ""
        
        # 本批次内去重（标题+时间组合）
        check_pair = (title, timestamp_str_for_check)
        if check_pair in seen_pairs:
            print(f"  ⚠️ 跳过重复文档（本批次内重复）: {title[:50]}...")
            continue
        
        seen_pairs.add(check_pair)
        seen_titles.add(title)
        seen_contents.add(content)
        
        # 读取报告系列
        report_series = ""
        if report_series_col is not None:
            rs_value = row.get(report_series_col) if hasattr(row, 'get') else row[report_series_col]
            if pd.notna(rs_value):
                report_series = str(rs_value).strip()
                if report_series.lower() in ['null', 'none', 'nan']:
                    report_series = ""
        
        # 计算doc_id编号
        doc_id_number = max_doc_id_number + len(segments) + len(segments_existing) + 1
        
        seg = PolicySegment(
            doc_id=f"doc_{doc_id_number:04d}",
            content=content,
            title=title,
            timestamp=timestamp_value,
            industries=[],
            metadata={'report_series': report_series}
        )
        
        # ⭐ 检查Milvus中是否已存在：已存在的不入库，但仍需生成报告
        if check_pair in existing_pairs_in_milvus:
            print(f"  ℹ️ 已存在于Milvus（不入库，但生成报告）: {title[:50]}...")
            seg.metadata['skip_insert'] = True  # 标记不需要入库
            segments_existing.append(seg)
        else:
            segments.append(seg)
    except Exception as e:
        print(f"  ⚠️ 文档 {i} 转换失败: {e}")
        continue

print(f"✅ 转换完成: {len(segments)} 个新文档, {len(segments_existing)} 个已存在文档")

# 合并所有需要处理的文档（新文档 + 已存在文档）
all_segments = segments + segments_existing

# 行业分类（包含投资相关性判断）
all_segments = industry_agent.process(all_segments)
print(f"✅ 行业标签完成（含投资相关性判断）")

# ==========================================
# 步骤5: 向量化并存入Milvus（仅新文档需要入库）
# ==========================================
print(f"\n[步骤5] 向量化并存入Milvus...")
print("-" * 80)

# 只入库新文档
new_segments = [seg for seg in all_segments if not seg.metadata.get('skip_insert', False)]
if new_segments:
    for seg in new_segments:
        print(f"\n{'='*60}")
        print(f"入库文档: {seg.title[:50]}...")
        print(f"{'='*60}")
        print(f"  行业: {', '.join(seg.industries[:5]) if seg.industries else '无'}")
        print(f"  投资相关性: {seg.metadata.get('investment_relevance', '未判断')}")
        db.add_documents([seg], batch_size=1)
        print(f"  ✅ 文档已入库: {seg.doc_id}")
else:
    print("ℹ️ 没有新文档需要入库")

# ==========================================
# 步骤6: 生成分析报告（所有文档都需要生成报告）
# ==========================================
print(f"\n[步骤6] 生成分析报告...")
print("-" * 80)

for idx, seg in enumerate(all_segments):
    print(f"\n{'='*60}")
    print(f"生成报告: {seg.title[:50]}...")
    print(f"{'='*60}")
    
    # ========== 新流程：Meeting对比 → 主题词 → 分主题RAG ==========
    
    # 6.1 行业分析（简单输出）
    print(f"\n[6.1] 行业分类分析...")
    try:
        industry_section = investment_agent.generate_industry_section(seg)
        print(f"  ✅ 行业分析生成完成")
    except Exception as e:
        print(f"  ⚠️ 行业分析生成失败: {e}")
        industry_section = "## 行业分类分析\n\n（生成失败）"
    
    # 6.2 报告系列时间对比分析（先执行，提取主题词）
    print(f"\n[6.2] 报告系列时间对比分析（提取主题词）...")
    meeting_topics = []
    try:
        meeting_result = investment_agent.generate_meeting_section(seg, vector_db=db)
        # 新版返回字典，包含analysis和topics
        if isinstance(meeting_result, dict):
            meeting_section = meeting_result.get('analysis', '## 报告系列时间对比分析\n\n（生成失败）')
            meeting_topics = meeting_result.get('topics', [])
            print(f"  ✅ 报告系列对比完成，提取到 {len(meeting_topics)} 个主题词")
            if meeting_topics:
                print(f"  主题词: {meeting_topics[:10]}...")
        else:
            # 兼容旧版返回字符串
            meeting_section = meeting_result
            print(f"  ✅ 报告系列对比完成（旧版格式）")
    except Exception as e:
        print(f"  ⚠️ 报告系列时间对比分析失败: {e}")
        meeting_section = "## 报告系列时间对比分析\n\n（生成失败）"
    
    # 6.3 分主题RAG增量分析（使用meeting提取的主题词）
    print(f"\n[6.3] 分主题RAG增量分析...")
    try:
        # 将meeting提取的主题词传给novelty_agent
        novelty_result = novelty_agent.analyze_with_topics(seg, topics=meeting_topics)
        novelty_section = novelty_result.get('analysis', '## 分主题增量分析\n\n（生成失败）')
        topic_rag_results = novelty_result.get('topic_rag_results', {})
        print(f"  ✅ 分主题增量分析完成")
    except Exception as e:
        print(f"  ⚠️ 分主题增量分析失败: {e}")
        import traceback
        traceback.print_exc()
        novelty_section = "## 分主题增量分析\n\n（生成失败）"
    
    # 合并结果
    results = {
        'industry': industry_section,
        'meeting': meeting_section,
        'novelty': novelty_section,
        'investment': ''  # 投资建议暂时关闭
    }
    
    # 6.3 生成Markdown报告
    print(f"\n[6.3] 生成Markdown报告...")
    
    # 组装完整的Markdown报告（新顺序：行业 → Meeting对比 → 分主题增量分析）
    md_content = f"""# 政策分析报告

## 基本信息

- **标题**：{seg.title}
- **发布时间**：{seg.timestamp.strftime('%Y年%m月%d日') if seg.timestamp else 'N/A'}

---

{results.get('industry', '')}

---

{results.get('meeting', '')}

---

{results.get('novelty', '')}
"""
    
    # 6.4 生成总的投资建议
    print(f"\n[6.4] 生成投资建议总结...")
    try:
        investment_summary = investment_agent.generate_final_investment_summary(md_content, seg.title)
        md_content += f"""

---

{investment_summary}
"""
        print(f"  ✅ 投资建议总结生成完成")
    except Exception as e:
        print(f"  ⚠️ 投资建议总结生成失败: {e}")
    
    # 保存为Markdown文件
    report_file = output_dir / f"report_{idx+1:04d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"  ✅ 报告已保存: {report_file}")
    
    # （备用）如需生成Word报告，取消下面注释：
    # doc = report_generator.generate_report(
    #     segment=seg,
    #     industry_section=results.get('industry', ''),
    #     novelty_section=results.get('novelty', ''),
    #     meeting_section=results.get('meeting', ''),
    #     investment_section=results.get('investment', '')
    # )
    # word_file = output_dir / f"report_{idx+1:04d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
    # report_generator.save(str(word_file))

print(f"\n{'='*80}")
print("✅ 所有政策处理完成！")
print(f"{'='*80}")
