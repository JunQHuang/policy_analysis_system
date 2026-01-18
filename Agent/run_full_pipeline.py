"""
一键运行：建立RAG知识库
"""
import pandas as pd
import sys
import argparse
from datetime import datetime
from vector_db import MilvusVectorDatabase
from agents import IndustryAgent
from models import PolicySegment
from pathlib import Path

# ==========================================
# 解析命令行参数
# ==========================================
parser = argparse.ArgumentParser(description='建立RAG知识库')
parser.add_argument('--data', '-d', type=str, default='./合并数据_20251202_161356.parquet',
                    help='数据文件路径（parquet格式）')
args = parser.parse_args()

print("="*80)
print("政策分析RAG系统 - 建立知识库")
print("="*80)

# ==========================================
# 步骤1: 加载历史数据
# ==========================================
print(f"\n[步骤1] 加载历史数据...")
data_file = args.data
print(f"数据文件: {data_file}")

# 检查文件是否存在
if not Path(data_file).exists():
    print(f"❌ 错误: 数据文件不存在: {data_file}")
    print(f"💡 提示: 使用 --data 参数指定数据文件路径")
    print(f"   例如: python run_full_pipeline.py --data ./输出文件.parquet")
    sys.exit(1)

df = pd.read_parquet(data_file)
print(f"✅ 数据加载完成: {len(df)} 个文档")

# ==========================================
# 步骤2: 合并附件内容到政策内容
# ==========================================
print(f"\n[步骤2] 合并附件内容...")

# 查找各个列（按列名匹配）
attachment_col = None
content_col = None
report_series_col = None

for col in df.columns:
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
if content_col is None and len(df.columns) > 7:
    content_col = df.columns[7]

# ⭐ 显示识别到的列
print(f"   识别到的列:")
print(f"   - 政策全文列: {content_col}")
print(f"   - 附件内容列: {attachment_col or '无'}")
print(f"   - 报告系列列: {report_series_col or '无'}")

# 合并附件内容
if attachment_col is not None and content_col is not None:
    merged_count = 0
    for idx in df.index:
        policy_content = str(df.at[idx, content_col]) if pd.notna(df.at[idx, content_col]) else ""
        attachment_content = str(df.at[idx, attachment_col]) if pd.notna(df.at[idx, attachment_col]) else ""
        
        if attachment_content and attachment_content.strip() not in ['None', 'nan', 'NaN', '']:
            if policy_content.strip():
                merged_content = f"{policy_content}\n\n---附件内容---\n{attachment_content}"
            else:
                merged_content = attachment_content
            df.at[idx, content_col] = merged_content
            merged_count += 1
    
    print(f"✅ 合并完成: {merged_count} 个文档")

# ==========================================
# 步骤3: 连接Milvus并检查已存在的文档
# ==========================================
print(f"\n[步骤3] 连接Milvus并检查已存在的文档...")

db = MilvusVectorDatabase(
    collection_name="policy_documents",
    embedding_model="./models/xiaobu-embedding-v2",
    dim=1792,
    enable_chunking=True,
    chunk_only=True
)

# ⭐ 获取Milvus中最大的doc_id编号，用于继续编号
max_doc_id_number = db.get_max_doc_id_number()
print(f"✅ 当前Milvus中最大doc_id编号: {max_doc_id_number}")

# ⭐ 获取Milvus中已存在的 (标题, 时间) 组合（用于去重）
existing_pairs_in_milvus = db.get_existing_title_timestamp_pairs()
print(f"✅ Milvus中已存在 {len(existing_pairs_in_milvus)} 个唯一文档")

# ==========================================
# 步骤4: 转换为PolicySegment并打行业标签
# ==========================================
print(f"\n[步骤4] 转换为PolicySegment并打行业标签...")

segments = []
seen_titles = set()
seen_contents = set()
seen_pairs = set()  # 本批次内的 (title, timestamp) 去重
skipped_existing_count = 0  # 统计跳过的已存在文档数量

for i, row in df.iterrows():
    try:
        title = str(row.iloc[0]) if len(row) > 0 else "未命名文档"
        
        if content_col is not None:
            content = str(row[content_col]) if pd.notna(row[content_col]) else ""
        else:
            content = str(row.iloc[7]) if len(row) > 7 else ""
        
        # 转换时间戳
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
        
        # ⭐ 检查Milvus中是否已存在相同 (标题, 时间) 的文档
        check_pair = (title, timestamp_str_for_check)
        if check_pair in existing_pairs_in_milvus:
            skipped_existing_count += 1
            continue
        
        # 本批次内去重（标题+时间组合）
        if check_pair in seen_pairs:
            continue
        
        # 旧的去重逻辑（标题或内容相同）
        if title in seen_titles or content in seen_contents:
            continue
        
        seen_pairs.add(check_pair)
        seen_titles.add(title)
        seen_contents.add(content)
        
        # ⭐ 计算新的doc_id编号：从最大编号+1开始，按顺序递增
        doc_id_number = max_doc_id_number + len(segments) + 1
        
        # ⭐ 读取报告系列
        # - 如果parquet有"报告系列"列：直接使用该列的值（null则为空）
        # - 如果parquet没有"报告系列"列：设为空字符串
        report_series = ""
        if report_series_col is not None:
            rs_value = row.get(report_series_col) if hasattr(row, 'get') else row[report_series_col]
            if pd.notna(rs_value):
                report_series = str(rs_value).strip()
                # 处理字符串形式的 null/None
                if report_series.lower() in ['null', 'none', 'nan']:
                    report_series = ""
        
        seg = PolicySegment(
            doc_id=f"doc_{doc_id_number:04d}",
            content=content,
            title=title,
            timestamp=timestamp_value,
            industries=[],
            metadata={'report_series': report_series}  # ⭐ 存入报告系列
        )
        segments.append(seg)
    except Exception as e:
        print(f"  ⚠️ 文档 {i} 转换失败: {e}")
        continue

print(f"✅ 转换完成: {len(segments)} 个新文档需要入库")
if skipped_existing_count > 0:
    print(f"   跳过 {skipped_existing_count} 个已存在于Milvus的文档（标题+时间相同）")

# 如果没有新文档需要入库，直接退出
if not segments:
    print(f"\n✅ 没有新文档需要入库，所有数据已存在于Milvus中")
    stats = db.get_stats()
    print(f"   当前Chunk总数: {stats['total_chunks']}")
    print("\n" + "="*80)
    print("✅ 增量入库完成！")
    print("="*80)
    sys.exit(0)

# 行业分类（包含投资相关性判断，一次DS32B调用完成）
industry_agent = IndustryAgent()
segments = industry_agent.process(segments)
print(f"✅ 行业标签完成（含投资相关性判断）")

# ==========================================
# 步骤5: 向量化并存入Milvus
# ==========================================
print(f"\n[步骤5] 向量化并存入Milvus...")

db.add_documents(segments, batch_size=32)
stats = db.get_stats()

print(f"✅ 入库完成")
print(f"   本次新增文档数: {len(segments)}")
print(f"   当前Chunk总数: {stats['total_chunks']}")
print(f"   GPU: {stats['gpu_device']}")

print("\n" + "="*80)
print("✅ 知识库增量入库完成！")
print("="*80)
