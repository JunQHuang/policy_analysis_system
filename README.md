# policy_analysis_system📖 项目简介

本项目是一个面向金融投资领域的端到端政策分析智能系统，融合了**RAG（检索增强生成）**、**向量数据库**、**大语言模型**和**Agent工作流**等前沿技术。系统从政策文档爬取、解析、向量化存储，到智能检索、语义分析、投资建议生成，实现了政策分析的全流程自动化。
例如：
<img width="1089" height="1455" alt="image" src="https://github.com/user-attachments/assets/e3c4d39f-fe8a-41e6-b92d-6756a8212599" />

### 技术亮点

- **MinerU文档解析引擎**：支持PDF/DOC/DOCX/OFD等多格式政策文档的高精度解析和结构化提取
- **分布式爬虫系统**：多线程异步爬取政策附件，支持断点续传和智能去重
- **Milvus向量数据库**：百万级政策文档的高性能向量检索，支持混合查询和相似度排序
- **BERT语义匹配**：基于预训练模型的行业分类和语义相似度计算
- **Agent工作流编排**：模块化的Agent协同处理，支持灵活的任务编排和结果聚合
- **LLM增强分析**：集成火山引擎大模型API，实现政策增量分析和投资建议生成

## 🎯 核心功能

- **政策文档采集**：自动爬取政府网站政策文档及附件，支持多种文件格式
- **文档智能解析**：使用MinerU引擎解析PDF/DOC等格式，提取结构化文本
- **向量化知识库**：将历史政策文档切分、向量化并存储到Milvus数据库
- **语义检索增强**：基于向量相似度的RAG检索，快速定位相关历史政策
- **行业智能分类**：使用BERT模型进行中信行业分类体系的自动匹配
- **增量分析生成**：LLM对比历史政策，自动生成政策创新点和变化分析
- **投资建议输出**：生成结构化Word报告，包含投资机会、板块配置、风险提示

## 🚀 快速开始

### 前置条件

```bash
# 1. 启动Milvus
docker-compose -f docker-compose-milvus-only.yml up -d

# 2. 激活Python环境
conda activate quant
```

### 两步运行

#### Step 1: 建立RAG知识库

```bash
python run_full_pipeline.py
```

- 读取历史政策数据
- 向量化存入Milvus
- **输出**：Milvus向量库

#### Step 2: 生成投资建议报告

```bash
python generate_insights.py
```

- 读取新政策数据
- RAG检索 + 增量分析 + 会议分析
- LLM生成投资建议
- **输出**：Word报告（`output/report_*.docx`）

---

## 📁 项目结构

```
gent/
├── main/                          # 主要执行脚本
│   ├── build_knowledge_base.py   # 建立知识库（原run_full_pipeline.py）
│   └── generate_report.py        # 生成报告（原generate_insights.py）
│
├── agents/                        # Agent工作流系统
│   ├── base.py                   # Agent基类
│   ├── industry_agent.py         # 行业分类Agent（BERT）
│   ├── novelty_agent.py          # 增量分析Agent（LLM生成）
│   ├── investment_agent.py       # 投资分析Agent
│   └── enhanced_rag_agent.py    # RAG检索Agent
│
├── core/                          # 核心功能模块
│   ├── models.py                 # 数据模型
│   ├── config.py                 # 配置文件
│   ├── vector_db.py              # Milvus向量数据库
│   └── clients/                  # LLM客户端
│       └── volcengine_client.py  # 火山引擎客户端
│
├── utils/                        # 工具模块
│   └── chunking.py               # 文档切分
│
├── scripts/                      # 工具脚本
│   ├── cleanup.py               # 数据库清理
│   └── test_milvus.py           # Milvus连接测试
│
├── report_generator.py          # 报告生成器（LLM生成投资建议）
├── citic_industries.py          # 中信行业分类
│
├── data/                        # 数据文件
│   └── examples/                # 示例数据
│
├── output/                      # 输出目录
└── docs/                        # 文档
```

---

## ⚙️ 配置说明

### LLM配置（config.py）

```python
# 火山引擎API配置
VOLCENGINE_API_KEY = "your_key"
VOLCENGINE_MODEL = "deepseek-r1-250120"
VOLCENGINE_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
```

---

## 🔧 核心技术架构

### 1. 数据采集与解析层

#### MinerU文档解析引擎

- **多格式支持**：PDF、DOC、DOCX、OFD、XLS、XLSX等政策文档格式
- **结构化提取**：保留文档标题、段落、表格等结构信息
- **OCR集成**：支持扫描版PDF的文字识别
- **批量处理**：支持大规模文档的并行解析和进度管理

#### 分布式爬虫系统

- **多线程异步爬取**：基于线程池的高并发下载，支持数千个附件的快速采集
- **智能去重机制**：基于URL和文件哈希的去重，避免重复下载
- **断点续传**：支持爬取任务的中断恢复，checkpoint机制保证数据完整性
- **格式转换**：自动将DOC转PDF，统一文档格式便于后续处理

### 2. 向量化存储层

#### Milvus向量数据库架构

- **版本**：Milvus 2.3+，支持GPU加速的向量检索
- **存储引擎**：基于RocksDB的持久化存储，支持数据快照和备份
- **索引算法**：HNSW（Hierarchical Navigable Small World）
  - M参数：16（每个节点的最大连接数）
  - efConstruction：200（构建索引时的搜索深度）
  - 检索复杂度：O(log N)，支持百万级向量的毫秒级响应
- **向量维度**：1792维（与xiaobu-embedding-v2匹配）
- **距离度量**：IP（Inner Product，内积），归一化向量下等价于Cosine相似度
- **混合查询**：支持向量检索 + 标量过滤的组合查询
  - 元数据过滤：行业分类、发布时间、文档类型等
  - 表达式语言：支持复杂的布尔逻辑和范围查询
- **分片策略**：支持Collection分片和数据分区，实现水平扩展
- **数据模型**：
  - Chunk级索引：每个文档切分为多个chunk，独立向量化和索引
  - 字段设计：chunk_id, doc_id, embedding(1792维), content(VARCHAR 5000), chunk_index, chunk_type, title, timestamp, industries等
  - 主键：chunk_id（唯一标识每个chunk）
- **性能指标**：
  - 插入吞吐：10000+ vectors/s（批量插入）
  - 查询延迟：<50ms（Top-K=20，百万级数据）
  - 召回率：>95%（HNSW参数优化后）

#### 文档切分策略（Chunking）

- **切分算法**：基于句子边界的递归切分（RecursiveCharacterTextSplitter）
  - 优先按段落分割（\n\n）
  - 其次按句子分割（。！？）
  - 最后按字符分割（保底策略）
- **Chunk大小**：
  - 目标大小：500字符（chunk_size=500）
  - 绝对上限：1200字符（absolute_max=1200）
  - 实际限制：450字符（Milvus VARCHAR字段限制，与embedding token限制对齐）
- **滑动窗口重叠**：
  - 重叠长度：150字符（overlap=150）
  - 重叠率：30%（150/500）
  - 目的：保证跨chunk的语义连贯性，避免关键信息被切断
- **Token对齐**：
  - Embedding模型限制：512 tokens
  - 中文字符-Token比例：约1:1.2（考虑标点和特殊字符）
  - 安全阈值：450字符 ≈ 540 tokens（留有余量）
- **元数据继承**：
  - 每个chunk继承文档级元数据：doc_id, title, publish_date, source, industries
  - Chunk级元数据：chunk_id（doc_id + chunk_index）, chunk_index, chunk_type
- **UTF-8编码处理**：
  - Milvus VARCHAR按字节长度限制（UTF-8编码）
  - 中文字符：3字节/字符
  - 截断策略：按字节边界截断，避免乱码

### 3. Agent工作流层

#### 模块化Agent设计

系统采用专业化Agent协同工作的架构，每个Agent负责特定的分析任务：

- **IndustryAgent（行业分类）**

  - 基于BERT预训练模型（chinese-bert-wwm-ext）
  - 计算政策文本与中信30个一级行业的语义相似度
  - 支持多级行业分类（一级/二级/三级）
  - 缓存机制加速重复查询
- **EnhancedRAGAgent（检索增强）**

  - Chunk级向量检索：基于cosine相似度的Top-K检索
  - 文档级聚合：按doc_id合并相关chunks，还原完整文档上下文
  - 行业过滤增强：同行业政策相似度加权（1.5x boost）
  - 时间衰减：考虑政策发布时间的相关性衰减
- **NoveltyAgent（增量分析）**

  - LLM驱动的政策对比分析
  - 自动识别政策创新点、变化点、延续点
  - 生成结构化的增量分析报告
  - 支持多轮对话式深度分析
- **InvestmentAgent（投资分析）**

  - 基于RAG检索结果的投资机会挖掘
  - 板块配置建议生成
  - 风险因素识别和提示
  - 投资时间窗口判断
- **ReportGenerator（报告生成）**

  - 整合所有Agent的分析结果
  - 生成结构化Word文档（python-docx）
  - 支持自定义报告模板和样式
  - 自动添加目录、页眉页脚等元素

### 4. 语义理解层

#### BERT语义匹配引擎

- **预训练模型**：chinese-bert-wwm-ext（Chinese BERT with Whole Word Masking）
  - 模型架构：BERT-base（12层Transformer，768维隐藏层）
  - 预训练语料：中文维基百科 + 其他中文语料（5GB+）
  - 词表大小：21128（中文字符 + WordPiece子词）
  - 全词遮罩：训练时对完整词进行遮罩，提升中文语义理解
- **句向量提取**：
  - 方法：[CLS] token的最后一层隐藏状态（pooled output）
  - 维度：768维稠密向量
  - 后处理：可选的mean pooling或max pooling
- **相似度计算**：
  - 度量方式：Cosine Similarity = cos(θ) = (A·B) / (||A|| × ||B||)
  - 取值范围：[-1, 1]，实际应用中归一化到[0, 1]
  - 阈值设定：>0.75视为高相似度，0.5-0.75中等相似度
- **推理优化**：
  - 批量推理：batch_size=32，充分利用GPU并行计算
  - 序列长度：最大512 tokens，超长截断
  - 精度：FP16混合精度推理（GPU），速度提升2x
  - 缓存策略：相同文本的向量结果缓存，避免重复计算
- **应用场景**：
  - 行业分类：计算政策文本与30个中信一级行业描述的相似度
  - 语义检索：计算查询文本与候选文档的语义匹配度
  - 去重检测：识别内容高度相似的重复政策文档

#### Embedding向量化技术栈

- **模型架构**：xiaobu-embedding-v2（基于Sentence-BERT架构的中文优化模型）
- **向量维度**：1792维高维稠密向量表示，提供更丰富的语义信息
- **归一化处理**：L2归一化（normalize_embeddings=True），将向量映射到单位超球面
- **相似度度量**：Cosine Similarity = dot(v1, v2)，归一化后等价于内积，计算效率高
- **Token限制**：最大512 tokens输入（中文约450-500字符），超长文本自动截断
- **批量编码**：支持batch encoding，GPU加速（CUDA），显著提升向量化吞吐量
- **模型加载**：基于SentenceTransformer框架，支持本地模型缓存和离线部署
- **精度优化**：FP32浮点精度，保证向量检索的准确性

### 5. LLM增强层

#### 火山引擎LLM API集成

- **模型选择**：DeepSeek-R1-250528（推理优化版本）
  - 模型规模：未公开（估计100B+参数）
  - 上下文窗口：32768 tokens（约24000中文字符）
  - 训练数据：截止2025年1月的多语言语料
  - 特点：强化学习优化，推理能力强，适合复杂分析任务
- **API配置**：
  - Endpoint：https://ark.cn-beijing.volces.com/api/v3
  - 认证方式：API Key（Bearer Token）
  - SDK：volcenginesdkarkruntime（官方Python SDK）
  - 协议：OpenAI兼容接口（Chat Completions API）
- **生成参数**：
  - temperature：0.1-0.3（低温度，减少随机性，保证输出稳定性和一致性）
  - max_tokens：32768（最大输出长度，支持长篇报告生成）
  - top_p：0.95（核采样，保留累积概率95%的token候选集）
  - frequency_penalty：0.0（不惩罚重复，允许术语重复出现）
  - presence_penalty：0.0（不惩罚新话题，允许自由展开）
- **可靠性保障**：
  - 重试机制：指数退避（1s, 2s, 4s, 8s, 16s），最多5次重试
  - 超时设置：60s连接超时，300s读取超时
  - 错误处理：区分4xx客户端错误（不重试）和5xx服务端错误（重试）
  - 限流应对：检测429状态码，自动延迟重试
- **流式输出**：
  - 支持Server-Sent Events（SSE）协议
  - 实时返回生成的token，提升用户体验
  - 支持中断和取消操作
- **Token计费**：
  - 输入token：按实际消耗计费
  - 输出token：按实际生成计费
  - 成本优化：缓存常用prompt，减少重复输入

#### Prompt工程与优化

- **角色设定（Role Prompting）**：
  - System Message：设定为"资深金融政策分析师"角色
  - 专业背景：10年+政策研究经验，熟悉宏观经济和行业分析
  - 输出风格：专业、客观、数据驱动
- **Few-shot学习**：
  - 提供2-3个高质量示例（输入-输出对）
  - 示例覆盖不同政策类型（财政、货币、产业等）
  - 引导模型学习输出格式和分析框架
- **Chain-of-Thought（思维链）**：
  - 引导模型分步推理："首先...其次...最后..."
  - 显式要求中间步骤："请先总结政策要点，再分析影响，最后给出建议"
  - 提升复杂分析任务的准确性和可解释性
- **结构化输出约束**：
  - 格式要求：Markdown格式，包含标题、列表、表格
  - 长度控制：增量分析800-1500字，投资建议500-1000字
  - 必需字段：政策要点、创新点、影响分析、投资建议、风险提示
- **上下文注入（Context Injection）**：
  - RAG检索结果：Top-K相关历史政策（截断到8000 tokens）
  - 元数据：政策标题、发布时间、发布机构、行业分类
  - 格式：XML标签包裹，便于模型识别和引用
- **温度与采样策略**：
  - 分析任务：temperature=0.2（低温度，保证客观性）
  - 创意任务：temperature=0.5（中等温度，增加多样性）
  - Top-p采样：0.95（保留高概率token，过滤低质量输出）
- **Prompt模板管理**：
  - 模板化设计：使用Jinja2模板引擎
  - 变量替换：{policy_content}, {rag_results}, {industry}等
  - 版本控制：Git管理prompt版本，支持A/B测试

### 6. 系统优化

#### 性能优化策略

- **多级缓存机制**：
  - L1缓存：内存缓存（LRU，最大1000条），缓存行业分类结果和BERT向量
  - L2缓存：磁盘缓存（JSON文件），持久化缓存，重启后可复用
  - 缓存键：基于内容哈希（MD5），保证唯一性
  - 命中率：>80%（重复政策和相似文本较多）
- **批量处理（Batching）**：
  - 向量化：batch_size=64，GPU批量编码，吞吐量提升10x
  - API调用：合并多个请求，减少网络往返（RTT）
  - 数据库插入：批量插入（batch_size=1000），减少事务开销
- **并行计算**：
  - 多线程：文档解析和爬取使用ThreadPoolExecutor（max_workers=16）
  - 多进程：CPU密集型任务（文本预处理）使用ProcessPoolExecutor
  - GPU加速：BERT和Embedding模型推理使用CUDA
- **异步IO（Async I/O）**：
  - 文件读写：使用aiofiles异步读写，避免阻塞
  - 网络请求：使用aiohttp异步HTTP客户端
  - 数据库操作：Milvus支持异步查询（async/await）
- **连接池管理**：
  - Milvus连接池：最大10个连接，复用TCP连接
  - HTTP连接池：requests.Session()，Keep-Alive复用
  - 超时设置：连接超时10s，读取超时60s
- **内存优化**：
  - 流式处理：大文件分块读取，避免一次性加载到内存
  - 垃圾回收：显式调用gc.collect()，释放大对象内存
  - 向量压缩：可选的PQ（Product Quantization）压缩，减少内存占用
- **索引优化**：
  - Milvus索引：HNSW参数调优（M=16, efConstruction=200）
  - 数据库索引：doc_id、timestamp等字段建立B-tree索引
  - 查询优化：避免全表扫描，使用索引覆盖查询

#### 可靠性与容错设计

- **异常处理体系**：
  - 分层异常：区分业务异常、系统异常、网络异常
  - Try-Catch覆盖：所有外部调用（API、数据库、文件IO）包裹异常处理
  - 错误日志：使用logging模块，记录异常堆栈和上下文信息
  - 日志级别：DEBUG（开发）、INFO（生产）、ERROR（异常）、CRITICAL（严重故障）
- **数据校验（Validation）**：
  - 输入校验：使用Pydantic模型，自动校验字段类型和格式
  - 必填字段：doc_id, content, title等核心字段非空检查
  - 长度限制：content ≤ 5000字符，title ≤ 500字符
  - 编码检查：UTF-8编码校验，拒绝非法字符
- **降级策略（Graceful Degradation）**：
  - API降级：LLM API失败时，返回基于规则的简化分析
  - 模型降级：GPU不可用时，自动切换到CPU推理
  - 检索降级：Milvus不可用时，使用本地缓存或简单关键词匹配
  - 功能降级：非核心功能失败时，跳过该步骤，继续执行主流程
- **重试机制（Retry Logic）**：
  - 指数退避：1s, 2s, 4s, 8s, 16s（最多5次）
  - 抖动（Jitter）：随机延迟±20%，避免雷鸣羊群效应
  - 幂等性保证：重试操作不产生副作用（如重复插入）
  - 断路器（Circuit Breaker）：连续失败5次后，暂停10分钟
- **数据一致性**：
  - 事务支持：Milvus批量插入使用事务，保证原子性
  - 幂等性：基于doc_id去重，避免重复插入
  - 校验和：文件下载后校验MD5，确保完整性
  - 备份恢复：定期备份Milvus数据，支持快速恢复
- **监控与告警**：
  - 性能指标：API响应时间、向量检索延迟、内存使用率
  - 业务指标：处理文档数、生成报告数、错误率
  - 告警规则：错误率>5%、响应时间>5s、内存使用>80%
  - 告警渠道：邮件、钉钉、Slack（可配置）
- **健康检查（Health Check）**：
  - Milvus连接：定期ping检查（每30s）
  - API可用性：定期调用测试接口（每5分钟）
  - 磁盘空间：检查剩余空间>10GB
  - 自动恢复：检测到故障后，自动重启服务

---

## 📝 使用示例

```python
from agents import IndustryAgent, NoveltyAgent, InvestmentAgent
from agents.enhanced_rag_agent import SimplifiedRAGAgent
from report_generator import ReportGenerator
from vector_db import MilvusVectorDatabase

# 初始化Agent工作流
db = MilvusVectorDatabase(collection_name="policy_documents", chunk_only=True)
rag_agent = SimplifiedRAGAgent(db)
industry_agent = IndustryAgent()
novelty_agent = NoveltyAgent()
investment_agent = InvestmentAgent()
report_generator = ReportGenerator()

# Agent工作流处理政策
seg = PolicySegment(...)

# Step 1: 行业分类
seg = industry_agent.classify_single(seg)

# Step 2: RAG检索相关历史政策
rag_results = rag_agent.search_enhanced(query_text=seg.content, top_k=10)

# Step 3: 增量分析
novelty_result = novelty_agent.process(seg, rag_results)

# Step 4: 投资分析
investment_result = investment_agent.process(seg, rag_results)

# Step 5: 生成完整报告
doc = report_generator.generate_report(
    segment=seg,
    industry_result={},
    novelty_result=novelty_result,
    investment_result=investment_result,
    rag_results=rag_results
)
report_generator.save("output/report.docx")
```

---

## 🛠️ 工具脚本

```bash
# 清理Milvus数据库
python scripts/cleanup.py

# 测试Milvus连接
python scripts/test_milvus.py
```

---

## 📊 输出格式

生成的Word报告包含：

1. **行业分类分析** - 中信行业分类结果
2. **政策增量分析** - LLM生成的增量分析报告
3. **会议关联分析** - 检测到的会议及相关文档
4. **投资建议** - LLM生成的投资机会、板块配置、风险提示

---

## ✅ 系统要求

- Python 3.8+
- Milvus 2.3+
- CUDA (可选，用于GPU加速)
- PyTorch
- transformers

---

## 📄 许可证

内部使用
