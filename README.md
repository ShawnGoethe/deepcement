# DeepCement — 固井质量评测报告系统

基于 **DeepAgent** + **LlamaIndex** 的固井历史资料智能检索与质量评测系统。自动查询固井历史资料，生成质量评测报告，输出结论和改进建议。

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI 服务层 (api/)                      │
│   /health  │  /index/build  │  /search  │  /chat  │  /docs  │
└─────────────────────────────────────────────────────────────┘
                              ↑
┌─────────────────────────────────────────────────────────────┐
│                     Agent 编排层 (agent/)                     │
│   CementAgent 编排  │  工具路由  │  报告生成  │  技能管理      │
└─────────────────────────────────────────────────────────────┘
                              ↑
┌─────────────────────────────────────────────────────────────┐
│                      核心引擎层 (core/)                       │
│  文档摄入  │  索引管理  │  语义检索  │  质量评测  │  分层评估   │
│  规则引擎  │  XGBoost  │           │           │            │
└─────────────────────────────────────────────────────────────┘
                              ↑
┌─────────────────────────────────────────────────────────────┐
│                      模型层 (models/)                         │
│   BGE-Large-ZH (Embedding)  │  BGE-Reranker-v2-M3 (重排)    │
│              XGBoost (固井质量预测)  │  LLM (评测/对话)        │
└─────────────────────────────────────────────────────────────┘
                              ↑
┌─────────────────────────────────────────────────────────────┐
│                      混合存储层                              │
│  InfluxDB     │  Milvus  │  Neo4j  │  MinIO                 │
│   (时序库)    │ (向量库)  │ (图库)  │ (对象存储)              │
└─────────────────────────────────────────────────────────────┘
```



## 评测维度


| 维度        | 评估要点                                      |
| --------- | ----------------------------------------- |
| **水泥浆性能** | 密度（1.85-1.95 g/cm³）、失水量（<50ml）、稠化时间、配方规范性 |
| **施工参数**  | 泵速/排量平稳性、施工压力、替浆量计算、施工连续性                 |
| **固井效果**  | 水泥返高、候凝时间、声幅测井(CBL)、胶结质量                  |
| **异常情况**  | 漏失、窜槽风险、气侵、其他异常及处理                        |




## 快速开始



### 1. 环境准备

```bash
# 克隆项目
git clone <repo-url>
cd DeepCement

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows

# 安装依赖
pip install -r requirements.txt
```



### 2. 配置模型



### 3. 导入固井资料

将固井资料文件放入 `data/raw/` 目录：

```
data/raw/
├── XX-1井固井施工报告.pdf
├── XX-2井完井报告.pdf
├── 固井数据表.xlsx
└── 施工参数记录.csv
```

支持格式：

- **PDF** — 固井施工报告、完井报告（PyMuPDF 解析）
- **Word** — 固井文档（python-docx 解析）
- **Excel** — 固井数据表、施工参数记录表
- **CSV** — 结构化数据



### 4. 构建索引

```bash
python main.py --build-index
```

输出示例：

```
14:30:15 |    INFO | 解析完成: XX-1井固井施工报告.pdf → 1 个文档片段
14:30:16 |    INFO | 解析完成: XX-2井完井报告.pdf → 1 个文档片段
14:30:16 |    INFO | 共解析 2 个文档片段
14:30:18 |    INFO | 开始构建索引，共 2 个文档...
14:30:25 |    INFO | 索引构建完成
14:30:25 |    INFO | 索引已保存到: data/index
```

### 5. 运行评测



#### 交互模式（推荐）

```bash
python main.py --interactive
```

```
==================================================
  DeepCement 固井质量评测系统
  输入井名和查询，系统将自动检索和评测
  输入 'quit' 退出
==================================================

就绪！请输入查询：

>>> 评测 XX-1 井的固井质量

=== 固井质量评测报告: XX-1 ===
综合得分: 82.5分 (良好)

--- 各维度评分 ---
  水泥浆性能: 85.0分 (良好)
  施工参数: 80.0分 (良好)
  固井效果: 83.0分 (良好)
  异常情况: 82.0分 (良好)

--- 结论 ---
该井固井质量整体良好，各项指标基本达标...

--- 改进建议 ---
  1. 建议优化水泥浆配方，降低失水量
  2. 加强施工过程压力监控
  3. 适当延长候凝时间
```



#### 单次查询

```bash
# 评测指定井
python main.py --well "XX-1"

# 带自定义查询
python main.py --well "XX-1" --query "分析该井固井质量问题"
```

生成的报告自动保存到 `agent/report/output/` 目录。

---



## 项目结构

```
DeepCement/
├── main.py                          # FastAPI 服务入口
├── config.py                        # 配置管理（pydantic-settings）
├── requirements.txt                 # Python 依赖
├── pyproject.toml                   # 项目元数据 + 工具配置
├── .env.example                     # 环境变量模板
├── format.bat / format.sh           # 代码格式化脚本（black + isort + ruff）
│
├── api/                             # FastAPI 服务层
│   ├── app.py                       # FastAPI 应用实例 + 路由挂载
│   ├── routes.py                    # API 路由定义（/health, /index/build, /search, /chat）
│   ├── schemas.py                   # 请求/响应 Pydantic 模型
│   └── dependencies.py              # 依赖注入（Agent 初始化、索引状态检查）
│
├── core/                            # 核心业务模块
│   ├── ingester.py                  # 数据摄入（PDF/Word/Excel/CSV 解析 + 元数据提取）
│   ├── indexer.py                   # LlamaIndex 向量索引管理
│   ├── retriever.py                 # 语义检索引擎（含 BGE Reranker）
│   ├── evaluator.py                 # LLM 四维度质量评测
│   ├── extractor.py                 # LLM 结构化数据抽取（三元组）
│   └── graph_builder.py             # 知识图谱 + SQLite 存储
│
├── agent/                           # 智能体层
│   ├── orchestrator.py              # CementAgent 编排（含降级模式）
│   ├── tools.py                     # Agent 工具定义（search, evaluate, compare）
│   ├── report/
│   │   ├── generator.py             # Markdown 报告生成器
│   │   └── templates/
│   │       └── quality_report.md    # Jinja2 报告模板
│   └── workspace/                   # Agent 工作空间
│       ├── memory/                  # 长期/短期记忆
│       ├── prompts/                 # 系统提示词
│       └── skills/                  # 技能定义（5 个技能）
│
├── config/
│
├── models/                          # 预训练模型
│   ├── bge-large-zh/                # BGE 中文 Embedding 模型
│   └── bge-reranker-v2-m3/          # BGE 重排序模型
│
├── data/
│   └── raw/                         # 原始固井资料（PDF/Word/Excel/CSV）
│
├── agent/report/output/             # 生成的报告输出
│
└── tests/
    ├── test_basic.py                # 基础测试
    └── test_pdf_query.py            # PDF 查询测试
```

---



## 核心模块说明

### 数据摄入 (`core/ingester.py`)

- 解析 PDF/Word/Excel/CSV 文件，输出标准化 `CementDocument`
- 自动提取元数据：井名、日期、井深、固井井段
- 支持按段落拆分 PDF 以保留上下文完整性

### 索引管理 (`core/indexer.py`)

- 基于 LlamaIndex `VectorStoreIndex` 构建向量索引
- 支持持久化存储和增量更新
- 延迟加载 LLM 和 Embedding 模型

### 语义检索 (`core/retriever.py`)

- 语义搜索 + 元数据过滤（井名、时间段）
- 提供 `search()`、`query()`、`search_by_well()` 等接口
- 集成 BGE Reranker 进行结果重排序，提升检索精度

### 质量评测 (`core/evaluator.py`)

- 4 维度独立评测（水泥浆性能、施工参数、固井效果、异常情况），每个维度由 LLM 评分
- 综合评分 = 各维度均分
- 自动生成结论和改进建议

### Agent 编排 (`agent/orchestrator.py`)

- 基于 DeepAgent 编排工具链
- 工具：`search_history`、`evaluate_quality`、`archive_file`
- 未安装 DeepAgent 时自动降级为规则匹配模式

---



## 使用流程图

```
┌─────────────────────────────────────────────────────────┐
│                    首次使用                               │
├─────────────────────────────────────────────────────────┤
│  1. 安装依赖        pip install -r requirements.txt     │
│  2. 配置模型        cp .env.example .env → 填入 API Key  │
│  3. 导入资料        将 PDF/Word/Excel 放入 data/raw/     │
│  4. 构建索引        python main.py --build-index         │
│  5. 开始评测        python main.py --interactive         │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                    日常使用                               │
├─────────────────────────────────────────────────────────┤
│  新增资料 → 放入 data/raw/ → 重新 build-index            │
│  查询评测 → python main.py --interactive                 │
│  生成报告 → 自动保存到 agent/report/output/               │
└─────────────────────────────────────────────────────────┘
```

---



## 运行测试

```bash
python tests/test_basic.py
```

---



## API 接口

服务启动后访问 `http://localhost:8000/docs` 查看 Swagger 文档。

| 方法   | 路径              | 说明                |
| ---- | --------------- | ----------------- |
| GET  | `/health`       | 健康检查 + 索引状态       |
| POST | `/index/build`  | 构建向量索引（从 data/raw/） |
| POST | `/search`       | 语义检索历史资料          |
| POST | `/chat`         | Agent 对话           |

---

## 技术栈

| 组件        | 选型                          | 用途               |
| --------- | --------------------------- | ---------------- |
| Web 框架    | FastAPI                      | REST API 服务      |
| Agent 框架  | DeepAgent                    | 智能体编排，串联检索→分析→报告 |
| RAG 框架    | LlamaIndex                   | 文档索引、语义检索        |
| LLM       | DeepSeek / Qwen              | 质量评测、报告生成        |
| Embedding | BGE-Large-ZH                  | 中文文档向量化          |
| 重排序       | BGE-Reranker-v2-M3            | 检索结果重排序          |
| 机器学习      | XGBoost                      | 固井质量预测          |
| 文档解析      | PyMuPDF / python-docx / pandas | PDF / Word / Excel 解析 |
| 报告格式      | Markdown + Jinja2             | 结构化评测报告输出        |


