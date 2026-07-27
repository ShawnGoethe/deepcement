# DeepCement — 固井质量评测报告系统

基于 **DeepAgent** + **LlamaIndex** 的固井历史资料智能检索与质量评测系统。自动查询固井历史资料，生成质量评测报告，输出结论和改进建议。

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                      应用层                                  │
│   岩性解释  │  胶结评价  │  风险预警  │  方案推荐  │  报告生成  │
└─────────────────────────────────────────────────────────────┘
                              ↑
┌─────────────────────────────────────────────────────────────┐
│                     DeepAgent 编排层                         │
│   Harness 调度  │  工具路由  │  多Agent协作  │  工作流引擎     │
└─────────────────────────────────────────────────────────────┘
                              ↑
┌─────────────────────────────────────────────────────────────┐
│                      工具封装                                │
│  检索融合  │  评分/报告  │  数据分析  │  异常检测 |  多模态    │
└─────────────────────────────────────────────────────────────┘                             
                              ↑
┌─────────────────────────────────────────────────────────────┐
│                      LlamaIndex 数据层                       │
│  文档解析  │  索引管理  │  检索融合  │  多模态处理             │
└─────────────────────────────────────────────────────────────┘
                              ↑
┌─────────────────────────────────────────────────────────────┐
│                      模型推理层                              │
│  规则引擎  │  XGBoost+windows/CNN+Transformer│ LLM │  KG推理 │
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

- **PDF**（重点支持）— 固井施工报告、完井报告
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

XGBoost 17个特征

```
# 全量训练（原有功能）
python scripts/train_xgboost.py --mode train

# 增量学习（新功能）
python scripts/train_xgboost.py --mode incremental

# 从单个文件增量学习
python scripts/train_xgboost.py --mode incremental --file data/raw/新井数据.TXT

# 自定义参数
python scripts/train_xgboost.py --mode incremental \
    --model-path models/xgboost_cement.json \
    --增量-trees 30 \
    --learning-rate 0.03
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
├── main.py                          # 入口文件
├── config.py                        # 配置管理
├── requirements.txt                 # Python 依赖
├── .env.example                     # 环境变量模板
├── .gitignore
│
├── core/                            # 核心模块
│   ├── ingester.py                  # 数据摄入（PDF/Excel/CSV 解析 + 元数据提取）
│   ├── indexer.py                   # LlamaIndex 向量索引管理
│   ├── retriever.py                 # 语义检索引擎
│   └── evaluator.py                 # 质量评测引擎
│
├── agent/                           # 智能体层
│   ├── orchestrator.py              # DeepAgent 编排（含降级模式）
│   ├── tools.py                     # Agent 工具定义
│   └── report/
│       ├── generator.py             # Markdown 报告生成器
│       └── templates/
│           └── quality_report.md    # 报告模板
│
├── data/
│   ├── raw/                         # 原始固井资料（PDF/Excel/CSV）
│   └── index/                       # 向量索引持久化存储
│
├── agent/report/output/             # 生成的报告输出
│
└── tests/
    └── test_basic.py                # 基础测试
```

---



## 核心模块说明



### 数据摄入 (`core/ingester.py`)

- 解析 PDF/Excel/CSV 文件，输出标准化 `CementDocument`
- 自动提取元数据：井名、日期、井深、固井井段
- 支持按段落拆分 PDF 以保留上下文完整性



### 索引管理 (`core/indexer.py`)

- 基于 LlamaIndex `VectorStoreIndex` 构建向量索引
- 支持持久化存储和增量更新
- 延迟加载 LLM 和 Embedding 模型



### 语义检索 (`core/retriever.py`)

- 语义搜索 + 元数据过滤（井名、时间段）
- 提供 `search()`、`query()`、`search_by_well()` 等接口
- `query()` 由 LLM 基于检索结果生成回答



### 质量评测 (`core/evaluator.py`)

- 4 维度独立评测，每个维度由 LLM 评分
- 综合评分 = 各维度均分
- 自动生成结论和改进建议



### Agent 编排 (`agent/orchestrator.py`)

- 基于 DeepAgent 编排工具链
- 工具：`search_history`、`evaluate_quality`、`compare_data`
- 未安装 DeepAgent 时自动降级为规则匹配模式

---



## 使用流程图

```
┌─────────────────────────────────────────────────────────┐
│                    首次使用                               │
├─────────────────────────────────────────────────────────┤
│  1. 安装依赖        pip install -r requirements.txt     │
│  2. 配置模型        cp .env.example .env → 填入 API Key  │
│  3. 导入资料        将 PDF/Excel 放入 data/raw/          │
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



## 技术栈


| 组件        | 选型               | 用途               |
| --------- | ---------------- | ---------------- |
| Agent 框架  | DeepAgent        | 智能体编排，串联检索→分析→报告 |
| RAG 框架    | LlamaIndex       | 文档索引、语义检索        |
| LLM       | DeepSeek / Qwen  | 质量评测、报告生成        |
| Embedding | 国产 Embedding 模型  | 文档向量化            |
| 文档解析      | PyMuPDF / pandas | PDF / Excel 解析   |
| 报告格式      | Markdown         | 结构化评测报告输出        |


