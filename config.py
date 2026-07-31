"""
DeepCement 配置管理
支持 DeepSeek / Qwen 等国产模型（兼容 OpenAI API 格式）
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings

# 项目根目录
BASE_DIR = Path(__file__).parent


class LLMConfig(BaseSettings):
    """LLM 模型配置"""

    base_url: str = Field(default="https://api.deepseek.com/v1", description="API 基础地址")
    api_key: str = Field(default="", description="API 密钥")
    model: str = Field(default="deepseek-chat", description="模型名称")
    temperature: float = Field(default=0.3, description="温度参数（评测任务用低温度）")
    max_tokens: int = Field(default=4096, description="最大输出 token 数")

    model_config = {"env_prefix": "LLM_"}


class EmbedConfig(BaseSettings):
    """Embedding 模型配置"""

    base_url: str = Field(default="https://api.deepseek.com/v1", description="API 基础地址")
    api_key: str = Field(default="", description="API 密钥")
    model: str = Field(default="deepseek-embedding", description="Embedding 模型名称")

    model_config = {"env_prefix": "EMBED_"}


class RetrieverConfig(BaseSettings):
    """检索配置"""

    top_k: int = Field(default=5, description="默认返回结果数量")

    model_config = {"env_prefix": "RETRIEVER_"}


class RerankerConfig(BaseSettings):
    """重排序模型配置（BGE Reranker）"""

    enabled: bool = Field(default=True, description="是否启用重排序")
    model_path: str = Field(
        default="models/bge-reranker-v2-m3",
        description="重排序模型路径（相对于项目根目录）",
    )
    top_n: int = Field(default=5, description="重排序后返回的结果数量")
    candidate_multiplier: int = Field(
        default=3,
        description="候选倍数：初始检索 top_k = top_n * candidate_multiplier",
    )

    model_config = {"env_prefix": "RERANKER_"}


class OCRConfig(BaseSettings):
    """PaddleOCR VL 配置（扫描件 PDF 识别）"""

    enabled: bool = Field(default=True, description="是否启用 OCR 识别（扫描件 PDF 兜底）")
    model_path: str = Field(
        default="models/paddleOCR",
        description="PaddleOCR VL 模型路径（相对于项目根目录）",
    )
    device: str = Field(default="auto", description="推理设备：auto / cuda / cpu")
    prompt: str = Field(
        default="请识别图片中的所有文字内容，保持原始排版格式。",
        description="OCR 提示词",
    )
    min_text_length: int = Field(
        default=50,
        description="页面文本长度低于此值时触发 OCR（视为扫描件）",
    )

    model_config = {"env_prefix": "OCR_"}


class GraphConfig(BaseSettings):
    """知识图谱 + 结构化数据配置"""

    enabled: bool = Field(default=True, description="是否启用结构化数据抽取")
    graph_dir: str = Field(default="data/graph", description="PropertyGraphIndex 存储目录")
    sqlite_path: str = Field(default="data/cement.db", description="SQLite 数据库路径")

    model_config = {"env_prefix": "GRAPH_"}


class LangSmithConfig(BaseSettings):
    """LangSmith 可观测性配置"""

    tracing: bool = Field(default=False, description="是否启用 LangSmith tracing")
    api_key: str = Field(default="", description="LangSmith API Key (lsv2_pt_...)")
    project: str = Field(default="deepcement", description="LangSmith 项目名称")
    endpoint: str = Field(
        default="https://api.smith.langchain.com", description="LangSmith API 地址"
    )

    model_config = {"env_prefix": "LANGSMITH_"}


class MilvusConfig(BaseSettings):
    """Milvus/Zilliz 向量数据库配置"""

    uri: str = Field(default="", description="Zilliz Cloud endpoint URL")
    token: str = Field(default="", description="Zilliz token (username:password)")
    collection_name: str = Field(default="cement_docs", description="集合名称")
    dim: int = Field(default=1024, description="向量维度（bge-large-zh = 1024）")

    model_config = {"env_prefix": "MILVUS_"}


class PathConfig(BaseSettings):
    """路径配置"""

    data_raw_dir: str = Field(default="data/raw", description="原始数据目录")
    data_index_dir: str = Field(default="data/index", description="索引存储目录")
    report_output_dir: str = Field(default="agent/report/output", description="报告输出目录")

    model_config = {"env_prefix": "DATA_"}


class Settings:
    """全局配置聚合"""

    def __init__(self):
        self.llm = LLMConfig()
        self.embed = EmbedConfig()
        self.retriever = RetrieverConfig()
        self.reranker = RerankerConfig()
        self.ocr = OCRConfig()
        self.graph = GraphConfig()
        self.paths = PathConfig()
        self.milvus = MilvusConfig()
        self.langsmith = LangSmithConfig()

    @property
    def raw_dir(self) -> Path:
        return BASE_DIR / self.paths.data_raw_dir

    @property
    def index_dir(self) -> Path:
        return BASE_DIR / self.paths.data_index_dir

    @property
    def report_dir(self) -> Path:
        return BASE_DIR / self.paths.report_output_dir


# 全局单例
settings = Settings()
