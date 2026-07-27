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
        self.paths = PathConfig()
        self.milvus = MilvusConfig()

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
