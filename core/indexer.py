"""
索引管理模块
基于 LlamaIndex 构建和管理向量索引
"""

from pathlib import Path
from typing import List, Optional

from loguru import logger

from config import settings
from core.ingester import CementDocument


class IndexManager:
    """LlamaIndex 向量索引管理器

    负责：
    - 将 CementDocument 转为 LlamaIndex Document
    - 构建 VectorStoreIndex
    - 持久化 / 加载索引
    - 增量更新
    """

    def __init__(self):
        self._index = None
        self._llm = None
        self._embed_model = None

    def _init_llm(self):
        """初始化 LLM（延迟加载）"""
        if self._llm is not None:
            return

        from llama_index.llms.openai_like import OpenAILike

        self._llm = OpenAILike(
            model=settings.llm.model,
            api_base=settings.llm.base_url,
            api_key=settings.llm.api_key,
            temperature=settings.llm.temperature,
            max_tokens=settings.llm.max_tokens,
        )
        logger.info(f"LLM 初始化完成: {settings.llm.model}")

    def _init_embed(self):
        """初始化 Embedding 模型（延迟加载）"""
        if self._embed_model is not None:
            return

        # from FlagEmbedding import FlagModel
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        model_path = str(Path(__file__).parent.parent / "models" / "bge-large-zh")
        self._embed_model = HuggingFaceEmbedding(model_path)
        logger.info(f"Embedding 初始化完成: {model_path}")

    def _to_llama_documents(self, cement_docs: List[CementDocument]) -> list:
        """将 CementDocument 转为 LlamaIndex Document"""
        from llama_index.core import Document

        llama_docs = []
        for doc in cement_docs:
            llama_doc = Document(
                text=doc.content,
                metadata=doc.metadata,
                metadata_seperator="\n",
            )
            llama_docs.append(llama_doc)
        return llama_docs

    def build_index(self, documents: List[CementDocument]):
        """构建向量索引

        Args:
            documents: 解析后的固井文档列表
        """
        from llama_index.core import Settings, VectorStoreIndex

        self._init_llm()
        self._init_embed()

        # 配置全局设置
        Settings.llm = self._llm
        Settings.embed_model = self._embed_model

        llama_docs = self._to_llama_documents(documents)
        if not llama_docs:
            logger.warning("没有文档可索引")
            return

        logger.info(f"开始构建索引，共 {len(llama_docs)} 个文档...")
        self._index = VectorStoreIndex.from_documents(llama_docs, show_progress=True)
        logger.info("索引构建完成")

    def save_index(self, index_dir: Optional[Path] = None):
        """持久化索引到磁盘"""
        if self._index is None:
            logger.warning("索引未构建，无法保存")
            return

        save_dir = Path(index_dir) if index_dir else settings.index_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        self._index.storage_context.persist(persist_dir=str(save_dir))
        logger.info(f"索引已保存到: {save_dir}")

    def load_index(self, index_dir: Optional[Path] = None) -> bool:
        """从磁盘加载索引

        Returns:
            是否加载成功
        """
        from llama_index.core import StorageContext, load_index_from_storage

        load_dir = Path(index_dir) if index_dir else settings.index_dir
        if not load_dir.exists():
            logger.warning(f"索引目录不存在: {load_dir}")
            return False

        try:
            self._init_llm()
            self._init_embed()

            from llama_index.core import Settings

            Settings.llm = self._llm
            Settings.embed_model = self._embed_model

            storage_context = StorageContext.from_defaults(persist_dir=str(load_dir))
            self._index = load_index_from_storage(storage_context)
            logger.info(f"索引加载成功: {load_dir}")
            return True
        except Exception as e:
            logger.error(f"索引加载失败: {e}")
            return False

    def add_documents(self, documents: List[CementDocument]):
        """增量添加文档到已有索引"""
        from llama_index.core import Settings

        if self._index is None:
            logger.warning("索引未加载，将构建新索引")
            self.build_index(documents)
            return

        self._init_llm()
        self._init_embed()
        Settings.llm = self._llm
        Settings.embed_model = self._embed_model

        llama_docs = self._to_llama_documents(documents)
        for doc in llama_docs:
            self._index.insert(doc)
        logger.info(f"增量添加 {len(llama_docs)} 个文档")

    @property
    def index(self):
        return self._index

    @property
    def is_ready(self) -> bool:
        return self._index is not None
