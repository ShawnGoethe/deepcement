"""
索引管理模块
基于 LlamaIndex + Milvus (Zilliz Cloud) 构建和管理向量索引
"""

from typing import List, Optional

from loguru import logger

from config import settings
from core.ingester import CementDocument


class IndexManager:
    """LlamaIndex 向量索引管理器（Zilliz Cloud 后端）

    负责：
    - 将 CementDocument 转为 LlamaIndex Document
    - 构建 VectorStoreIndex（存储到 Zilliz Cloud）
    - 连接已有集合 / 增量更新
    """

    def __init__(self):
        self._index = None
        self._llm = None
        self._embed_model = None
        self._vector_store = None

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

        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from pathlib import Path

        model_path = str(Path(__file__).parent.parent / "models" / "bge-large-zh")
        self._embed_model = HuggingFaceEmbedding(model_path)
        logger.info(f"Embedding 初始化完成: {model_path}")

    def _init_vector_store(self):
        """初始化 Milvus 向量存储（延迟加载）"""
        if self._vector_store is not None:
            return

        from llama_index.vector_stores.milvus import MilvusVectorStore

        self._vector_store = MilvusVectorStore(
            uri=settings.milvus.uri,
            token=settings.milvus.token,
            collection_name=settings.milvus.collection_name,
            dim=settings.milvus.dim,
        )
        logger.info(
            f"Milvus 连接成功: collection={settings.milvus.collection_name}"
        )

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
        """构建向量索引（写入 Zilliz Cloud）

        Args:
            documents: 解析后的固井文档列表
        """
        from llama_index.core import Settings, VectorStoreIndex

        self._init_llm()
        self._init_embed()
        self._init_vector_store()

        Settings.llm = self._llm
        Settings.embed_model = self._embed_model

        llama_docs = self._to_llama_documents(documents)
        if not llama_docs:
            logger.warning("没有文档可索引")
            return

        logger.info(f"开始构建索引，共 {len(llama_docs)} 个文档...")
        from llama_index.core import StorageContext

        storage_context = StorageContext.from_defaults(
            vector_store=self._vector_store,
        )
        self._index = VectorStoreIndex.from_documents(
            llama_docs,
            storage_context=storage_context,
            show_progress=True,
        )
        logger.info("索引构建完成（已写入 Zilliz Cloud）")

    def connect(self) -> bool:
        """连接已有的 Zilliz 集合并加载索引

        Returns:
            是否连接成功
        """
        from llama_index.core import Settings, VectorStoreIndex

        try:
            self._init_llm()
            self._init_embed()
            self._init_vector_store()

            Settings.llm = self._llm
            Settings.embed_model = self._embed_model

            self._index = VectorStoreIndex.from_vector_store(
                self._vector_store,
            )
            logger.info(
                f"已连接 Zilliz 集合: {settings.milvus.collection_name}"
            )
            return True
        except Exception as e:
            logger.error(f"连接 Zilliz 失败: {e}")
            return False

    def add_documents(self, documents: List[CementDocument]):
        """增量添加文档到已有索引"""
        from llama_index.core import Settings

        if self._index is None:
            logger.warning("索引未连接，将构建新索引")
            self.build_index(documents)
            return

        self._init_llm()
        self._init_embed()
        Settings.llm = self._llm
        Settings.embed_model = self._embed_model

        llama_docs = self._to_llama_documents(documents)
        for doc in llama_docs:
            self._index.insert(doc)
        logger.info(f"增量添加 {len(llama_docs)} 个文档到 Zilliz")

    @property
    def index(self):
        return self._index

    @property
    def is_ready(self) -> bool:
        return self._index is not None
