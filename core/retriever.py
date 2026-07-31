"""
检索引擎模块
基于 LlamaIndex 进行语义检索，支持按井名/井段/时间过滤
集成 BGE Reranker 进行重排序以提高检索精度
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from loguru import logger

from config import settings
from core.indexer import IndexManager
from core.tracing import traceable


@dataclass
class RetrievalResult:
    """检索结果"""

    content: str  # 文本内容
    score: float = 0.0  # 相似度分数
    metadata: dict = field(default_factory=dict)  # 元数据
    source: str = ""  # 来源


class HistoryRetriever:
    """固井历史资料检索器

    支持：
    - 语义检索：自然语言查询
    - 元数据过滤：井名、井段、时间段
    - 结构化查询：按字段精确匹配
    - BGE Reranker 重排序：提高检索精度
    """

    def __init__(self, index_manager: IndexManager):
        self.index_manager = index_manager
        self._reranker = None

    def _ensure_ready(self):
        """确保索引已就绪"""
        if not self.index_manager.is_ready:
            raise RuntimeError("索引未就绪，请先构建或加载索引")

    def init_reranker(self):
        """初始化 BGE Reranker

        可在启动时主动调用，也可延迟到首次检索时自动调用。
        """
        if self._reranker is not None:
            return

        if not settings.reranker.enabled:
            return

        try:
            from pathlib import Path

            from llama_index.core.postprocessor import SentenceTransformerRerank

            model_path = str(Path(__file__).parent.parent / settings.reranker.model_path)
            self._reranker = SentenceTransformerRerank(
                model=model_path,
                top_n=settings.reranker.top_n,
            )
            logger.info(f"Reranker 初始化完成: {model_path}, top_n={settings.reranker.top_n}")
        except ImportError:
            logger.warning(
                "sentence-transformers 未安装，Reranker 不可用。"
                "请运行: pip install sentence-transformers"
            )
        except Exception as e:
            logger.warning(f"Reranker 初始化失败: {e}，将使用原始排序")

    def _init_reranker(self):
        """内部调用，委托给 init_reranker"""
        self.init_reranker()

    def _get_retriever(self, top_k: int = 5):
        """获取 LlamaIndex retriever（每次根据 top_k 创建新实例）"""
        self._ensure_ready()
        return self.index_manager.index.as_retriever(
            similarity_top_k=top_k,
        )

    def _get_query_engine(self, top_k: int = 5):
        """获取 LlamaIndex query engine（带 Reranker 后处理器）"""
        self._ensure_ready()

        # 构建后处理器列表
        node_postprocessors = []
        self._init_reranker()
        if self._reranker is not None:
            node_postprocessors.append(self._reranker)

        return self.index_manager.index.as_query_engine(
            similarity_top_k=top_k,
            node_postprocessors=node_postprocessors,
        )

    def _rerank_nodes(self, query: str, nodes: list, top_n: int) -> list:
        """对检索结果进行重排序

        Args:
            query: 查询文本
            nodes: 初始检索结果（NodeWithScore 列表）
            top_n: 重排序后返回的结果数量

        Returns:
            重排序后的结果列表
        """
        self._init_reranker()
        if self._reranker is None:
            return nodes[:top_n]

        try:
            # 动态调整 top_n（Reranker 实例的 top_n 可能与请求不同）
            original_top_n = self._reranker.top_n
            self._reranker.top_n = top_n
            reranked = self._reranker.postprocess_nodes(nodes, query_str=query)
            self._reranker.top_n = original_top_n
            return reranked
        except Exception as e:
            logger.warning(f"重排序失败: {e}，使用原始排序")
            return nodes[:top_n]

    @traceable(name="HistoryRetriever.search")
    def search(
        self,
        query: str,
        top_k: int = settings.retriever.top_k,
        well_name: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """语义检索固井历史资料（带 Reranker 重排序）

        流程：向量检索（多取候选）→ Reranker 重排序 → 元数据过滤

        Args:
            query: 查询文本（自然语言）
            top_k: 返回结果数量（默认从配置读取）
            well_name: 过滤井名（可选）
            date_from: 起始日期 YYYY-MM-DD（可选）
            date_to: 结束日期 YYYY-MM-DD（可选）

        Returns:
            检索结果列表
        """
        # 计算初始检索数量：reranker 启用时多取候选
        self._init_reranker()
        if self._reranker is not None:
            candidate_k = top_k * settings.reranker.candidate_multiplier
        else:
            candidate_k = top_k

        retriever = self._get_retriever(top_k=candidate_k)

        # 构建查询（将过滤条件加入查询文本以提高召回）
        enhanced_query = query
        if well_name:
            enhanced_query = f"井名：{well_name}。{query}"

        try:
            nodes = retriever.retrieve(enhanced_query)
        except Exception as e:
            logger.error(f"检索失败: {e}")
            return []

        # 重排序
        if self._reranker is not None and len(nodes) > top_k:
            nodes = self._rerank_nodes(enhanced_query, nodes, top_n=top_k)
            logger.info(f"Reranker 重排序完成: {len(nodes)} 条结果")

        results = []
        for node in nodes:
            # 应用元数据过滤
            meta = node.metadata or {}

            if well_name and meta.get("well_name"):
                if well_name not in str(meta["well_name"]):
                    continue

            if date_from and meta.get("date"):
                if str(meta["date"]) < date_from:
                    continue

            if date_to and meta.get("date"):
                if str(meta["date"]) > date_to:
                    continue

            results.append(
                RetrievalResult(
                    content=node.get_content(),
                    score=getattr(node, "score", 0.0),
                    metadata=meta,
                    source=meta.get("filename", "unknown"),
                )
            )

        logger.info(f"检索完成: query='{query[:30]}...' → {len(results)} 条结果")
        return results

    # query search为不同的检索方式
    @traceable(name="HistoryRetriever.query")
    def query(self, question: str, top_k: int = settings.retriever.top_k) -> str:
        """自然语言查询（由 LLM 生成回答，带 Reranker 后处理）

        Args:
            question: 自然语言问题
            top_k: 检索结果数量（默认从配置读取）

        Returns:
            LLM 生成的回答文本
        """
        # query_engine 内部已集成 reranker
        engine = self._get_query_engine(top_k=top_k)

        try:
            response = engine.query(question)
            return str(response)
        except Exception as e:
            logger.error(f"查询失败: {e}")
            return f"查询失败: {e}"

    def search_by_well(
        self, well_name: str, top_k: int = settings.retriever.top_k
    ) -> List[RetrievalResult]:
        """按井名检索所有相关资料"""
        return self.search(
            query=f"{well_name} 固井 施工 质量",
            top_k=top_k,
            well_name=well_name,
        )

    def search_quality_issues(self, well_name: Optional[str] = None) -> List[RetrievalResult]:
        """检索质量问题相关资料"""
        query = "固井质量问题 漏失 窜槽 气侵 异常 事故"
        if well_name:
            query = f"{well_name} {query}"
        return self.search(query=query, well_name=well_name)
