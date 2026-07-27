"""
检索引擎模块
基于 LlamaIndex 进行语义检索，支持按井名/井段/时间过滤
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

from loguru import logger

from config import settings
from core.indexer import IndexManager


@dataclass
class RetrievalResult:
    """检索结果"""
    content: str                          # 文本内容
    score: float = 0.0                    # 相似度分数
    metadata: dict = field(default_factory=dict)  # 元数据
    source: str = ""                      # 来源


class HistoryRetriever:
    """固井历史资料检索器

    支持：
    - 语义检索：自然语言查询
    - 元数据过滤：井名、井段、时间段
    - 结构化查询：按字段精确匹配
    """

    def __init__(self, index_manager: IndexManager):
        self.index_manager = index_manager

    def _ensure_ready(self):
        """确保索引已就绪"""
        if not self.index_manager.is_ready:
            raise RuntimeError("索引未就绪，请先构建或加载索引")

    def _get_retriever(self, top_k: int = 5):
        """获取 LlamaIndex retriever（每次根据 top_k 创建新实例）"""
        self._ensure_ready()
        return self.index_manager.index.as_retriever(
            similarity_top_k=top_k,
        )

    def _get_query_engine(self, top_k: int = 5):
        """获取 LlamaIndex query engine（每次根据 top_k 创建新实例）"""
        self._ensure_ready()
        return self.index_manager.index.as_query_engine(
            similarity_top_k=top_k,
        )

    def search(
        self,
        query: str,
        top_k: int = settings.retriever.top_k,
        well_name: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """语义检索固井历史资料

        Args:
            query: 查询文本（自然语言）
            top_k: 返回结果数量（默认从配置读取）
            well_name: 过滤井名（可选）
            date_from: 起始日期 YYYY-MM-DD（可选）
            date_to: 结束日期 YYYY-MM-DD（可选）

        Returns:
            检索结果列表
        """
        retriever = self._get_retriever(top_k=top_k)

        # 构建查询（将过滤条件加入查询文本以提高召回）
        enhanced_query = query
        if well_name:
            enhanced_query = f"井名：{well_name}。{query}"

        try:
            nodes = retriever.retrieve(enhanced_query)
        except Exception as e:
            logger.error(f"检索失败: {e}")
            return []

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

            results.append(RetrievalResult(
                content=node.get_content(),
                score=getattr(node, "score", 0.0),
                metadata=meta,
                source=meta.get("filename", "unknown"),
            ))

        logger.info(f"检索完成: query='{query[:30]}...' → {len(results)} 条结果")
        return results

    def query(self, question: str, top_k: int = settings.retriever.top_k) -> str:
        """自然语言查询（由 LLM 生成回答）

        Args:
            question: 自然语言问题
            top_k: 检索结果数量（默认从配置读取）

        Returns:
            LLM 生成的回答文本
        """
        engine = self._get_query_engine(top_k=top_k)

        try:
            response = engine.query(question)
            return str(response)
        except Exception as e:
            logger.error(f"查询失败: {e}")
            return f"查询失败: {e}"

    def search_by_well(self, well_name: str, top_k: int = settings.retriever.top_k) -> List[RetrievalResult]:
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
