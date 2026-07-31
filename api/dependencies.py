"""
FastAPI 依赖注入
提供 CementAgent 单例和索引加载
"""

from loguru import logger

from agent.orchestrator import CementAgent

_agent: CementAgent | None = None


def init_agent() -> CementAgent:
    """初始化 CementAgent、加载向量索引和 Reranker（应用启动时调用）

    Returns:
        CementAgent 实例
    """
    global _agent
    logger.info("[API] 正在初始化 CementAgent...")
    _agent = CementAgent()

    logger.info("[API] 正在加载向量索引...")
    if _agent.load_index():
        logger.info("[API] 向量索引加载成功")
    else:
        logger.warning("[API] 向量索引加载失败，请稍后通过 /index/build 构建")

    logger.info("[API] 正在加载 Reranker 模型...")
    _agent.retriever.init_reranker()

    logger.info("[API] CementAgent 初始化完成")
    return _agent


def get_agent() -> CementAgent:
    """获取 CementAgent 单例

    Returns:
        CementAgent 实例

    Raises:
        RuntimeError: Agent 尚未初始化
    """
    if _agent is None:
        raise RuntimeError("CementAgent 未初始化，服务启动可能失败")
    return _agent


def ensure_index_loaded(agent: CementAgent) -> bool:
    """确保向量索引已加载

    Args:
        agent: CementAgent 实例

    Returns:
        索引是否就绪
    """
    if agent.indexer.is_ready:
        return True

    logger.info("[API] 索引未就绪，尝试连接 Zilliz Cloud...")
    return agent.load_index()
