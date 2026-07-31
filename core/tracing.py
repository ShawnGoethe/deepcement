"""
LangSmith 可观测性模块
初始化 tracing 环境变量，提供 @traceable 装饰器复用
"""

import os

from loguru import logger

from config import settings


def setup_tracing() -> bool:
    """根据配置初始化 LangSmith tracing 环境变量

    在应用启动时调用一次。LangChain 的 ChatOpenAI 会自动读取这些变量，
    非 LangChain 代码通过 @traceable 装饰器接入。

    Returns:
        tracing 是否成功启用
    """
    cfg = settings.langsmith

    if not cfg.tracing:
        logger.info("[tracing] LangSmith tracing 未启用（LANGSMITH_TRACING=false）")
        return False

    if not cfg.api_key:
        logger.warning("[tracing] LANGSMITH_API_KEY 未配置，tracing 跳过")
        return False

    # 设置 LangSmith 所需的环境变量
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = cfg.api_key
    os.environ["LANGSMITH_PROJECT"] = cfg.project
    if cfg.endpoint:
        os.environ["LANGSMITH_ENDPOINT"] = cfg.endpoint

    # LangChain 自动追踪也需要这些变量
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = cfg.api_key
    os.environ["LANGCHAIN_PROJECT"] = cfg.project
    if cfg.endpoint:
        os.environ["LANGCHAIN_ENDPOINT"] = cfg.endpoint

    logger.info(f"[tracing] LangSmith tracing 已启用 → project={cfg.project}")
    return True


# 便捷复用：其他模块 from core.tracing import traceable
try:
    from langsmith import traceable  # noqa: F401
except ImportError:
    # langsmith 未安装时提供无操作的装饰器占位
    def traceable(func=None, **kwargs):  # type: ignore[misc]
        """占位装饰器 — langsmith 未安装时原样透传"""

        def decorator(fn):
            return fn

        if func is not None:
            return decorator(func)
        return decorator
