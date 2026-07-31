"""
API 路由定义
"""

from fastapi import APIRouter, HTTPException

from api.dependencies import ensure_index_loaded, get_agent
from api.schemas import (
    BuildIndexResponse,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
)
from pipeline.builder import build_index as pipeline_build_index

router = APIRouter()


# ─── 健康检查 ───────────────────────────────────────────────


@router.get("/v1/health", response_model=HealthResponse, tags=["系统"])
def health_check():
    """健康检查 + 索引状态"""
    try:
        agent = get_agent()
        return HealthResponse(
            status="ok",
            index_ready=agent.indexer.is_ready,
        )
    except Exception as e:
        return HealthResponse(
            status=f"error: {e}",
            index_ready=False,
        )


# ─── 索引管理 ───────────────────────────────────────────────


@router.post("/v1/index/build", response_model=BuildIndexResponse, tags=["索引"])
def build_index_route():
    """构建向量索引（从 data/raw/ 读取文档，抽取结构化数据 + 知识图谱）"""
    success = pipeline_build_index()
    if success:
        return BuildIndexResponse(success=True, message="索引构建完成")
    raise HTTPException(status_code=500, detail="索引构建失败，请检查 data/raw/ 目录")


# ─── 历史检索 ───────────────────────────────────────────────


@router.post("/search", response_model=SearchResponse, tags=["检索"])
def search_history(req: SearchRequest):
    """语义检索固井历史资料"""
    agent = get_agent()
    if not ensure_index_loaded(agent):
        raise HTTPException(status_code=503, detail="索引未就绪，请先调用 /index/build")

    results = agent.retriever.search(
        query=req.query,
        top_k=req.top_k,
        well_name=req.well_name,
    )

    return SearchResponse(
        results=[
            SearchResultItem(
                source=r.source,
                content=r.content,
                metadata=r.metadata,
            )
            for r in results
        ],
        total=len(results),
    )


# ─── Agent 对话 ─────────────────────────────────────────────


@router.post("/chat", response_model=ChatResponse, tags=["对话"])
def chat(req: ChatRequest):
    """Agent 对话接口（转发给 CementAgent）"""
    agent = get_agent()

    try:
        reply = agent.run(req.query)
        return ChatResponse(reply=reply)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent 执行失败: {e}")
