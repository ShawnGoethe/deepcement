"""
API 请求/响应数据模型
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# ─── 请求模型 ───────────────────────────────────────────────


class EvaluateRequest(BaseModel):
    """固井质量评测请求"""

    well_name: str = Field(..., description="井名", examples=["XX-1"])
    query: Optional[str] = Field(None, description="额外查询条件")


class SearchRequest(BaseModel):
    """历史资料检索请求"""

    query: str = Field(..., description="查询内容", examples=["水泥浆密度"])
    well_name: Optional[str] = Field(None, description="井名过滤")
    top_k: int = Field(5, ge=1, le=50, description="返回结果数量")


class CompareRequest(BaseModel):
    """井数据对比请求"""

    well_name: str = Field(..., description="当前井名")
    target_well: str = Field(..., description="对比井名")


class ChatRequest(BaseModel):
    """Agent 对话请求（转发给 CementAgent.run）"""

    query: str = Field(..., description="用户查询", examples=["威202H16-6水泥密度？"])


class LasEvaluateRequest(BaseModel):
    """LAS 测井数据评测请求"""

    las_file: str = Field(..., description="LAS 文件路径或文件名", examples=["well_01.las"])
    well_name: Optional[str] = Field(None, description="井名（默认从文件解析）")


# ─── 响应模型 ───────────────────────────────────────────────


class DimensionResponse(BaseModel):
    """单维度评测结果"""

    name: str
    score: float
    grade: str
    details: str = ""
    issues: List[str] = []


class EvaluateResponse(BaseModel):
    """固井质量评测响应"""

    well_name: str
    overall_score: float
    overall_grade: str
    dimensions: List[DimensionResponse]
    conclusion: str
    suggestions: List[str]


class SearchResultItem(BaseModel):
    """单条检索结果"""

    source: str
    content: str
    metadata: Dict[str, Any] = {}


class SearchResponse(BaseModel):
    """历史资料检索响应"""

    results: List[SearchResultItem]
    total: int


class CompareResponse(BaseModel):
    """井数据对比响应"""

    well_a: str
    well_b: str
    data_a: List[SearchResultItem]
    data_b: List[SearchResultItem]


class ChatResponse(BaseModel):
    """Agent 对话响应"""

    reply: str


class BuildIndexResponse(BaseModel):
    """索引构建响应"""

    success: bool
    message: str


class HealthResponse(BaseModel):
    """健康检查响应"""

    status: str = "ok"
    index_ready: bool
    version: str = "0.1.0"
