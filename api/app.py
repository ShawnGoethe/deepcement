"""
FastAPI 应用定义
"""

import os
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from api.dependencies import init_agent
from api.routes import router
from core.tracing import setup_tracing


# ─── 生命周期 ───────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[API] DeepCement API 启动")
    setup_tracing()
    init_agent()
    yield
    logger.info("[API] DeepCement API 关闭")


# ─── 应用实例 ───────────────────────────────────────────
app = FastAPI(
    title="DeepCement API",
    description="固井质量评测系统 API — 提供质量评测、历史检索、数据对比等功能",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ─── CORS ───────────────────────────────────────────────
cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── 请求日志中间件 ─────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = time.time() - start
    logger.info(
        f"[API] {request.method} {request.url.path} -> {response.status_code} ({elapsed:.3f}s)"
    )
    return response


# ─── 全局异常处理 ───────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(exc: Exception):
    logger.error(f"[API] 未捕获异常: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"服务器内部错误: {type(exc).__name__}"},
    )


# ─── 注册路由 ───────────────────────────────────────────
app.include_router(router)
