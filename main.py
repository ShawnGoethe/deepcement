"""
DeepCement - 固井质量评测报告系统
FastAPI 服务入口

启动方式：
    fastapi dev main.py               # 开发模式（自动重载）
    fastapi run main.py               # 生产模式
    python main.py                    # 通过 uvicorn 直接启动
"""

import uvicorn

from api.app import app  # noqa: F401 — fastapi dev 需要顶层 app 对象


def main():
    """通过 uvicorn 启动服务"""
    import argparse

    parser = argparse.ArgumentParser(description="DeepCement 固井质量评测 API 服务")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8000, help="端口")
    args = parser.parse_args()

    uvicorn.run("api.app:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
