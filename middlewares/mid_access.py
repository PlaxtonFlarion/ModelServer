#     _                           __  __ _     _     _ _
#    / \   ___ ___ ___  ___ ___  |  \/  (_) __| | __| | | _____      ____ _ _ __ ___
#   / _ \ / __/ __/ _ \/ __/ __| | |\/| | |/ _` |/ _` | |/ _ \ \ /\ / / _` | '__/ _ \
#  / ___ \ (_| (_|  __/\__ \__ \ | |  | | | (_| | (_| | |  __/\ V  V / (_| | | |  __/
# /_/   \_\___\___\___||___/___/ |_|  |_|_|\__,_|\__,_|_|\___| \_/\_/ \__,_|_|  \___|
#

import time
import uuid
import typing
from loguru import logger
from fastapi import Request


async def access_middleware(
    request: Request,
    call_next: typing.Callable
) -> typing.Any:
    """
    访问日志中间件 (Access Middleware):
    - Trace-ID
    - 客户端 IP
    - 性能监控（精确）
    - 慢请求告警
    - Incoming / Outgoing 日志
    """

    # ---- Trace ID ----
    trace_id = str(uuid.uuid4())
    request.state.trace_id = trace_id

    # ---- 客户端真实 IP ----
    client_ip = (
        request.headers.get("CF-Connecting-IP") or
        request.headers.get("X-Real-IP") or
        request.headers.get("X-Forwarded-For") or
        (request.client.host if request.client else "unknown")
    )

    method = request.method
    path   = request.url.path

    # ---- 请求日志 ----
    logger.info(
        f"[{trace_id}] → Incoming {method} {path} (from={client_ip})"
    )

    # ---- 性能计时 ----
    start = time.perf_counter()

    response = await call_next(request)

    # ---- 耗时计算 ----
    cost_ms = round((time.perf_counter() - start) * 1000, 2)

    # ---- 响应头 ----
    response.headers["X-Trace-ID"] = trace_id
    response.headers["X-Process-Time"] = str(cost_ms)

    # ---- 慢请求警告 ----
    if cost_ms > 300:
        logger.warning(
            f"[{trace_id}] Slow request: {cost_ms}ms {method} {path}"
        )

    # ---- 响应日志 ----
    logger.info(
        f"[{trace_id}] ← Outgoing {response.status_code} {path} ({cost_ms}ms)"
    )

    return response


if __name__ == '__main__':
    pass
