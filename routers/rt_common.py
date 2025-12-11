#   ____                                        ____             _
#  / ___|___  _ __ ___  _ __ ___   ___  _ __   |  _ \ ___  _   _| |_ ___ _ __
# | |   / _ \| '_ ` _ \| '_ ` _ \ / _ \| '_ \  | |_) / _ \| | | | __/ _ \ '__|
# | |__| (_) | | | | | | | | | | | (_) | | | | |  _ < (_) | |_| | ||  __/ |
#  \____\___/|_| |_| |_|_| |_| |_|\___/|_| |_| |_| \_\___/ \__,_|\__\___|_|
#

import time
import modal
import asyncio
from loguru import logger
from fastapi import (
    APIRouter, Request
)
from fastapi.responses import JSONResponse
from utils import const

common_router = APIRouter(tags=["Common"])


@common_router.get(
    path="/service",
    response_class=JSONResponse,
    operation_id="api_service"
)
async def api_service(request: Request) -> JSONResponse:
    logger.info(f"Request: {request.method} {request.url}")

    tasks = [
        "CrossENC", "EmbeddingEN", "EmbeddingZH", "InferenceColor", "InferenceFaint"
    ]

    for resp in await asyncio.gather(
        *(modal.Cls.from_name(app_name=const.GROUP_FUNC, name=name)()
          .heartbeat.remote_aio() for name in tasks)
    ):
        logger.info(f"HeartBeat: {resp}")

    content = {
        "status"    : "OK",
        "message"   : {},
        "timestamp" : int(time.time()),
    }

    logger.info(content)
    return JSONResponse(content)


if __name__ == '__main__':
    pass
