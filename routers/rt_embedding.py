#  _____           _              _     _ _               ____             _
# | ____|_ __ ___ | |__   ___  __| | __| (_)_ __   __ _  |  _ \ ___  _   _| |_ ___ _ __
# |  _| | '_ ` _ \| '_ \ / _ \/ _` |/ _` | | '_ \ / _` | | |_) / _ \| | | | __/ _ \ '__|
# | |___| | | | | | |_) |  __/ (_| | (_| | | | | | (_| | |  _ < (_) | |_| | ||  __/ |
# |_____|_| |_| |_|_.__/ \___|\__,_|\__,_|_|_| |_|\__, | |_| \_\___/ \__,_|\__\___|_|
#                                                 |___/
#

import modal
from loguru import logger
from fastapi import (
    APIRouter, Request
)
from schemas.cognitive import TensorResponse
from schemas.errors import BizError
from utils import const

embedding_router = APIRouter(tags=["Embedding"])


@embedding_router.post(
    path="/tensor/en",
    response_model=TensorResponse,
    operation_id="api_tensor_en"
)
async def api_tensor_en(request: Request) -> TensorResponse:
    logger.info(f"<Tensor Begin>")
    logger.info(f"Request: {request.method} {request.url}")

    f = modal.Cls.from_name(app_name=const.GROUP_FUNC, name="EmbeddingEN")

    try:
        body     = await request.json()
        query    = body.get("query")
        elements = body.get("elements")
        mesh     = ([query] if query else []) + (elements or [])

        if not mesh: raise BizError(
            status_code=400, detail="query and elements required"
        )

        resp = await f().tensor.remote.aio(query, elements, mesh)
        return TensorResponse(**resp)

    finally:
        logger.info(f"<Tensor Final>")


@embedding_router.post(
    path="/tensor/zh",
    response_model=TensorResponse,
    operation_id="api_tensor_zh"
)
async def api_tensor_zh(request: Request) -> TensorResponse:
    logger.info(f"<Tensor Begin>")
    logger.info(f"Request: {request.method} {request.url}")

    f = modal.Cls.from_name(app_name=const.GROUP_FUNC, name="EmbeddingZH")

    try:
        body     = await request.json()
        query    = body.get("query")
        elements = body.get("elements")
        mesh     = ([query] if query else []) + (elements or [])

        if not mesh: raise BizError(
            status_code=400, detail="query and elements required"
        )

        resp = await f().remote.aio(query, elements, mesh)
        return TensorResponse(**resp)

    finally:
        logger.info(f"<Tensor Final>")


if __name__ == '__main__':
    pass
