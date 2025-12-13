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
from schemas.cognitive import (
    TensorRequest, TensorResponse
)
from schemas.errors import BizError
from utils import const

embedding_router = APIRouter(tags=["Embedding"])


@embedding_router.post(
    path="/tensor",
    response_model=TensorResponse,
    operation_id="api_tensor"
)
async def api_tensor_en(
    request: Request,
    payload: TensorRequest
) -> TensorResponse:
    logger.info(f"**> {request.method} {request.url}")

    f = modal.Cls.from_name(app_name=const.GROUP_FUNC, name="Embedding")

    try:
        query    = payload.query
        elements = payload.elements
        s        = payload.s
        k        = payload.k
        mesh     = ([query] if query else []) + (elements or [])

        if not mesh: raise BizError(
            status_code=400, detail="query and elements required"
        )

        resp = await f().tensor.remote.aio(query, elements, mesh, s, k)
        return TensorResponse(**resp)

    finally:
        logger.info(f"**> {('=' * 12)}")


if __name__ == '__main__':
    pass
