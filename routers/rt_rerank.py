#  ____                      _      ____             _
# |  _ \ ___ _ __ __ _ _ __ | | __ |  _ \ ___  _   _| |_ ___ _ __
# | |_) / _ \ '__/ _` | '_ \| |/ / | |_) / _ \| | | | __/ _ \ '__|
# |  _ <  __/ | | (_| | | | |   <  |  _ < (_) | |_| | ||  __/ |
# |_| \_\___|_|  \__,_|_| |_|_|\_\ |_| \_\___/ \__,_|\__\___|_|
#

import modal
from loguru import logger
from fastapi import (
    APIRouter, Request
)
from schemas.cognitive import RerankResponse
from schemas.errors import BizError
from utils import const

rerank_router = APIRouter(tags=["Rerank"])


@rerank_router.post(
    path="/rerank",
    response_model=RerankResponse,
    operation_id="api_rerank"
)
async def api_rerank(request: Request) -> RerankResponse:
    logger.info(f"**> {request.method} {request.url}")

    f = modal.Cls.from_name(app_name=const.GROUP_FUNC, name="CrossENC")

    try:
        body      = await request.json()
        query     = body.get("query")
        candidate = body.get("candidate")

        if not query or not isinstance(candidate, list) or not candidate:
            raise BizError(
                status_code=400, detail="query and candidate (list) are required"
            )

        resp = await f().rerank.remote.aio(
            query, candidate
        )
        return RerankResponse(**resp)

    finally:
        logger.info(f"**> {('=' * 12)}")


if __name__ == '__main__':
    pass
