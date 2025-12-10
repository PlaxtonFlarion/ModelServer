#   ____                     _____ _   _  ____
#  / ___|_ __ ___  ___ ___  | ____| \ | |/ ___|
# | |   | '__/ _ \/ __/ __| |  _| |  \| | |
# | |___| | | (_) \__ \__ \ | |___| |\  | |___
#  \____|_|  \___/|___/___/ |_____|_| \_|\____|
#

import modal
import typing
from loguru import logger
from fastapi import Request
from sentence_transformers import CrossEncoder
from middlewares.mid_auth import auth_middleware
from middlewares.mid_exception import exception_middleware
from schemas.cognitive import RerankResponse
from schemas.errors import BizError
from images.emb_image import (
    image, secret
)
from utils import toolset

# Notes: https://huggingface.co/cross-encoder
# ms-marco-MiniLM-L-12-v2

app = modal.App("cross-encoder")
src = "/root/models/cross_encoder"

toolset.init_logger()


@app.cls(
    image=image,
    secrets=[secret],
    memory=4096,
    max_containers=5,
    scaledown_window=300
)
class CrossENC(object):

    reranker: typing.Optional[CrossEncoder] = None

    @modal.enter()
    def startup(self) -> None:
        logger.info("🔥 Cross Encoder model loading ...")
        self.reranker = CrossEncoder(src)
        logger.info("🔥 Cross Encoder model loaded")

    @modal.fastapi_endpoint(method="POST")
    @exception_middleware
    @auth_middleware("X-Token")
    async def rerank(self, request: Request) -> RerankResponse:
        logger.info(f"========== Rerank Begin ==========")
        logger.info(f"Request: {request.method} {request.url}")

        try:
            body      = await request.json()
            query     = body.get("query")
            candidate = body.get("candidate")

            if not query or not isinstance(candidate, list) or not candidate:
                raise BizError(
                    status_code=400, detail="query and candidate (list) are required"
                )

            logger.info(f"Rerank 计算逻辑")
            pairs = [[query, t] for t in candidate]
            rerank_scores = self.reranker.predict(pairs)
            scores = [float(s) for s in rerank_scores]
            logger.info(f"Rerank 最终得分 {scores}")

            logger.info(f"Rerank 下发结果 RerankResponse")
            return RerankResponse(
                scores=scores, count=len(scores)
            )
        finally:
            logger.info(f"========== Rerank Final ==========")


if __name__ == '__main__':
    pass
