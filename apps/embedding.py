#  _____           _              _     _ _
# | ____|_ __ ___ | |__   ___  __| | __| (_)_ __   __ _
# |  _| | '_ ` _ \| '_ \ / _ \/ _` |/ _` | | '_ \ / _` |
# | |___| | | | | | |_) |  __/ (_| | (_| | | | | | (_| |
# |_____|_| |_| |_|_.__/ \___|\__,_|\__,_|_|_| |_|\__, |
#                                                 |___/
#

import modal
import numpy
import typing
from loguru import logger
from fastapi import Request
from fastapi.responses import JSONResponse
from sentence_transformers import (
    SentenceTransformer, CrossEncoder
)
from middlewares.auth import auth_middleware
from middlewares.exception import exception_middleware
from schemas.meta import (
    TensorRequest, TensorResponse
)
from utils import (
    const, toolset
)


app = modal.App("embedding")

toolset.init_logger()

image = modal.Image.debian_slim(
    "3.11"
).pip_install(
    const.EMBEDDING_DEPENDENCIES
).add_local_dir(
    "", "/root", ignore=["**/.venv", "**/venv"]
)
secret = modal.Secret.from_name("SHARED_SECRET")


@app.cls(
    image=image,
    secrets=[secret],
    memory=4096,
    max_containers=2,
    scaledown_window=300
)
class EmbeddingService(object):

    embedder: typing.Optional[SentenceTransformer] = None
    reranker: typing.Optional[CrossEncoder]        = None

    @modal.enter()
    def startup(self) -> None:
        logger.info("BGE embedding model loading ...")
        self.embedder = SentenceTransformer("/root/models/bge_base")
        logger.info("🔥 BGE embedding model loaded")

        logger.info("Cross Encoder model loading ...")
        self.reranker = CrossEncoder("/root/models/cross_encoder")
        logger.info("🔥 Cross Encoder model loaded")

    @modal.method(is_generator=False)
    async def encode(self, origin: typing.Union[str, list[str]]) -> numpy.ndarray:
        logger.info(f"======== Begin ========")
        if isinstance(origin, list): sentences, batch_size = origin, 16
        else: sentences, batch_size = [origin], 32

        # 🔥 workflow: 批量 embedding（GPU/CPU向量化）
        embeds = self.embedder.encode(
            sentences, batch_size=batch_size, convert_to_numpy=True
        )

        # 🔥 workflow: 归一化 → 更适合向量检索
        embeds = embeds / (numpy.linalg.norm(embeds, axis=1, keepdims=True) + 1e-8)
        logger.info(f"Normalization: {embeds}")

        for embed in embeds:
            logger.info(f"Embed: {embed}")

        logger.info(f"======== Final ========")
        return embeds

    @modal.fastapi_endpoint(method="POST")
    @exception_middleware
    @auth_middleware("X-Token")
    async def tensor(self, req: TensorRequest) -> TensorResponse | JSONResponse:
        logger.info(f"Request: {req.method} {req.url}")

        query    = req.query
        elements = req.elements

        # ✦ 1) 必须至少给一个，否则报错
        if not query and not elements:
            return JSONResponse(
                content={"error": "query or elements required"}, status_code=400
            )

        # ✦ 2) 向量化逻辑
        query_vec    = None
        page_vectors = None
        # ---- Encode Query ----
        if query: query_vec = self.encode.remote_gen(query)
        # ---- Encode Elements ----
        if elements: page_vectors = self.encode.remote_gen(elements)

        # ✦ 3) 动态 count/dim 统计
        count = (
            (1 if query_vec else 0) + (len(page_vectors) if page_vectors else 0)
        )
        dim = (
            len(query_vec.shape[1]) if query_vec else len(page_vectors.shape[1]) if page_vectors else 0
        )

        resp_body = TensorResponse(
            query=query,
            query_vec=query_vec.astype("float32").tolist(),
            elements=elements,
            page_vectors=page_vectors.astype("float32").tolist(),
            count=count,
            dim=dim,
            model="bge-base-en-v1.5"
        )

        logger.info(f"Response: {resp_body}")
        return resp_body

    @modal.fastapi_endpoint(method="POST")
    @exception_middleware
    @auth_middleware("X-Token")
    async def rerank(self, request: Request) -> JSONResponse:
        logger.info(f"Request: {request.method} {request.url}")

        body      = await request.json()
        query     = body.get("query")
        candidate = body.get("candidate")

        if not query or not isinstance(candidate, list) or not candidate:
            return JSONResponse(
                content={"error": "query and candidate (list) are required"}, status_code=400,
            )

        candidate_pairs = [[query, t] for t in candidate]
        rerank_scores   = self.reranker.predict(candidate_pairs)

        scores = [float(s) for s in rerank_scores]
        logger.info(f"Rerank scores: {scores}")

        resp_body = {"scores": scores, "count": len(scores)}
        logger.info(f"Response body: {resp_body}")

        return JSONResponse(
            content=resp_body, status_code=200
        )


if __name__ == '__main__':
    pass
