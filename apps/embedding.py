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
from sentence_transformers import (
    SentenceTransformer, CrossEncoder
)
from middlewares.auth import auth_middleware
from middlewares.exception import exception_middleware
from schemas.cognitive import (
    TensorResponse, RerankResponse
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
    ".", "/root", ignore=["**/.venv", "**/venv"]
)
secret = modal.Secret.from_name("SHARED_SECRET")


@app.cls(
    image=image,
    secrets=[secret],
    memory=4096,
    max_containers=2,
    scaledown_window=300,
    concurrency_limit=5
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
    def enc_character(self, origin: list[str]) -> numpy.ndarray:
        if not origin: return numpy.empty((0, 768))

        try:
            logger.info(f"========== Encode Begin ==========")

            # 🔥 批量 embedding（GPU/CPU向量化）
            embeds = self.embedder.encode(
                origin, batch_size=16, convert_to_numpy=True
            )

            # 🔥 归一化 → 更适合向量检索
            embeds = embeds / (numpy.linalg.norm(embeds, axis=1, keepdims=True) + 1e-8)
            logger.info(f"Normalization: {embeds}")

            for index, embed in enumerate(embeds, start=1):
                logger.info(f"Embed-{index:04}: {embed.shape}")

            return embeds

        except Exception as e:
            logger.error(e); return numpy.empty((0, 768))

        finally:
            logger.info(f"========== Encode Final ==========")

    @modal.fastapi_endpoint(method="POST")
    @exception_middleware
    @auth_middleware("X-Token")
    async def tensor(self, request: Request) -> TensorResponse:
        logger.info(f"Request: {request.method} {request.url.path}")

        body     = await request.json()
        query    = body.get("query")
        elements = body.get("elements")

        logger.info(f"✦ 1) 输入文本融合 (保证顺序 Query → Elements)")
        mesh = ([query] if query else []) + (elements or [])
        if not mesh: return TensorResponse(
            query=query,
            query_vec=numpy.array([]).tolist(),
            elements=elements,
            page_vectors=numpy.array([]).tolist(),
            count=0,
            dim=0,
            model="",
            error="query and elements required"
        )

        logger.info("✦ 2) 调用嵌入")
        embeds = self.enc_character.remote_async(mesh)
        embeds = numpy.asarray(embeds, dtype="float32")

        logger.info(f"✦ 3) 拆分恢复结构")
        query_vec    = embeds[0] if query else None
        page_vectors = embeds[1:] if elements else None

        logger.info(f"✦ 4) 统计")
        count = len(mesh)
        dim   = embeds.shape[-1] if mesh else 0

        logger.info(f"✦ 5) 下发结果 TensorResponse")
        return TensorResponse(
            query=query,
            query_vec=query_vec.tolist(),
            elements=elements,
            page_vectors=page_vectors.tolist(),
            count=count,
            dim=dim,
            model="bge-base-en-v1.5",
        )

    @modal.fastapi_endpoint(method="POST")
    @exception_middleware
    @auth_middleware("X-Token")
    async def rerank(self, request: Request) -> RerankResponse:
        logger.info(f"Request: {request.method} {request.url.path}")

        body      = await request.json()
        query     = body.get("query")
        candidate = body.get("candidate")

        if not query or not isinstance(candidate, list) or not candidate:
            return RerankResponse(
                scores=None,
                count=None,
                error="query and candidate (list) are required"
            )

        logger.info(f"Rerank 计算逻辑")
        pairs = [[query, t] for t in candidate]
        rerank_scores = self.reranker.predict(pairs)
        scores = [float(s) for s in rerank_scores]
        logger.info(f"Rerank 最终得分 {scores}")

        logger.info(f"Rerank 下发结果 RerankResponse")
        return RerankResponse(
            scores=scores,
            count=len(scores)
        )


if __name__ == '__main__':
    pass
