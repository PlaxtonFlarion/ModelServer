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
    scaledown_window=300
)
class EmbeddingService(object):

    embedder: typing.Optional["SentenceTransformer"] = None
    reranker: typing.Optional["CrossEncoder"]        = None

    @modal.enter()
    def startup(self) -> None:
        logger.info("BGE embedding model loading ...")
        self.embedder = SentenceTransformer("/root/models/bge_base")
        logger.info("🔥 BGE embedding model loaded")

        logger.info("Cross Encoder model loading ...")
        self.reranker = CrossEncoder("/root/models/cross_encoder")
        logger.info("🔥 Cross Encoder model loaded")

    @modal.fastapi_endpoint(method="POST")
    @exception_middleware
    @auth_middleware("X-Token")
    async def tensor(self, request: "Request") -> "JSONResponse":
        logger.info(f"Request: {request.method} {request.url}")

        body = await request.json()
        text = [body["text"]] if "text" in body else body.get("texts", [])

        if not text or not isinstance(text, list):
            return JSONResponse(content={"error": "text or texts required"}, status_code=400)

        # 🔥 Notes: 批量 embedding（GPU/CPU向量化）
        embeds = self.embedder.encode(
            text, batch_size=16, convert_to_numpy=True
        )
        for embed in embeds:
            logger.info(f"Embed: {embed}")

        # 🔥 Notes: 归一化 → 更适合向量检索
        embeds = embeds / (numpy.linalg.norm(embeds, axis=1, keepdims=True) + 1e-8)
        logger.info(f"Normalization: {embeds}")

        resp_body = {
            "vectors" : embeds.astype("float32").tolist(),
            "count"   : len(embeds),
            "dim"     : embeds.shape[1],
            "model"   : "bge-base-en-v1.5"
        }
        logger.info(f"Response body: {resp_body}")

        return JSONResponse(content=resp_body, status_code=200)

    @modal.fastapi_endpoint(method="POST")
    @exception_middleware
    @auth_middleware("X-Token")
    async def rerank(self, request: "Request") -> "JSONResponse":
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
    # Notes: ==== https://modal.com/ ====
    # modal run apps/embedding.py
    # modal deploy apps/embedding.py
    pass
