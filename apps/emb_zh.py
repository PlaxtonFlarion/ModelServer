#  _____           _       ______   _
# | ____|_ __ ___ | |__   |__  / | | |
# |  _| | '_ ` _ \| '_ \    / /| |_| |
# | |___| | | | | | |_) |  / /_|  _  |
# |_____|_| |_| |_|_.__/  /____|_| |_|
#

import modal
import numpy
import typing
import asyncio
from loguru import logger
from fastapi import Request
from sentence_transformers import SentenceTransformer
from middlewares.mid_auth import auth_middleware
from middlewares.mid_exception import exception_middleware
from schemas.cognitive import TensorResponse
from schemas.errors import BizError
from images.emb_image import (
    image, secret, mounts
)
from utils import toolset

# Notes: https://huggingface.co/collections/BAAI/bge
# bge-base-zh-v1.5

app = modal.App("embedding-zh")
src = "/root/models/bge_base_zh"
dst = modal.Volume.from_name("bge-zh-cache")

toolset.init_logger()


@app.cls(
    image=image,
    secrets=[secret],
    mounts=mounts,
    volumes={src: dst},
    memory=4096,
    max_containers=5,
    scaledown_window=300
)
class EmbeddingZH(object):

    embedder: typing.Optional[SentenceTransformer] = None

    @modal.enter()
    def startup(self) -> None:
        logger.info("🔥 BGE embedding model loading ...")
        self.embedder = SentenceTransformer(src)
        logger.info("🔥 BGE embedding model loaded")

    @modal.fastapi_endpoint(method="POST")
    @exception_middleware
    @auth_middleware("X-Token")
    async def tensor_zh(self, request: Request) -> TensorResponse:
        logger.info(f"========== Tensor Begin ==========")
        logger.info(f"Request: {request.method} {request.url}")

        try:
            body = await request.json()
            query = body.get("query")
            elements = body.get("elements")
            mesh = ([query] if query else []) + (elements or [])

            if not mesh: raise BizError(
                status_code=400, detail="query and elements required"
            )

            logger.info("✦ 1) 调用嵌入")
            embeds = await asyncio.to_thread(
                self.embedder.encode, mesh, batch_size=16, convert_to_numpy=True
            )

            logger.info("✦ 2) 归一化 → 更适合向量检索")
            embeds = embeds / (numpy.linalg.norm(embeds, axis=1, keepdims=True) + 1e-8)
            logger.info(f"Normalization: {embeds}")

            for index, embed in enumerate(embeds, start=1):
                logger.info(f"Embed-{index:04}: {embed.shape}")
            embeds = numpy.asarray(embeds, dtype="float32")

            logger.info(f"✦ 3) 拆分恢复结构")
            query_vec = embeds[0] if query else None
            page_vectors = embeds[1:] if elements else None

            logger.info(f"✦ 4) 统计")
            count = len(mesh)
            dim = embeds.shape[-1] if mesh else 0

            logger.info(f"✦ 5) 下发结果 TensorResponse")
            return TensorResponse(
                query=query,
                query_vec=query_vec.tolist(),
                elements=elements,
                page_vectors=page_vectors.tolist(),
                count=count,
                dim=dim,
                model="bge-base-zh-v1.5",
            )

        finally:
            logger.info(f"========== Tensor Final ==========")


if __name__ == '__main__':
    pass
