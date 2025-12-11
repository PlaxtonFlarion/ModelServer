#  _____           _              _   ______   _
# | ____|_ __ ___ | |__   ___  __| | |__  / | | |
# |  _| | '_ ` _ \| '_ \ / _ \/ _` |   / /| |_| |
# | |___| | | | | | |_) |  __/ (_| |  / /_|  _  |
# |_____|_| |_| |_|_.__/ \___|\__,_| /____|_| |_|
#

import modal
import numpy
import typing
import asyncio
from loguru import logger
from sentence_transformers import SentenceTransformer
from images.embed_image import (
    image, secret
)
from utils import toolset

# Notes: https://huggingface.co/collections/BAAI/bge
# bge-base-zh-v1.5

app = modal.App("embedding-zh")
src = "/root/models/bge_base_zh"

toolset.init_logger()


@app.cls(
    image=image,
    secrets=[secret],
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

    @modal.method()
    async def heartbeat(self) -> str:
        return self.embedder.__str__()

    @modal.method()
    async def tensor(self, query: str, elements: list[str], mesh: list[str]) -> dict:
        logger.info("✦ 1) 调用嵌入")
        embeds = await asyncio.to_thread(
            self.embedder.encode, mesh, batch_size=16, convert_to_numpy=True
        )

        logger.info("✦ 2) 归一化 → 更适合向量检索")
        embeds = embeds / (numpy.linalg.norm(embeds, axis=1, keepdims=True) + 1e-8)

        for index, embed in enumerate(embeds, start=1):
            logger.info(f"Embed-{index:04}: {embed.shape}")
        embeds = numpy.asarray(embeds, dtype="float32")

        logger.info(f"✦ 3) 拆分恢复结构")
        query_vec    = embeds[0] if query else numpy.array([])
        page_vectors = embeds[1:] if elements else numpy.array([])

        logger.info(f"✦ 4) 统计")
        count = len(mesh)
        dim   = embeds.shape[-1] if mesh else 0

        logger.info(f"✦ 5) 下发结果")
        return {
            "query"        : query,
            "query_vec"    : query_vec.tolist(),
            "elements"     : elements,
            "page_vectors" : page_vectors.tolist(),
            "count"        : count,
            "dim"          : dim,
            "model"        : "bge-base-en-v1.5"
        }


if __name__ == '__main__':
    pass
