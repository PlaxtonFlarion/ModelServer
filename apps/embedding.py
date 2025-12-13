#  _____           _              _     _ _
# | ____|_ __ ___ | |__   ___  __| | __| (_)_ __   __ _
# |  _| | '_ ` _ \| '_ \ / _ \/ _` |/ _` | | '_ \ / _` |
# | |___| | | | | | |_) |  __/ (_| | (_| | | | | | (_| |
# |_____|_| |_| |_|_.__/ \___|\__,_|\__,_|_|_| |_|\__, |
#                                                 |___/
#

import time
import modal
import numpy
import typing
import asyncio
from loguru import logger
from sentence_transformers import SentenceTransformer
from images.embed_image import (
    image, secrets
)
from utils import toolset

# Notes: https://huggingface.co/collections/BAAI/bge
# BAAI/bge-m3

app = modal.App("embedding")
src = "/root/models/bge_m3"

toolset.init_logger()


@app.cls(
    image=image,
    secrets=secrets,
    gpu="A10G",
    memory=16384,
    max_containers=5,
    scaledown_window=300
)
class Embedding(object):

    embedder: typing.Optional[SentenceTransformer] = None

    @modal.enter()
    def startup(self) -> None:
        logger.info("🔥 BGE embedding model loading ...")
        self.embedder = SentenceTransformer(src)
        logger.info("🔥 BGE embedding model loaded")

    @modal.method()
    async def heartbeat(self) -> dict:
        return {
            "status"  : "ok",
            "service" : "tensor",
            "model"   : "BAAI/bge-m3"
        }

    @modal.method()
    async def tensor(
        self,
        query: str,
        elements: list[str],
        mesh: list[str],
        s: bool = False,
        k: typing.Optional[int] = 5
    ) -> dict:

        start_ts = time.time()

        logger.info(f"🟡 [BEGIN] Embedding tensor start")
        logger.info(f"🟢 Input stats | query | elements | mesh")

        try:
            # ===== 1) 调用嵌入 =====
            t1 = time.time()
            logger.info(
                f"🟢 1/5) 调用 SentenceTransformer.encode()"
            )
            embeds = await asyncio.to_thread(
                self.embedder.encode, mesh, batch_size=16, convert_to_numpy=True
            )
            logger.info(f"   └ done | shape={embeds.shape} | cost={time.time() - t1:.3f}s")

            # ===== 2) 归一化 =====
            t2 = time.time()
            logger.info(f"🟢 2/5) 向量归一化（L2）")
            embeds = embeds / (numpy.linalg.norm(embeds, axis=1, keepdims=True) + 1e-8)
            logger.info(f"   └ done | cost={time.time() - t2:.3f}s")

            # ===== 3) 转 dtype =====
            logger.info("🟢 3/5) 转 float32")
            embeds = numpy.asarray(embeds, dtype="float32")

            # ===== 4) 拆分结构 =====
            logger.info("🟢 4/5) 拆分 query / page vectors")
            query_vec    = embeds[0] if query else numpy.array([], dtype="float32")
            page_vectors = embeds[1:] if elements else numpy.array([], dtype="float32")

            scored: typing.Optional[list[dict[str, str | float]]] = None
            if s:
                t_2 = time.time()
                logger.info(
                    f"🟡 Score enabled | mode=cosine | elements={len(elements)} | k={k or 5}"
                )
                scores = (page_vectors @ query_vec).tolist()
                scored = [
                    {
                        "score" : float(scores[i]),
                        "text"  : elements[i]
                    }
                    for i in range(len(elements))
                ]
                scored.sort(key=lambda x: x["score"], reverse=True)
                scored = scored[:k or 5]

                v = [x["score"] for x in scored]
                logger.info(
                    f"🟢 Score done | avg={sum(v) / len(v):.4f} | cost={time.time() - t_2:.3f}s"
                )
                for i, x in enumerate(scored, start=1):
                    logger.info(
                        f"   └ Top-{i}: score={x['score']:.4f} | {x['text'][:10]}"
                    )

            # ===== 5) 统计 =====
            count = len(mesh)
            dim    = embeds.shape[-1] if count else 0

            logger.info(
                f"🟢 5/5) 统计完成 | count={count} | dim={dim}"
            )
            logger.info(
                f"✅ [FINAL] Embedding tensor finished | elapsed={time.time() - start_ts:.3f}s"
            )

            return {
                "query"        : query,
                "query_vec"    : query_vec.tolist(),
                "elements"     : elements,
                "page_vectors" : page_vectors.tolist(),
                "scores"       : scored,
                "count"        : count,
                "dim"          : dim,
                "model"        : "BAAI/bge-m3"
            }

        except Exception as e:
            logger.exception("❌ [ERROR] Embedding tensor failed")
            raise e


if __name__ == '__main__':
    pass
