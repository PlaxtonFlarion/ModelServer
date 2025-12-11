#   ____                     _____ _   _  ____
#  / ___|_ __ ___  ___ ___  | ____| \ | |/ ___|
# | |   | '__/ _ \/ __/ __| |  _| |  \| | |
# | |___| | | (_) \__ \__ \ | |___| |\  | |___
#  \____|_|  \___/|___/___/ |_____|_| \_|\____|
#

import modal
import typing
from loguru import logger
from sentence_transformers import CrossEncoder
from images.embed_image import (
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
    memory=2048,
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

    @modal.method()
    async def heartbeat(self) -> str:
        return self.reranker.__str__()

    @modal.method()
    async def rerank(self, query: str, candidate: list[str]) -> dict:
        logger.info(f"✦ 1) 计算逻辑")
        pairs         = [[query, t] for t in candidate]
        rerank_scores = self.reranker.predict(pairs)
        scores        = [float(s) for s in rerank_scores]
        logger.info(f"✦ 2) 最终得分 {scores}")

        logger.info(f"✦ 3) 下发结果")
        return {
            "scores" : scores,
            "count"  : len(scores)
        }


if __name__ == '__main__':
    pass
