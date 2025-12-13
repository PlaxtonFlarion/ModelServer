#   ____                     _____ _   _  ____
#  / ___|_ __ ___  ___ ___  | ____| \ | |/ ___|
# | |   | '__/ _ \/ __/ __| |  _| |  \| | |
# | |___| | | (_) \__ \__ \ | |___| |\  | |___
#  \____|_|  \___/|___/___/ |_____|_| \_|\____|
#

import time
import modal
import typing
from loguru import logger
from sentence_transformers import CrossEncoder
from images.embed_image import (
    image, secrets
)
from utils import toolset

# Notes: https://huggingface.co/cross-encoder
# ms-marco-MiniLM-L-12-v2

app = modal.App("cross-encoder")
src = "/root/models/cross_encoder"

toolset.init_logger()


@app.cls(
    image=image,
    secrets=secrets,
    memory=2048,
    max_containers=5,
    scaledown_window=300
)
class CrossENC(object):

    reranker: typing.Optional[CrossEncoder] = None

    @modal.enter()
    def startup(self) -> None:
        logger.info("🔥 CrossEncoder model loading ...")
        self.reranker = CrossEncoder(src)
        logger.info("🔥 CrossEncoder model loaded")

    @modal.method()
    async def heartbeat(self) -> dict:
        return {
            "status"  : "ok",
            "service" : "rerank",
            "model"   : "ms-marco-MiniLM-L-12-v2"
        }

    @modal.method()
    async def rerank(self, query: str, candidate: list[str]) -> dict:
        start_ts = time.time()

        logger.info(f"🟡 [BEGIN] Rerank start")
        logger.info(f"🟢 Input query length={len(query)} chars")
        logger.info(f"🟢 Candidate count={len(candidate)}")

        # ---------- 预览候选（防止刷屏） ----------
        preview_n = min(3, len(candidate))
        for i in range(preview_n):
            logger.info(f"   cand[{i}]={candidate[i][:120]}")

        try:
            # ===== 1) 构造 pair =====
            logger.info("🟢 1/3) 构造 query-candidate pairs")
            pairs = [[query, t] for t in candidate]

            # ===== 2) 推理 =====
            logger.info("🟡 2/3) CrossEncoder 推理中...")
            rerank_scores = self.reranker.predict(pairs)

            scores = [float(s) for s in rerank_scores]

            # ===== 3) 输出 =====
            logger.info("🟢 3/3) 推理完成，得分如下（前几项）")
            for i, s in enumerate(scores[:preview_n]):
                logger.info(f"   score[{i}]={s:.6f}")

            logger.info(
                f"✅ [FINAL] Rerank finished | count={len(scores)} | elapsed={time.time() - start_ts:.3f}s"
            )

            return {
                "scores": scores,
                "count": len(scores)
            }

        except Exception as e:
            logger.exception("❌ [ERROR] Rerank failed")
            raise e


if __name__ == '__main__':
    pass
