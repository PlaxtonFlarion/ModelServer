#   __  __       _
#  |  \/  | __ _(_)_ __
#  | |\/| |/ _` | | '_ \
#  | |  | | (_| | | | | |
#  |_|  |_|\__,_|_|_| |_|
#

import io
import json
import time
import modal
import numpy
import typing
from loguru import logger
from fastapi import (
    UploadFile, Request
)
from fastapi.responses import (
    JSONResponse, StreamingResponse
)
from starlette.datastructures import FormData
from sentence_transformers import (
    SentenceTransformer, CrossEncoder
)
from services.sequential.classifier.keras_classifier import KerasStruct
from services.sequential.cutter.cut_range import VideoCutRange
from services.sequential.video import (
    VideoFrame, VideoObject
)
from middlewares.auth import auth_middleware
from middlewares.exception import exception_middleware
from schema.meta import FrameMeta
from utils import toolset


app = modal.App("inference")

toolset.init_logger()

image = modal.Image.debian_slim(
    python_version="3.11"
).pip_install_from_requirements(
    "requirements.txt"
).apt_install(
    "libgl1", "libglib2.0-0", "ffmpeg"
).add_local_dir(
    local_path=".",
    remote_path="/root",
    ignore=["**/.venv", "**/venv"]
)
secret = modal.Secret.from_name("SHARED_SECRET")


@app.cls(
    image=image,
    # gpu="A10G",
    secrets=[secret],
    memory=8192,
    max_containers=2,
    scaledown_window=300
)
class InferenceService(object):
    """InferenceService"""

    kf: typing.Optional["KerasStruct"] = None
    kc: typing.Optional["KerasStruct"] = None

    embedder: typing.Optional["SentenceTransformer"] = None
    reranker: typing.Optional["CrossEncoder"]        = None

    @modal.enter()
    def startup(self):
        """预热"""
        logger.info("KF model loading ...")
        self.kf = KerasStruct()
        self.kf.load_model("/root/models/sequence/Keras_Gray_W256_H256")
        logger.info("✅ KF model loaded")

        logger.info("KC model loading ...")
        self.kc = KerasStruct()
        self.kc.load_model("/root/models/sequence/Keras_Hued_W256_H256")
        logger.info("✅ KC model loaded")

        logger.info("BGE embedding model loading ...")
        self.embedder = SentenceTransformer("/root/models/bge_base")
        logger.info("🔥 BGE embedding model loaded")

        logger.info("Cross Encoder model loading ...")
        self.reranker = CrossEncoder("/root/models/cross_encoder")
        logger.info("🔥 Cross Encoder model loaded")

    @modal.method(is_generator=True)
    def classify_stream(self, file_bytes: bytes, meta_dict: dict):
        """推理"""
        try:
            logger.info(f"========== Overflow Begin ==========")
            meta = FrameMeta(**meta_dict)
            npz_data = numpy.load(io.BytesIO(file_bytes), allow_pickle=False)

            logger.info(f"video name: {meta.video_name}")
            logger.info(f"video path: {meta.video_path}")
            logger.info(f"frame count: {meta.frame_count}")
            logger.info(f"frame shape: {meta.frame_shape}")
            logger.info(f"frames data: {len(meta.frames_data)}")
            logger.info(f"valid range: {len(meta.valid_range)}")
            for cut_range in meta.valid_range:
                logger.info(f"Cut Range: {cut_range}")
            logger.info(f"step: {meta.step}")
            logger.info(f"keep data: {meta.keep_data}")
            logger.info(f"boost mode: {meta.boost_mode}")

            keep_data    = False
            frame_arrays = [npz_data[key] for key in npz_data.files]
            frame_list   = [
                VideoFrame(frame["frame_id"], frame["timestamp"], data)
                for frame, data in zip(meta.frames_data, frame_arrays)
            ]

            video = VideoObject(
                meta.video_name, meta.video_path, meta.frame_count, tuple(frame_list)
            )

            cut_ranges = [
                VideoCutRange(
                    video=video,
                    start=cr["start"],
                    end=cr["end"],
                    ssim=cr["ssim"],
                    psnr=cr["psnr"],
                    mse=cr["mse"],
                    start_time=cr["start_time"],
                    end_time=cr["end_time"]
                )
                for cr in meta.valid_range
            ]

            frame_channel = toolset.judge_channel(
                meta.frame_shape
            ) or toolset.judge_channel(video.frame_detail()[-1])
            logger.info(f"Frame channel: {frame_channel}")

            final         = self.kc if frame_channel != 1 else self.kf
            model_channel = final.model.input_shape[-1]
            logger.info(f"Model channel: {model_channel}")

            matched: typing.Callable[[], bool] = lambda: frame_channel == model_channel
            if not matched():
                stream = {
                    "fatal": (
                        message := f"通道数不匹配 FCH={frame_channel} MCH={model_channel} 回退分析模式"
                    )
                }
                yield f"FATAL: {json.dumps(stream, ensure_ascii=False)}\n\n"
                return logger.error(message)

            yield from final.classify(
                video, cut_ranges, meta.step, keep_data, meta.boost_mode
            )
        except Exception as e:
            yield f"FATAL: {json.dumps({'fatal': str(e)}, ensure_ascii=False)}\n\n"
            return logger.error(e)

        finally:
            logger.info(f"========== Overflow Final ==========")

    @modal.fastapi_endpoint(method="POST")
    @exception_middleware
    @auth_middleware("X-Token")
    async def predict(self, request: "Request"):
        """推理接口"""

        logger.info(f"Request: {request.method} {request.url}")

        form: "FormData"         = await request.form()
        frame_meta: str          = form["frame_meta"]
        frame_file: "UploadFile" = form["frame_file"]

        meta_dict  = json.loads(frame_meta)
        file_bytes = await frame_file.read()

        return StreamingResponse(
            self.classify_stream.remote_gen(file_bytes, meta_dict),
            media_type="text/event-stream"
        )

    @modal.fastapi_endpoint(method="GET")
    @exception_middleware
    @auth_middleware("X-Token")
    async def service(self, request: "Request"):
        """心跳接口"""

        logger.info(f"Request: {request.method} {request.url}")

        faint_model_dict = {
            "fettle" : "Online",
            "dazzle" : {
                k: v for k, v in json.loads(self.kf.model.to_json()).items() if k != "config"
            }
        } if self.kf.model else {"fettle": "Offline"}

        color_model_dict = {
            "fettle" : "Online",
            "dazzle" : {
                k: v for k, v in json.loads(self.kc.model.to_json()).items() if k != "config"
            }
        } if self.kc.model else {"fettle": "Offline"}

        content = {
            "status"    : "OK",
            "message"   : {
                "AquilaSequence-F" : {**faint_model_dict},
                "AquilaSequence-C" : {**color_model_dict}
            },
            "timestamp" : int(time.time()),
        }

        logger.info(content)
        return JSONResponse(content=content, status_code=200)

    @modal.fastapi_endpoint(method="POST")
    @exception_middleware
    @auth_middleware("X-Token")
    async def embedding(self, request: "Request"):
        """
        批量Embedding接口（函数名不变，兼容单输入/批量输入）

        支持两种参数：
        1) { "text": "立即支付 按钮" }
        2) { "texts": ["立即支付", "继续支付", ...] }

        返回：
        {
            "vectors":[[...],[...]],
            "count":N,
            "dim":768,
            "model":"bge-base"
        }
        """

        body = await request.json()

        # 兼容单文本 → 自动转列表
        if "text" in body:
            texts = [body["text"]]
        else:
            texts = body.get("texts", [])

        if not texts or not isinstance(texts, list):
            return JSONResponse(content={"error": "text or texts required"}, status_code=400)

        # 🔥 批量 embedding（GPU/CPU向量化）
        embeddings = self.embedder.encode(
            texts, batch_size=16, convert_to_numpy=True
        )

        # 归一化 → 更适合向量检索
        embeddings = embeddings / (numpy.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

        return JSONResponse({
            "vectors" : embeddings.astype("float32").tolist(),
            "count"   : len(embeddings),
            "dim"     : embeddings.shape[1],
            "model"   : "bge-base"
        })

    @modal.fastapi_endpoint(method="POST")
    @exception_middleware
    @auth_middleware("X-Token")
    async def rerank(self, request: "Request"):
        """
        Cross-Encoder 重排接口

        请求 JSON:
        {
          "query": "立即支付 按钮",
          "candidate": [
            "继续支付 按钮",
            "去结算 按钮",
            "立即支付 按钮"
          ]
        }

        响应 JSON:
        {
          "scores": [0.12, 0.34, 0.98],
          "count": 3
        }
        """

        body      = await request.json()
        query     = body.get("query")
        candidate = body.get("candidate")

        if not query or not isinstance(candidate, list) or not candidate:
            return JSONResponse(
                content={"error": "query and candidates (list) are required"}, status_code=400,
            )

        candidate_pairs = [[query, t] for t in candidate]
        rerank_scores   = self.reranker.predict(candidate_pairs)

        scores = [float(s) for s in rerank_scores]
        return JSONResponse(
            content={"scores": scores, "count": len(scores)}, status_code=200
        )


if __name__ == "__main__":
    # Notes: ==== https://modal.com/ ====
    # modal run main.py
    # modal deploy main.py
    # uvicorn main:app --host 0.0.0.0 --port 8000 --reload

    # ==== Volume ====
    # modal volume create model-cache
    # modal volume delete model-cache
    # modal volume list
    # modal volume put model-cache ./models/bge_base /root/models/bge_base
    # modal volume put model-cache ./models/cross_encoder /root/models/cross_encoder
    # modal volume put model-cache ./models/sequence/Keras_Gray_W256_H256 /root/models/sequence/Keras_Gray_W256_H256
    # modal volume put model-cache ./models/sequence/Keras_Hued_W256_H256 /root/models/sequence/Keras_Hued_W256_H256

    pass
