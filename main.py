import os
import io
import json
import hmac
import time
import modal
import numpy
import base64
import typing
import hashlib
from fastapi import (
    FastAPI, UploadFile, Request
)
from fastapi.responses import (
    JSONResponse, StreamingResponse
)
from starlette.requests import ClientDisconnect
from loguru import logger
from functools import wraps
from pydantic import BaseModel
from services.sequential.classifier.keras_classifier import KerasStruct
from services.sequential.cutter.cut_range import VideoCutRange
from services.sequential.video import VideoFrame, VideoObject
from common import craft

app = modal.App("inference")
web_app = FastAPI()
craft.init_logger()

# 构建镜像 & 模型挂载目录
image = modal.Image.debian_slim(
    python_version="3.11"
).pip_install_from_requirements(
    "requirements.txt"
).apt_install("libgl1", "libglib2.0-0", "ffmpeg",).add_local_dir(
    local_path=".", remote_path="/root"
)
secret = modal.Secret.from_name("SHARED_SECRET")


def require_token(header_key: str = "X-Token"):
    """
    参数化装饰器：校验请求头中的签名 Token。
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(self, request, *args, **kwargs):
            token = request.headers.get(header_key)
            self.verify_token(token)
            return await func(self, request, *args, **kwargs)
        return wrapper
    return decorator


def with_exception_handling(func):
    """
    装饰器：用于 Modal 的 fastapi_endpoint 接口，统一异常处理
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except ClientDisconnect as e:
            logger.error(e)
            return JSONResponse(
                content={"error": "Client disconnected during upload"}, status_code=499
            )
        except json.JSONDecodeError as e:
            logger.error(e)
            return JSONResponse(
                content={"error": "Invalid JSON payload"}, status_code=400
            )
        except modal.exception.InvalidError as e:
            logger.error(e)
            return JSONResponse(
                content={"error": f"Modal error: {str(e)}"}, status_code=500
            )
        except Exception as e:
            logger.error(e)
            return JSONResponse(
                content={"error": f"Unexpected error: {str(e)}"}, status_code=500
            )
    return wrapper


class FrameMeta(BaseModel):
    video_name: str
    video_path: str
    frame_count: int
    frames_data: list
    valid_range: list
    step: typing.Optional[int] = None
    keep_data: typing.Optional[bool] = None
    boost_mode: typing.Optional[bool] = None


@app.cls(
    image=image,
    memory=16384,
    max_containers=5,
    scaledown_window=600,
    secrets=[secret]
)
class InferenceService(object):

    kf: typing.Optional["KerasStruct"] = None
    kc: typing.Optional["KerasStruct"] = None

    shared_secret: typing.Optional[str] = None

    def verify_token(self, token: str) -> typing.Union["JSONResponse", bool]:
        logger.info(f"Verify token: {token}")
        if not token:
            return JSONResponse(
                content={"error": "Unauthorized", "message": "Token missing"}, status_code=401
            )

        try:
            payload, sig = token.rsplit('.', 1)
            app_id, expire_at = payload.split(':')

            if time.time() > int(expire_at):
                logger.warning(f"Token has expired")
                return JSONResponse(
                    content={"error": "Token has expired"}, status_code=401
                )

            expected_sig = hmac.new(
                self.shared_secret.encode(), payload.encode(), hashlib.sha256
            ).digest()
            expected_b64 = base64.b64encode(expected_sig).decode()

            if not (compare := hmac.compare_digest(expected_b64, sig)):
                logger.warning("Token signature mismatch")
                return JSONResponse(
                    content={"error": "Invalid token signature"}, status_code=401
                )
            return compare

        except ValueError as e:
            logger.error(e)
            return JSONResponse(
                content={"error": "Malformed token"}, status_code=401
            )
        except Exception as e:
            logger.error(e)
            return JSONResponse(
                content={"error": "Unauthorized", "message": str(e)}, status_code=401
            )

    @modal.enter()
    def startup(self):
        logger.info("Faint model loading ...")
        self.kf = KerasStruct()
        self.kf.load_model("/root/models/Keras_Gray_W256_H256")
        logger.info("✅ Faint model loaded")

        logger.info("Color model loading ...")
        self.kc = KerasStruct()
        self.kc.load_model("/root/models/Keras_Hued_W256_H256")
        logger.info("✅ Color model loaded")

        self.shared_secret = os.environ["SHARED_SECRET"]

    @modal.method(is_generator=True)
    def classify_stream(self, file_bytes: bytes, meta_dict: dict):
        meta = FrameMeta(**meta_dict)
        npz_data = numpy.load(io.BytesIO(file_bytes), allow_pickle=False)

        logger.info(f"video name: {meta.video_name}")
        logger.info(f"video path: {meta.video_path}")
        logger.info(f"frame count: {meta.frame_count}")
        logger.info(f"frames data: {len(meta.frames_data)}")
        logger.info(f"valid range: {len(meta.valid_range)}")
        for cut_range in meta.valid_range:
            logger.info(f"Cut Range: {cut_range}")
        logger.info(f"step: {meta.step}")
        logger.info(f"keep data: {meta.keep_data}")
        logger.info(f"boost mode: {meta.boost_mode}")

        keep_data = False  # Notes: 服务端不返回图像数据，仅返回分类结果

        frame_arrays = [npz_data[key] for key in npz_data.files]
        frame_list = [
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

        frame_shape: tuple[int, ...] = video.frame_detail()[-1]
        logger.info(f"Frame shape: {frame_shape}")
        # blend_model = self.kf if len(frame_shape) == 2 else (
        #     self.kf if frame_shape[0] == 1 else self.kc
        # )
        blend_model = self.kc

        logger.info(f"Classifier: {blend_model.__class__.__name__}")
        yield from blend_model.classify(
            video, cut_ranges, meta.step, keep_data, meta.boost_mode
        )

    @modal.fastapi_endpoint(method="POST")
    @with_exception_handling
    @require_token(header_key="X-Token")
    async def predict(self, request: "Request"):
        form = await request.form()
        frame_meta = form["frame_meta"]
        frame_file: "UploadFile" = form["frame_file"]

        meta_dict = json.loads(frame_meta)
        file_bytes = await frame_file.read()

        return StreamingResponse(
            self.classify_stream.remote_gen(file_bytes, meta_dict),
            media_type="text/event-stream"
        )

    @modal.fastapi_endpoint(method="GET")
    @with_exception_handling
    async def service(self):
        """
        轻量级心跳接口，用于保持容器活跃并返回模型加载状态。
        """
        return JSONResponse(
            content={
                "status": "OK",
                "message": {
                    "QuantumWeave": {
                        "name": self.kf.model.name,
                        "input_shape": self.kf.model.input_shape,
                        "output_shape": self.kf.model.output_shape,
                        "layers": self.kf.model.layers,
                    },
                    "AquilaSequence-X": {
                        "name": self.kc.model.name,
                        "input_shape": self.kc.model.input_shape,
                        "output_shape": self.kc.model.output_shape,
                        "layers": self.kc.model.layers,
                    },
                },
                "timestamp": int(time.time()),
            }, status_code=200
        )


if __name__ == "__main__":
    # modal run main.py
    # modal deploy main.py
    # uvicorn main:app --host 0.0.0.0 --port 8000 --reload
    pass
