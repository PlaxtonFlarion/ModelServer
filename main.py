#   __  __       _
#  |  \/  | __ _(_)_ __
#  | |\/| |/ _` | | '_ \
#  | |  | | (_| | | | | |
#  |_|  |_|\__,_|_|_| |_|
#

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

from functools import wraps

from loguru   import logger
from pydantic import BaseModel

from fastapi           import UploadFile, Request
from fastapi.responses import JSONResponse, StreamingResponse

from starlette.datastructures import FormData
from starlette.requests       import ClientDisconnect

from services.sequential.classifier.keras_classifier import KerasStruct
from services.sequential.cutter.cut_range            import VideoCutRange
from services.sequential.video                       import VideoFrame, VideoObject

from common import craft

# Notes: ==== 启动应用 ====
app = modal.App("inference")

# Notes: ==== 启动日志 ====
craft.init_logger()

# Notes: ==== 构建镜像 ====
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


def require_token(header_key: str = "X-Token"):
    """鉴权中间件"""

    def decorator(func):
        @wraps(func)
        async def wrapper(self, request, *args, **kwargs):
            token = request.headers.get(header_key)
            self.verify_token(token)
            return await func(self, request, *args, **kwargs)
        return wrapper
    return decorator


def with_exception_handling(func):
    """异常中间件"""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except ClientDisconnect as e:
            logger.error(e)
            return JSONResponse(
                content={
                    "Error": "Client disconnected during upload"
                },
                status_code=499
            )
        except json.JSONDecodeError as e:
            logger.error(e)
            return JSONResponse(
                content={
                    "Error": "Invalid JSON payload"
                },
                status_code=400
            )
        except modal.exception.InvalidError as e:
            logger.error(e)
            return JSONResponse(
                content={
                    "Error": f"Modal error: {str(e)}"
                },
                status_code=500
            )
        except Exception as e:
            logger.error(e)
            return JSONResponse(
                content={
                    "Error": f"Unexpected error: {str(e)}"
                },
                status_code=500
            )
    return wrapper


class FrameMeta(BaseModel):
    video_name: str
    video_path: str
    frame_count: int
    frame_shape: tuple[int, ...]
    frames_data: list
    valid_range: list
    step: typing.Optional[int] = None
    keep_data: typing.Optional[bool] = None
    boost_mode: typing.Optional[bool] = None


@app.cls(
    image=image,
    gpu="A10G",
    secrets=[secret],
    memory=16384,
    max_containers=3,
    scaledown_window=300
)
class InferenceService(object):
    """InferenceService"""

    kf: typing.Optional["KerasStruct"] = None
    kc: typing.Optional["KerasStruct"] = None

    shared_secret: typing.Optional[str] = None

    def verify_token(self, token: str) -> typing.Union["JSONResponse", bool]:
        """鉴权"""

        logger.info(f"Verify token: {token}")
        if not token:
            return JSONResponse(
                content={
                    "Error"   : "Unauthorized",
                    "Message" : "Token missing"
                },
                status_code=401
            )

        try:
            payload, sig      = token.rsplit(".", 1)
            app_id, expire_at = payload.split(":")

            if time.time() > int(expire_at):
                logger.warning("Token has expired")
                return JSONResponse(
                    content={
                        "Error": "Token has expired"
                    },
                    status_code=401
                )

            expected_sig = hmac.new(
                self.shared_secret.encode(), payload.encode(), hashlib.sha256
            ).digest()
            expected_b64 = base64.b64encode(expected_sig).decode()

            if not (compare := hmac.compare_digest(expected_b64, sig)):
                logger.warning("Token signature mismatch")
                return JSONResponse(
                    content={
                        "Error": "Invalid token signature"
                    },
                    status_code=401
                )
            return compare

        except ValueError as e:
            logger.error(e)
            return JSONResponse(
                content={
                    "Error": "Malformed token"
                },
                status_code=401
            )

        except Exception as e:
            logger.error(e)
            return JSONResponse(
                content={
                    "Error"   : "Unauthorized",
                    "Message" : str(e)
                },
                status_code=401
            )

    @staticmethod
    def judge_channel(shape: tuple[int, ...]) -> int:
        """色彩"""
        return shape[2] if len(shape) == 3 and shape[2] in (1, 3, 4) else \
            shape[0] if len(shape) == 3 and shape[0] in (1, 3, 4) else \
                1 if len(shape) == 2 else None

    @modal.enter()
    def startup(self):
        """预热"""
        logger.info("KF model loading ...")
        self.kf = KerasStruct()
        self.kf.load_model("/root/models/Keras_Gray_W256_H256")
        logger.info("✅ KF model loaded")

        logger.info("KC model loading ...")
        self.kc = KerasStruct()
        self.kc.load_model("/root/models/Keras_Hued_W256_H256")
        logger.info("✅ KC model loaded")

        self.shared_secret = os.environ["SHARED_SECRET"]

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

            frame_channel = self.judge_channel(
                meta.frame_shape
            ) or self.judge_channel(video.frame_detail()[-1])
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
    @with_exception_handling
    @require_token(header_key="X-Token")
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
    @with_exception_handling
    @require_token(header_key="X-Token")
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


if __name__ == "__main__":
    # Notes: ==== https://modal.com/ ====
    # modal run main.py
    # modal deploy main.py
    # uvicorn main:app --host 0.0.0.0 --port 8000 --reload
    pass
