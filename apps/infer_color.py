#  ___        __              ____      _
# |_ _|_ __  / _| ___ _ __   / ___|___ | | ___  _ __
#  | || '_ \| |_ / _ \ '__| | |   / _ \| |/ _ \| '__|
#  | || | | |  _|  __/ |    | |__| (_) | | (_) | |
# |___|_| |_|_|  \___|_|     \____\___/|_|\___/|_|
#

import io
import json
import modal
import numpy
import typing
from loguru import logger
from services.sequential.classifier.keras_classifier import KerasStruct
from services.sequential.cutter.cut_range import VideoCutRange
from services.sequential.video import (
    VideoFrame, VideoObject
)
from schemas.cognitive import FrameMeta
from images.infer_image import (
    image, secrets
)
from utils import toolset

# Notes: https://keras.io/
# Sequential

app = modal.App("inference-color")
src = "/root/models/sequence/Keras_Hued_W256_H256"

toolset.init_logger()


@app.cls(
    image=image,
    # gpu="A10G",
    secrets=secrets,
    memory=8192,
    max_containers=5,
    scaledown_window=300
)
class InferenceColor(object):

    keras_sequential: typing.Optional[KerasStruct] = None

    @modal.enter()
    def startup(self) -> None:
        logger.info("🔥 Keras color model loading ...")
        self.keras_sequential = KerasStruct()
        self.keras_sequential.load_model(src)
        logger.info("🔥 Keras color model loaded")

    @modal.method()
    async def heartbeat(self) -> str:
        return self.keras_sequential.__str__()

    @modal.method(is_generator=True)
    def classify_stream(self, meta_dict: dict, file_bytes: bytes) -> typing.Generator[str, None, None]:
        logger.info(f"========== Overflow Begin ==========")

        try:
            meta     = FrameMeta(**meta_dict)
            npz_data = numpy.load(io.BytesIO(file_bytes), allow_pickle=False)

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

            model_channel = self.keras_sequential.model.input_shape[-1]
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

            yield from self.keras_sequential.classify(
                video, cut_ranges, meta.step, keep_data, meta.boost_mode
            )
        except Exception as e:
            yield f"FATAL: {json.dumps({'fatal': str(e)}, ensure_ascii=False)}\n\n"
            return logger.error(e)

        finally:
            logger.info(f"========== Overflow Final ==========")


if __name__ == '__main__':
    pass
