# __   __    _         _   _ _ _
# \ \ / /__ | | ___   | | | | | |_ _ __ __ _
#  \ V / _ \| |/ _ \  | | | | | __| '__/ _` |
#   | | (_) | | (_) | | |_| | | |_| | | (_| |
#   |_|\___/|_|\___/   \___/|_|\__|_|  \__,_|
#

import io
import modal
import numpy
import typing
from PIL import Image
from loguru import logger
from ultralytics import YOLO
from images.yolo_image import (
    image, secrets
)
from utils import toolset

# Notes: https://huggingface.co/Ultralytics/
# yolo11s

app = modal.App("yolo")
src = "/root/models/yolo_11/yolo11s.pt"

toolset.init_logger()


@app.cls(
    image=image,
    secrets=secrets,
    memory=4096,
    max_containers=5,
    scaledown_window=300
)
class YoloUltra(object):

    yolo_model: typing.Optional[YOLO] = None

    @modal.enter()
    def startup(self) -> None:
        logger.info("🔥 Yolo model loading ...")
        self.yolo_model = YOLO(src)
        logger.info("🔥 Yolo model loaded")

    @modal.method()
    async def heartbeat(self) -> dict:
        return {
            "status"  : "ok",
            "service" : "detection",
            "model"   : "yolo11s"
        }

    @modal.method()
    async def detection(self, image_bytes: bytes) -> dict:
        logger.info("🟡 [BEGIN] Detection start")
        logger.info(
            f"🟢 [YOLO - 1/5] Image bytes size = {len(image_bytes)}"
        )

        # ===== Step 1: bytes -> PIL =====
        image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        logger.info(
            f"🟢 [YOLO - 2/5] Image decoded (PIL) size={image_pil.size}"
        )

        # ===== Step 2: PIL -> numpy =====
        image_arr = numpy.array(image_pil)
        logger.info(
            f"🟢 [YOLO - 3/5] Image converted to numpy shape={image_arr.shape} dtype={image_arr.dtype}"
        )

        # ===== Step 3: YOLO inference =====
        values = self.yolo_model(image_arr, verbose=False)
        result = values[0]
        logger.info(
            f"🟢 [YOLO - 4/5] YOLO inference done"
        )

        # ===== Step 4: Parse results =====
        objects: list[dict] = []

        if result.boxes is None: return {"objects": objects}

        for box in result.boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cfg = float(box.conf[0])

            objects.append({
                "label" : self.yolo_model.names[cls],
                "bbox"  : [x1, y1, x2, y2],
                "score" : round(cfg, 4),
            })

        logger.info(
            f"🟢 [YOLO - 5/5] Result parsed objects={len(objects)}"
        )
        logger.info(f"✅ [FINAL] Detection finished")

        return {
            "objects" : objects,
            "count"   : len(objects),
        }


if __name__ == '__main__':
    pass
