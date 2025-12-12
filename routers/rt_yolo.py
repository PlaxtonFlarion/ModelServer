# __   __    _         ____             _
# \ \ / /__ | | ___   |  _ \ ___  _   _| |_ ___ _ __
#  \ V / _ \| |/ _ \  | |_) / _ \| | | | __/ _ \ '__|
#   | | (_) | | (_) | |  _ < (_) | |_| | ||  __/ |
#   |_|\___/|_|\___/  |_| \_\___/ \__,_|\__\___|_|
#

import time
import modal
from loguru import logger
from fastapi import (
    APIRouter, Request, UploadFile, File
)
from schemas.cognitive import (
    YoloObject, YoloDetectionResponse
)
from schemas.errors import BizError
from utils import const

yolo_router = APIRouter(tags=["Yolo"])


@yolo_router.get(
    path="/yolo-detection",
    response_model=YoloDetectionResponse,
    operation_id="api_yolo_detection"
)
async def api_yolo_detection(
    request: Request,
    file: UploadFile = File(...)
) -> YoloDetectionResponse:

    logger.info(f"**> {request.method} {request.url}")

    try:
        if not (image_bytes := await file.read()):
            raise BizError(
                status_code=400, detail="empty image file"
            )

        f = modal.Cls.from_name(app_name=const.GROUP_FUNC, name="YoloUltra")

        objects_raw = await f().detection.remote.aio(image_bytes)

        objects = [
            YoloObject(
                index=i,
                label=obj["label"],
                bbox=obj["bbox"],
                score=obj["score"]
            ) for i, obj in enumerate(objects_raw)
        ]

        return YoloDetectionResponse(
            status="ok",
            model="yolo11s",
            objects=objects,
            count=len(objects),
            ts=int(time.time())
        )

    finally:
        logger.info(f"**> {('=' * 12)}")


if __name__ == '__main__':
    pass
