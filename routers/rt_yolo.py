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
    APIRouter, Request
)
from schemas.cognitive import (
    YoloObject, YoloDetectionRequest, YoloDetectionResponse
)
from schemas.errors import BizError
from utils import (
    const, toolset
)

yolo_router = APIRouter(tags=["Yolo"])


@yolo_router.post(
    path="/yolo-detection",
    response_model=YoloDetectionResponse,
    operation_id="api_yolo_detection"
)
async def api_yolo_detection(
    request: Request,
    payload: YoloDetectionRequest
) -> YoloDetectionResponse:

    logger.info(f"**> {request.method} {request.url}")

    try:
        # ✅ 1. Base64 → bytes（只在 HTTP 边界）
        if not (image_bytes := toolset.secure_b64decode(payload.data)):
            raise BizError(
                status_code=400, detail="empty image file"
            )

        f = modal.Cls.from_name(app_name=const.GROUP_FUNC, name="YoloUltra")

        # ✅ 2. Modal 调用（只传 bytes）
        objects_raw = await f().detection.remote.aio(image_bytes)

        # ✅ 3. 结构化结果
        objects = [
            YoloObject(
                index=i,
                label=obj["label"],
                bbox=obj["bbox"],
                score=obj["score"]
            ) for i, obj in enumerate(objects_raw)
        ]

        # ✅ 4. 下发结构化结果
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
