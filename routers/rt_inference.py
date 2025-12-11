#  ___        __                                ____             _
# |_ _|_ __  / _| ___ _ __ ___ _ __   ___ ___  |  _ \ ___  _   _| |_ ___ _ __
#  | || '_ \| |_ / _ \ '__/ _ \ '_ \ / __/ _ \ | |_) / _ \| | | | __/ _ \ '__|
#  | || | | |  _|  __/ | |  __/ | | | (_|  __/ |  _ < (_) | |_| | ||  __/ |
# |___|_| |_|_|  \___|_|  \___|_| |_|\___\___| |_| \_\___/ \__,_|\__\___|_|
#

import json
import modal
from loguru import logger
from fastapi import (
    APIRouter, Request, UploadFile, Form
)
from fastapi.responses import StreamingResponse

from schemas.errors import BizError
from utils import (
    const, toolset
)

inference_router = APIRouter(tags=["Inference"])


@inference_router.post(
    path="/predict",
    response_class=StreamingResponse,
    operation_id="api_predict"
)
async def api_predict(request: Request) -> StreamingResponse:
    logger.info(f"**> {request.method} {request.url}")

    form: Form             = await request.form()
    frame_meta: str        = form["frame_meta"]
    frame_file: UploadFile = form["frame_file"]

    meta_dict  = json.loads(frame_meta)
    file_bytes = await frame_file.read()

    for k, v in meta_dict.items():
        match k:
            case "frames_data":
                logger.info(f"frames data: {len(v)}")
            case "valid_range":
                logger.info(f"valid range: {len(v)}")
                for c in v: logger.info(c)
            case _:
                logger.info(f"{k}: {v}")

    frame_current = meta_dict["frames_data"][0]
    frame_channel = toolset.judge_channel(
        meta_dict["frame_shape"]
    ) or toolset.judge_channel(frame_current.data.shape[::-1])

    match frame_channel:
        case 3: f = modal.Cls.from_name(
            app_name=const.GROUP_FUNC, name="InferenceColor"
        )
        case 1: f = modal.Cls.from_name(
            app_name=const.GROUP_FUNC, name="InferenceFaint"
        )
        case _: raise BizError(
            status_code=400, detail="Bad Request"
        )

    return StreamingResponse(
        f().classify_stream.remote_gen(meta_dict, file_bytes),
        media_type="text/event-stream"
    )


if __name__ == '__main__':
    pass
