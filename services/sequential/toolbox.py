#   _____           _ ____
#  |_   _|__   ___ | | __ )  _____  __
#    | |/ _ \ / _ \| |  _ \ / _ \ \/ /
#    | | (_) | (_) | | |_) | (_) >  <
#    |_|\___/ \___/|_|____/ \___/_/\_\
#

import os
import cv2
import math
import time
import random
import typing
import contextlib
import subprocess
import numpy as np
from loguru import logger
from findit import FindIt
from base64 import b64encode
from skimage.feature import (
    hog, local_binary_pattern
)
from skimage.metrics import (
    normalized_root_mse as compare_nrmse,
    peak_signal_noise_ratio as compare_psnr,
    structural_similarity as origin_compare_ssim
)


@contextlib.contextmanager
def video_capture(video_path: str):
    video_cap = cv2.VideoCapture(video_path)
    try:
        yield video_cap
    finally:
        video_cap.release()


def video_jump(video_cap: "cv2.VideoCapture", frame_id: int) -> None:
    video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id - 1 - 1)
    video_cap.read()


def compare_ssim(pic1: "np.ndarray", pic2: "np.ndarray") -> float:
    pic1, pic2 = [turn_grey(i) for i in [pic1, pic2]]

    return origin_compare_ssim(pic1, pic2)


def multi_compare_ssim(
    pic1_list: list,
    pic2_list: list,
    hooks: typing.Optional[list] = None
) -> list[float]:

    from services.sequential.video import VideoFrame

    if isinstance(pic1_list[0], VideoFrame):
        if hooks:
            for each in hooks:
                pic1_list = [each.do(each_frame) for each_frame in pic1_list]
        pic1_list = [i.data for i in pic1_list]

    if isinstance(pic2_list[0], VideoFrame):
        if hooks:
            for each in hooks:
                pic2_list = [each.do(each_frame) for each_frame in pic2_list]
        pic2_list = [i.data for i in pic2_list]

    return [compare_ssim(a, b) for a, b in zip(pic1_list, pic2_list)]


def get_current_frame_id(video_cap: "cv2.VideoCapture") -> int:
    return int(video_cap.get(cv2.CAP_PROP_POS_FRAMES))


def get_current_frame_time(video_cap: "cv2.VideoCapture") -> float:
    return video_cap.get(cv2.CAP_PROP_POS_MSEC) / 1000


def imread(img_path: str, *_, **__) -> "np.ndarray":
    assert os.path.isfile(img_path), f"file {img_path} is not existed"

    return cv2.imread(img_path, *_, **__)


def get_frame_time(
    video_cap: "cv2.VideoCapture",
    frame_id: int,
    recover: bool = None
) -> float:

    current = get_current_frame_id(video_cap)
    video_jump(video_cap, frame_id)
    result = get_current_frame_time(video_cap)
    logger.debug(f"frame {frame_id} -> {result}")

    if recover:
        video_jump(video_cap, current + 1)
    return result


def get_frame_count(video_cap: "cv2.VideoCapture") -> int:
    return int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))


def get_frame_size(video_cap: "cv2.VideoCapture") -> tuple[int, int]:
    h = video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    w = video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    return int(w), int(h)


def get_frame(
    video_cap: "cv2.VideoCapture",
    frame_id: int,
    recover: typing.Optional[bool] = None
) -> "np.ndarray":

    current = get_current_frame_id(video_cap)
    video_jump(video_cap, frame_id)
    ret, frame = video_cap.read()
    assert ret, f"read frame failed, frame id: {frame_id}"

    if recover:
        video_jump(video_cap, current + 1)
    return frame


def turn_grey(old: "np.ndarray") -> "np.ndarray":
    try:
        return cv2.cvtColor(old, cv2.COLOR_RGB2GRAY)
    except cv2.error:
        return old


def turn_binary(old: "np.ndarray") -> "np.ndarray":
    grey = turn_grey(old).astype("uint8")

    return cv2.adaptiveThreshold(
        grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )


def turn_hog_desc(old: "np.ndarray") -> "np.ndarray":
    fd, _ = hog(
        old,
        orientations=8,
        pixels_per_cell=(16, 16),
        cells_per_block=(1, 1),
        block_norm="L2-Hys",
        visualize=True,
    )

    return fd


def turn_lbp_desc(old: "np.ndarray", radius: typing.Optional[int] = None) -> "np.ndarray":
    radius = radius if radius else 3
    n_points = 8 * radius

    grey = turn_grey(old)
    lbp = local_binary_pattern(grey, n_points, radius, method="default")

    return lbp


def turn_blur(old: "np.ndarray") -> "np.ndarray":
    return cv2.GaussianBlur(old, (7, 7), 0)


def sharpen_frame(old: "np.ndarray") -> "np.ndarray":
    blur = turn_blur(old)
    smooth = cv2.addWeighted(blur, 1.5, old, -0.5, 0)
    canny = cv2.Canny(smooth, 50, 150)

    return canny


def calc_mse(pic1: "np.ndarray", pic2: "np.ndarray") -> float:
    return compare_nrmse(pic1, pic2)


def calc_psnr(pic1: "np.ndarray", pic2: "np.ndarray") -> float:
    psnr = compare_psnr(pic1, pic2)
    if math.isinf(psnr):
        psnr = 100.0

    return psnr / 100


def compress_frame(
    frame: "np.ndarray",
    compress_rate: typing.Optional[typing.Union[int, float]] = None,
    target_size: typing.Optional[tuple] = None,
    not_grey: typing.Optional[bool] = None,
    interpolation: typing.Optional[int] = None,
    *_,
    **__,
) -> "np.ndarray":

    target = frame if not_grey else turn_grey(frame)

    interpolation = interpolation or cv2.INTER_AREA

    if target_size:
        return cv2.resize(target, target_size, interpolation=interpolation)

    if not compress_rate:
        return target

    return cv2.resize(target, (0, 0), fx=compress_rate, fy=compress_rate, interpolation=interpolation)


def get_timestamp_str() -> str:
    time_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
    salt     = random.randint(10, 99)

    return f"{time_str}{salt}"


def np2b64str(frame: "np.ndarray") -> str:
    buffer = cv2.imencode(".png", frame)[1]

    return b64encode(buffer).decode()


def fps_convert(
    target_fps: int,
    source_path: str,
    target_path: str,
    ffmpeg_exe: typing.Optional[str] = None
) -> int:

    ffmpeg_exe = ffmpeg_exe if ffmpeg_exe else r"ffmpeg"

    command: list[str] = [
        ffmpeg_exe, "-i", source_path, "-r", str(target_fps), target_path,
    ]
    logger.debug(f"convert video: {command}")

    return subprocess.check_call(command)


def match_template_with_object(
    template: "np.ndarray",
    target: "np.ndarray",
    engine_template_cv_method_name: str = None,
    **kwargs,
) -> dict[str, typing.Any]:

    if not engine_template_cv_method_name:
        engine_template_cv_method_name = "cv2.TM_CCOEFF_NORMED"

    fi = FindIt(
        engine=["html"],
        engine_template_cv_method_name=engine_template_cv_method_name,
        **kwargs,
    )

    fi_template_name = "default"
    fi.load_template(fi_template_name, pic_object=template)

    result = fi.find(target_pic_name="", target_pic_object=target, **kwargs)
    logger.debug(f"findit result: {result}")

    return result["data"][fi_template_name]["TemplateEngine"]


def match_template_with_path(
    template: str,
    target: "np.ndarray",
    **kwargs
) -> dict[str, typing.Any]:

    assert os.path.isfile(template), f"image {template} not existed"
    template_object = turn_grey(imread(template))

    return match_template_with_object(template_object, target, **kwargs)


if __name__ == '__main__':
    pass
