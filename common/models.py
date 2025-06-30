#   __  __           _      _
#  |  \/  | ___   __| | ___| |___
#  | |\/| |/ _ \ / _` |/ _ \ / __|
#  | |  | | (_) | (_| |  __/ \__ \
#  |_|  |_|\___/ \__,_|\___|_|___/
#

import typing
from pydantic import BaseModel


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


if __name__ == '__main__':
    pass