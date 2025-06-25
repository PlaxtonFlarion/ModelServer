#  __     ___     _
#  \ \   / (_) __| | ___  ___
#   \ \ / /| |/ _` |/ _ \/ _ \
#    \ V / | | (_| |  __/ (_) |
#     \_/  |_|\__,_|\___|\___/
#

import numpy
import typing
from loguru import logger
from services.sequential import toolbox


class VideoFrame(object):

    def __init__(self, frame_id: int, timestamp: float, data: "numpy.ndarray"):
        self.frame_id: int = frame_id
        self.timestamp: float = timestamp
        self.data: "numpy.ndarray" = data

    def __str__(self):
        return f"<VideoFrame id={self.frame_id} timestamp={self.timestamp}>"

    def copy(self) -> "VideoFrame":
        return VideoFrame(self.frame_id, self.timestamp, self.data[:])

    def contain_image(
        self, *, image_path: str = None, image_data: "numpy.ndarray" = None, **kwargs
    ) -> dict[str, typing.Any]:
        assert image_path or (
                image_data is not None
        ), "should fill image_path or image_data"

        if image_path:
            logger.debug(f"found image path, use it first: {image_path}")
            return toolbox.match_template_with_path(image_path, self.data, **kwargs)

        image_data = toolbox.turn_grey(image_data)
        return toolbox.match_template_with_object(image_data, self.data, **kwargs)


class _BaseFrameOperator(object):

    def __init__(self, video: "VideoObject"):
        self.cur_ptr: int = 0
        self.video: "VideoObject" = video

    def get_frame_by_id(self, frame_id: int) -> typing.Optional["VideoFrame"]:
        raise NotImplementedError

    def get_length(self) -> int:
        return self.video.frame_count


class MemFrameOperator(_BaseFrameOperator):

    def get_frame_by_id(self, frame_id: int) -> typing.Optional["VideoFrame"]:
        if frame_id > self.get_length():
            return None

        frame_id = frame_id - 1

        return self.video.frames_data[frame_id].copy()


class DocFrameOperator(_BaseFrameOperator):
    pass


class VideoObject(object):

    def __init__(
            self,
            name: str,
            path: str,
            frame_count: int,
            frames_data: typing.Optional[tuple["VideoFrame", ...]]
    ):

        self.name = name
        self.path = path
        self.frame_count = frame_count
        self.frames_data = frames_data

    def __str__(self):
        return f"<VideoObject name={self.name} path={self.path}>"

    __repr__ = __str__

    def clean_frames(self) -> None:
        self.frames_data = tuple()

    def frame_detail(self) -> tuple:
        frame = self.frames_data[0]

        every_cost = frame.data.nbytes / (1024 ** 2)
        total_cost = every_cost * len(self.frames_data)
        frame_size = frame.data.shape[::-1]

        return every_cost, total_cost, frame_size

    def _read_from_doc(self) -> typing.Generator["VideoFrame", None, None]:
        pass

    def _read_from_mem(self) -> typing.Generator["VideoFrame", None, None]:
        for each_frame in self.frames_data:
            yield each_frame

    def _read(self) -> typing.Generator["VideoFrame", None, None]:
        if self.frames_data:
            yield from self._read_from_mem()
        else:
            yield from self._read_from_doc()

    def get_iterator(self) -> typing.Generator["VideoFrame", None, None]:
        return self._read()

    def get_operator(self) -> "_BaseFrameOperator":
        if self.frames_data:
            return MemFrameOperator(self)
        else:
            return DocFrameOperator(self)

    def __iter__(self):
        return self.get_iterator()


if __name__ == '__main__':
    pass
