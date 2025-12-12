#   ____                  _ _   _
#  / ___|___   __ _ _ __ (_) |_(_)_   _____
# | |   / _ \ / _` | '_ \| | __| \ \ / / _ \
# | |__| (_) | (_| | | | | | |_| |\ V /  __/
#  \____\___/ \__, |_| |_|_|\__|_| \_/ \___|
#             |___/
#

import typing
from pydantic import (
    BaseModel, Field, ConfigDict
)


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

    model_config = ConfigDict(from_attributes=True)


class TensorResponse(BaseModel):
    query: typing.Optional[str] = Field(None, description="原始输入文本")
    query_vec: typing.Optional[list[float]] = Field(
        None,
        description="query 的向量表达结果 (二维数组)",
        examples=[[0.104, -0.223, 0.873]]
    )

    elements: typing.Optional[list[str]] = Field(
        None,
        description="输入的对照文本列表"
    )
    page_vectors: typing.Optional[list[list[float]]] = Field(
        None,
        description="elements 批量向量结果 (二维数组)",
        examples=[[0.298, -0.111, 0.552]]
    )

    count: int = Field(..., description="向量数量", examples=[2])
    dim: int = Field(..., description="向量维度", examples=[768])
    model: str = Field(..., description="使用的 embedding 模型", examples=["bge-m3"])

    model_config = ConfigDict(from_attributes=True)


class RerankResponse(BaseModel):
    scores: typing.Optional[list[float]] = Field(
        ..., description="Rerank 打分结果，按输入顺序对应 candidate"
    )
    count: typing.Optional[int] = Field(
        ..., description="评分条数，等于 candidate 数量"
    )

    model_config = ConfigDict(from_attributes=True)


class YoloObject(BaseModel):
    index: int = Field(..., description="对象索引，用于 LLM / Action 引用")
    label: str = Field(..., description="YOLO 识别的类别名称")
    bbox: list[int] = Field(
        ..., min_length=4, max_length=4, description="边界框 [x1, y1, x2, y2]"
    )
    score: float = Field(..., ge=0.0, le=1.0, description="置信度")

    model_config = ConfigDict(from_attributes=True)


class YoloDetectionResponse(BaseModel):
    status: str = Field("ok", description="接口状态")
    model: str = Field(..., description="YOLO 模型名称")
    count: int = Field(..., description="检测到的对象数量")
    objects: typing.List[YoloObject] = Field(default_factory=list)
    ts: int = Field(..., description="Unix 时间戳（秒）")

    model_config = ConfigDict(from_attributes=True)


class Mix(BaseModel):
    app: dict[str, typing.Any] = Field(default_factory=dict)
    white_list: list[str] = Field(default_factory=list)
    rate_config: dict[str, typing.Any] = Field(default_factory=dict)

    model_config = ConfigDict(from_attributes=True)


if __name__ == '__main__':
    pass
