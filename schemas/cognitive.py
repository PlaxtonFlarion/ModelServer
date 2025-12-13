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


class ScoreItem(BaseModel):
    score: float = Field(
        ...,
        description="query 与 element 的余弦相似度，已归一化，范围 [0,1]"
    )
    text: str = Field(
        ...,
        description="对应的元素文本（element 描述）"
    )

    model_config = ConfigDict(from_attributes=True)


class TensorRequest(BaseModel):
    query: typing.Optional[str] = Field(
        None,
        description="用于相似度计算的查询文本"
    )

    elements: typing.Optional[list[str]] = Field(
        None,
        description="候选文本列表，用于向量化或相似度计算",
        examples=[["立即支付", "确认付款", "取消订单"]]
    )

    s: bool = Field(
        False,
        description="是否返回 query 与 elements 的相似度 scores（Top-K）"
    )

    k: int = Field(
        5,
        ge=1,
        le=50,
        description="返回的 K 相似结果数量，仅在 s=true 时生效"
    )


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

    scores: typing.Optional[list[ScoreItem]] = Field(
        None,
        description=(
            "query 与 elements 的相似度结果列表，"
            "按 score 从高到低排序，结构与 Milvus search 返回一致。"
            "仅在 return_scores=true 时返回"
        )
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


class YoloDetectionRequest(BaseModel):
    image_base64: str
    image_format: typing.Optional[str] = "png"


class YoloDetectionResponse(BaseModel):
    status: str = Field("ok", description="接口状态")
    model: str = Field(..., description="YOLO 模型名称")
    count: int = Field(..., description="检测到的对象数量")
    objects: list[YoloObject] = Field(default_factory=list)
    ts: int = Field(..., description="Unix 时间戳（秒）")

    model_config = ConfigDict(from_attributes=True)


class Mix(BaseModel):
    app: dict[str, typing.Any] = Field(default_factory=dict)
    white_list: list[str] = Field(default_factory=list)
    rate_config: dict[str, typing.Any] = Field(default_factory=dict)

    model_config = ConfigDict(from_attributes=True)


if __name__ == '__main__':
    pass
