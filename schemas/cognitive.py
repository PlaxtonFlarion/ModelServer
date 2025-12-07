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
    """
    Tensor Embedding Response

    返回 query 向量及 elements 对应向量结果，用于向量召回/相似度计算。
    """

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
    error: typing.Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class RerankResponse(BaseModel):
    """
    Rerank Response

    Returns
    -------
    scores : List[float]
        每个候选文本的相关性得分，数值越大越相关。
    count : int
        返回得分数量（候选文本数量）。
    """

    scores: typing.Optional[list[float]] = Field(..., description="Rerank 打分结果，按输入顺序对应 candidate")
    count: typing.Optional[int] = Field(..., description="评分条数，等于 candidate 数量")
    error: typing.Optional[str] = None


if __name__ == '__main__':
    pass
