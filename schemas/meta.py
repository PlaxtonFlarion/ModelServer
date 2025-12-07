#  __  __      _
# |  \/  | ___| |_ __ _
# | |\/| |/ _ \ __/ _` |
# | |  | |  __/ || (_| |
# |_|  |_|\___|\__\__,_|
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


class TensorRequest(BaseModel):
    """
    Tensor Embedding Request

    向量编码 / 相似度计算任务输入数据结构

    Parameters
    ----------
    query : Optional[str]
        需要进行编码的查询文本
    elements : Optional[List[str]]
        候选文本集合，可用于向量对比或CrossEncoder重排
    """

    query: typing.Optional[str] = Field(
        None,
        description="要计算向量的输入文本，可为空",
        examples=["如何做向量召回？"]
    )

    elements: typing.Optional[list[str]] = Field(
        None,
        description="候选文本列表，可为空",
        examples=["向量召回是什么？", "如何提高搜索效果", "使用CrossEncoder更精准"]
    )

    model_config = ConfigDict(from_attributes=True)


class TensorResponse(BaseModel):
    """
    Tensor Embedding Response

    返回 query 向量及 elements 对应向量结果，用于向量召回/相似度计算。
    """

    query: typing.Optional[str] = Field(None, description="原始输入文本")
    query_vec: typing.Optional[list[list[float]]] = Field(
        None,
        description="query 的向量表达结果 (二维数组)",
        examples=[[0.104, -0.223, 0.873, ...]]
    )

    elements: typing.Optional[list[str]] = Field(
        None,
        description="输入的对照文本列表"
    )
    page_vectors: typing.Optional[list[list[float]]] = Field(
        None,
        description="elements 批量向量结果 (二维数组)",
        examples=[[0.298, -0.111, 0.552, ...]]
    )

    count: int = Field(..., description="向量数量", examples=[2])
    dim: int = Field(..., description="向量维度", examples=[768])
    model: str = Field(..., description="使用的 embedding 模型", examples=["bge-m3"])

    model_config = ConfigDict(from_attributes=True)




if __name__ == '__main__':
    pass
