#  ____          _ _        ____           _
# |  _ \ ___  __| (_)___   / ___|__ _  ___| |__   ___
# | |_) / _ \/ _` | / __| | |   / _` |/ __| '_ \ / _ \
# |  _ <  __/ (_| | \__ \ | |__| (_| | (__| | | |  __/
# |_| \_\___|\__,_|_|___/  \____\__,_|\___|_| |_|\___|
#

import json
import typing
import redis.asyncio as redis


class RedisCache(object):
    """
    Redis 异步缓存封装（基于官方 redis-py / redis.asyncio）。

    Notes
    -----
    本类用于统一封装 Redis 的常见操作，避免在业务代码中
    直接操作 redis 客户端，提升可维护性与可读性。

    适用于以下场景：
    - Agent 会话状态缓存
    - 任务 / 工作流状态（FSM）
    - Prompt / 配置缓存
    - 幂等控制 / 去重
    """

    def __init__(self, host: str, password: str) -> None:
        self.client = redis.Redis(
            host=host,
            port=14381,
            decode_responses=True,
            username="default",
            password=password
        )

    async def get(self, key: str) -> typing.Optional[typing.Union[dict, list, str, int, float]]:
        """获取字符串类型的值。"""
        if (val := await self.client.get(key)) is None:
            return None
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError) as e:
            from loguru import logger
            logger.error(e)
            return val

    async def set(self, key: str, value: str, *, ttl: typing.Optional[int] = None) -> typing.Optional[bool]:
        """设置字符串类型的值。"""
        json.dumps(value, ensure_ascii=False)
        return bool(await self.client.set(key, value, ex=ttl))

    async def delete(self, key: str) -> typing.Optional[int]:
        """删除指定 Key。"""
        return int(await self.client.delete(key))

    async def exists(self, key: str) -> bool:
        """判断 Key 是否存在。"""
        return bool(await self.client.exists(key))

    async def expire(self, key: str, seconds: int) -> bool:
        """设置 Key 的过期时间。"""
        return bool(await self.client.expire(key, seconds))

    async def ttl(self, key: str) -> int:
        """获取 Key 剩余生存时间。"""
        return int(await self.client.ttl(key))


if __name__ == '__main__':
    pass
