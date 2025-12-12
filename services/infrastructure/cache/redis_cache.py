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
        self.pool = redis.ConnectionPool(
            host=host,
            port=14381,
            username="default",
            password=password,
            decode_responses=True,
            max_connections=50,
        )
        self.client = redis.Redis(connection_pool=self.pool)

    async def get(self, key: str) -> typing.Optional[str]:
        """
        获取字符串类型的值。

        Parameters
        ----------
        key : str
            Redis Key

        Returns
        -------
        typing.Optional[str]
            Key 对应的字符串值；若不存在则返回 None
        """

        return await self.client.get(key)

    async def set(
        self,
        key: str,
        value: str,
        *,
        ttl: typing.Optional[int] = None,
    ) -> bool:
        """
        设置字符串类型的值。

        Parameters
        ----------
        key : str
            Redis Key
        value : str
            要存储的字符串值
        ttl : typing.Optional[int], optional
            过期时间（秒），None 表示不过期

        Returns
        -------
        bool
            是否设置成功
        """

        return bool(await self.client.set(key, value, ex=ttl))

    async def delete(self, key: str) -> int:
        """
        删除指定 Key。

        Parameters
        ----------
        key : str
            Redis Key

        Returns
        -------
        int
            实际删除的 Key 数量（0 或 1）
        """

        return int(await self.client.delete(key))

    async def exists(self, key: str) -> bool:
        """
        判断 Key 是否存在。

        Parameters
        ----------
        key : str
            Redis Key

        Returns
        -------
        bool
            是否存在
        """

        return bool(await self.client.exists(key))

    async def get_json(self, key: str) -> typing.Optional[dict[str, typing.Any]]:
        """
        获取 JSON 对象（底层以字符串形式存储）。

        Parameters
        ----------
        key : str
            Redis Key

        Returns
        -------
        typing.Optional[dict[str, Any]]
            解析后的字典对象；若不存在则返回 None
        """

        raw: typing.Optional[str] = await self.get(key)
        return json.loads(raw) if raw else None

    async def set_json(
        self,
        key: str,
        value: dict[str, typing.Any],
        *,
        ttl: typing.Optional[int] = None,
    ) -> bool:
        """
        设置 JSON 对象（自动序列化为字符串）。

        Parameters
        ----------
        key : str
            Redis Key
        value : dict[str, typing.Any]
            要存储的字典对象
        ttl : typing.Optional[int], optional
            过期时间（秒），None 表示不过期

        Returns
        -------
        bool
            是否设置成功
        """

        return await self.set(key, json.dumps(value, ensure_ascii=False), ttl=ttl)

    async def hset(
        self,
        key: str,
        mapping: dict[str, typing.Any],
    ) -> int:
        """
        设置 Hash 类型字段（字段值自动 JSON 序列化）。

        Parameters
        ----------
        key : str
            Redis Hash Key
        mapping : dict[str, typing.Any]
            字段映射表（field -> value）

        Returns
        -------
        int
            新增字段数量
        """

        data = {k: json.dumps(v, ensure_ascii=False) for k, v in mapping.items()}
        return int(await self.client.hset(key, mapping=data))

    async def hget(self, key: str, field: str) -> typing.Optional[typing.Any]:
        """
        获取 Hash 中的单个字段值。

        Parameters
        ----------
        key : str
            Redis Hash Key
        field : str
            字段名

        Returns
        -------
        typing.Optional[typing.Any]
            反序列化后的字段值；若不存在返回 None
        """

        raw = await self.client.hget(key, field)
        return json.loads(raw) if raw else None

    async def hgetall(self, key: str) -> dict[str, typing.Any]:
        """
        获取 Hash 中的所有字段。

        Parameters
        ----------
        key : str
            Redis Hash Key

        Returns
        -------
        dict[str, typing.Any]
            字段名 -> 反序列化后的值
        """

        raw: dict[str, str] = await self.client.hgetall(key)
        return {k: json.loads(v) for k, v in raw.items()}

    async def hdel(self, key: str, *fields: str) -> int:
        """
        删除 Hash 中的指定字段。

        Parameters
        ----------
        key : str
            Redis Hash Key
        *fields : str
            要删除的字段名列表

        Returns
        -------
        int
            实际删除的字段数量
        """

        return int(await self.client.hdel(key, *fields))

    async def lpush(self, key: str, *values: str) -> int:
        """
        从左侧推入列表（栈 / 队列）。

        Returns
        -------
        int
            列表当前长度
        """
        return int(await self.client.lpush(key, *values))

    async def rpush(self, key: str, *values: str) -> int:
        """
        从右侧推入列表。

        Returns
        -------
        int
            列表当前长度
        """

        return int(await self.client.rpush(key, *values))

    async def lpop(self, key: str) -> typing.Optional[str]:
        """
        从左侧弹出一个元素。

        Returns
        -------
        typing.Optional[str]
            弹出的值；若列表为空返回 None
        """

        return await self.client.lpop(key)

    async def rpop(self, key: str) -> typing.Optional[str]:
        """
        从右侧弹出一个元素。

        Returns
        -------
        typing.Optional[str]
            弹出的值；若列表为空返回 None
        """

        return await self.client.rpop(key)

    async def lrange(self, key: str, start: int = 0, end: int = -1) -> list[str]:
        """
        获取列表范围内的元素。

        Returns
        -------
        list[str]
            列表元素集合
        """

        return list(await self.client.lrange(key, start, end))

    async def sadd(self, key: str, *members: str) -> int:
        """
        向集合中添加成员。

        Returns
        -------
        int
            新增成员数量
        """

        return int(await self.client.sadd(key, *members))

    async def srem(self, key: str, *members: str) -> int:
        """
        从集合中移除成员。

        Returns
        -------
        int
            实际移除的成员数量
        """

        return int(await self.client.srem(key, *members))

    async def smembers(self, key: str) -> typing.Set[str]:
        """
        获取集合中的所有成员。

        Returns
        -------
        Set[str]
            成员集合
        """

        return set(await self.client.smembers(key))

    async def sismember(self, key: str, member: str) -> bool:
        """
        判断成员是否存在于集合中。

        Returns
        -------
        bool
            是否存在
        """

        return bool(await self.client.sismember(key, member))

    async def expire(self, key: str, seconds: int) -> bool:
        """
        设置 Key 的过期时间。

        Returns
        -------
        bool
            是否设置成功
        """

        return bool(await self.client.expire(key, seconds))

    async def ttl(self, key: str) -> int:
        """
        获取 Key 剩余生存时间。

        Returns
        -------
        int
            剩余秒数；-1 表示永不过期；-2 表示不存在
        """

        return int(await self.client.ttl(key))

    async def publish(self, channel: str, message: str) -> int:
        """
        向频道发布消息。

        Returns
        -------
        int
            接收到消息的订阅者数量
        """

        return int(await self.client.publish(channel, message))

    async def subscribe(self, channel: str) -> redis.client.PubSub:
        """
        订阅频道。

        Returns
        -------
        redis.client.PubSub
            PubSub 对象，用于监听消息
        """

        await (pubsub := self.client.pubsub()).subscribe(channel)
        return pubsub

    async def close(self) -> None:
        """
        关闭 Redis 连接。
        """

        await self.client.close()


if __name__ == '__main__':
    pass
