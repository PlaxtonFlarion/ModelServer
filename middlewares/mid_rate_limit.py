#  ____       _         _     _           _ _     __  __ _     _     _ _
# |  _ \ __ _| |_ ___  | |   (_)_ __ ___ (_) |_  |  \/  (_) __| | __| | | _____      ____ _ _ __ ___
# | |_) / _` | __/ _ \ | |   | | '_ ` _ \| | __| | |\/| | |/ _` |/ _` | |/ _ \ \ /\ / / _` | '__/ _ \
# |  _ < (_| | ||  __/ | |___| | | | | | | | |_  | |  | | | (_| | (_| | |  __/\ V  V / (_| | | |  __/
# |_| \_\__,_|\__\___| |_____|_|_| |_| |_|_|\__| |_|  |_|_|\__,_|\__,_|_|\___| \_/\_/ \__,_|_|  \___|
#

import json
import time
import typing
import asyncio
from loguru import logger
from fastapi import (
    Request, HTTPException
)
from services.infrastructure.cache.redis_cache import RedisCache
from utils import const


async def rate_limit_middleware(
    request: Request,
    call_next: typing.Callable
) -> typing.Any:
    """限流中间件"""

    cache: RedisCache = request.app.state.cache

    cache_key = f"RateConfig"

    if cached := await cache.get(cache_key): config = json.loads(cached)
    else: config = const.RATE_CONFIG
    logger.info(f"远程限流配置表 -> {config}")

    route = request.url.path
    ip    = request.client.host
    key   = f"tb:{hash(route)}:{ip}"

    default_config = config.get("default", {})
    route_config   = config.get("routes", {}).get(route, {})
    ip_config      = config.get("ip", {}).get(ip, {})

    # ==== 生成最终配置 (IP > route > default) ===
    final = {**default_config, **route_config, **ip_config}
    logger.info(f"RateRule={final}")

    burst    = final.get("burst",   10)
    rate     = final.get("rate",     2)
    max_wait = final.get("max_wait", 1)

    while True:
        now    = int(time.time() * 1000)
        tokens = await cache.client.eval(
            const.TOKEN_BUCKET_LUA, 1, key, burst, rate, now
        )

        if tokens >= 0:
            resp = await call_next(request)
            resp.headers["X-Rate-Limit"] = f"{burst} burst / {rate}/s"
            resp.headers["X-Rate-Remaining"] = str(round(tokens, 2))
            return resp

        if (wait := 1 / rate) > max_wait:
            raise HTTPException(
                status_code=429,
                detail={"error": "RATE_LIMIT_HIT", "rule": final, "retry": wait},
                headers={"Retry-After": str(round(wait))}
            )
        await asyncio.sleep(wait)


if __name__ == '__main__':
    pass
