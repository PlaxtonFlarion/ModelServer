#     _         _   _       __  __ _     _     _ _
#    / \  _   _| |_| |__   |  \/  (_) __| | __| | | _____      ____ _ _ __ ___
#   / _ \| | | | __| '_ \  | |\/| | |/ _` |/ _` | |/ _ \ \ /\ / / _` | '__/ _ \
#  / ___ \ |_| | |_| | | | | |  | | | (_| | (_| | |  __/\ V  V / (_| | | |  __/
# /_/   \_\__,_|\__|_| |_| |_|  |_|_|\__,_|\__,_|_|\___| \_/\_/ \__,_|_|  \___|
#

import typing
from loguru import logger
from fastapi import Request
from schemas.cognitive import Mix
from schemas.errors import AuthorizationError
from services.infrastructure.cache.redis_cache import RedisCache
from utils import (
    const,toolset
)


async def auth_middleware(
    request: Request,
    call_next: typing.Callable
) -> typing.Any:
    """鉴权中间件"""

    cache: RedisCache = request.app.state.cache

    if mixed := await cache.get(const.K_MIX): mix = Mix(**mixed)
    else: mix = Mix(**const.V_MIX)

    public_paths = mix.white_list
    logger.info(f"远程鉴权白名单 -> {public_paths}")

    if request.url.path in public_paths:
        return await call_next(request)

    if not (token := request.headers.get(const.AUTH_KEY)):
        logger.error(
            f"🚫 Missing credentials — {const.AUTH_KEY}={token}"
        )
        raise AuthorizationError(
            status_code=401, detail="Missing authentication credentials"
        )

    try:
        toolset.verify_token(request, token)
    except Exception as e:
        logger.error(f"❗ Token verification failed, Reason={e}")
        raise AuthorizationError(
            status_code=403, detail="Invalid token"
        )

    return await call_next(request)


if __name__ == '__main__':
    pass
