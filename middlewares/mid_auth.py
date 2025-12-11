#     _         _   _       __  __ _     _     _ _
#    / \  _   _| |_| |__   |  \/  (_) __| | __| | | _____      ____ _ _ __ ___
#   / _ \| | | | __| '_ \  | |\/| | |/ _` |/ _` | |/ _ \ \ /\ / / _` | '__/ _ \
#  / ___ \ |_| | |_| | | | | |  | | | (_| | (_| | |  __/\ V  V / (_| | | |  __/
# /_/   \_\__,_|\__|_| |_| |_|  |_|_|\__,_|\__,_|_|\___| \_/\_/ \__,_|_|  \___|
#

import typing
from loguru import logger
from fastapi import Request
from schemas.errors import AuthorizationError
from utils import (
    const,toolset
)


async def auth_middleware(
    request: Request,
    call_next: typing.Callable
) -> typing.Any:
    """鉴权中间件"""

    if not (token := request.headers.get(const.AUTH_KEY)):
        logger.error(
            f"🚫 Missing credentials — {const.AUTH_KEY}={token}"
        )
        raise AuthorizationError(
            status_code=401, detail="Missing authentication credentials"
        )

    try:
        toolset.verify_token(token)
    except Exception as e:
        logger.error(f"❗ Token verification failed, Reason={e}")
        raise AuthorizationError(
            status_code=403, detail="Invalid token"
        )

    return await call_next(request)


if __name__ == '__main__':
    pass
