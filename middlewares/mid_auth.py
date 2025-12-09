#     _         _   _       __  __ _     _     _ _
#    / \  _   _| |_| |__   |  \/  (_) __| | __| | | _____      ____ _ _ __ ___
#   / _ \| | | | __| '_ \  | |\/| | |/ _` |/ _` | |/ _ \ \ /\ / / _` | '__/ _ \
#  / ___ \ |_| | |_| | | | | |  | | | (_| | (_| | |  __/\ V  V / (_| | | |  __/
# /_/   \_\__,_|\__|_| |_| |_|  |_|_|\__,_|\__,_|_|\___| \_/\_/ \__,_|_|  \___|
#

from functools import wraps
from loguru import logger
from schemas.errors import AuthorizationError
from utils import toolset


def auth_middleware(key: str = "X-Token"):
    """鉴权中间件"""

    def decorator(func):

        @wraps(func)
        async def wrapper(*args, **kwargs):
            if "request" in kwargs: request = kwargs["request"]
            else: request = args[1] if hasattr(args[0], "__dict__") else args[0]

            if not (token := request.headers.get(key)):
                logger.error(
                    f"🚫 Missing credentials — {key}={token}"
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

            return await func(*args, **kwargs)

        return wrapper

    return decorator


if __name__ == '__main__':
    pass
