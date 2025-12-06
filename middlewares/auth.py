#     _         _   _
#    / \  _   _| |_| |__
#   / _ \| | | | __| '_ \
#  / ___ \ |_| | |_| | | |
# /_/   \_\__,_|\__|_| |_|
#

from functools import wraps
from fastapi import Request
from utils import toolset


def auth_middleware(key: str = "X-Token"):
    """鉴权中间件"""

    def decorator(func):

        @wraps(func)
        async def wrapper(self, request: Request, *args, **kwargs):
            token = request.headers.get(key)
            toolset.verify_token(token)
            return await func(self, request, *args, **kwargs)
        return wrapper

    return decorator


if __name__ == '__main__':
    pass
