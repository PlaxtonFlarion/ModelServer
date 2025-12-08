#     _         _   _
#    / \  _   _| |_| |__
#   / _ \| | | | __| '_ \
#  / ___ \ |_| | |_| | | |
# /_/   \_\__,_|\__|_| |_|
#

import uuid
from functools import wraps
from fastapi import HTTPException
from utils import toolset


def auth_middleware(key: str = "X-Token"):
    """鉴权中间件"""

    def decorator(func):

        @wraps(func)
        async def wrapper(*args, **kwargs):
            if "request" in kwargs: request = kwargs["request"]
            else: request = args[1] if hasattr(args[0], "__dict__") else args[0]

            trace_id = uuid.uuid4().hex[:8]

            if not (token := request.headers.get(key)):
                raise HTTPException(
                    status_code=401,
                    detail={"error": "TOKEN_MISSING", "trace_id": trace_id}
                )

            try:
                toolset.verify_token(token)
            except Exception as e:
                raise HTTPException(
                    status_code=403,
                    detail={"error": "TOKEN_INVALID", "msg": str(e), "trace_id": trace_id}
                )

            return await func(*args, **kwargs)

        return wrapper

    return decorator


if __name__ == '__main__':
    pass
