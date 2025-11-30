#     _         _   _       __  __ _     _     _ _
#    / \  _   _| |_| |__   |  \/  (_) __| | __| | | _____      ____ _ _ __ ___
#   / _ \| | | | __| '_ \  | |\/| | |/ _` |/ _` | |/ _ \ \ /\ / / _` | '__/ _ \
#  / ___ \ |_| | |_| | | | | |  | | | (_| | (_| | |  __/\ V  V / (_| | | |  __/
# /_/   \_\__,_|\__|_| |_| |_|  |_|_|\__,_|\__,_|_|\___| \_/\_/ \__,_|_|  \___|
#

import os
import hmac
import time
import base64
import typing
import hashlib
from functools import wraps
from loguru import logger
from fastapi import Request
from fastapi.responses import JSONResponse


def auth_middleware(key: str = "X-Token"):
    """鉴权中间件"""

    def decorator(func):
        @wraps(func)
        async def wrapper(self, request: "Request", *args, **kwargs):
            token = request.headers.get(key)
            verify_token(token)
            return await func(self, request, *args, **kwargs)
        return wrapper
    return decorator


def verify_token(token: str) -> typing.Union["JSONResponse", bool]:
    """鉴权"""

    shared_secret = os.environ["SHARED_SECRET"]

    logger.info(f"Verify token: {token}")
    if not token:
        return JSONResponse(content={"error": "Unauthorized", "message": "Token missing"}, status_code=401)

    try:
        payload, sig      = token.rsplit(".", 1)
        app_id, expire_at = payload.split(":")

        if time.time() > int(expire_at):
            logger.warning("Token has expired")
            return JSONResponse(content={"error": "Token has expired"}, status_code=401)

        expected_sig = hmac.new(
            shared_secret.encode(), payload.encode(), hashlib.sha256
        ).digest()
        expected_b64 = base64.b64encode(expected_sig).decode()

        if not (compare := hmac.compare_digest(expected_b64, sig)):
            logger.warning("Token signature mismatch")
            return JSONResponse(content={"error": "Invalid token signature"}, status_code=401)
        return compare

    except ValueError as e:
        logger.error(e)
        return JSONResponse(content={"error": "Malformed token"}, status_code=401)
    except Exception as e:
        logger.error(e)
        return JSONResponse(content={"error": "Unauthorized", "message" : str(e)}, status_code=401)


if __name__ == '__main__':
    pass
