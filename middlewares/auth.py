import hmac
import time
import base64
import typing
import hashlib
from functools import wraps
from loguru import logger
from fastapi.responses import JSONResponse


def require_token(shared_secret: str, header_key: str = "X-Token"):
    """鉴权中间件"""

    def decorator(func):
        @wraps(func)
        async def wrapper(request, *args, **kwargs):
            token = request.headers.get(header_key)
            verify_token(shared_secret, token)
            return await func(request, *args, **kwargs)
        return wrapper
    return decorator


def verify_token(shared_secret: str, token: str) -> typing.Union["JSONResponse", bool]:
    """鉴权"""
    logger.info(f"==== @ ====")
    logger.info(f"Verify token: {token}")
    if not token:
        return JSONResponse(
            content={
                "Error"   : "Unauthorized",
                "Message" : "Token missing"
            },
            status_code=401
        )

    try:
        payload, sig      = token.rsplit(".", 1)
        app_id, expire_at = payload.split(":")

        if time.time() > int(expire_at):
            logger.warning("Token has expired")
            return JSONResponse(
                content={
                    "Error": "Token has expired"
                },
                status_code=401
            )

        expected_sig = hmac.new(
            shared_secret.encode(), payload.encode(), hashlib.sha256
        ).digest()
        expected_b64 = base64.b64encode(expected_sig).decode()

        if not (compare := hmac.compare_digest(expected_b64, sig)):
            logger.warning("Token signature mismatch")
            return JSONResponse(
                content={
                    "Error": "Invalid token signature"
                },
                status_code=401
            )
        return compare

    except ValueError as e:
        logger.error(e)
        return JSONResponse(
            content={
                "Error": "Malformed token"
            },
            status_code=401
        )

    except Exception as e:
        logger.error(e)
        return JSONResponse(
            content={
                "Error"   : "Unauthorized",
                "Message" : str(e)
            },
            status_code=401
        )


if __name__ == '__main__':
    pass
