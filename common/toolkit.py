#  _____           _ _    _ _
# |_   _|__   ___ | | | _(_) |_
#   | |/ _ \ / _ \| | |/ / | __|
#   | | (_) | (_) | |   <| | |_
#   |_|\___/ \___/|_|_|\_\_|\__|
#

import os
import time
import hmac
import base64
import typing
import hashlib
from loguru import logger
from fastapi.responses import JSONResponse


def judge_channel(shape: tuple[int, ...]) -> int:
    """色彩通道"""

    return shape[2] if len(shape) == 3 and shape[2] in (1, 3, 4) else \
        shape[0] if len(shape) == 3 and shape[0] in (1, 3, 4) else \
            1 if len(shape) == 2 else None


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
