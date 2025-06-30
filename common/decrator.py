#   ____                      _
#  |  _ \  ___  ___ _ __ __ _| |_ ___  _ __
#  | | | |/ _ \/ __| '__/ _` | __/ _ \| '__|
#  | |_| |  __/ (__| | | (_| | || (_) | |
#  |____/ \___|\___|_|  \__,_|\__\___/|_|
#

import json
import modal
from loguru import logger
from functools import wraps
from fastapi.responses import JSONResponse
from starlette.requests import ClientDisconnect


def require_token(header_key: str = "X-Token"):
    """
    参数化装饰器：校验请求头中的签名 Token。
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(self, request, *args, **kwargs):
            token = request.headers.get(header_key)
            self.verify_token(token)
            return await func(self, request, *args, **kwargs)
        return wrapper
    return decorator


def with_exception_handling(func):
    """
    装饰器：用于 Modal 的 fastapi_endpoint 接口，统一异常处理
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except ClientDisconnect as e:
            logger.error(e)
            return JSONResponse(
                content={"error": "Client disconnected during upload"}, status_code=499
            )
        except json.JSONDecodeError as e:
            logger.error(e)
            return JSONResponse(
                content={"error": "Invalid JSON payload"}, status_code=400
            )
        except modal.exception.InvalidError as e:
            logger.error(e)
            return JSONResponse(
                content={"error": f"Modal error: {str(e)}"}, status_code=500
            )
        except Exception as e:
            logger.error(e)
            return JSONResponse(
                content={"error": f"Unexpected error: {str(e)}"}, status_code=500
            )
    return wrapper


if __name__ == '__main__':
    pass