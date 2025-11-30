import json
import modal
from functools import wraps
from loguru import logger
from fastapi.responses import JSONResponse
from starlette.requests import ClientDisconnect


def with_exception_handling(func):
    """异常中间件"""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except ClientDisconnect as e:
            logger.error(e)
            return JSONResponse(
                content={
                    "Error": "Client disconnected during upload"
                },
                status_code=499
            )
        except json.JSONDecodeError as e:
            logger.error(e)
            return JSONResponse(
                content={
                    "Error": "Invalid JSON payload"
                },
                status_code=400
            )
        except modal.exception.InvalidError as e:
            logger.error(e)
            return JSONResponse(
                content={
                    "Error": f"Modal error: {str(e)}"
                },
                status_code=500
            )
        except Exception as e:
            logger.error(e)
            return JSONResponse(
                content={
                    "Error": f"Unexpected error: {str(e)}"
                },
                status_code=500
            )
    return wrapper


if __name__ == '__main__':
    pass
