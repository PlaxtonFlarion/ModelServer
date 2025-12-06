#  _____                    _   _
# | ____|_  _____ ___ _ __ | |_(_) ___  _ __
# |  _| \ \/ / __/ _ \ '_ \| __| |/ _ \| '_ \
# | |___ >  < (_|  __/ |_) | |_| | (_) | | | |
# |_____/_/\_\___\___| .__/ \__|_|\___/|_| |_|
#                    |_|
#

import json
import modal
from functools import wraps
from loguru import logger
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.requests import ClientDisconnect


def exception_middleware(func):
    """异常中间件"""

    @wraps(func)
    async def wrapper(self, request: Request, *args, **kwargs):
        try:
            return await func(self, request, *args, **kwargs)
        except ClientDisconnect as e:
            logger.error(e)
            return JSONResponse(content={"error": f"Client disconnected during upload: {str(e)}"}, status_code=499)
        except json.JSONDecodeError as e:
            logger.error(e)
            return JSONResponse(content={"error": f"Invalid JSON payload: {str(e)}"}, status_code=400)
        except modal.exception.InvalidError as e:
            logger.error(e)
            return JSONResponse(content={"error": f"Modal error: {str(e)}"}, status_code=500)
        except Exception as e:
            logger.error(e)
            return JSONResponse(content={"error": f"Unexpected error: {str(e)}"}, status_code=500)

    return wrapper


if __name__ == '__main__':
    pass
