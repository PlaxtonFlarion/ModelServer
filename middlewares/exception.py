#  _____                    _   _
# | ____|_  _____ ___ _ __ | |_(_) ___  _ __
# |  _| \ \/ / __/ _ \ '_ \| __| |/ _ \| '_ \
# | |___ >  < (_|  __/ |_) | |_| | (_) | | | |
# |_____/_/\_\___\___| .__/ \__|_|\___/|_| |_|
#                    |_|
#

import json
import uuid
import inspect
import traceback
from functools import wraps
from loguru import logger
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from modal.exception import (
    ClientClosed, InvalidError
)


def exception_middleware(func):
    """异常中间件"""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        sig = inspect.signature(func)
        has_self = "self" in sig.parameters

        if "request" in kwargs: request = kwargs["request"]
        else: request = args[1] if has_self else args[0]

        trace_id = uuid.uuid4().hex[:8]
        request.state.trace_id = trace_id

        try:
            return await func(*args, **kwargs)

        except HTTPException as e:
            logger.error(f"[{trace_id}] HTTPException: {e.detail}")
            return JSONResponse(
                status_code=e.status_code,
                content={"error": str(e.detail), "detail": str(e), "trace_id": trace_id}
            )

        except ClientClosed as e:
            logger.error(f"[{trace_id}] ClientClosed: {e}")
            return JSONResponse(
                status_code=499,
                content={"error": "CLIENT_CLOSED", "detail": str(e), "trace_id": trace_id}
            )

        except json.JSONDecodeError as e:
            logger.error(f"[{trace_id}] JSONDecodeError: {e}")
            return JSONResponse(
                status_code=400,
                content={"error": "INVALID_JSON", "detail": str(e), "trace_id": trace_id}
            )

        except InvalidError as e:
            logger.error(f"[{trace_id}] ModalError: {e}")
            return JSONResponse(
                status_code=502,
                content={"error": "MODAL_CALL_FAILED", "detail": str(e), "trace_id": trace_id}
            )

        except Exception as e:
            logger.error(f"[{trace_id}] Unexpected: {e}\n" + traceback.format_exc())
            return JSONResponse(
                status_code=500,
                content={"error": "INTERNAL_ERROR", "detail": str(e), "trace_id": trace_id}
            )

    return wrapper


if __name__ == '__main__':
    pass
