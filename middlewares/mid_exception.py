#  _____                    _   _               __  __ _     _     _ _
# | ____|_  _____ ___ _ __ | |_(_) ___  _ __   |  \/  (_) __| | __| | | _____      ____ _ _ __ ___
# |  _| \ \/ / __/ _ \ '_ \| __| |/ _ \| '_ \  | |\/| | |/ _` |/ _` | |/ _ \ \ /\ / / _` | '__/ _ \
# | |___ >  < (_|  __/ |_) | |_| | (_) | | | | | |  | | | (_| | (_| | |  __/\ V  V / (_| | | |  __/
# |_____/_/\_\___\___| .__/ \__|_|\___/|_| |_| |_|  |_|_|\__,_|\__,_|_|\___| \_/\_/ \__,_|_|  \___|
#                    |_|
#

import uuid
import inspect
from functools import wraps
from loguru import logger
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from schemas.errors import (
    AuthorizationError, BizError
)
from modal.exception import InvalidError


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

        except (AuthorizationError, BizError) as e:
            logger.error(
                f"[{trace_id}] ⚠️ {e.status_code} {request.method} {request.url.path} → {e.detail}"
            )
            return JSONResponse(
                content={
                    "error"    : "FATAL",
                    "details"  : e.detail,
                    "type"     : e.__class__.__name__,
                    "trace_id" : trace_id
                },
                status_code=e.status_code
            )

        except InvalidError as e:
            logger.error(
                f"[{trace_id}] ⚠️ {request.method} {request.url.path} → {e}"
            )
            return JSONResponse(
                status_code=502,
                content={
                    "error"    : "MODAL CALL FAILED",
                    "details"  : str(e),
                    "type"     : e.__class__.__name__,
                    "trace_id" : trace_id
                }
            )

        except HTTPException as e:
            logger.error(
                f"[{trace_id}] ⚠️ {e.status_code} {request.method} {request.url.path} → {e.detail}"
            )
            return JSONResponse(
                content={
                    "error"    : "HTTP EXCEPTION",
                    "details"  : e.detail,
                    "type"     : e.__class__.__name__,
                    "trace_id" : trace_id
                },
                status_code=e.status_code
            )

        except Exception as e:
            logger.error(
                f"[{trace_id}] ❌ Unhandled Exception: {e}"
            )
            return JSONResponse(
                content={
                    "error"    : "INTERNAL_ERROR",
                    "details"  : str(e),
                    "type"     : e.__class__.__name__,
                    "trace_id" : trace_id
                },
                status_code=500
            )

    return wrapper


if __name__ == '__main__':
    pass
