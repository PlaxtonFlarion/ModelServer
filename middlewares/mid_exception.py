#  _____                    _   _               __  __ _     _     _ _
# | ____|_  _____ ___ _ __ | |_(_) ___  _ __   |  \/  (_) __| | __| | | _____      ____ _ _ __ ___
# |  _| \ \/ / __/ _ \ '_ \| __| |/ _ \| '_ \  | |\/| | |/ _` |/ _` | |/ _ \ \ /\ / / _` | '__/ _ \
# | |___ >  < (_|  __/ |_) | |_| | (_) | | | | | |  | | | (_| | (_| | |  __/\ V  V / (_| | | |  __/
# |_____/_/\_\___\___| .__/ \__|_|\___/|_| |_| |_|  |_|_|\__,_|\__,_|_|\___| \_/\_/ \__,_|_|  \___|
#                    |_|
#

import uuid
import typing
from loguru import logger
from fastapi import (
    Request, HTTPException
)
from fastapi.responses import JSONResponse
from schemas.errors import (
    AuthorizationError, BizError
)
from modal.exception import InvalidError


async def exception_middleware(
    request: Request,
    call_next: typing.Callable
) -> typing.Any:
    """异常中间件"""

    trace_id = getattr(request.state, "trace_id", str(uuid.uuid4()))
    request.state.trace_id = trace_id

    try:
        return await call_next(request)

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
                "error"    : "INVALID ERROR",
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
                "error"    : "INTERNAL ERROR",
                "details"  : str(e),
                "type"     : e.__class__.__name__,
                "trace_id" : trace_id
            },
            status_code=500
        )


if __name__ == '__main__':
    pass
