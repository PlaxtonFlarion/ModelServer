#  __  __ _     _     _ _
# |  \/  (_) __| | __| | | _____      ____ _ _ __ ___  ___
# | |\/| | |/ _` |/ _` | |/ _ \ \ /\ / / _` | '__/ _ \/ __|
# | |  | | | (_| | (_| | |  __/\ V  V / (_| | | |  __/\__ \
# |_|  |_|_|\__,_|\__,_|_|\___| \_/\_/ \__,_|_|  \___||___/
#

from fastapi import FastAPI

from .mid_auth      import auth_middleware
from .mid_exception import exception_middleware
from .mid_trace     import trace_middleware


def register_middlewares(web_app: FastAPI):
    web_app.middleware("http")(trace_middleware      )
    web_app.middleware("http")(auth_middleware       )
    web_app.middleware("http")(exception_middleware  )


if __name__ == '__main__':
    pass
