#  __  __ _     _     _ _
# |  \/  (_) __| | __| | | _____      ____ _ _ __ ___  ___
# | |\/| | |/ _` |/ _` | |/ _ \ \ /\ / / _` | '__/ _ \/ __|
# | |  | | | (_| | (_| | |  __/\ V  V / (_| | | |  __/\__ \
# |_|  |_|_|\__,_|\__,_|_|\___| \_/\_/ \__,_|_|  \___||___/
#

from fastapi import FastAPI

from .mid_access     import access_middleware
from .mid_auth       import auth_middleware
from .mid_exception  import exception_middleware
from .mid_rate_limit import rate_limit_middleware


def register_middlewares(web_app: FastAPI):
    # inbound 1（最外层）
    web_app.middleware("http")(rate_limit_middleware  )
    # inbound 2
    web_app.middleware("http")(auth_middleware        )
    # inbound 3
    web_app.middleware("http")(access_middleware      )
    # inbound 4（最内层）
    web_app.middleware("http")(exception_middleware   )


if __name__ == '__main__':
    pass
