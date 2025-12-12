#  __  __       _
# |  \/  | __ _(_)_ __
# | |\/| |/ _` | | '_ \
# | |  | | (_| | | | | |
# |_|  |_|\__,_|_|_| |_|
#
# Notes: ==== https://modal.com/ ====

import os
import modal

from fastapi import FastAPI

from services.infrastructure.cache.redis_cache import RedisCache

from middlewares import register_middlewares
from routers     import register_routers

from images.base_image import (
    image, secrets
)
from utils import (
    const, toolset
)

app = modal.App(const.GROUP_MAIN)


@app.function(
    image=image,
    secrets=secrets,
    memory=4096,
    max_containers=5,
    scaledown_window=300
)
@modal.asgi_app(label="web-app")
def api_main():

    web_app = FastAPI()
    web_app.state.cache = RedisCache(
        os.environ["REDIS_URL"], os.environ["REDIS_KEY"]
    )
    web_app.state.shared_secret = os.environ["SHARED_SECRET"]

    toolset.init_logger()

    register_middlewares(web_app)
    register_routers(web_app)

    return web_app

"""
部署方式:
    modal deploy main.py
    modal deploy apps/app.py

启动本地调试:
    modal run main.py
    modal run apps/app.py

列出函数/cls:
    modal app list
"""


if __name__ == '__main__':
    pass
