#  ____             _
# |  _ \ ___  _   _| |_ ___ _ __ ___
# | |_) / _ \| | | | __/ _ \ '__/ __|
# |  _ < (_) | |_| | ||  __/ |  \__ \
# |_| \_\___/ \__,_|\__\___|_|  |___/
#

from fastapi import FastAPI

from .rt_common    import common_router
from .rt_embedding import embedding_router
from .rt_inference import inference_router
from .rt_rerank    import rerank_router


def register_routers(web_app: FastAPI):
    web_app.include_router(common_router     )
    web_app.include_router(embedding_router  )
    web_app.include_router(inference_router  )
    web_app.include_router(rerank_router     )


if __name__ == '__main__':
    pass
