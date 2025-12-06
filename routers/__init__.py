#  ____             _
# |  _ \ ___  _   _| |_ ___ _ __ ___
# | |_) / _ \| | | | __/ _ \ '__/ __|
# |  _ < (_) | |_| | ||  __/ |  \__ \
# |_| \_\___/ \__,_|\__\___|_|  |___/
#

import modal

from apps.embedding import app as embedding_app
from apps.inference import app as inference_app


def register_routers(app: "modal.App") -> None:
    app.include(embedding_app)
    app.include(inference_app)

    @app.function()
    def health():
        return {
            "status": "ok",
            "services": ["embedding_app", "inference_app"]
        }


if __name__ == '__main__':
    pass
