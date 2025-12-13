#     _
#    / \   _ __  _ __
#   / _ \ | '_ \| '_ \
#  / ___ \| |_) | |_) |
# /_/   \_\ .__/| .__/
#         |_|   |_|
#

import modal

from apps.cross_enc   import app as cross_enc_app
from apps.embedding   import app as embedding_app
from apps.infer_color import app as infer_color_app
from apps.infer_faint import app as infer_faint_app
from apps.yolo        import app as yolo_app

from utils import const

app = modal.App(const.GROUP_FUNC)

app.include(cross_enc_app    )
app.include(embedding_app    )
app.include(infer_color_app  )
app.include(infer_faint_app  )
app.include(yolo_app   )


if __name__ == '__main__':
    pass
