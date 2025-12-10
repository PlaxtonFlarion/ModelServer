#  __  __       _
# |  \/  | __ _(_)_ __
# | |\/| |/ _` | | '_ \
# | |  | | (_| | | | | |
# |_|  |_|\__,_|_|_| |_|
#
# Notes: ==== https://modal.com/ ====

"""
统一入口 — 支持 modal deploy main.py 一键部署所有服务

注意：
  - 只负责 import，不包含业务逻辑
  - 每个 service 文件中必须自带相同的 modal.App("xxx")
  - deploy 时所有相关 cls 都会在 Modal 上创建/更新
"""


# Notes: ==== 合并部署 ====

import modal
from apps.cross_enc import app as cross_enc_app
from apps.emb_en    import app as embedding_en
from apps.emb_zh    import app as embedding_zh
from apps.inference import app as inference_app

app = modal.App("apps")
app.include(cross_enc_app  )
app.include(embedding_en   )
app.include(embedding_zh   )
app.include(inference_app  )

"""
部署方式:
    modal deploy main.py

启动本地调试:
    modal run main.py

列出函数/cls:
    modal app list
"""


if __name__ == '__main__':
    pass
