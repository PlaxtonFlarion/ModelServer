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
from apps.embedding import app as embedding_app
from apps.inference import app as inference_app

app = modal.App("apps")
app.include(embedding_app)
app.include(inference_app)

"""
部署方式：
    modal deploy main.py

启动本地调试：
    modal run main.py
"""


# Notes: ==== 单独部署 ====

"""
部署方式：
    modal deploy apps/embedding.py
    modal deploy apps/inference.py

启动本地调试：
    modal run apps/embedding.py
    modal run apps/inference.py
"""


# Notes: ==== 部署说明 ====

"""
列出函数/cls：
    modal app list
"""


if __name__ == '__main__':
    pass
