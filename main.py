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

# Notes: ==== Volume ====

"""
# 查看 Modal 全部 volume:
modal volume list

# 查看某一个 volume 的内容:
modal volume contents <volume-name>

# 删除 volume（慎用⚠️）:
modal volume delete <volume-name>

# 创建 Volume 目录:
modal volume create cross-enc-cache
modal volume create bge-en-cache
modal volume create bge-zh-cache
modal volume create sequence-cache
modal volume create ocr-cache

# 本地上传文件到 Volume:
modal volume put cross-enc-cache ./models/cross_encoder /root/models/cross_encoder
modal volume put bge-en-cache ./models/bge_base_en /root/models/bge_base_en
modal volume put bge-zh-cache ./models/bge_base_zh /root/models/bge_base_zh
modal volume put sequence-cache ./models/sequence /root/models/sequence
modal volume put ocr-cache ./models/paddle /root/models/paddle
"""


if __name__ == '__main__':
    pass
