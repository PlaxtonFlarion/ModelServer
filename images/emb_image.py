#  _____           _       ___
# | ____|_ __ ___ | |__   |_ _|_ __ ___   __ _  __ _  ___
# |  _| | '_ ` _ \| '_ \   | || '_ ` _ \ / _` |/ _` |/ _ \
# | |___| | | | | | |_) |  | || | | | | | (_| | (_| |  __/
# |_____|_| |_| |_|_.__/  |___|_| |_| |_|\__,_|\__, |\___|
#                                              |___/

from modal import (
    Image, Secret
)
from utils import const

image = Image.debian_slim(
    "3.11"
).pip_install(
    const.EMBEDDING_DEPENDENCIES
).add_local_dir(
    ".", "/root", ignore=["*.venv", "*venv", "*models"]
).add_local_dir(
    "models/bge_base_en", "/root/models/bge_base_en"
).add_local_dir(
    "models/bge_base_zh", "/root/models/bge_base_zh"
).add_local_dir(
    "models/cross_encoder", "/root/models/cross_encoder"
)
secret = Secret.from_name("SHARED_SECRET")


if __name__ == '__main__':
    pass
