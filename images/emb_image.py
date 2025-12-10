#  _____           _       ___
# | ____|_ __ ___ | |__   |_ _|_ __ ___   __ _  __ _  ___
# |  _| | '_ ` _ \| '_ \   | || '_ ` _ \ / _` |/ _` |/ _ \
# | |___| | | | | | |_) |  | || | | | | | (_| | (_| |  __/
# |_____|_| |_| |_|_.__/  |___|_| |_| |_|\__,_|\__, |\___|
#                                              |___/

import modal
from utils import const

image = modal.Image.debian_slim(
    "3.11"
).pip_install(
    const.EMBEDDING_DEPENDENCIES
)
secret = modal.Secret.from_name("SHARED_SECRET")
mounts = [
    modal.Mount.from_local_dir("apps", remote_path="/root/apps"),
    modal.Mount.from_local_dir("images", remote_path="/root/images"),
    modal.Mount.from_local_dir("middlewares", remote_path="/root/middlewares"),
    modal.Mount.from_local_dir("schemas", remote_path="/root/schemas"),
    modal.Mount.from_local_dir("services", remote_path="/root/services"),
    modal.Mount.from_local_dir("utils", remote_path="/root/utils")
]


if __name__ == '__main__':
    pass
