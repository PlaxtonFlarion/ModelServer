# __   __    _         ___
# \ \ / /__ | | ___   |_ _|_ __ ___   __ _  __ _  ___
#  \ V / _ \| |/ _ \   | || '_ ` _ \ / _` |/ _` |/ _ \
#   | | (_) | | (_) |  | || | | | | | (_| | (_| |  __/
#   |_|\___/|_|\___/  |___|_| |_| |_|\__,_|\__, |\___|
#                                          |___/
#

import modal
from utils import const

image = modal.Image.debian_slim(
    "3.11"
).pip_install(
    const.BASE_DEPENDENCIES
).pip_install(
    const.YOLO_DEPENDENCIES
).add_local_dir(
    ".", "/root", ignore=const.IGNORE
).add_local_dir(
    "models/yolo_11", "/root/models/yolo_11"
)
secrets = [
    modal.Secret.from_name("SHARED_SECRET")
]


if __name__ == '__main__':
    pass
