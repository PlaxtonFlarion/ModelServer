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
).run_commands(
    const.COMMANDS
).apt_install(
    "libgl1", "libglib2.0-0", "ffmpeg"
).pip_install(
    const.YOLO_DEPENDENCIES
).env(
    {"YOLO_CONFIG_DIR": "/tmp/Ultralytics"}
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
