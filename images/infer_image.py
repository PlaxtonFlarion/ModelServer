#  ___        __             ___
# |_ _|_ __  / _| ___ _ __  |_ _|_ __ ___   __ _  __ _  ___
#  | || '_ \| |_ / _ \ '__|  | || '_ ` _ \ / _` |/ _` |/ _ \
#  | || | | |  _|  __/ |     | || | | | | | (_| | (_| |  __/
# |___|_| |_|_|  \___|_|    |___|_| |_| |_|\__,_|\__, |\___|
#                                                |___/
#

import modal
from utils import const

image = modal.Image.debian_slim(
    "3.11"
).pip_install(
    const.BASE_DEPENDENCIES
).pip_install(
    const.INFERENCE_DEPENDENCIES
).apt_install(
    "libgl1", "libglib2.0-0", "ffmpeg"
).add_local_dir(
    ".", "/root", ignore=const.IGNORE
).add_local_dir(
    "models/sequence", "/root/models/sequence"
)
secrets = [
    modal.Secret.from_name("SHARED_SECRET")
]


if __name__ == '__main__':
    pass
