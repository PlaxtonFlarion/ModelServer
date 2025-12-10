#  ___        __                                ___
# |_ _|_ __  / _| ___ _ __ ___ _ __   ___ ___  |_ _|_ __ ___   __ _  __ _  ___
#  | || '_ \| |_ / _ \ '__/ _ \ '_ \ / __/ _ \  | || '_ ` _ \ / _` |/ _` |/ _ \
#  | || | | |  _|  __/ | |  __/ | | | (_|  __/  | || | | | | | (_| | (_| |  __/
# |___|_| |_|_|  \___|_|  \___|_| |_|\___\___| |___|_| |_| |_|\__,_|\__, |\___|
#                                                                   |___/
#

from modal import (
    Image, Secret
)
from utils import const

image = Image.debian_slim(
    "3.11"
).pip_install(
    const.INFERENCE_DEPENDENCIES
).apt_install(
    "libgl1", "libglib2.0-0", "ffmpeg"
).add_local_dir(
    ".", "/root", ignore=["models/", "venv/", ".venv/"]
).add_local_dir(
    "models/sequence", "/root/models/sequence"
)
secret = Secret.from_name("SHARED_SECRET")


if __name__ == '__main__':
    pass
