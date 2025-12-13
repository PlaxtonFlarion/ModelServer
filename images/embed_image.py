#  _____           _              _   ___
# | ____|_ __ ___ | |__   ___  __| | |_ _|_ __ ___   __ _  __ _  ___
# |  _| | '_ ` _ \| '_ \ / _ \/ _` |  | || '_ ` _ \ / _` |/ _` |/ _ \
# | |___| | | | | | |_) |  __/ (_| |  | || | | | | | (_| | (_| |  __/
# |_____|_| |_| |_|_.__/ \___|\__,_| |___|_| |_| |_|\__,_|\__, |\___|
#                                                         |___/
#

import modal
from utils import const

image = modal.Image.debian_slim(
    "3.11"
).run_commands(
    const.COMMANDS
).pip_install(
    const.EMBEDDING_DEPENDENCIES
).add_local_dir(
    ".", "/root", ignore=const.IGNORE
).add_local_dir(
    "models/bge_m3", "/root/models/bge_m3"
).add_local_dir(
    "models/cross_encoder", "/root/models/cross_encoder"
)
secrets = [
    modal.Secret.from_name("SHARED_SECRET")
]


if __name__ == '__main__':
    pass
