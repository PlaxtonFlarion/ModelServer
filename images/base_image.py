#  ____                   ___
# | __ )  __ _ ___  ___  |_ _|_ __ ___   __ _  __ _  ___
# |  _ \ / _` / __|/ _ \  | || '_ ` _ \ / _` |/ _` |/ _ \
# | |_) | (_| \__ \  __/  | || | | | | | (_| | (_| |  __/
# |____/ \__,_|___/\___| |___|_| |_| |_|\__,_|\__, |\___|
#                                             |___/
#

import modal
from utils import const


image = modal.Image.debian_slim(
    "3.11"
).pip_install(
    const.BASE_DEPENDENCIES
).add_local_dir(
    ".", "/root", ignore=const.IGNORE
)
secrets = [
    modal.Secret.from_name("SHARED_SECRET"),
    modal.Secret.from_name("REDIS")
]


if __name__ == '__main__':
    pass
