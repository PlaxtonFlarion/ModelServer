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


if __name__ == '__main__':
    pass
