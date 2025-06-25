#    ____            __ _
#   / ___|_ __ __ _ / _| |_
#  | |   | '__/ _` | |_| __|
#  | |___| | | (_| |  _| |_
#   \____|_|  \__,_|_|  \__|
#

import sys
from loguru import logger
from common import const


def init_logger() -> None:
    logger.remove()
    logger.add(sys.stdout, level=const.SHOW_LEVEL, format=const.PRINT_FORMAT)


if __name__ == '__main__':
    pass
