#  _____           _ _    _ _
# |_   _|__   ___ | | | _(_) |_
#   | |/ _ \ / _ \| | |/ / | __|
#   | | (_) | (_) | |   <| | |_
#   |_|\___/ \___/|_|_|\_\_|\__|
#


def judge_channel(shape: tuple[int, ...]) -> int:
    """色彩通道"""

    return shape[2] if len(shape) == 3 and shape[2] in (1, 3, 4) else \
        shape[0] if len(shape) == 3 and shape[0] in (1, 3, 4) else \
            1 if len(shape) == 2 else None


if __name__ == '__main__':
    pass
