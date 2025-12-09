#  _____
# | ____|_ __ _ __ ___  _ __ ___
# |  _| | '__| '__/ _ \| '__/ __|
# | |___| |  | | | (_) | |  \__ \
# |_____|_|  |_|  \___/|_|  |___/
#

import typing
from fastapi import HTTPException


class AuthorizationError(HTTPException):

    def __init__(self, status_code: int = 401, detail: typing.Any = None):
        super().__init__(
            status_code=status_code, detail=detail
        )


class BizError(HTTPException):

    def __init__(self, status_code: int = 400, detail: typing.Any = None):
        super().__init__(
            status_code=status_code, detail=detail
        )


if __name__ == '__main__':
    pass
