#  _____           _          _
# |_   _|__   ___ | |___  ___| |_
#   | |/ _ \ / _ \| / __|/ _ \ __|
#   | | (_) | (_) | \__ \  __/ |_
#   |_|\___/ \___/|_|___/\___|\__|
#

import re
import sys
import time
import hmac
import base64
import typing
import hashlib
from loguru import logger
from fastapi import Request
from fastapi.responses import JSONResponse
from schemas.errors import BizError
from utils import const


def init_logger() -> None:
    logger.remove()
    logger.add(sys.stdout, level=const.SHOW_LEVEL, format=const.PRINT_FORMAT)


def secure_b64decode(data: str) -> bytes:
    """
    安全 Base64 解码，自动处理 padding / data:image 前缀
    """

    if "," in data: data = data.split(",", 1)[1]

    data = data.strip()
    pad = len(data) % 4
    if pad:
        data += "=" * (4 - pad)

    try:
        return base64.b64decode(data, validate=False)
    except Exception as e:
        raise BizError(status_code=400, detail=f"invalid base64 image: {e}")


def desensitize(value: typing.Optional[str]) -> typing.Optional[str]:
    """
    字符串脱敏工具。

    根据内容自动做不同策略：
    - 邮箱：保留用户名首字母 + 全部域名
    - 手机号（11 位）：保留前三后四，中间打星
    - 纯字母数字且较长（token、id 等）：保留前三后两
    - 普通字符串：保留首尾各 1 个字符

    Parameters
    ----------
    value : str or None
        原始字符串。

    Returns
    -------
    str or None
        脱敏后的字符串；如果传入 None，原样返回 None。
    """

    # 简单的邮箱和手机号正则，可按需要调整
    email_re = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    phone_re = re.compile(r"^1[3-9]\d{9}$")  # 中国大陆 11 位手机号

    def mask_middle(text: str, left: int = 2, right: int = 2, fill: str = "*") -> str:
        """
        对字符串中间部分做脱敏，保留两端可见字符。

        Parameters
        ----------
        text : str
            原始字符串。
        left : int
            左侧需要保留的明文长度。
        right : int
            右侧需要保留的明文长度。
        fill : str
            脱敏使用的填充字符。

        Returns
        -------
        str
            脱敏后的字符串。
        """

        if not text: return text

        # 太短就全脱敏
        if (length := len(text)) <= left + right: return fill * length

        return text[:left] + fill * (length - left - right) + text[-right:]

    if value is None: return None

    if not (s := value.strip()): return s

    # 1) 邮箱脱敏：a***@domain.com
    if email_re.fullmatch(s):
        name, domain = s.split("@", 1)
        masked_name  = mask_middle(name, left=1, right=0, fill="*")
        return f"{masked_name}@{domain}"

    # 2) 手机号脱敏：138****1234
    if phone_re.fullmatch(s):
        return re.sub(pattern=r"(\d{3})\d{4}(\d{4})", repl=r"\1****\2", string=s)

    # 3) 长 token / id：abcdefg1234 → abc*****34
    if len(s) >= 8 and re.fullmatch(r"[0-9a-zA-Z]+", s):
        return mask_middle(s, left=3, right=2, fill="*")

    # 4) 默认策略：首尾各保留 1 位
    return mask_middle(s, left=1, right=1, fill="*")


def judge_channel(shape: tuple[int, ...]) -> int:
    """色彩通道"""

    return shape[2] if len(shape) == 3 and shape[2] in (1, 3, 4) else \
        shape[0] if len(shape) == 3 and shape[0] in (1, 3, 4) else \
            1 if len(shape) == 2 else None


def verify_token(request: Request, token: str) -> typing.Union["JSONResponse", bool]:
    """鉴权"""

    shared_secret = request.app.state.shared_secret

    logger.info(f"Verify token: {desensitize(token)}")
    if not token:
        return JSONResponse(
            content={"error": "Unauthorized", "message": "Token missing"},
            status_code=401
        )

    try:
        payload, sig      = token.rsplit(sep=".", maxsplit=1)
        app_id, expire_at = payload.split(":")

        if time.time() > int(expire_at):
            logger.warning("Token has expired")
            return JSONResponse(
                content={"error": "Token has expired"},
                status_code=401
            )

        expected_sig = hmac.new(
            shared_secret.encode(), payload.encode(), hashlib.sha256
        ).digest()
        expected_b64 = base64.b64encode(expected_sig).decode()

        if not (compare := hmac.compare_digest(expected_b64, sig)):
            logger.warning("Token signature mismatch")
            return JSONResponse(
                content={"error": "Invalid token signature"},
                status_code=401
            )
        return compare

    except ValueError as e:
        logger.error(e)
        return JSONResponse(
            content={"error": "Malformed token"},
            status_code=401
        )
    except Exception as e:
        logger.error(e)
        return JSONResponse(
            content={"error": "Unauthorized", "message" : str(e)},
            status_code=401
        )


if __name__ == '__main__':
    pass
