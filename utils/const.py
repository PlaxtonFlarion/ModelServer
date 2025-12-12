#    ____                _
#   / ___|___  _ __  ___| |_
#  | |   / _ \| '_ \/ __| __|
#  | |__| (_) | | | \__ \ |_
#   \____\___/|_| |_|___/\__|
#

# ==== Notes: 日志 ====
SHOW_LEVEL   = r"INFO"
PRINT_FORMAT = r"<bold><level>{level}</level></bold>: <bold><cyan>{message}</cyan></bold>"
WRITE_FORMAT = r"{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"

# ==== Notes: Redis Token Bucket ====
TOKEN_BUCKET_LUA = """
local key   = KEYS[1]
local burst = tonumber(ARGV[1])
local rate  = tonumber(ARGV[2])
local now   = tonumber(ARGV[3])

local data = redis.call("HMGET", key, "tokens", "time")
local tokens = tonumber(data[1])
local last   = tonumber(data[2])

if tokens == nil then
    tokens = burst
    last   = now
else
    local delta = (now - last) / 1000.0
    if delta > 0 then
        tokens = math.min(burst, tokens + delta * rate)
    end
end

if tokens >= 1 then
    tokens = tokens - 1
    redis.call("HMSET", key, "tokens", tokens, "time", now)
    redis.call("EXPIRE", key, math.ceil(burst/rate)+2)
    return tokens
else
    return -1
end
"""

# ==== Notes: Notes: Redis Hot Key ====
K_MIX = "Mix"
V_MIX = {
  "app": {},
  "white_list": [
    "/",
    "/status"
  ],
  "rate_config": {
    "default": {
      "burst": 10,
      "rate": 2,
      "max_wait": 1
    },
    "routes": {
      "/service": {
        "burst": 2,
        "rate": 0.2
      },
      "/tensor/en": {
        "burst": 2,
        "rate": 0.2
      },
      "/tensor/zh": {
        "burst": 2,
        "rate": 0.2
      },
      "/predict": {
        "burst": 2,
        "rate": 0.2
      },
      "/rerank": {
        "burst": 2,
        "rate": 0.2
      },
    },
    "ip": {}
  }
}

# ==== Notes: 鉴权 ====
AUTH_KEY = r"X-Token"

# ==== Notes: 分组 ====
GROUP_MAIN = r"apps"
GROUP_FUNC = r"functions"

# ==== Notes: 过滤 ====
IGNORE = [
    "*venv",
    "*.venv",
    "models/",
    "__pycache__/",
    ".idea/",
    ".DS_Store"
]

# ==== Notes: 依赖 ====
BASE_DEPENDENCIES = [
    "modal==1.2.4",
    "synchronicity==0.10.5",
    "propcache==0.4.1",
    "grpclib==0.4.8",
    "types-toml==0.10.8.20240310",
    "python-multipart==0.0.20",
    "fastapi==0.123.9",
    "starlette==0.50.0",
    "pydantic==2.12.5",
    "uvicorn==0.38.0",
    "httpx==0.28.1",
    "loguru==0.7.3",
    "redis==5.0.3",
    "hiredis==2.3.2"
]

# ==== Notes: embedding 依赖 ====
EMBEDDING_DEPENDENCIES = [
    "modal==1.2.4",
    "synchronicity==0.10.5",
    "propcache==0.4.1",
    "grpclib==0.4.8",
    "types-toml==0.10.8.20240310",
    "python-multipart==0.0.20",
    "fastapi==0.123.9",
    "starlette==0.50.0",
    "pydantic==2.12.5",
    "uvicorn==0.38.0",
    "httpx==0.28.1",
    "loguru==0.7.3",
    "redis==5.0.3",
    "hiredis==2.3.2",
    "sentence-transformers==5.1.2",
    "transformers==4.57.3",
    "tokenizers==0.22.1",
    "numpy==1.26.4",
    "torch==2.9.1",
    "scipy==1.11.4",
    "joblib==1.4.2",
    "threadpoolctl==3.6.0",
    "scikit-learn==1.4.2 ",
    "Pillow==9.5.0",
    "accelerate==1.12.0"
]

# ==== Notes: inference 依赖 ====
INFERENCE_DEPENDENCIES = [
    "modal==1.0.4",
    "synchronicity==0.9.16",
    "propcache==0.3.2",
    "grpclib==0.4.7",
    "types-toml==0.10.8.20240310",
    "python-multipart==0.0.20",
    "fastapi==0.110.2",
    "starlette==0.37.2",
    "pydantic==2.11.4",
    "uvicorn==0.29.0",
    "httpx==0.27.0",
    "loguru==0.7.3",
    "redis==5.0.3",
    "hiredis==2.3.2",
    "watchfiles==1.1.0",
    "shellingham==1.5.4",
    "typer==0.16.0",
    "rich==14.0.0",
    "markdown-it-py==3.0.0",
    "mdurl==0.1.2",
    "Pygments==2.19.2",
    "numpy==1.26.4",
    "scipy==1.11.4",
    "joblib==1.4.2",
    "threadpoolctl==3.6.0",
    "Pillow==9.5.0",
    "imageio==2.34.0",
    "opencv-python==4.8.1.78",
    "scikit-learn==1.4.2",
    "scikit-image==0.18.3",
    "keras==2.14.0",
    "tensorflow==2.14.0",
    "tensorflow-estimator==2.14.0",
    "tensorboard==2.14.1",
    "tensorboard-data-server==0.7.2",
    "absl-py==2.2.2",
    "gast==0.6.0",
    "flatbuffers==25.2.10",
    "opt_einsum==3.4.0",
    "protobuf==4.25.7",
    "h5py==3.13.0",
    "ml-dtypes==0.2.0",
    "wrapt==1.14.1",
    "typing_extensions==4.13.2",
    "findit==0.5.9",
    "win32_setctime==1.2.0; sys_platform == 'win32'",
    "pyobjc-core==11.0; sys_platform == 'darwin'",
    "pyobjc-framework-Cocoa==11.0; sys_platform == 'darwin'",
    "tensorflow-intel==2.14.0; sys_platform == 'win32'",
    "tensorflow-io-gcs-filesystem==0.31.0; sys_platform == 'win32'",
    "tensorflow-macos==2.14.0; sys_platform == 'darwin'",
    "tensorflow-metal==1.1.0; sys_platform == 'darwin'",
    "tensorflow-io-gcs-filesystem==0.37.1; sys_platform == 'darwin'",
    "sentence-transformers==5.1.2",
    "transformers==4.57.3",
    "tokenizers==0.22.1",
    "accelerate==1.12.0"
]

# ==== Notes: yolo 依赖 ====
YOLO_DEPENDENCIES = [
    "modal==1.2.4",
    "synchronicity==0.10.5",
    "propcache==0.4.1",
    "grpclib==0.4.8",
    "types-toml==0.10.8.20240310",
    "python-multipart==0.0.20",
    "fastapi==0.123.9",
    "starlette==0.50.0",
    "pydantic==2.12.5",
    "uvicorn==0.38.0",
    "httpx==0.28.1",
    "loguru==0.7.3",
    "redis==5.0.3",
    "hiredis==2.3.2",
    "ultralytics==8.3.237",
    "opencv-python==4.12.0.88",
    "Pillow==9.5.0",
    "numpy==1.26.4"
]


if __name__ == '__main__':
    pass
