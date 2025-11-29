# 📘 Model Inference Service — API Documentation

本项目提供模型推理与模型健康检查服务，基于 Modal、FastAPI 与流式推理框架构建。  
所有接口均通过 `X-Token` 进行鉴权，并包含全局异常处理机制。

---

# 🧩 API Summary
| Method | Endpoint   | Description        |
|--------|------------|--------------------|
| POST   | `/predict` | 图像推理接口（SSE 流式推理返回） |
| GET    | `/service` | 模型状态与容器心跳接口        |

---

# 🔐 Authentication
## 所有接口均必须提供认证 Header：
### X-Token
示例失败返回：
```json
{
  "error"  : "unauthorized",
  "detail" : "invalid or missing token"
}
```

---

# 🛠 Global Exception Handling
接口包含统一异常捕获 @with_exception_handling，未处理异常格式化为：
```json
{
  "error"    : "internal_error",
  "detail"   : "Exception message",
  "trace_id" : "<trace-id>"
}
```

---

# 🚀 POST /predict — 模型推理接口
## Endpoint
```
POST /predict
Content-Type  : multipart/form-data
Authorization : X-Token required
Response-Type : text/event-stream
```

## 描述
该接口接收图像帧与对应元数据，并调用模型执行推理。
推理过程中通过 Server-Sent Events (SSE) 实时返回：
- 预处理阶段信息
- 推理过程输出
- 识别结果
- 后处理结果

适用于实时场景：
如连续帧检测、直播流分析、帧级 AI 推理等。

## 🔧 Request Parameters
| Field      | Type        | Required | Description                   |
|------------|-------------|----------|-------------------------------|
| frame_meta | JSON string | ✔        | 帧元信息（frame_id、width、height 等） |
| frame_file | binary file | ✔        | 图像文件（JPG/PNG）                 |

## 📥 示例请求（cURL）
```
curl -X POST "https://your-service/predict" \
  -H "X-Token: <YourToken>" \
  -F 'frame_meta={"frame_id":1,"width":720,"height":1280}' \
  -F "frame_file=@/path/to/frame.jpg"
```

## 📡 Response — SSE 推理流
Response Header：
```
Content-Type: text/event-stream
```

示例 SSE 输出：
```
data: {"stage":"preprocess","cost_ms":12}
data: {"stage":"inference","label":"Person","prob":0.98}
data: {"stage":"postprocess","objects":[ ... ]}
```
推理结束后自动关闭流连接。

---

# 🚦 GET /service — 模型状态与容器心跳接口
## Endpoint
```
GET /service
Authorization: X-Token required
```

## 描述
该接口用于：
- 容器存活探测（keep-alive）
- 模型加载状态查询
- readiness / liveness 检查
- 调试模型状态


## 📤 Response Example
```json
{
  "status": "OK",
  "message": {
    "AquilaSequence-F": {
      "fettle": "Online",
      "dazzle": {
        "layers": ["..."],
        "architecture": "CNN",
        "params": 12498312
      }
    },
    "AquilaSequence-C": {
      "fettle": "Online",
      "dazzle": {
        "layers": ["..."],
        "architecture": "Transformer",
        "params": 32400000
      }
    }
  },
  "timestamp": 1737990701
}
```

## 字段说明
| Field     | Type   | Meaning                  |
|-----------|--------|--------------------------|
| status    | string | 服务状态（固定为 "OK"）           |
| fettle    | string | 模型是否加载成功（Online/Offline） |
| dazzle    | object | 模型拓扑结构摘要（不含 config）      |
| timestamp | int    | Unix 时间戳                 |

## ❌ Error Codes
| HTTP Code | Meaning        | 	Example                   |
|-----------|----------------|----------------------------|
| 401       | Token 缺失/无效	   | {"error":"unauthorized"}   |
| 403       | Token 校验失败	    | {"error":"forbidden"}      |
| 500       | 未处理异常	         | {"error":"internal_error"} |
| 503       | 模型未加载 / 推理不可用	 | {"error":"model_offline"}  |

---

# 📐 Response Requirements
- JSON 使用 UTF-8
- 字段名统一使用 snake_case
- timestamp 为秒级 Unix 时间戳
- 所有错误需包含 trace_id
- 推理接口返回 SSE 格式，不返回 JSON

---

# 📄 License
    This project is proprietary and confidential.
    Unauthorized redistribution is prohibited.
