下面我整理成**直接给 Codex 的工程开发文档格式**。重点不是介绍方案，而是告诉 Codex：

* 做什么
* 不做什么
* 技术选型
* Repo结构
* 开发任务
* API定义
* 验收标准

你可以直接复制给 Codex 开始生成项目。

---

# AirGPT Embodied AI Model Service v1.0

## Codex 开发设计文档

版本：v1.0
项目类型：Backend / AI Infrastructure
目标：构建面向科研教学场景的具身智能模型统一推理服务平台

---

# 0. 开发目标（给 Codex）

请实现一个：

> 基于 vLLM + Transformers + Diffusion Runtime 的具身智能模型服务平台。

定位类似：

```
vLLM
+
Ollama
+
ModelScope model hosting

for Embodied AI
```

第一阶段目标：

在单台 GPU Server / Workstation 上：

* Docker 部署
* 提供统一 API
* 支持多个具身智能模型
* 提供 Python SDK
* 支持 Isaac Sim / AirSim 调用

---

# 1. 明确边界

## 本阶段必须实现

### Backend

* FastAPI 服务
* Model Registry
* Model Adapter
* VLA Runtime
* World Model Runtime
* API Gateway
* Python SDK
* Docker部署

---

## 本阶段不要实现

不要开发：

```
用户认证
权限系统
课程系统
GPU调度
Kubernetes
计费系统
Jetson部署
国产芯片适配
```

这些属于 AirGPT 上层平台。

---

# 2. 总体架构

```
                 AirGPT Platform


              User / Course System


                       |

                       |

              Embodied Model API


                       |

        --------------------------------

        |                              |

 VLA Model Service          World Model Service


        |                              |

      vLLM                  Diffusion Runtime


        |                              |

 OpenVLA / Octo          Cosmos / Video Model

 Qwen-VL


                       |

                 GPU Server


                       |

          Isaac Sim / AirSim / Robot

```

---

# 3. 技术栈

## Backend

使用：

```
Python 3.11

FastAPI

Pydantic

Uvicorn

PyTorch

Transformers

vLLM

Diffusers

Docker

```

---

# 4. Repo结构

创建：

```
airgpt-model-service/


├── README.md
├── requirements.txt
├── docker-compose.yml
├── Dockerfile


├── app/


│
├── api/

│   ├── models.py

│   ├── robot.py

│   └── world.py


│
├── core/


│   ├── config.py

│   ├── registry.py

│   └── manager.py


│
├── runtime/


│   ├── vllm_runtime.py

│   └── world_runtime.py


│
├── adapters/


│
│   ├── vla/

│   │
│   ├── openvla.py

│   ├── octo.py

│
│
│   └── world/

│       ├── cosmos.py

│       └── diffusion.py



├── sdk/


│   └── client.py


├── configs/


│   └── models.yaml


└── examples/


    ├── openvla_demo.py

    └── world_demo.py

```

---

# 5. 核心模块设计

# 5.1 Model Registry

功能：

管理模型。

配置：

`configs/models.yaml`

示例：

```yaml

models:


  openvla:

    type: vla

    backend: vllm

    adapter:
      openvla


  octo:

    type: vla

    backend: vllm


  cosmos:

    type: world

    backend: diffusion

```

---

# 5.2 Model Manager

负责：

* 模型发现
* 模型加载
* 模型调用

接口：

```python

class ModelManager:


    def load(model_name):

        pass



    def predict(
        model_name,
        input
    ):

        pass

```

---

# 6. VLA Runtime实现

## 目标

基于 vLLM。

调用链：

```
API

 ↓

VLA Adapter

 ↓

vLLM Runtime

 ↓

OpenVLA

 ↓

Action Decoder

```

---

实现：

```
runtime/vllm_runtime.py

```

提供：

```python

class VLLMRuntime:


    load_model()


    generate()


```

---

# 7. VLA Adapter

示例：

```
adapters/vla/openvla.py

```

接口统一：

```python

class OpenVLAAdapter:


    def preprocess(
        image,
        instruction
    ):


        pass



    def postprocess(
        output
    ):


        return action

```

---

# 8. World Model Runtime

注意：

世界模型不要走 vLLM。

独立：

```
Diffusion Runtime

```

支持：

* Diffusers
* PyTorch

接口：

```python

class WorldRuntime:


    load_model()


    generate_future()

```

---

# 9. API设计

## 9.1 获取模型列表

GET

```
/v1/models

```

返回：

```json

{

"models":[

"openvla",

"octo",

"cosmos"

]

}

```

---

# 9.2 Robot Action API

POST

```
/v1/robot/action

```

请求：

```json

{

"model":"openvla",

"image":"base64",

"instruction":

"pick the red cube"


}

```

返回：

```json

{

"action":[

0.1,

0.2,

0.3

],

"model":

"openvla"


}

```

---

# 9.3 World Prediction API

POST

```
/v1/world/predict

```

请求：

```json

{

"model":"cosmos",

"scene":

"warehouse",

"action":

"move forward"

}

```

返回：

```json

{

"frames":[

"xxx"

]

}

```

---

# 10. Python SDK

实现：

```
sdk/client.py

```

使用方式：

```python

from airgpt import Client



client = Client(
    "http://localhost:8000"
)



result = client.robot.action(

    model="openvla",

    image="test.jpg",

    instruction="pick cup"

)


```

---

# 11. Docker部署

目标：

一条命令启动。

## docker-compose.yml

包含：

```
gateway

vla-runtime

world-runtime

```

示例：

```yaml

services:


 gateway:

  image:
   airgpt-service


  ports:

   - 8000:8000



 vla:

  image:
   airgpt-vla



 world:

  image:
   airgpt-world

```

---

# 12. Demo要求

完成三个Demo。

## Demo 1

OpenVLA：

输入：

```
image

instruction

```

输出：

```
robot action

```

---

## Demo 2

World Model：

输入：

```
scene

action

```

输出：

```
future frames

```

---

## Demo 3

Python SDK调用：

```python

client.robot.action()

```

---

# 13. 开发阶段

## Sprint 1

目标：

基础服务

完成：

* FastAPI
* Registry
* Docker
* API

---

## Sprint 2

目标：

VLA支持

完成：

* vLLM Runtime
* OpenVLA Adapter
* Action API

---

## Sprint 3

目标：

完整Demo

完成：

* World Runtime
* SDK
* Isaac/AirSim Example

---

# 14. 验收标准

## 服务

启动：

```
docker compose up

```

成功。

---

## API

访问：

```
localhost:8000/docs

```

看到：

* models API
* robot API
* world API

---

## 模型

至少支持：

```
OpenVLA

Octo

一个World Model

```

---

## SDK

能够：

```python

client.robot.action()

```

返回结果。

---

# 最终目标

构建：

> AirGPT Embodied AI Model Service

成为：

```
具身智能领域的 vLLM/Ollama 服务层
```

第一阶段重点：

**服务器端统一服务，不做端侧部署。**

---

这个版本直接丢 Codex 最合适。
我建议下一步让 Codex 先生成 **MVP skeleton（FastAPI + Docker + Registry + Mock Adapter）**，然后再接 OpenVLA，因为这样工程风险最低。
