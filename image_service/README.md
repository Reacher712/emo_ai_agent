# 原版 Z-Image-Turbo 本地推理

独立于 `agent/.venv` 的 Python 3.12 环境，官方 `Tongyi-MAI/Z-Image-Turbo` 权重，不加 LoRA、不使用第三方微调或量化权重。原始 safetensors 按 BF16 加载推理，下载约 32.8 GB。

依赖起点为 PyTorch 2.14.0＋CUDA 13.0、Diffusers 0.42.0、Transformers 5.20.2。不是每日构建版；安装后还需实际 CUDA 与出图验证。

## 本机已验证结果（2026-10-26）

- RTX 5080 Laptop 16 GB，Python 3.12.13，CUDA BF16 矩阵运算通过。
- 原版权重提交：`f332072aa78be7aecdf3ee76d5c247082da564a6`；7 个 safetensors 分片可读取。
- 768×768、seed 42、9 步设置、CFG 0、staged 加载：加载及文本编码 43.7 秒，图像推理及解码 15.8 秒，图像阶段峰值分配显存 11.87 GiB。仅代表本次运行，不保证其他尺寸或提示词速度。
- 验证图片：`outputs/9230f63200dd4263b2b9f20313765aa2.png`，同名 JSON 记录完整参数。
- HTTP 中文提示词生成也已验证：`outputs/eda8375a229b444f975d8d72a2afd650.png`，768×768，总计约 49.4 秒（加载/文本编码 33.8 秒＋图像推理/解码 15.6 秒）。
- 依赖检查通过；Agent 原有 21 项测试仍通过。`requirements-lock.txt` 保存本次图像环境实际版本，不包含模型权重。

## 目录

```text
image_service/
  .venv/                 独立推理依赖（不提交）
  models/Z-Image-Turbo/   官方权重与部署版本清单（不提交）
  outputs/               PNG 与同名参数 JSON（不提交）
  download_model.py      官方下载，可重跑续传
  generate.py            单图命令和模型生命周期
  server.py              本地 HTTP 接口
  install.ps1            环境安装
  start.ps1              启动本地服务
  requirements.txt       推理依赖版本
```

## 安装与下载

从项目根目录执行。已有环境可以跳过安装步骤；脚本使用独立 `.venv`，不修改 Agent 依赖。

```powershell
powershell -ExecutionPolicy Bypass -File .\image_service\install.ps1
.\image_service\.venv\Scripts\python.exe .\image_service\download_model.py
```

无需浏览器手动下载。脚本从官方 Hugging Face 仓库解析 `main` 的具体提交，然后按该提交下载推理组件；版本写入模型目录 `deployment_manifest.json`。复现时可传 `--revision <提交哈希>`，更新权重时先备份现有清单并重新验证。

## 单张生成

```powershell
.\image_service\.venv\Scripts\python.exe .\image_service\generate.py --width 768 --height 768 --seed 42
.\image_service\.venv\Scripts\python.exe .\image_service\generate.py --prompt "中国水墨山水，远山云雾，溪水竹林，淡墨，疏朗构图，大片留白，无文字" --width 1024 --height 1024
```

默认 9 个推理步设置、`guidance_scale=0`，起点来自官方示例。默认 `staged`：先用原版文本编码器生成提示词嵌入并释放它，再加载原版图像 Transformer 与 VAE，以 model CPU offload 运行；避免在 32 GB 内存上同时驻留整套模型。每次生成需要重新分阶段加载，适合先验证正确性，不代表最高吞吐。

显存不足时先减至 512×512；内存充裕时可试 `--offload model` 保留整条管线，显存更紧张且内存充裕时可试 `--offload sequential`，速度可能明显下降。`--offload none` 会把整条管线常驻 GPU，不建议在 16 GB 显卡上直接使用。

PNG 同名 JSON 分别记录加载阶段（staged 包含文本编码）与图像推理耗时、seed、尺寸、实际参数、图像阶段显存峰值和模型提交。生成阶段先让 Agent 使用的 Qwen3.5 释放 GPU，尚未实现自动显存协调；Z-Image 自带的文本编码器与 Agent 模型是不同组件。

## 本地 HTTP 服务

```powershell
powershell -ExecutionPolicy Bypass -File .\image_service\start.ps1
```

- 文档：`http://127.0.0.1:8001/docs`
- 状态：`GET /health`，`loaded=false` 表示尚未加载，不代表模型出图已验证。
- 生成：`POST /generate`，参数为 `prompt`、`width`、`height`、`seed`、`steps`，返回图片绝对路径和元数据。

```powershell
$imageRequest = @{ prompt='中国水墨山水，远山云雾，淡墨留白'; width=768; height=768; seed=42; steps=9 } | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8001/generate -Method Post -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($imageRequest)) -TimeoutSec 900
```

服务仅监听本机，单进程单 worker，生成请求串行使用 GPU；不要增加 worker 数导致重复加载。关闭运行终端或 Ctrl+C 退出后释放 GPU。此接口是原始图像工具，尚未连接 LangGraph，也没有执行作品评审/内容检查，不将它直接作为已审核发布接口。

不要同时运行服务和另一个 CLI 生成进程，以免重复加载权重占满内存。服务会保留图像管线供下次调用；需要把内存交给其他模型时先停止服务。

来源：[官方模型卡](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)、[PyTorch CUDA 安装](https://pytorch.org/get-started/locally/)。
