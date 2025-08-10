# RMBG-2.0 背景抠图（GUI 与 CLI）

基于 Hugging Face 上的 `briaai/RMBG-2.0` 模型实现的背景去除工具，提供 PyQt6 图形界面与命令行两种使用方式。

模型卡与用法参考：[Hugging Face: briaai/RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0)

## 环境要求
- macOS 15.x（Apple Silicon 或 Intel）
- Python 3.9+（建议使用系统 `/usr/bin/python3`）
- 首次运行需联网下载模型权重（数百 MB）

## 目录结构
- 工作目录：`/Users/mengfs/rmbg`
- 关键文件：
  - `rmbg/gui_pyqt.py`：图形界面（PyQt6）
  - `rmbg/rmbg_cli.py`：命令行工具
  - `rmbg/requirements.txt`：依赖列表

## 安装与准备
```bash
cd /Users/mengfs/rmbg
/usr/bin/python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r rmbg/requirements.txt
```

首轮运行会自动下载模型与远程代码：
- 权重：`~/.cache/huggingface/hub/models--briaai--RMBG-2.0/snapshots/<revision>/`
- 远程代码：`~/.cache/huggingface/modules/transformers_modules/briaai/RMBG-2.0/<commit>/`

可通过环境变量自定义缓存目录（可选）：
```bash
export HF_HOME="$HOME/.hfhome"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
```

## 启动图形界面（PyQt6）
```bash
python rmbg/gui_pyqt.py
```

### 界面说明
- 选择图片…：添加单张或多张图片
- 选择文件夹…：将文件夹内所有图片（含子目录）加入队列
- 清空：清空队列与预览
- 设备：选择推理设备
  - auto（默认）：优先 MPS（Apple Silicon），遇到不支持算子自动回退到 CPU
  - cpu：最稳定
  - mps：Apple Silicon 加速，个别算子不支持时会自动回退 CPU
  - cuda：如有 NVIDIA GPU + CUDA 环境
- 输入尺寸：默认 1024（越大越清晰，耗时也更长）
- 输出目录：不选则输出在每张图片同目录
- 预览：左侧原图，右侧抠图结果（带透明通道 PNG）
- 进度日志（文字版）：
  - 每个文件处理过程与错误信息会按行追加，如：
    - `[3/10] 正在处理: /path/img.jpg`
    - `[3/10] 完成: /path/img.jpg -> /path/img_rmbg.png`
    - `[3/10] 失败: /path/img.jpg | 错误: <异常详情>`

### 输出命名
- 每张输入 `xxx.ext` 对应输出 `xxx_rmbg.png`（带透明通道）

## 命令行用法（可选）
- 单图（输出到同目录）
```bash
python rmbg/rmbg_cli.py /path/to/input.jpg
```
- 指定输出
```bash
python rmbg/rmbg_cli.py /path/to/input.jpg -o /path/to/output.png
```
- 批量处理目录
```bash
python rmbg/rmbg_cli.py /path/to/dir -o /path/to/out_dir
```
- 指定设备与尺寸
```bash
python rmbg/rmbg_cli.py /path/to/input.jpg --device cpu --size 1024
```

## 设备与性能建议
- 在 Apple Silicon 上，`auto` 或 `cpu` 推荐。遇到 MPS 未实现算子时会自动回退到 CPU。
- 大图与大输入尺寸会显著增加处理时间与内存占用；建议在 768–1024 之间权衡。

## 常见问题（FAQ）
- urllib3/LibreSSL 警告（不影响运行）：
  - 现象：`urllib3 NotOpenSSLWarning`
  - 处理：在虚拟环境内升级相关包
    ```bash
    pip install --upgrade pip urllib3 certifi
    ```
- MPS 不支持的算子（Apple Silicon）：
  - 现象：`NotImplementedError: torchvision::deform_conv2d is not implemented for MPS`
  - 处理：GUI/CLI 已默认设置 `PYTORCH_ENABLE_MPS_FALLBACK=1` 并在出错时切到 CPU；或手动使用 `--device cpu`
- 离线/内网使用：
  - 方案一：联网环境运行一次，后续离线复用缓存
  - 方案二：自定义缓存目录（便于迁移/外置硬盘）
  - 方案三：离线加载本地模型目录，示例（在代码中切换）：
    ```python
    from transformers import AutoModelForImageSegmentation
    model = AutoModelForImageSegmentation.from_pretrained(
        "/absolute/path/to/local/RMBG-2.0",  # 包含 config.json 与权重
        trust_remote_code=True,
        local_files_only=True,
    )
    ```

## 高级用法
- 固定版本（避免上游更新影响）：
  ```python
  AutoModelForImageSegmentation.from_pretrained(
      "briaai/RMBG-2.0", trust_remote_code=True, revision="main"
  )
  ```
- 自定义缓存目录（仅本次调用）：
  ```python
  AutoModelForImageSegmentation.from_pretrained(
      "briaai/RMBG-2.0", trust_remote_code=True, cache_dir="/path/to/cache"
  )
  ```

## 许可证与引用
- 模型遵循非商业使用许可（CC BY-NC 4.0）；商业使用需另购授权。
- 模型与背景说明：[Hugging Face: briaai/RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0)

## 维护
- 升级依赖：
```bash
source .venv/bin/activate
pip install --upgrade -r rmbg/requirements.txt
```
- 清理模型缓存：
```bash
rm -rf ~/.cache/huggingface
```

如需增加功能（自定义阈值、透明度调节、背景替换、批量导出设置、ONNX 导出等），欢迎提出需求。


