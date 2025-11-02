# Translate Agent

该项目用于将英文视频（例如课程）自动翻译成带有中文配音和中英双语字幕的成片。整体流程包括 Whisper 语音识别、OpenAI 大模型翻译、OpenAI TTS 中文配音、以及 MoviePy 音视频混流，最终输出完整的中文版本视频。

## 核心功能
- Whisper 自动识别英文语音，生成带时间戳的文本片段。
- 利用大语言模型将英文口语翻译成自然流畅的中文口语。
- 基于 GPT-4o mini TTS 生成逐段中文配音，并保证时间对齐。
- 可选保留原英文音轨，生成中英混合音效。
- 自动导出中英文双语 SRT 字幕。
- 输出 JSON 格式的转写与翻译元数据，方便后续处理。

## 环境准备
- **Python** 版本 ≥ 3.9。
- 系统已安装 **ffmpeg**（Whisper、MoviePy、PyDub 都需要）。
- 至少准备一种「翻译 + 配音」服务：
  - **OpenAI**：`gpt-4o-mini`（翻译）+ `gpt-4o-mini-tts`（配音），需要 `OPENAI_API_KEY`。
  - **DeepSeek + Edge TTS**：`deepseek-chat`（翻译）+ Microsoft Edge 神经语音（配音），只需 `DEEPSEEK_API_KEY`，Edge TTS 无需额外账号。
  - 也可接入其他提供商，修改 `TranslationConfig` / `TTSConfig` 实现。
- 建议使用 GPU（远程或本地）加速 Whisper 推理，CPU 也可以但速度会慢。

## 安装方式
```bash
python -m venv .venv
source .venv/bin/activate  # Windows 使用 .venv\Scripts\activate
pip install -e .
```

安装完成后，会得到 CLI 命令 `translate-video`。

### API Key 配置
- 项目会自动读取 `.env` 或环境变量，可根据使用的服务写入：
  ```
  # 使用 OpenAI
  OPENAI_API_KEY=sk-...

  # 使用 DeepSeek
  DEEPSEEK_API_KEY=sk-...
  ```
- 如果两种服务都需要，可以同时设置，运行时通过 CLI 参数控制调用哪一个。

## 使用步骤

### OpenAI 翻译 + OpenAI TTS（默认）
1. 在项目根目录创建 `.env`（可选）：
   ```
   OPENAI_API_KEY=sk-...
   ```
2. 执行：
   ```bash
   translate-video path/to/lesson.mp4 \
     --run-name lesson1 \
     --whisper-model medium \
     --translation-provider openai \
     --tts-provider openai \
     --mix-level 0.25
   ```

### DeepSeek 翻译 + Edge TTS（无需 OpenAI）
1. `.env` 中写入 DeepSeek 密钥（Edge TTS 无需密钥）：
   ```
   DEEPSEEK_API_KEY=sk-...
   ```
2. 运行命令：
   ```bash
   translate-video path/to/lesson.mp4 \
     --run-name lesson1 \
     --translation-provider deepseek \
     --translation-model deepseek-chat \
     --translation-api-key-env DEEPSEEK_API_KEY \
     --tts-provider edge \
     --tts-format mp3 \
     --edge-voice zh-CN-XiaoxiaoNeural \
     --mix-level 0.15
   ```
   - Edge TTS 输出为 mp3，脚本会自动拼接并生成最终视频。
   - 如果你希望 Edge 语速更快，可增加 `--edge-rate +10%`。

### 结果目录
- `video/<run-name>_zh.mp4`：最终中文配音视频文件；
- `audio/<run-name>_zh.wav`：中文配音音轨；
- `subtitles/<run-name>_bilingual.srt`：中英双语字幕；
- `transcript/<run-name>.json`：转写和翻译元数据；
- `tts_segments/segment_XXXX.*`：每个片段的中文 TTS 音频（便于抽查）。

### 常用参数
- `--translation-provider`：选择翻译后端（`openai` / `deepseek`）。
- `--translation-api-key-env`：指定读取密钥的环境变量名称，默认根据后端自动选择。
- `--tts-provider`：选择配音后端（`openai` / `edge`）。
- `--edge-voice` / `--edge-rate` / `--edge-volume`：Edge TTS 使用的声音、语速、音量。
- `--no-mix`：完全去掉英文原声，只保留中文配音。
- `--mix-level`：控制英文原声的保留音量（默认 0.25）。
- `--tts-voice`：选择 GPT-4o mini TTS 可用的声音，例如 `alloy`、`ember`、`verse` 等。
- `--speaking-rate`：调整中文配音的语速，方便微调对齐。
- `--sample-rate`：指定最终输出音频采样率，便于与其他流程保持一致。

### 流程概览
1. **Transcribe**：Whisper 生成英文文本及时间轴。
2. **Translate**：逐段调用大模型翻译成中文口语。
3. **Synthesize**：按片段生成中文配音音频文件。
4. **Assemble Audio**：将配音按时间轴拼接，可选叠加英文原声。
5. **Mux Video**：MoviePy 将中文音轨写入新的视频文件。
6. **Subtitle Export**：生成中文在上、英文在下的双语字幕。

## Docker 一键运行（推荐）
为了在阿里云 GPU ECS 上最少操作即可运行，项目已提供 `Dockerfile`：

```bash
# 1. 在具有 Docker 与 NVIDIA Container Toolkit 的服务器上构建镜像
docker build -t translate-agent:latest .

# 2. 挂载视频与输出目录运行
docker run --rm \
  --gpus all \
  -e OPENAI_API_KEY=sk-... \
  -v /path/to/videos:/videos \
  -v /path/to/artifacts:/artifacts \
  translate-agent:latest \
  /videos/lesson.mp4 \
  --run-name lesson1 \
  --output-dir /artifacts \
  --whisper-model medium
```

说明：
- `--gpus all` 需要在阿里云实例安装 NVIDIA 驱动及 `nvidia-container-toolkit`，然后将 Docker `default-runtime` 设置为 `nvidia`。
- 视频目录和输出目录通过 `-v` 挂载到容器中，避免大文件复制。
- 容器入口就是 `translate-video` 命令，可直接传入 CLI 参数。
- 首次运行会在容器内下载 Whisper 模型；如需加速，可在构建后将镜像推送至阿里云 ACR 复用。

若改用 DeepSeek + Edge TTS，可在运行命令中切换参数：

```bash
docker run --rm \
  --gpus all \
  -e DEEPSEEK_API_KEY=sk-... \
  -v /path/to/videos:/videos \
  -v /path/to/artifacts:/artifacts \
  translate-agent:latest \
  /videos/lesson.mp4 \
  --run-name lesson1 \
  --output-dir /artifacts \
  --translation-provider deepseek \
  --translation-model deepseek-chat \
  --tts-provider edge \
  --tts-format mp3 \
  --edge-voice zh-CN-XiaoxiaoNeural
```

如需批量处理，可将上述命令写成脚本或结合阿里云容器服务（ACK/ROS）调度。

## 扩展方向
- 如果想换成其他翻译服务，可在 `translate_agent/translation.py` 中替换实现。
- 若需要本地或其他云端语音合成模型，可改写 `translate_agent/tts.py`。
- 调整字幕格式（例如英文在上）可以修改 `write_bilingual_srt`。

## 常见问题
- **转写速度慢**：尝试 `--whisper-model small`，或在 GPU 环境运行（需要安装 CUDA 版本的 torch）。
- **音频略有错位**：可以微调 `--speaking-rate`，或对个别段落使用外部音频软件拉伸。
- **系统找不到 ffmpeg**：macOS 使用 `brew install ffmpeg`，Linux 用发行版自带包管理器，Windows 可安装预编译版本并添加到 PATH。
- **Edge TTS 失败或限流**：稍等片刻重试，或切换到其他 Edge 声音；若频繁调用，可考虑自建缓存或使用付费 TTS 服务。

## 注意事项
- 音视频处理消耗 CPU/GPU 资源和磁盘空间，建议提前预估。
- 使用翻译和配音时请遵守课程版权及相关法律法规。
