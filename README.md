# 多端录制（multi-end recording）

协调多个采集端（模拟摄像头 / 麦克风）在同一时刻开始录制：
准备屏障 → 设备确认 → 共同起点 → 各端首个真实样本即录制起点。

## 图形入口（主入口）

多端录制从制作面板发起，命令行仅起辅助作用：

    python3 -m uvicorn server:app --host 127.0.0.1 --port 8000

浏览器打开 http://127.0.0.1:8000/ 进入多端制作面板：

- 在“采集端”勾选参与录制的端，点击“发起录制”；
- 过程记录严格区分三类事件：开始命令发出 / 设备确认收到 / 首个真实样本（录制起点）；
- 录制结束后面板自动实际运算校验各端真实片段与起点（解码帧序号、脉冲位置、
  颜色 / 音调身份、真实起录位置），异常直接在结果区提示；
- “会话 / 回归样例”列表可打开历史会话与入库的回归样例
  （如 recordings/543f4e7adbd0），同样可触发实际运算校验。

## 命令行（仅辅助）

命令行只用于自动验收与回归，不是录制入口：

    python3 -m pytest tests/test_start_barrier.py -v   # 启动屏障验收（正反例）
    python3 -m pytest tests/ -v                        # 全部测试

## 依赖

    python3 -m pip install numpy pytest fastapi uvicorn imageio-ffmpeg
