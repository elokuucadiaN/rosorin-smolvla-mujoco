# ROSOrin SmolVLA × MuJoCo：首帧预测夹木块

用训练好的 **SmolVLA**，在 ROSOrin Pro 原厂几何的 MuJoCo 场景中完成：**张开 → 接近纸面红色方块 → 下降 → 闭合 → 原地保持**。

每轮只输入 **一张初始腕部 RGB + 初始六关节角**，一次生成 **108×6 个绝对关节角目标，10Hz，共10.8秒**。没有开头5秒等待，没有执行中重拍、专家纠偏、动作平滑或保持段投影。不抬起、不搬运、不释放方块。红块是具有明确质量、尺寸和摩擦的仿真物体，未标定真实木材参数。

这是从训练工作区整理出的独立运行版。**不需要 ROS、不需要原作者的训练服务器账号**；模型放在 [Release](https://github.com/elokuucadiaN/rosorin-smolvla-mujoco/releases/tag/v1.0.0)，不混进 Git 历史。

[查看实跑视频](docs/demo.mp4) · [模型与验证范围](docs/MODEL_CARD.zh-CN.md) · [故障排查](docs/TROUBLESHOOTING.zh-CN.md)

## 1. 安装

验证环境：Ubuntu 22.04 / Windows WSL2 Ubuntu 22.04，Python 3.10、MuJoCo 3.14.0、PyTorch 2.7.1、CUDA GPU。建议 NVIDIA 显卡至少8GB空闲显存、16GB系统内存、10GB以上空闲磁盘。WSL2 的图形窗口使用 WSLg；运行前确认 `nvidia-smi` 正常。

```bash
git clone https://github.com/elokuucadiaN/rosorin-smolvla-mujoco.git
cd rosorin-smolvla-mujoco
sudo apt-get update
sudo apt-get install -y git python3-venv libgl1 libegl1 libosmesa6
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[inference]'
```

请使用仓库内的 editable 安装 `-e`，仿真通过仓库相对路径加载世界和原厂网格。安装会固定 LeRobot 到已验证提交，而不是跟随最新主分支。CUDA/PyTorch 安装问题见故障排查。原生 Windows 和 macOS 图形模式没有验收；推荐 Ubuntu 或 WSL2。

## 2. 下载模型

```bash
python -m rosorin_vla.download
```

下载约729MB，解压到 `models/smolvla-block/`；程序验证压缩包及每个模型文件的 SHA256。之后模型推理可离线运行，不另行下载完整 VLM 基座。已有模型包可执行：

```bash
python -m rosorin_vla.download --asset /path/to/smolvla-block-v1.tar.gz
```

## 3. 打开 MuJoCo 并运行

```bash
python -m rosorin_vla.demo --seed 0
```

先加载模型，再从 t=0 的初始输入预测并执行一轮；完成后窗口保留夹持姿态，关闭窗口或 Ctrl+C 退出。推理等待期间暂停物理步进，加载耗时不插入动作序列。结果和原始预测保存在 `outputs/`。

六个位置按 `seed % 6` 选择：x={0.205,0.212,0.219}m，y={0.025,0.045}m，先x后y排序。无界面批量检查：

```bash
MUJOCO_GL=egl python -m rosorin_vla.demo --headless --episodes 6
```

无 NVIDIA 渲染设备时可用 `MUJOCO_GL=osmesa`。`--device cpu` 提供较慢的实验运行方式；报告中的预测一致性和成功率使用 CUDA 验证，CPU 的随机数与浮点数行为可能不同，不保证相同成绩。

## 可选：仿真电脑和 GPU 服务器分开

服务器安装同一仓库、下载模型后运行：

```bash
python -m rosorin_vla.server --port 8767
```

它只监听服务器的 `127.0.0.1`。在仿真电脑另一个终端建立自己的 SSH 隧道：

```bash
ssh -N -L 8767:127.0.0.1:8767 YOUR_USER@YOUR_GPU_HOST
```

仿真电脑只需基础依赖，无需安装 Torch/LeRobot 或下载模型：

```bash
python -m pip install -e .
python -m rosorin_vla.demo --endpoint http://127.0.0.1:8767 --seed 0
```

同样只传一次初始 RGB/状态。没有服务、模型不匹配、动作越界或非法时会报告错误，**不会悄悄切换专家**。此独立项目不依赖原工作区的 ROS 话题；原 ROS 接入未包含在这个精简发行版中。

## 验证与目录

CUDA 下六组初始输入的可移植推理与原运行方式逐项一致，原模型文件没有修改。发布版的实际服务→MuJoCo六场景结果见 [验收记录](docs/ACCEPTANCE.json)。此前的320场景评估只覆盖窄工作区，不能当作真实机器人或任意位置的成功率；参见模型说明。

```text
rosorin_vla/          模型下载、推理、独立演示、可选推理服务
rosorin_mujoco_sim/   原厂仿真核心、关节伺服、首帧动作执行器
worlds/ assets/      MuJoCo世界、机器人网格、URDF与参数来源
tests/              协议与故障检查；fixtures仅为初始观测
docs/               视频、模型说明、验收记录
models/ outputs/    下载模型与本地结果，Git忽略
```

协议检查（不下载模型，不把假策略当作抓取成功）：

```bash
python -m pip install -e '.[test]'
MUJOCO_GL=osmesa python -m pytest -q
```

代码及第三方材料的授权范围见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。本项目用于纯仿真，未部署真实机器人。
