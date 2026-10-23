# 故障排查

## CUDA不可用或显存不足

先检查 `nvidia-smi` 和 `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`。PyTorch固定版本为2.7.1；按本机驱动安装对应CUDA构建，避免直接升级LeRobot/Transformers改变预测结果。多卡服务器可以 `CUDA_VISIBLE_DEVICES=1 python -m rosorin_vla.demo`，只使用自己可用的GPU。CPU可尝试 `--device cpu`，速度与结果不属于CUDA验收范围。

## 无法打开窗口

WSL2需要WSLg，并从WSL终端运行。不要继承旧Gazebo的 `LIBGL_ALWAYS_SOFTWARE=1`。只跑结果时使用 `MUJOCO_GL=egl ... --headless`；CPU渲染需要 `sudo apt-get install libosmesa6`，再用 `MUJOCO_GL=osmesa`。图形窗口不能在没有显示服务的SSH终端直接打开，可用可选远程推理方式，在本地电脑显示MuJoCo。

## 缺少模型或校验失败

运行 `python -m rosorin_vla.download`。压缩包必须来自对应版本的Release，不能拿其他模型的stats混用。如果目标目录已经存在但不完整，先将其移到别处，再重新下载；程序不会自动覆盖不匹配的已有模型。网络受限时可通过浏览器下载Release中的tar.gz，使用 `--asset` 从本地导入。

## 安装或导入失败

使用Python3.10/3.11和独立venv，在仓库根目录 `python -m pip install -e '.[inference]'`。必须先安装Git，固定LeRobot依赖从GitHub获取；不要手动跳过它的依赖。基础仿真环境只安装 `-e .` 时可以连接推理服务，但不能本地加载SmolVLA。

## 运行失败如何查看

结果在 `outputs/episode_*_seed_*.json`，其中有 `failure_reasons`、双指/纸面比例、纸面最长缺口、抬升、碰撞和穿透等字段；对应npz保存初始输入与完整原始预测。推理错误发生在动作开始前，会直接报告并退出，不回退到专家。

模拟步长为0.5ms，关节目标每100ms切换一次；2rad/s低层限速是原仿真伺服设置，不是输出平滑或真实舵机标定。执行后保持画面时物理暂停，这是为了保留本轮终止状态。
