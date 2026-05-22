# ChineseChess-AI

一个基于 [ChineseChess-AlphaZero](https://github.com/NeymarL/ChineseChess-AlphaZero) 的中国象棋 AI 实验项目。仓库在原项目的象棋环境、棋盘资源和教师模型基础上，加入了 PyTorch 学生网络、教师蒸馏、自博弈/教师对战训练，以及可直接运行的 PyGame 人机对弈界面。

## 功能特性

- 使用上游 AlphaZero 模型作为教师，蒸馏得到 PyTorch policy-value 学生模型。
- 支持学生自博弈、学生对教师对局、MCTS 搜索和镜像增强训练。
- 提供 PyGame 图形界面，可加载训练好的 checkpoint 与 AI 对弈。
- 可输出训练指标 CSV/曲线图，以及训练棋局 GIF 和终局 PNG。

## 项目结构

```text
.
├── play.py                         # 图形界面人机对弈入口
├── distill_cchess_alphazero.py      # 教师模型蒸馏入口
├── train.py                        # 自博弈 + 教师对战训练入口
├── ChineseChess-AlphaZero/          # 上游项目代码、棋盘资源和教师模型
├── models/                         # 学生模型 checkpoint
└── runs/                           # 训练日志、指标图和棋局可视化输出
```

## 环境准备

建议使用 Python 3.10+，并优先在虚拟环境中安装依赖。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch numpy pygame pillow matplotlib h5py
```

如果要使用 CUDA，请按你的显卡和 CUDA 版本安装对应的 PyTorch。项目默认的教师后端为 `original-pytorch-h5`，会直接读取上游 `.h5` 权重，不需要安装 TensorFlow 1.x / Keras 2.0.8。只有在显式使用 `--teacher-backend legacy-keras` 时，才需要考虑上游旧版依赖。

运行前请确认上游教师模型文件存在：

```text
ChineseChess-AlphaZero/data/model/model_best_config.json
ChineseChess-AlphaZero/data/model/model_best_weight.h5
```

## 人机对弈

直接启动默认 checkpoint：

```powershell
python play.py
```

常用参数：

```powershell
python play.py --device cuda --mcts-sims 64
python play.py --ai-move-first
python play.py --checkpoint models/cchess_adversarial_resattn.pt
python play.py --piece-style POLISH --bg-style CANVAS
```

说明：

- 默认加载 `models/cchess_adversarial_resattn.pt`。
- `--mcts-sims 0` 表示直接使用策略网络选招；数值越大，AI 每步思考越久。
- `--temperature 0` 总是选择当前最优招；大于 0 时会按策略分布采样。
- 对局记录会保存到 `ChineseChess-AlphaZero/data/play_record/`。

## 教师蒸馏

使用上游教师模型生成教师对局，并训练学生网络：

```powershell
python distill_cchess_alphazero.py --device cuda --iterations 5 --games 16
```

保存棋局 GIF 和终局图：

```powershell
python distill_cchess_alphazero.py --device cuda --iterations 1 --games 4 --visualize-games
```

快速检查环境是否可用：

```powershell
python distill_cchess_alphazero.py --dry-run
```

默认学生模型输出为：

```text
models/cchess_distilled_resattn.pt
```

## 自博弈与教师对战训练

在已有学生模型基础上继续通过自博弈和教师对战训练：

```powershell
python train.py --device cuda --iterations 10 --self-play-games 8 --teacher-games 8
```

记录训练指标并生成可视化：

```powershell
python train.py --device cuda --visualize --visualize-games
```

快速 dry-run：

```powershell
python train.py --dry-run
```

默认训练 checkpoint 为：

```text
models/cchess_adversarial_resattn.pt
```

## 常用输出

- `models/*.pt`：PyTorch 学生模型 checkpoint。
- `runs/cchess_adversarial.csv`：训练指标日志。
- `runs/cchess_adversarial.png`：训练曲线图，需要 `matplotlib`。
- `runs/distill_games/`：教师蒸馏棋局 GIF/PNG。
- `runs/adversarial_games/`：自博弈和教师对战棋局 GIF/PNG。

## 备注

- CPU 可以运行对弈和 dry-run，但完整蒸馏/训练会比较慢。
- 增大 `--mcts-sims`、`--teacher-mcts-sims`、`--games` 和 `--iterations` 通常会提升训练样本质量，也会显著增加耗时。
- 如果想从零开始训练学生模型，可以加 `--fresh` 忽略已有 checkpoint。

## 致谢

感谢 [NeymarL/ChineseChess-AlphaZero](https://github.com/NeymarL/ChineseChess-AlphaZero) 提供中国象棋 AlphaZero 的基础实现、环境、资源和预训练模型，本项目在其基础上进行学习、改造和扩展。
