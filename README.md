# ChineseChess-AI

基于 [ChineseChess-AlphaZero](https://github.com/NeymarL/ChineseChess-AlphaZero) 的中国象棋 AI 实验项目。当前仓库在上游象棋环境、棋盘资源和预训练教师模型基础上，补充了现代 PyTorch 学生网络、教师蒸馏、学生自博弈/教师对抗训练、SQLite 棋谱数据集监督训练，以及可直接运行的 PyGame 人机对弈界面。

## 功能实现

- 直接读取上游 `model_best_weight.h5`，不依赖 TensorFlow 1.x。
- 支持 PyTorch 学生网络：残差块、SE、可选空间自注意力、policy-value 双头。
- 支持三条训练路线：
  - `distill_cchess_alphazero.py`：原版 AlphaZero 教师对局蒸馏。
  - `train_adversarial.py`：学生自博弈 + 学生与教师对抗强化训练。
  - `train_dataset.py`：SQLite 大规模棋谱/局面数据监督训练。
- 支持训练棋局 GIF、终局 PNG、训练指标 CSV/PNG。
- 支持人机对弈 UI：MCTS、全局 temperature、AI 落子前闪烁提示、将军大字提示、终局不自动关闭。

## 项目结构

```text
.
├── play.py                         # 加载学生模型与人对弈
├── distill_cchess_alphazero.py      # ChineseChess-AlphaZero 教师蒸馏
├── train_adversarial.py            # 自博弈 + 与教师对抗训练
├── train_dataset.py                # SQLite 棋谱数据集监督训练
├── ChineseChess-AlphaZero/          # 上游项目、棋盘资源、教师模型
├── train_data/                     # SQLite 棋谱数据集
├── models/                         # PyTorch checkpoint
└── runs/                           # 指标、曲线、GIF、PNG
```

## 环境准备

建议 Python 3.10+。

```powershell
cd D:\Python\ChineseChess-AI

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch numpy pygame pillow matplotlib h5py tqdm
```

如果要用 CUDA，请安装与你显卡匹配的 PyTorch CUDA 版。项目里的训练命令通常使用：

```powershell
--device cuda --require-cuda
```

教师模型默认路径：

```text
ChineseChess-AlphaZero/data/model/model_best_config.json
ChineseChess-AlphaZero/data/model/model_best_weight.h5
```

默认教师后端是 `original-pytorch-h5`，会直接用 PyTorch 读取 Keras H5 权重，不需要 TensorFlow。只有使用 `--teacher-backend legacy-keras` 时才需要旧版 TensorFlow/Keras 环境。

## 推荐启动命令

### 1. 人机对弈

```powershell
python play.py `
  --checkpoint models/cchess_dataset_midgame.pt `
  --device cuda `
  --require-cuda `
  --mcts-sims 128 `
  --global-temperature 0.05 `
  --ai-move-preview-seconds 2
```

如果只是快速启动：

```powershell
python play.py --device cuda --require-cuda --mcts-sims 64
```

### 2. 教师蒸馏

```powershell
python distill_cchess_alphazero.py `
  --device cuda `
  --require-cuda `
  --teacher-backend original-pytorch-h5 `
  --fresh `
  --student models/cchess_distilled_resattn.pt `
  --iterations 20 `
  --games 64 `
  --max-game-length 160 `
  --temperature-moves 18 `
  --temperature 1.0 `
  --eval-temperature 0.05 `
  --learn-after-step 4 `
  --replay-size 120000 `
  --channels 192 `
  --blocks 10 `
  --attention-every 4 `
  --dropout 0.05 `
  --batch-size 128 `
  --epochs 2 `
  --lr 2e-4 `
  --weight-decay 1e-4 `
  --value-loss-weight 0.35 `
  --visualize-games `
  --visual-games-dir runs/distill_games `
  --visual-games-every 1 `
  --visual-games-max 4 `
  --visual-gif-duration 420 `
  --visual-gif-max-frames 120
```

### 3. 自博弈 + 教师对抗训练

```powershell
python train_adversarial.py `
  --device cuda `
  --require-cuda `
  --model models/cchess_adversarial_resattn.pt `
  --channels 192 `
  --blocks 10 `
  --attention-every 4 `
  --dropout 0.05 `
  --iterations 20 `
  --self-play-games 16 `
  --teacher-games 16 `
  --max-game-length 220 `
  --mcts-sims 64 `
  --teacher-mcts-sims 0 `
  --cpuct 1.5 `
  --teacher-cpuct 1.5 `
  --temperature-moves 18 `
  --temperature 1.0 `
  --eval-temperature 0.05 `
  --dirichlet-alpha 0.3 `
  --exploration-frac 0.25 `
  --learn-after-step 4 `
  --batch-size 128 `
  --epochs 2 `
  --replay-size 120000 `
  --lr 2e-4 `
  --weight-decay 1e-4 `
  --value-loss-weight 0.5 `
  --teacher-backend original-pytorch-h5 `
  --teacher-side alternate `
  --visualize `
  --visual-dir runs `
  --visual-name cchess_adversarial `
  --visualize-games `
  --visual-games-dir runs/adversarial_games `
  --visual-games-every 1 `
  --visual-games-max 4 `
  --visual-gif-max-frames 120
```

### 4. SQLite 数据集监督训练

推荐先用 `depth_11_12m` 做开中盘训练：

```powershell
python train_dataset.py `
  --train-db train_data/train_dat_pkf_depth_11_12m/train/XQDB.db `
  --val-db train_data/train_dat_pkf_depth_11_12m/val/XQDB.db `
  --device cuda `
  --require-cuda `
  --model models/cchess_dataset_midgame.pt `
  --init-from models/cchess_adversarial_resattn.pt `
  --epochs 10 `
  --batch-size 512 `
  --max-games-per-epoch 20000 `
  --val-games 5000 `
  --lr 2e-4 `
  --weight-decay 1e-4 `
  --value-loss-weight 0.35 `
  --value-source score `
  --score-scale 600 `
  --mirror-augment `
  --metrics-csv runs/cchess_dataset_midgame.csv `
  --eval-teacher-games 16 `
  --eval-student-mcts-sims 64 `
  --eval-teacher-mcts-sims 0 `
  --eval-max-game-length 1000 `
  --eval-temperature-moves 18 `
  --eval-opening-temperature 1.0 `
  --eval-temperature 0.25 `
  --eval-visualize-games `
  --eval-games-dir runs/dataset_eval_games `
  --eval-games-max 10 `
  --eval-gif-max-frames 500
```

小规模测试命令：

```powershell
python train_dataset.py `
  --train-db train_data/train_dat_pkf_depth_11_12m/train/XQDB.db `
  --val-db train_data/train_dat_pkf_depth_11_12m/val/XQDB.db `
  --device cuda `
  --require-cuda `
  --model models/cchess_dataset_5k_test.pt `
  --init-from models/cchess_adversarial_resattn.pt `
  --epochs 1 `
  --batch-size 256 `
  --max-games-per-epoch 5000 `
  --val-games 500 `
  --value-source score `
  --score-scale 600 `
  --mirror-augment `
  --metrics-csv runs/cchess_dataset_5k_test.csv `
  --eval-teacher-games 4 `
  --eval-student-mcts-sims 64 `
  --eval-teacher-mcts-sims 0 `
  --eval-max-game-length 500 `
  --eval-temperature-moves 18 `
  --eval-opening-temperature 1.0 `
  --eval-temperature 0.25 `
  --eval-visualize-games `
  --eval-games-dir runs/dataset_eval_games `
  --eval-games-max 4 `
  --eval-gif-max-frames 120
```

## 数据集说明

本项目使用的象棋训练数据来自 ModelScope 数据集 [nowcan/xiangqi_train_data](https://modelscope.cn/datasets/nowcan/xiangqi_train_data)。如需复现实验，请先从该页面下载数据，并放入 `train_data/` 下对应目录。

当前支持的数据集格式为 SQLite：

```text
table: XQPGN
column: PGN
```

每条 `PGN` 是 JSON 字符串，包含：

- `start_fen`：起始局面，黑上红下。
- `skip_step`：开头随机步数，这些随机步默认不学习。
- `result`：`1` 红胜，`0` 和棋，`-1` 黑胜，`2` 含随机招法，最终结果无参考价值。
- `mvs`：招法列表。
- `mv`：2 字节无符号整数，低字节为起点，高字节为终点。
- `score`：走完该招后的局面分。

已观察到的数据集差异：

| 数据集 | 倾向 | 规模 | 特点 |
|---|---|---:|---|
| `train_dat_pkf_depth_11_12m` | 开中盘/全子 | 约 1200 万局 | 子力接近完整，适合先训练整体棋感 |
| `train_dat_pkf_depth_11_7m` | 残局 | 约 746 万局 | 残局更多，深度 11 |
| `train_dat_pkf_depth_16` | 残局 | 约 401 万局 | 深度 16，残局质量更高 |

注意 `train_dat_pkf_depth_11_7m` 的验证库路径是：

```text
train_data/train_dat_pkf_depth_11_7m/val/XQDB_v.db
```

## 脚本参数详解

### play.py

用途：加载学生 checkpoint，与人类在 PyGame UI 中对弈。

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--checkpoint` | `models/cchess_adversarial_resattn.pt` | 要加载的学生模型 checkpoint |
| `--device` | auto | `cuda` 或 `cpu`；省略时自动选择 |
| `--require-cuda` | false | 要求必须使用 CUDA，否则报错 |
| `--ai-move-first` | false | AI 执红先走 |
| `--piece-style` | `WOOD` | 棋子样式：`WOOD/POLISH/DELICATE` |
| `--bg-style` | `WOOD` | 棋盘样式 |
| `--temperature`, `--global-temperature` | `0.0` | 全局走子温度，直出 policy 和 MCTS 都生效；0 表示贪心 |
| `--top-k` | `5` | 右侧显示前 N 个候选招 |
| `--mcts-sims` | `0` | AI 每步 MCTS 次数；0 表示直接 policy |
| `--cpuct` | `1.5` | MCTS 探索系数 |
| `--ai-move-preview-seconds` | `2.0` | AI 落子前闪烁提示秒数 |

UI 行为：

- AI 每步落子前会闪烁起点、终点和移动线。
- 被将军会显示大字提示。
- 棋局结束后窗口不会自动关闭。
- 棋谱保存到 `ChineseChess-AlphaZero/data/play_record/`。

### distill_cchess_alphazero.py

用途：让 ChineseChess-AlphaZero 教师模型互相对弈，蒸馏训练 PyTorch 学生。

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--teacher-a-config` | `ChineseChess-AlphaZero/data/model/model_best_config.json` | 教师 A config |
| `--teacher-a-weight` | `ChineseChess-AlphaZero/data/model/model_best_weight.h5` | 教师 A 权重 |
| `--teacher-b-config` | 同 A | 教师 B config |
| `--teacher-b-weight` | 同 A | 教师 B 权重 |
| `--teacher-backend` | `original-pytorch-h5` | `original-pytorch-h5/legacy-keras/fallback` |
| `--student` | `models/cchess_distilled_resattn.pt` | 学生 checkpoint 输出路径 |
| `--fresh` | false | 忽略已有学生，从头创建 |
| `--device` | auto | `cuda` 或 `cpu` |
| `--require-cuda` | false | 不允许回退 CPU |
| `--iterations` | `5` | 训练迭代轮数 |
| `--games` | `16` | 每轮教师对局数 |
| `--max-game-length` | `160` | 单局最大半回合数 |
| `--temperature-moves` | `18` | 前 N 个半回合使用开局温度 |
| `--temperature` | `1.0` | 前 N 步采样温度 |
| `--eval-temperature` | `0.0` | N 步之后采样温度 |
| `--learn-after-step` | `4` | 前 N 步不收训练样本 |
| `--replay-size` | `120000` | replay buffer 最大样本数 |
| `--epochs` | `2` | 每轮训练 epoch 数 |
| `--batch-size` | `128` | batch 大小 |
| `--lr` | `2e-4` | 学习率 |
| `--weight-decay` | `1e-4` | AdamW 权重衰减 |
| `--channels` | `192` | 网络通道数 |
| `--blocks` | `10` | 残差块数 |
| `--attention-every` | `4` | 每 N 个残差块插入一次空间注意力；0 关闭 |
| `--dropout` | `0.05` | dropout |
| `--value-loss-weight` | `0.35` | value loss 权重 |
| `--mirror-augment` | true | 左右镜像增强 |
| `--no-mirror-augment` | false | 关闭镜像增强 |
| `--seed` | `11` | 随机种子 |
| `--save-examples` | None | 保存最近样本为 `.npz` |
| `--visualize-games` | false | 保存教师对局 GIF/PNG |
| `--visual-games-dir` | `runs/distill_games` | GIF 输出目录 |
| `--visual-games-every` | `1` | 每 N 轮保存一次 |
| `--visual-games-max` | `1` | 每轮最多保存几盘 |
| `--visual-gif-duration` | `420` | GIF 每帧毫秒数 |
| `--visual-gif-max-frames` | `0` | 每个 GIF 最大帧数；0 不限制 |
| `--dry-run` | false | 小规模检查，不保存 checkpoint |

### train_adversarial.py

用途：学生自博弈，以及学生与教师对抗训练。

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--model`, `--student` | `models/cchess_adversarial_resattn.pt` | checkpoint 路径 |
| `--fresh` | false | 忽略已有 checkpoint |
| `--device` | auto | `cuda/cpu` |
| `--require-cuda` | false | 要求 CUDA |
| `--channels` | `192` | 网络通道数 |
| `--blocks` | `10` | 残差块数 |
| `--attention-every` | `4` | 空间注意力间隔 |
| `--dropout` | `0.05` | dropout |
| `--iterations` | `10` | 训练迭代轮数 |
| `--self-play-games` | `8` | 每轮学生自博弈盘数 |
| `--teacher-games` | `8` | 每轮学生打教师盘数 |
| `--max-game-length` | `180` | 单局最大半回合 |
| `--epochs` | `2` | 每轮训练 epoch 数 |
| `--batch-size` | `128` | batch 大小 |
| `--replay-size` | `120000` | replay buffer 大小 |
| `--lr` | `2e-4` | 学习率 |
| `--weight-decay` | `1e-4` | 权重衰减 |
| `--value-loss-weight` | `0.5` | value loss 权重 |
| `--gradient-clip` | `5.0` | 梯度裁剪 |
| `--mcts-sims` | `64` | 学生每步 MCTS 次数 |
| `--teacher-mcts-sims` | `0` | 教师每步 MCTS 次数；0 为直接 policy |
| `--cpuct` | `1.5` | 学生 MCTS 探索系数 |
| `--teacher-cpuct` | `1.5` | 教师 MCTS 探索系数 |
| `--temp-threshold`, `--temperature-moves` | `18` | 前 N 步使用开局温度 |
| `--temperature` | `1.0` | 前 N 步温度 |
| `--eval-temperature` | `0.0` | N 步之后温度 |
| `--dirichlet-alpha` | `0.3` | 自博弈根节点 Dirichlet 噪声 alpha |
| `--exploration-frac` | `0.25` | 根节点噪声混合比例 |
| `--learn-after-step` | `4` | 前 N 步不收训练样本 |
| `--mirror-augment` | true | 镜像增强 |
| `--no-mirror-augment` | false | 关闭镜像增强 |
| `--learn-teacher-moves` | false | 不管输赢都学习教师招法 |
| `--no-learn-teacher-moves` | true | 关闭上述行为 |
| `--learn-teacher-wins` | true | 学生输棋时学习教师赢法 |
| `--no-learn-teacher-wins` | false | 只让教师当陪练，不模仿教师招 |
| `--teacher-side` | `alternate` | `alternate/student-first/teacher-first/random` |
| `--teacher-a-config` | model config | 教师 config |
| `--teacher-a-weight` | model weight | 教师权重 |
| `--teacher-backend` | `original-pytorch-h5` | 教师后端 |
| `--visualize` | false | 保存训练 CSV/PNG |
| `--visual-dir` | `runs` | 指标输出目录 |
| `--visual-name` | `cchess_adversarial` | 指标文件名 |
| `--visualize-games` | false | 保存训练棋局 GIF/PNG |
| `--visual-games-dir` | `runs/adversarial_games` | 棋局 GIF 目录 |
| `--visual-games-every` | `1` | 每 N 轮保存一次 |
| `--visual-games-max` | `1` | 每轮最多保存几盘 |
| `--visual-gif-duration` | `420` | GIF 每帧毫秒 |
| `--visual-gif-max-frames` | `0` | GIF 最大帧数；0 不限制 |
| `--seed` | `7` | 随机种子 |
| `--save-examples` | None | 保存本轮样本 `.npz` |
| `--dry-run` | false | 小规模检查，不保存 |

默认训练节奏不是一局一练，而是一轮收集 `self-play-games + teacher-games` 盘，再集中训练 `epochs` 次。

### train_dataset.py

用途：从 SQLite 大规模棋谱数据流式训练。不会把整个数据库加载进内存或显存，只会按 `chunk-size` 读 PGN，攒够 `batch-size` 后送入 GPU。

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--train-db` | `train_data/train_dat_pkf_depth_16/train/XQDB.db` | 训练 SQLite |
| `--val-db` | `train_data/train_dat_pkf_depth_16/val/XQDB.db` | 验证 SQLite |
| `--model` | `models/cchess_dataset_resattn.pt` | 输出 checkpoint |
| `--init-from` | None | 仅初始化用的 checkpoint；若 `--model` 已存在且未 `--fresh`，优先加载 `--model` |
| `--fresh` | false | 忽略已有 `--model` |
| `--device` | auto | `cuda/cpu` |
| `--require-cuda` | false | 要求 CUDA |
| `--channels` | `192` | 网络通道数 |
| `--blocks` | `10` | 残差块数 |
| `--attention-every` | `4` | 注意力间隔 |
| `--dropout` | `0.05` | dropout |
| `--epochs` | `1` | 训练 epoch 数 |
| `--batch-size` | `256` | batch 大小 |
| `--lr` | `2e-4` | 学习率 |
| `--weight-decay` | `1e-4` | 权重衰减 |
| `--value-loss-weight` | `0.35` | value loss 权重 |
| `--gradient-clip` | `5.0` | 梯度裁剪 |
| `--max-games-per-epoch` | `200000` | 每轮最多读取多少盘；0 表示扫完整库 |
| `--max-positions-per-epoch` | `0` | 每轮最多训练局面数；0 不限制 |
| `--val-games` | `2000` | 每轮验证棋局数 |
| `--save-every` | `1` | 每 N 个 epoch 保存一次 |
| `--value-source` | `score` | `score/result/mixed` |
| `--score-scale` | `600` | `tanh(score / scale)` 的缩放 |
| `--result-mix` | `0.2` | mixed 模式最终胜负权重 |
| `--exclude-result-2` | false | 跳过 `result=2` 棋局；通常不建议开 |
| `--learn-random-prefix` | false | 学习 `skip_step` 随机开局招；默认不学 |
| `--mirror-augment` | true | 镜像增强 |
| `--no-mirror-augment` | false | 关闭镜像增强 |
| `--check-legal` | false | 用规则生成器检查合法招；很慢，只建议抽查 |
| `--start-rowid` | `0` | 从指定 rowid 后开始读 |
| `--chunk-size` | `4096` | SQLite 每次读取 PGN 数 |
| `--metrics-csv` | `runs/cchess_dataset_train.csv` | 指标 CSV |
| `--progress` | true | 显示进度条 |
| `--no-progress` | false | 关闭进度条 |
| `--eval-teacher-games` | `0` | 每轮后与教师评估盘数 |
| `--eval-student-mcts-sims` | `64` | 评估时学生 MCTS 次数 |
| `--eval-teacher-mcts-sims` | `0` | 评估时教师 MCTS 次数 |
| `--eval-cpuct` | `1.5` | 评估学生 cpuct |
| `--eval-teacher-cpuct` | `1.5` | 评估教师 cpuct |
| `--eval-max-game-length` | `220` | 评估单局最大半回合 |
| `--eval-temperature-moves` | `0` | 评估前 N 步使用开局温度 |
| `--eval-opening-temperature` | `1.0` | 评估前 N 步温度 |
| `--eval-temperature` | `0.0` | N 步之后温度 |
| `--eval-teacher-side` | `alternate` | 评估谁执红 |
| `--eval-teacher-config` | model config | 评估教师 config |
| `--eval-teacher-weight` | model weight | 评估教师权重 |
| `--eval-teacher-backend` | `original-pytorch-h5` | 评估教师后端 |
| `--eval-visualize-games` | false | 保存评估对局 GIF |
| `--eval-games-dir` | `runs/dataset_eval_games` | 评估 GIF 目录 |
| `--eval-games-max` | `2` | 每轮最多保存几盘评估 GIF |
| `--eval-gif-duration` | `420` | 评估 GIF 每帧毫秒 |
| `--eval-gif-max-frames` | `80` | 评估 GIF 最大帧数 |
| `--eval-seed` | `17` | 评估随机种子 |
| `--dry-run` | false | 小规模检查，不保存 |

## 训练日志

示例：

```text
epoch 2 | train 1266608 | policy_loss 2.4106 value_loss 0.5679 | acc 0.306 | val_acc 0.233 top5 0.593 | eval W/D/L 0/2/2 | saved models\cchess_dataset_5k_test.pt
```

- `train`：本 epoch 实际训练局面数，镜像增强后会翻倍。
- `policy_loss`：走法监督损失，越低越好。
- `value_loss`：局面价值损失，使用 `score` 时可能波动较大。
- `acc`：训练集 top-1 招法命中率。
- `val_acc`：验证集 top-1 命中率。
- `top5`：验证集前 5 候选包含原招的比例。
- `eval W/D/L`：当前学生与教师评估对局的胜/未分胜负/负。

`D` 常常表示走到 `max-game-length` 后截断，并不一定是规则和棋。

## 输出文件

- `models/*.pt`：学生模型 checkpoint。
- `runs/*.csv`：训练指标日志。
- `runs/*.png`：训练曲线。
- `runs/distill_games/`：蒸馏教师对局 GIF/PNG。
- `runs/adversarial_games/`：自博弈/教师对抗训练 GIF/PNG。
- `runs/dataset_eval_games/`：数据集训练后评估对局 GIF/PNG。
- `ChineseChess-AlphaZero/data/play_record/`：人机对弈棋谱。

## 注意

- `--fresh` 会忽略已有模型，从头创建；继续训练时不要加。
- 如果想从旧模型初始化但保存到新文件，使用 `--init-from old.pt --model new.pt`。
- `--eval-teacher-mcts-sims 0` 表示教师直接 policy，不做 MCTS；要让教师更强可设为 32/64，但会慢很多。
- `--visual-gif-max-frames 0` 会保存完整 GIF，长棋局文件可能很大。
- 数据集训练完整大库很耗时，建议先用 `--max-games-per-epoch 5000` 或 `20000` 做速度和曲线测试。
- 如果显存不足，优先降低 `--batch-size`，其次降低 `--channels`。

## 致谢

感谢 [NeymarL/ChineseChess-AlphaZero](https://github.com/NeymarL/ChineseChess-AlphaZero) 提供中国象棋 AlphaZero 基础实现、环境、资源和预训练模型。本项目在其基础上进行现代化改造、训练路线扩展和可视化实验。

感谢 ModelScope 数据集 [nowcan/xiangqi_train_data](https://modelscope.cn/datasets/nowcan/xiangqi_train_data) 的整理与发布，为本项目的数据集监督训练路线提供了大规模中国象棋棋局数据支持。
