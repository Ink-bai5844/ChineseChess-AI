from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sqlite3
import sys
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Any, Iterator

import numpy as np
import torch
from torch import optim
from torch.nn import functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR / "ChineseChess-AlphaZero"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import cchess_alphazero.environment.static_env as senv
from cchess_alphazero.environment.lookup_tables import ActionLabelsRed, flip_move
from distill_cchess_alphazero import (
    ACTION_SIZE,
    ACTION_TO_INDEX,
    CChessDistillNet,
    CChessTrainingGameVisualizer,
    DEFAULT_MODEL_DIR,
    StudentConfig,
    create_teacher,
    fixed_visual_state_action,
    mirror_policy,
    parameter_count,
)
from train_adversarial import MCTS, StudentPolicy


DEFAULT_DATA_ROOT = SCRIPT_DIR / "train_data/train_dat_pkf_depth_16"
DEFAULT_MODEL = SCRIPT_DIR / "models/cchess_dataset_resattn.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supervised training from SQLite PGN Chinese-chess game data.")
    parser.add_argument("--train-db", type=Path, default=DEFAULT_DATA_ROOT / "train/XQDB.db")
    parser.add_argument("--val-db", type=Path, default=DEFAULT_DATA_ROOT / "val/XQDB.db")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--init-from", type=Path, default=None, help="optional checkpoint used only for initialization")
    parser.add_argument("--fresh", action="store_true", help="ignore --model and start a new network")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--require-cuda", action="store_true")

    parser.add_argument("--channels", type=int, default=192)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--attention-every", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.05)

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--amp", choices=["off", "fp16", "bf16"], default="bf16", help="mixed precision mode on CUDA")
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True, help="allow TF32 matmul/cudnn on CUDA")
    parser.add_argument("--compile", action="store_true", help="use torch.compile for the train/validation forward path")
    parser.add_argument(
        "--compile-mode",
        choices=["default", "reduce-overhead", "max-autotune"],
        default="default",
        help="torch.compile mode; max-autotune may print Triton resource warnings on some GPUs",
    )
    parser.add_argument("--channels-last", action=argparse.BooleanOptionalAction, default=True, help="use channels_last memory format")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--value-loss-weight", type=float, default=0.35)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--max-games-per-epoch", type=int, default=200000, help="0 means scan the whole train DB")
    parser.add_argument("--max-positions-per-epoch", type=int, default=0, help="0 means no limit")
    parser.add_argument("--val-games", type=int, default=2000)
    parser.add_argument("--save-every", type=int, default=1)

    parser.add_argument(
        "--value-source",
        choices=["score", "result", "mixed"],
        default="score",
        help="score uses per-move engine score; result uses final result; mixed blends both",
    )
    parser.add_argument("--score-scale", type=float, default=600.0, help="centipawn scale before tanh(score / scale)")
    parser.add_argument("--result-mix", type=float, default=0.2, help="mixed mode: final-result weight")
    parser.add_argument("--exclude-result-2", action="store_true", help="skip games whose result is 2")
    parser.add_argument(
        "--learn-random-prefix",
        action="store_true",
        help="also learn the first skip_step random moves; their value target stays 0",
    )
    parser.add_argument("--mirror-augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--check-legal", action="store_true", help="validate moves with the project rule generator")
    parser.add_argument("--start-rowid", type=int, default=0)
    parser.add_argument("--epoch-rowid-stride", type=int, default=0, help="advance train start rowid by this amount each epoch")
    parser.add_argument("--shuffle-start-rowid", action="store_true", help="randomize train start rowid each epoch")
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--prefetch-batches", type=int, default=2, help="CPU batch prefetch queue size; 0 disables prefetch")

    parser.add_argument("--metrics-csv", type=Path, default=SCRIPT_DIR / "runs/cchess_dataset_train.csv")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True, help="show tqdm progress bars")

    parser.add_argument("--eval-teacher-games", type=int, default=0, help="student-vs-ChineseChess-AlphaZero games after each epoch")
    parser.add_argument("--eval-every", type=int, default=1, help="run teacher evaluation every N epochs; 1 means every epoch")
    parser.add_argument("--eval-student-mcts-sims", type=int, default=64)
    parser.add_argument("--eval-teacher-mcts-sims", type=int, default=0)
    parser.add_argument("--eval-cpuct", type=float, default=1.5)
    parser.add_argument("--eval-teacher-cpuct", type=float, default=1.5)
    parser.add_argument("--eval-max-game-length", type=int, default=220)
    parser.add_argument(
        "--eval-temperature-moves",
        type=int,
        default=0,
        help="use --eval-opening-temperature for the first N eval plies",
    )
    parser.add_argument("--eval-opening-temperature", type=float, default=1.0)
    parser.add_argument("--eval-temperature", type=float, default=0.0, help="eval temperature after --eval-temperature-moves")
    parser.add_argument(
        "--eval-teacher-side",
        choices=["alternate", "student-first", "teacher-first", "random"],
        default="alternate",
        help="who plays red in post-epoch teacher evaluation",
    )
    parser.add_argument("--eval-teacher-config", type=Path, default=DEFAULT_MODEL_DIR / "model_best_config.json")
    parser.add_argument("--eval-teacher-weight", type=Path, default=DEFAULT_MODEL_DIR / "model_best_weight.h5")
    parser.add_argument(
        "--eval-teacher-backend",
        choices=["original-pytorch-h5", "legacy-keras", "fallback"],
        default="original-pytorch-h5",
    )
    parser.add_argument("--eval-visualize-games", action="store_true", help="save GIFs for post-epoch teacher evaluation")
    parser.add_argument("--eval-games-dir", type=Path, default=SCRIPT_DIR / "runs/dataset_eval_games")
    parser.add_argument("--eval-games-max", type=int, default=2, help="max eval GIFs captured per epoch")
    parser.add_argument("--eval-gif-duration", type=int, default=420)
    parser.add_argument("--eval-gif-max-frames", type=int, default=80)
    parser.add_argument("--eval-seed", type=int, default=17)
    parser.add_argument("--dry-run", action="store_true", help="run a tiny train/val pass without saving")
    args = parser.parse_args()
    if args.eval_temperature_moves < 0:
        parser.error("--eval-temperature-moves must be non-negative")
    if args.eval_opening_temperature < 0 or args.eval_temperature < 0:
        parser.error("eval temperatures must be non-negative")
    return args


def square_to_xy(square: int) -> tuple[int, int]:
    x = (square & 0x0F) - 3
    y = 12 - ((square >> 4) & 0x0F)
    if x < 0 or x > 8 or y < 0 or y > 9:
        raise ValueError(f"dataset square out of board: 0x{square:02X}")
    return x, y


def decode_dataset_move(encoded: int) -> str:
    src = encoded & 0xFF
    dst = (encoded >> 8) & 0xFF
    sx, sy = square_to_xy(src)
    dx, dy = square_to_xy(dst)
    return f"{sx}{sy}{dx}{dy}"


def raw_step_red_orientation(state: str, action: str) -> str:
    sx, sy, dx, dy = [int(ch) for ch in action]
    board = senv.state_to_board(state)
    if board[sy][sx] == ".":
        raise ValueError(f"no piece at {action[:2]} in state {state}")
    board[dy][dx] = board[sy][sx]
    board[sy][sx] = "."
    return senv.board_to_state(board)


def side_from_fen(fen: str) -> str:
    parts = fen.split()
    if len(parts) < 2:
        return "w"
    return parts[1].lower()


def toggle_side(side: str) -> str:
    return "b" if side in {"w", "r", "red"} else "w"


def score_to_value(score: float, *, red_to_move: bool, scale: float) -> float:
    if scale <= 0:
        raise ValueError("--score-scale must be positive")
    red_value = math.tanh(float(score) / scale)
    return red_value if red_to_move else -red_value


def result_to_value(result: int, *, red_to_move: bool) -> float | None:
    if result == 2:
        return None
    value = float(result)
    return value if red_to_move else -value


def blend_value(
    *,
    score: float,
    result: int,
    red_to_move: bool,
    random_prefix: bool,
    args: argparse.Namespace,
) -> float:
    if random_prefix:
        return 0.0
    score_value = score_to_value(score, red_to_move=red_to_move, scale=args.score_scale)
    result_value = result_to_value(result, red_to_move=red_to_move)
    if args.value_source == "score" or result_value is None:
        return score_value
    if args.value_source == "result":
        return result_value
    mix = min(1.0, max(0.0, args.result_mix))
    return (1.0 - mix) * score_value + mix * result_value


def one_hot_policy(action: str) -> np.ndarray:
    policy = np.zeros(ACTION_SIZE, dtype=np.float32)
    policy[ACTION_TO_INDEX[action]] = 1.0
    return policy


def mirror_planes_policy(planes: np.ndarray, policy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.flip(planes, axis=2).copy().astype(np.float32), mirror_policy(policy).astype(np.float32)


def iter_rows(db_path: Path, *, start_rowid: int, chunk_size: int) -> Iterator[str]:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        last_rowid = start_rowid
        while True:
            rows = con.execute(
                "SELECT rowid, PGN FROM XQPGN WHERE rowid > ? ORDER BY rowid LIMIT ?",
                (last_rowid, chunk_size),
            ).fetchall()
            if not rows:
                break
            for rowid, pgn in rows:
                last_rowid = int(rowid)
                yield str(pgn)
    finally:
        con.close()


def sqlite_max_rowid(db_path: Path) -> int:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        value = con.execute("SELECT max(rowid) FROM XQPGN").fetchone()[0]
        return int(value or 0)
    finally:
        con.close()


def epoch_train_start_rowid(
    args: argparse.Namespace,
    *,
    epoch: int,
    max_rowid: int,
    rng: random.Random,
) -> int:
    base = max(0, int(args.start_rowid))
    if args.shuffle_start_rowid:
        if max_rowid <= 0:
            return base
        reserve = max(0, int(args.max_games_per_epoch))
        max_start = max(base, max_rowid - reserve) if reserve > 0 else max_rowid
        if max_start <= base:
            return base
        return rng.randint(base, max_start)
    stride = max(0, int(args.epoch_rowid_stride))
    if stride <= 0:
        return base
    start = base + (epoch - 1) * stride
    if max_rowid <= 0:
        return start
    reserve = max(0, int(args.max_games_per_epoch))
    max_start = max(base, max_rowid - reserve) if reserve > 0 else max_rowid
    span = max_start - base + 1
    if span <= 0:
        return base
    return base + ((start - base) % span)


def make_progress_bar(args: argparse.Namespace, *, total: int, desc: str):
    if not args.progress:
        return None
    try:
        from tqdm.auto import tqdm
    except Exception:
        return None
    return tqdm(total=total or None, desc=desc, unit="game", dynamic_ncols=True)


def iter_examples(
    db_path: Path,
    args: argparse.Namespace,
    *,
    train: bool,
    start_rowid_override: int | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray, float]]:
    games = 0
    positions = 0
    max_games = args.max_games_per_epoch if train else args.val_games
    start_rowid = start_rowid_override if train and start_rowid_override is not None else args.start_rowid if train else 0
    pbar = make_progress_bar(args, total=max_games, desc="train games" if train else "val games")
    try:
        for raw in iter_rows(db_path, start_rowid=start_rowid, chunk_size=args.chunk_size):
            if max_games and games >= max_games:
                break
            try:
                game = json.loads(raw)
                result = int(game.get("result", 2))
                games += 1
                if args.exclude_result_2 and result == 2:
                    continue
                state = senv.fen_to_state(str(game["start_fen"]))
                side = side_from_fen(str(game["start_fen"]))
                skip_step = int(game.get("skip_step", 0))
                for ply, move_info in enumerate(game.get("mvs", []), start=1):
                    red_to_move = side in {"w", "r", "red"}
                    action = decode_dataset_move(int(move_info["mv"]))
                    canonical_state = state if red_to_move else senv.fliped_state(state)
                    canonical_action = action if red_to_move else flip_move(action)
                    random_prefix = ply <= skip_step
                    if canonical_action not in ACTION_TO_INDEX:
                        state = raw_step_red_orientation(state, action)
                        side = toggle_side(side)
                        continue
                    if args.check_legal and canonical_action not in set(senv.get_legal_moves(canonical_state)):
                        state = raw_step_red_orientation(state, action)
                        side = toggle_side(side)
                        continue
                    if args.learn_random_prefix or not random_prefix:
                        planes = senv.state_to_planes(canonical_state).astype(np.float32)
                        policy = one_hot_policy(canonical_action)
                        value = blend_value(
                            score=float(move_info.get("score", 0.0)),
                            result=result,
                            red_to_move=red_to_move,
                            random_prefix=random_prefix,
                            args=args,
                        )
                        yield planes, policy, float(value)
                        positions += 1
                        if args.mirror_augment:
                            mirrored_planes, mirrored_policy = mirror_planes_policy(planes, policy)
                            yield mirrored_planes, mirrored_policy, float(value)
                            positions += 1
                        if train and args.max_positions_per_epoch and positions >= args.max_positions_per_epoch:
                            return
                    state = raw_step_red_orientation(state, action)
                    side = toggle_side(side)
            except Exception:
                continue
            finally:
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix(positions=positions)
    finally:
        if pbar is not None:
            pbar.close()


def batch_examples(
    examples: Iterator[tuple[np.ndarray, np.ndarray, float]],
    batch_size: int,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    states: list[np.ndarray] = []
    policies: list[np.ndarray] = []
    values: list[float] = []
    for state, policy, value in examples:
        states.append(state)
        policies.append(policy)
        values.append(value)
        if len(states) >= batch_size:
            yield np.stack(states), np.stack(policies), np.asarray(values, dtype=np.float32)
            states.clear()
            policies.clear()
            values.clear()
    if states:
        yield np.stack(states), np.stack(policies), np.asarray(values, dtype=np.float32)


def prefetch_batches(
    batches: Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]],
    max_prefetch: int,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if max_prefetch <= 0:
        yield from batches
        return

    sentinel = object()
    queue: Queue = Queue(maxsize=max_prefetch)

    def worker() -> None:
        try:
            for batch in batches:
                queue.put(batch)
        except BaseException as exc:
            queue.put(exc)
        finally:
            queue.put(sentinel)

    Thread(target=worker, name="dataset_batch_prefetch", daemon=True).start()
    while True:
        item = queue.get()
        if item is sentinel:
            break
        if isinstance(item, BaseException):
            raise item
        yield item


def make_train_batches(
    args: argparse.Namespace,
    *,
    train: bool,
    start_rowid_override: int | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    db_path = args.train_db if train else args.val_db
    batches = batch_examples(
        iter_examples(db_path, args, train=train, start_rowid_override=start_rowid_override),
        args.batch_size,
    )
    return prefetch_batches(batches, args.prefetch_batches)


def load_or_create_model(args: argparse.Namespace, device: torch.device) -> tuple[CChessDistillNet, StudentConfig, int]:
    source = None
    if args.model.exists() and not args.fresh:
        source = args.model
    elif args.init_from is not None:
        source = args.init_from
    if source is not None and source.exists():
        checkpoint = torch.load(source, map_location=device)
        config = StudentConfig(**checkpoint["student_config"])
        model = CChessDistillNet(config).to(device)
        model.load_state_dict(checkpoint["model"])
        iteration = 0 if source == args.init_from else int(checkpoint.get("iteration", 0))
        print(f"Loaded model from {source} at iteration {iteration}")
        return model, config, iteration
    config = StudentConfig(
        channels=args.channels,
        blocks=args.blocks,
        attention_every=args.attention_every,
        dropout=args.dropout,
    )
    model = CChessDistillNet(config).to(device)
    print(f"Created model with {parameter_count(model):,} parameters on {device}")
    return model, config, 0


def configure_torch_runtime(args: argparse.Namespace, device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.tf32)
    if args.tf32:
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def amp_dtype(args: argparse.Namespace, device: torch.device):
    if device.type != "cuda" or args.amp == "off":
        return None
    return torch.float16 if args.amp == "fp16" else torch.bfloat16


def autocast_context(args: argparse.Namespace, device: torch.device):
    dtype = amp_dtype(args, device)
    if dtype is None:
        return nullcontext()
    return torch.amp.autocast(device_type="cuda", dtype=dtype)


def prepare_runtime_model(model: CChessDistillNet, args: argparse.Namespace, device: torch.device):
    if args.channels_last and device.type == "cuda":
        model.to(memory_format=torch.channels_last)
    if args.compile:
        try:
            return torch.compile(model, mode=args.compile_mode)
        except Exception as exc:
            print(f"[train_dataset] torch.compile disabled: {exc}", flush=True)
    return model


def tensor_states(states_np: np.ndarray, args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    states = torch.tensor(states_np, dtype=torch.float32, device=device)
    if args.channels_last and device.type == "cuda":
        states = states.contiguous(memory_format=torch.channels_last)
    return states


def run_train_epoch(
    model: CChessDistillNet,
    args: argparse.Namespace,
    optimizer: optim.Optimizer,
    device: torch.device,
    *,
    start_rowid: int | None = None,
) -> dict[str, float]:
    model.train()
    policy_losses: list[float] = []
    value_losses: list[float] = []
    accuracies: list[float] = []
    examples_seen = 0
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.amp == "fp16")
    for states_np, policies_np, values_np in make_train_batches(args, train=True, start_rowid_override=start_rowid):
        states = tensor_states(states_np, args, device)
        values = torch.tensor(values_np, dtype=torch.float32, device=device)
        target_actions = torch.tensor(policies_np.argmax(axis=1), dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(args, device):
            logits, predicted_values = model(states)
            policy_loss = F.cross_entropy(logits, target_actions)
            value_loss = F.mse_loss(predicted_values.float(), values)
            loss = policy_loss + args.value_loss_weight * value_loss
        scaler.scale(loss).backward()
        if args.gradient_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        scaler.step(optimizer)
        scaler.update()

        examples_seen += int(states.size(0))
        policy_losses.append(float(policy_loss.detach().cpu()))
        value_losses.append(float(value_loss.detach().cpu()))
        accuracies.append(float((logits.argmax(dim=1) == target_actions).float().mean().detach().cpu()))
    return summarize_metrics(policy_losses, value_losses, accuracies, examples_seen)


def run_validation(model: CChessDistillNet, args: argparse.Namespace, device: torch.device) -> dict[str, float]:
    model.eval()
    policy_losses: list[float] = []
    value_losses: list[float] = []
    accuracies: list[float] = []
    top5s: list[float] = []
    examples_seen = 0
    with torch.inference_mode():
        for states_np, policies_np, values_np in make_train_batches(args, train=False):
            states = tensor_states(states_np, args, device)
            values = torch.tensor(values_np, dtype=torch.float32, device=device)
            target_actions = torch.tensor(policies_np.argmax(axis=1), dtype=torch.long, device=device)
            with autocast_context(args, device):
                logits, predicted_values = model(states)
            policy_losses.append(float(F.cross_entropy(logits, target_actions).detach().cpu()))
            value_losses.append(float(F.mse_loss(predicted_values.float(), values).detach().cpu()))
            predictions = logits.argmax(dim=1)
            top5 = logits.topk(k=5, dim=1).indices
            accuracies.append(float((predictions == target_actions).float().mean().detach().cpu()))
            top5s.append(float((top5 == target_actions.unsqueeze(1)).any(dim=1).float().mean().detach().cpu()))
            examples_seen += int(states.size(0))
    metrics = summarize_metrics(policy_losses, value_losses, accuracies, examples_seen)
    metrics["top5"] = float(np.mean(top5s)) if top5s else 0.0
    return metrics


def summarize_metrics(
    policy_losses: list[float],
    value_losses: list[float],
    accuracies: list[float],
    examples_seen: int,
) -> dict[str, float]:
    return {
        "examples": float(examples_seen),
        "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
        "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
        "policy_acc": float(np.mean(accuracies)) if accuracies else 0.0,
    }


def save_checkpoint(
    path: Path,
    *,
    model: CChessDistillNet,
    config: StudentConfig,
    optimizer: optim.Optimizer,
    iteration: int,
    args: argparse.Namespace,
    metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    train_args = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "student_config": asdict(config),
            "iteration": iteration,
            "replay_size": 0,
            "action_labels": ActionLabelsRed,
            "dataset": {
                "train_db": str(args.train_db),
                "val_db": str(args.val_db),
                "value_source": args.value_source,
            },
            "train_args": train_args,
            "metrics": metrics,
        },
        path,
    )


def append_metrics(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def eval_student_red(args: argparse.Namespace, epoch: int, game_index: int, rng: random.Random) -> bool:
    if args.eval_teacher_side == "student-first":
        return True
    if args.eval_teacher_side == "teacher-first":
        return False
    if args.eval_teacher_side == "random":
        return bool(rng.getrandbits(1))
    return (epoch + game_index) % 2 == 0


def record_eval_frame(
    frames: list[dict[str, Any]],
    *,
    state: str,
    action: str,
    ply: int,
    actor: str,
    value: float,
) -> None:
    visual_state, visual_action = fixed_visual_state_action(state, action, ply)
    frames.append(
        {
            "state": visual_state,
            "action": visual_action,
            "step": ply,
            "side": "red" if ply % 2 == 1 else "black",
            "teacher": actor,
            "value": value,
        }
    )


def eval_move_temperature(args: argparse.Namespace, ply: int) -> float:
    if ply <= args.eval_temperature_moves:
        return args.eval_opening_temperature
    return args.eval_temperature


def play_eval_teacher_game(
    *,
    args: argparse.Namespace,
    student_agent: MCTS,
    teacher_agent: MCTS,
    student_red: bool,
    visualizer: CChessTrainingGameVisualizer | None,
    epoch: int,
    game_index: int,
) -> dict[str, Any]:
    state = senv.INIT_STATE
    frames: list[dict[str, Any]] = []
    student_plies: list[int] = []
    final_value = 0
    last_ply = 0

    for ply in range(1, args.eval_max_game_length + 1):
        done, final_value, _ = senv.done(state)
        if done:
            break
        is_student = (ply % 2 == 1) == student_red
        actor = "student" if is_student else "teacher"
        agent = student_agent if is_student else teacher_agent
        temperature = eval_move_temperature(args, ply)
        action, _, value = agent.search(
            state,
            temperature=temperature,
            add_noise=False,
        )
        if action is None:
            final_value = 0
            break
        if is_student:
            student_plies.append(ply)
        if visualizer is not None:
            record_eval_frame(frames, state=state, action=action, ply=ply, actor=actor, value=value)
        state = senv.step(state, action)
        last_ply = ply
    else:
        final_value = 0

    if final_value == 0 or not student_plies:
        student_result = 0.0
    else:
        last_student_ply = student_plies[-1]
        student_result = float(final_value * ((-1) ** (last_ply - last_student_ply + 1)))

    if visualizer is not None:
        visualizer.render(
            frames,
            iteration=epoch,
            phase="dataset_eval",
            game_index=game_index,
            winner=final_value,
        )

    return {
        "plies": last_ply,
        "final_value": final_value,
        "student_result": student_result,
        "student_red": student_red,
    }


def evaluate_against_teacher(
    *,
    model: CChessDistillNet,
    teacher: Any,
    args: argparse.Namespace,
    device: torch.device,
    epoch: int,
    rng: random.Random,
) -> dict[str, Any]:
    if args.eval_teacher_games <= 0:
        return {
            "eval_games": 0,
            "eval_wins": 0,
            "eval_draws": 0,
            "eval_losses": 0,
            "eval_win_rate": 0.0,
            "eval_avg_plies": 0.0,
        }

    student_policy = StudentPolicy(model, device)
    student_agent = MCTS(
        student_policy,
        simulations=args.eval_student_mcts_sims,
        cpuct=args.eval_cpuct,
        rng=rng,
        dirichlet_alpha=0.0,
        exploration_frac=0.0,
    )
    teacher_agent = MCTS(
        teacher,
        simulations=args.eval_teacher_mcts_sims,
        cpuct=args.eval_teacher_cpuct,
        rng=rng,
        dirichlet_alpha=0.0,
        exploration_frac=0.0,
    )
    visualizer = (
        CChessTrainingGameVisualizer(
            args.eval_games_dir,
            every=1,
            max_games_per_iteration=args.eval_games_max,
            duration=args.eval_gif_duration,
            max_gif_frames=args.eval_gif_max_frames,
        )
        if args.eval_visualize_games
        else None
    )

    stats: list[dict[str, Any]] = []
    for game_index in range(args.eval_teacher_games):
        recorder = visualizer if visualizer and game_index < args.eval_games_max else None
        game_stats = play_eval_teacher_game(
            args=args,
            student_agent=student_agent,
            teacher_agent=teacher_agent,
            student_red=eval_student_red(args, epoch, game_index, rng),
            visualizer=recorder,
            epoch=epoch,
            game_index=game_index,
        )
        stats.append(game_stats)
        print(
            f"epoch {epoch} eval-teacher {game_index + 1}/{args.eval_teacher_games} "
            f"| student_result {game_stats['student_result']:+.0f} "
            f"| plies {game_stats['plies']}",
            flush=True,
        )

    wins = sum(1 for item in stats if item["student_result"] > 0)
    losses = sum(1 for item in stats if item["student_result"] < 0)
    draws = len(stats) - wins - losses
    return {
        "eval_games": len(stats),
        "eval_wins": wins,
        "eval_draws": draws,
        "eval_losses": losses,
        "eval_win_rate": wins / len(stats) if stats else 0.0,
        "eval_avg_plies": float(np.mean([item["plies"] for item in stats])) if stats else 0.0,
    }


def main() -> None:
    args = parse_args()
    if args.dry_run:
        args.epochs = 1
        args.max_games_per_epoch = min(args.max_games_per_epoch, 64) if args.max_games_per_epoch else 64
        args.val_games = min(args.val_games, 32)
        args.batch_size = min(args.batch_size, 32)
        if args.eval_teacher_games > 0:
            args.eval_teacher_games = min(args.eval_teacher_games, 2)
            args.eval_student_mcts_sims = min(args.eval_student_mcts_sims, 4)
            args.eval_teacher_mcts_sims = min(args.eval_teacher_mcts_sims, 2)
            args.eval_max_game_length = min(args.eval_max_game_length, 24)

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and not str(device_name).startswith("cuda"):
        raise RuntimeError("CUDA was required, but torch.cuda.is_available() is false.")
    device = torch.device(device_name)
    configure_torch_runtime(args, device)

    model, config, start_iteration = load_or_create_model(args, device)
    runtime_model = prepare_runtime_model(model, args, device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    eval_teacher = None
    if args.eval_teacher_games > 0:
        eval_teacher = create_teacher(
            args.eval_teacher_backend,
            args.eval_teacher_config,
            args.eval_teacher_weight,
            name="eval_teacher",
            device=device,
        )
    print(
        json.dumps(
            {
                "train_db": str(args.train_db),
                "val_db": str(args.val_db),
                "parameters": parameter_count(model),
                "device": str(device),
                "value_source": args.value_source,
                "mirror_augment": args.mirror_augment,
                "start_rowid": args.start_rowid,
                "epoch_rowid_stride": args.epoch_rowid_stride,
                "shuffle_start_rowid": args.shuffle_start_rowid,
                "amp": args.amp,
                "tf32": args.tf32,
                "compile": args.compile,
                "compile_mode": args.compile_mode,
                "channels_last": args.channels_last,
                "eval_teacher_games": args.eval_teacher_games,
                "eval_teacher_backend": None if eval_teacher is None else eval_teacher.backend,
            },
            ensure_ascii=True,
        ),
        flush=True,
    )

    eval_rng = random.Random(args.eval_seed)
    data_rng = random.Random(args.eval_seed + 1009)
    train_max_rowid = sqlite_max_rowid(args.train_db) if args.shuffle_start_rowid or args.epoch_rowid_stride > 0 else 0
    try:
        for epoch in range(start_iteration + 1, start_iteration + args.epochs + 1):
            train_start_rowid = epoch_train_start_rowid(args, epoch=epoch, max_rowid=train_max_rowid, rng=data_rng)
            print(f"epoch {epoch} train_start_rowid {train_start_rowid}", flush=True)
            train_metrics = run_train_epoch(runtime_model, args, optimizer, device, start_rowid=train_start_rowid)
            val_metrics = run_validation(runtime_model, args, device)
            should_eval = eval_teacher is not None and args.eval_every > 0 and epoch % args.eval_every == 0
            eval_metrics = (
                evaluate_against_teacher(
                    model=model,
                    teacher=eval_teacher,
                    args=args,
                    device=device,
                    epoch=epoch,
                    rng=eval_rng,
                )
                if should_eval
                else {
                    "eval_games": 0,
                    "eval_wins": 0,
                    "eval_draws": 0,
                    "eval_losses": 0,
                    "eval_win_rate": 0.0,
                    "eval_avg_plies": 0.0,
                }
            )
            row = {
                "epoch": epoch,
                "train_start_rowid": train_start_rowid,
                "train_examples": int(train_metrics["examples"]),
                "train_policy_loss": train_metrics["policy_loss"],
                "train_value_loss": train_metrics["value_loss"],
                "train_policy_acc": train_metrics["policy_acc"],
                "val_examples": int(val_metrics["examples"]),
                "val_policy_loss": val_metrics["policy_loss"],
                "val_value_loss": val_metrics["value_loss"],
                "val_policy_acc": val_metrics["policy_acc"],
                "val_top5": val_metrics["top5"],
                **eval_metrics,
            }
            append_metrics(args.metrics_csv, row)
            saved_label = "[dry-run]" if args.dry_run else "[not saved]"
            if not args.dry_run and args.save_every > 0 and epoch % args.save_every == 0:
                save_checkpoint(args.model, model=model, config=config, optimizer=optimizer, iteration=epoch, args=args, metrics=row)
                saved_label = str(args.model)
            print(
                f"epoch {epoch} | train {int(train_metrics['examples'])} "
                f"| policy_loss {train_metrics['policy_loss']:.4f} value_loss {train_metrics['value_loss']:.4f} "
                f"| acc {train_metrics['policy_acc']:.3f} | val_acc {val_metrics['policy_acc']:.3f} "
                f"top5 {val_metrics['top5']:.3f} "
                f"| eval W/D/L {eval_metrics['eval_wins']}/{eval_metrics['eval_draws']}/{eval_metrics['eval_losses']} "
                f"| saved {saved_label}",
                flush=True,
            )
    finally:
        if eval_teacher is not None:
            eval_teacher.close()


if __name__ == "__main__":
    main()
