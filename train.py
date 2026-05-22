from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

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
from cchess_alphazero.environment.lookup_tables import ActionLabelsRed
from distill_cchess_alphazero import (
    ACTION_SIZE,
    ACTION_TO_INDEX,
    BOARD_SHAPE,
    DEFAULT_MODEL_DIR,
    CChessDistillNet,
    CChessTrainingGameVisualizer,
    StudentConfig,
    augment_example,
    create_teacher,
    fixed_visual_state_action,
    parameter_count,
)


DEFAULT_MODEL = SCRIPT_DIR / "models/cchess_adversarial_resattn.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Chinese chess student with self-play and teacher-play.")
    parser.add_argument("--model", "--student", dest="model", type=Path, default=DEFAULT_MODEL, help="checkpoint path")
    parser.add_argument("--fresh", action="store_true", help="ignore an existing checkpoint and start from scratch")
    parser.add_argument("--device", type=str, default=None, help="cuda, cpu, or auto when omitted")
    parser.add_argument("--require-cuda", action="store_true", help="fail instead of falling back to CPU")

    parser.add_argument("--channels", type=int, default=192)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--attention-every", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.05)

    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--self-play-games", type=int, default=8, help="student self-play games per iteration")
    parser.add_argument("--teacher-games", type=int, default=8, help="student-vs-teacher games per iteration")
    parser.add_argument("--max-game-length", type=int, default=180)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--replay-size", type=int, default=120000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--value-loss-weight", type=float, default=0.5)
    parser.add_argument("--gradient-clip", type=float, default=5.0)

    parser.add_argument("--mcts-sims", type=int, default=64, help="student MCTS simulations per move")
    parser.add_argument("--teacher-mcts-sims", type=int, default=0, help="teacher MCTS simulations per move; 0 uses direct policy")
    parser.add_argument("--cpuct", type=float, default=1.5)
    parser.add_argument("--teacher-cpuct", type=float, default=1.5)
    parser.add_argument("--temp-threshold", type=int, default=18)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--eval-temperature", type=float, default=0.0, help="temperature after --temp-threshold")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--exploration-frac", type=float, default=0.25)
    parser.add_argument("--learn-after-step", type=int, default=4)
    parser.add_argument("--mirror-augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--learn-teacher-moves",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="also train on teacher moves in teacher-play games",
    )
    parser.add_argument(
        "--learn-teacher-wins",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="when the student loses to teacher, imitate the teacher's moves from that game",
    )
    parser.add_argument(
        "--teacher-side",
        choices=["alternate", "student-first", "teacher-first", "random"],
        default="alternate",
        help="who plays red in teacher-play games",
    )

    parser.add_argument("--teacher-a-config", type=Path, default=DEFAULT_MODEL_DIR / "model_best_config.json")
    parser.add_argument("--teacher-a-weight", type=Path, default=DEFAULT_MODEL_DIR / "model_best_weight.h5")
    parser.add_argument("--teacher-backend", choices=["original-pytorch-h5", "legacy-keras", "fallback"], default="original-pytorch-h5")

    parser.add_argument("--visualize", action="store_true", help="write metrics CSV and optional PNG curves")
    parser.add_argument("--visual-dir", type=Path, default=SCRIPT_DIR / "runs")
    parser.add_argument("--visual-name", type=str, default="cchess_adversarial")
    parser.add_argument("--visualize-games", action="store_true", help="save self-play/teacher-play GIFs and final-board PNGs")
    parser.add_argument("--visual-games-dir", type=Path, default=SCRIPT_DIR / "runs/adversarial_games")
    parser.add_argument("--visual-games-every", type=int, default=1)
    parser.add_argument("--visual-games-max", type=int, default=1)
    parser.add_argument("--visual-gif-duration", type=int, default=420)

    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save-examples", type=Path, default=None, help="optional .npz path for latest iteration examples")
    parser.add_argument("--dry-run", action="store_true", help="run one tiny iteration without saving checkpoint")
    args = parser.parse_args()
    if args.temperature < 0 or args.eval_temperature < 0:
        parser.error("temperatures must be non-negative")
    if args.exploration_frac < 0 or args.exploration_frac > 1:
        parser.error("--exploration-frac must be between 0 and 1")
    return args


@dataclass
class PendingExample:
    planes: np.ndarray
    policy: np.ndarray
    ply: int
    actor: str


class StudentPolicy:
    def __init__(self, model: CChessDistillNet, device: torch.device):
        self.model = model
        self.device = device
        self.name = "student"

    def policy_value(self, state: str) -> tuple[np.ndarray, float]:
        planes = torch.tensor(np.asarray([senv.state_to_planes(state)], dtype=np.float32), device=self.device)
        self.model.eval()
        with torch.inference_mode():
            logits, value = self.model(planes)
            policy = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        return mask_legal(policy, senv.get_legal_moves(state)), float(value.detach().cpu().item())


class MCTSNode:
    def __init__(self, prior: float = 1.0):
        self.prior = float(prior)
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: dict[str, MCTSNode] = {}

    @property
    def value(self) -> float:
        return 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count

    def expanded(self) -> bool:
        return bool(self.children)


class MCTS:
    def __init__(
        self,
        policy: Any,
        *,
        simulations: int,
        cpuct: float,
        rng: random.Random,
        dirichlet_alpha: float = 0.3,
        exploration_frac: float = 0.25,
    ):
        self.policy = policy
        self.simulations = max(0, int(simulations))
        self.cpuct = float(cpuct)
        self.rng = rng
        self.dirichlet_alpha = float(dirichlet_alpha)
        self.exploration_frac = float(exploration_frac)

    def search(self, root_state: str, *, temperature: float, add_noise: bool) -> tuple[str | None, np.ndarray, float]:
        policy, root_value = self.policy.policy_value(root_state)
        legal_moves = list(senv.get_legal_moves(root_state))
        if not legal_moves:
            return None, policy, root_value
        if self.simulations <= 0:
            return sample_action(policy, legal_moves, temperature, self.rng), policy, root_value

        root = MCTSNode()
        for move in legal_moves:
            root.children[move] = MCTSNode(float(policy[ACTION_TO_INDEX[move]]))
        if add_noise:
            self._add_root_noise(root)

        for _ in range(self.simulations):
            self._simulate(root, root_state)

        search_policy = np.zeros(ACTION_SIZE, dtype=np.float64)
        for move, child in root.children.items():
            search_policy[ACTION_TO_INDEX[move]] = child.visit_count
        total = search_policy.sum()
        if total > 0:
            search_policy /= total
        else:
            search_policy = policy
        return sample_action(search_policy, legal_moves, temperature, self.rng), search_policy, root_value

    def _simulate(self, root: MCTSNode, root_state: str) -> float:
        node = root
        state = root_state
        path = [node]
        while node.expanded():
            move, node = self._select_child(node)
            state = senv.step(state, move)
            path.append(node)
            done, value, _ = senv.done(state)
            if done:
                self._backpropagate(path, value)
                return value

        policy, value = self.policy.policy_value(state)
        for move in senv.get_legal_moves(state):
            node.children[move] = MCTSNode(float(policy[ACTION_TO_INDEX[move]]))
        self._backpropagate(path, value)
        return value

    def _select_child(self, node: MCTSNode) -> tuple[str, MCTSNode]:
        parent_sqrt = math.sqrt(max(1, node.visit_count))

        def score(item: tuple[str, MCTSNode]) -> float:
            _, child = item
            prior_score = self.cpuct * child.prior * parent_sqrt / (1 + child.visit_count)
            return -child.value + prior_score

        return max(node.children.items(), key=score)

    def _add_root_noise(self, root: MCTSNode) -> None:
        if not root.children or self.exploration_frac <= 0 or self.dirichlet_alpha <= 0:
            return
        moves = list(root.children)
        noise = np.asarray([self.rng.gammavariate(self.dirichlet_alpha, 1.0) for _ in moves], dtype=np.float64)
        total = noise.sum()
        if total <= 0:
            return
        noise /= total
        for move, sample in zip(moves, noise):
            child = root.children[move]
            child.prior = child.prior * (1.0 - self.exploration_frac) + float(sample) * self.exploration_frac

    @staticmethod
    def _backpropagate(path: list[MCTSNode], value: float) -> None:
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            value = -value


class MetricsVisualizer:
    def __init__(self, output_dir: Path, run_name: str):
        self.output_dir = output_dir
        self.run_name = run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / f"{run_name}.csv"
        self.rows: list[dict[str, Any]] = []

    def record(self, row: dict[str, Any]) -> None:
        self.rows.append(row)
        exists = self.csv_path.exists()
        with self.csv_path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        self._write_plot()

    def _write_plot(self) -> None:
        try:
            import matplotlib.pyplot as plt
        except Exception:
            return
        if not self.rows:
            return
        xs = [row["iteration"] for row in self.rows]
        plt.figure(figsize=(9, 5))
        plt.plot(xs, [row["policy_loss"] for row in self.rows], label="policy_loss")
        plt.plot(xs, [row["value_loss"] for row in self.rows], label="value_loss")
        plt.plot(xs, [row["teacher_student_win_rate"] for row in self.rows], label="vs_teacher_win_rate")
        plt.xlabel("iteration")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{self.run_name}.png")
        plt.close()


def mask_legal(policy: np.ndarray, legal_moves: Iterable[str]) -> np.ndarray:
    legal = list(legal_moves)
    masked = np.zeros(ACTION_SIZE, dtype=np.float64)
    for move in legal:
        masked[ACTION_TO_INDEX[move]] = max(float(policy[ACTION_TO_INDEX[move]]), 0.0)
    total = masked.sum()
    if total <= 0:
        if not legal:
            return masked
        for move in legal:
            masked[ACTION_TO_INDEX[move]] = 1.0 / len(legal)
    else:
        masked /= total
    return masked


def sample_action(policy: np.ndarray, legal_moves: list[str], temperature: float, rng: random.Random) -> str:
    legal_indexes = [ACTION_TO_INDEX[move] for move in legal_moves]
    if temperature <= 1e-6:
        return ActionLabelsRed[max(legal_indexes, key=lambda idx: policy[idx])]
    probs = np.asarray([policy[idx] for idx in legal_indexes], dtype=np.float64)
    probs = np.power(np.maximum(probs, 1e-12), 1.0 / temperature)
    probs /= probs.sum()
    return legal_moves[int(rng.choices(range(len(legal_moves)), weights=probs, k=1)[0])]


def move_temperature(args: argparse.Namespace, ply: int) -> float:
    return args.temperature if ply <= args.temp_threshold else args.eval_temperature


def finish_examples(
    pending: list[PendingExample],
    *,
    final_value: int,
    last_ply: int,
    args: argparse.Namespace,
    include_teacher: bool,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    examples: list[tuple[np.ndarray, np.ndarray, float]] = []
    for item in pending:
        if item.actor == "teacher" and not include_teacher:
            continue
        value = 0.0 if final_value == 0 else float(final_value * ((-1) ** (last_ply - item.ply + 1)))
        examples.extend(augment_example(item.planes, item.policy, value, mirror=args.mirror_augment))
    return examples


def student_result_from_pending(pending: list[PendingExample], final_value: int, last_ply: int) -> float:
    values = [
        float(final_value * ((-1) ** (last_ply - item.ply + 1)))
        for item in pending
        if item.actor == "student"
    ]
    return values[-1] if values else 0.0


def record_frame(frames: list[dict[str, Any]], state: str, action: str, ply: int, actor: str, value: float) -> None:
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


def play_game(
    *,
    args: argparse.Namespace,
    rng: random.Random,
    student_agent: MCTS,
    teacher_agent: MCTS | None,
    mode: str,
    student_red: bool,
    game_visualizer: CChessTrainingGameVisualizer | None,
    iteration: int,
    game_index: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray, float]], dict[str, Any]]:
    state = senv.INIT_STATE
    pending: list[PendingExample] = []
    frames: list[dict[str, Any]] = []
    final_value = 0
    last_ply = 0

    for ply in range(1, args.max_game_length + 1):
        done, final_value, _ = senv.done(state)
        if done:
            break
        is_student = mode == "self" or ((ply % 2 == 1) == student_red)
        actor = "student" if is_student else "teacher"
        agent = student_agent if is_student else teacher_agent
        if agent is None:
            raise RuntimeError("teacher_agent is required for teacher-play games")
        temperature = move_temperature(args, ply)
        action, policy, value = agent.search(
            state,
            temperature=temperature,
            add_noise=is_student and mode == "self",
        )
        if action is None:
            final_value = 0
            break
        if ply > args.learn_after_step:
            pending.append(PendingExample(senv.state_to_planes(state), policy, ply, actor))
        if game_visualizer is not None:
            record_frame(frames, state, action, ply, actor, value)
        state = senv.step(state, action)
        last_ply = ply
    else:
        final_value = 0

    student_result = student_result_from_pending(pending, final_value, last_ply) if mode == "teacher" else 0.0
    include_teacher = args.learn_teacher_moves or (args.learn_teacher_wins and student_result < 0)
    examples = finish_examples(
        pending,
        final_value=final_value,
        last_ply=last_ply,
        args=args,
        include_teacher=include_teacher,
    )
    if game_visualizer is not None:
        game_visualizer.render(
            frames,
            iteration=iteration,
            phase=f"{mode}_play",
            game_index=game_index,
            winner=final_value,
        )
    return examples, {
        "mode": mode,
        "plies": last_ply,
        "final_value": final_value,
        "student_result": student_result,
        "examples": len(examples),
        "include_teacher": include_teacher,
    }


def train_epoch(
    student: CChessDistillNet,
    replay: list[tuple[np.ndarray, np.ndarray, float]],
    args: argparse.Namespace,
    optimizer: optim.Optimizer,
    device: torch.device,
    rng: random.Random,
) -> tuple[float, float]:
    if not replay:
        return 0.0, 0.0
    student.train()
    policy_losses: list[float] = []
    value_losses: list[float] = []
    indices = list(range(len(replay)))
    for _ in range(args.epochs):
        rng.shuffle(indices)
        for start in range(0, len(indices), args.batch_size):
            batch = [replay[i] for i in indices[start : start + args.batch_size]]
            states = torch.tensor(np.stack([item[0] for item in batch]), dtype=torch.float32, device=device)
            policies = torch.tensor(np.stack([item[1] for item in batch]), dtype=torch.float32, device=device)
            values = torch.tensor([item[2] for item in batch], dtype=torch.float32, device=device)
            logits, predicted_values = student(states)
            log_probs = F.log_softmax(logits, dim=1)
            policy_loss = -(policies * log_probs).sum(dim=1).mean()
            value_loss = F.mse_loss(predicted_values, values)
            loss = policy_loss + args.value_loss_weight * value_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), args.gradient_clip)
            optimizer.step()
            policy_losses.append(float(policy_loss.detach().cpu()))
            value_losses.append(float(value_loss.detach().cpu()))
    return float(np.mean(policy_losses)), float(np.mean(value_losses))


def load_or_create_model(args: argparse.Namespace, device: torch.device) -> tuple[CChessDistillNet, StudentConfig, int]:
    if args.model.exists() and not args.fresh:
        checkpoint = torch.load(args.model, map_location=device)
        config = StudentConfig(**checkpoint["student_config"])
        model = CChessDistillNet(config).to(device)
        model.load_state_dict(checkpoint["model"])
        iteration = int(checkpoint.get("iteration", 0))
        print(f"Loaded model {args.model} at iteration {iteration}")
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


def save_checkpoint(
    path: Path,
    *,
    student: CChessDistillNet,
    student_config: StudentConfig,
    optimizer: optim.Optimizer,
    iteration: int,
    replay_size: int,
    args: argparse.Namespace,
    metrics: dict[str, Any],
    teacher: Any | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    train_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    torch.save(
        {
            "model": student.state_dict(),
            "optimizer": optimizer.state_dict(),
            "student_config": asdict(student_config),
            "iteration": iteration,
            "replay_size": replay_size,
            "action_labels": ActionLabelsRed,
            "teacher": None
            if teacher is None
            else {
                "config": str(teacher.config_path),
                "weight": str(teacher.weight_path),
                "digest": teacher.digest,
                "fallback": teacher.using_fallback,
                "backend": teacher.backend,
            },
            "train_args": train_args,
            "metrics": metrics,
        },
        path,
    )


def save_examples(path: Path, examples: list[tuple[np.ndarray, np.ndarray, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        states=np.stack([item[0] for item in examples]).astype(np.float32),
        policies=np.stack([item[1] for item in examples]).astype(np.float32),
        values=np.asarray([item[2] for item in examples], dtype=np.float32),
    )


def teacher_student_red(args: argparse.Namespace, iteration: int, game_index: int, rng: random.Random) -> bool:
    if args.teacher_side == "student-first":
        return True
    if args.teacher_side == "teacher-first":
        return False
    if args.teacher_side == "random":
        return bool(rng.getrandbits(1))
    return (iteration + game_index) % 2 == 0


def main() -> None:
    args = parse_args()
    if args.dry_run:
        args.iterations = 1
        args.self_play_games = min(args.self_play_games, 1)
        args.teacher_games = min(args.teacher_games, 1)
        args.max_game_length = min(args.max_game_length, 16)
        args.mcts_sims = min(args.mcts_sims, 4)
        args.teacher_mcts_sims = min(args.teacher_mcts_sims, 2)
        args.batch_size = min(args.batch_size, 8)
        args.epochs = 1

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and not str(device_name).startswith("cuda"):
        raise RuntimeError("CUDA was required, but torch.cuda.is_available() is false.")
    device = torch.device(device_name)

    student, student_config, start_iteration = load_or_create_model(args, device)
    optimizer = optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    replay: deque[tuple[np.ndarray, np.ndarray, float]] = deque(maxlen=args.replay_size)
    student_policy = StudentPolicy(student, device)
    teacher = None
    if args.teacher_games > 0:
        teacher = create_teacher(
            args.teacher_backend,
            args.teacher_a_config,
            args.teacher_a_weight,
            name="teacher",
            device=device,
        )

    metrics_visualizer = MetricsVisualizer(args.visual_dir, args.visual_name) if args.visualize else None
    game_visualizer = (
        CChessTrainingGameVisualizer(
            args.visual_games_dir,
            every=args.visual_games_every,
            max_games_per_iteration=args.visual_games_max,
            duration=args.visual_gif_duration,
        )
        if args.visualize_games
        else None
    )

    print(
        json.dumps(
            {
                "student_parameters": parameter_count(student),
                "board_shape": BOARD_SHAPE,
                "action_size": ACTION_SIZE,
                "device": str(device),
                "student_mcts_sims": args.mcts_sims,
                "teacher_mcts_sims": args.teacher_mcts_sims,
                "teacher_backend": None if teacher is None else teacher.backend,
            },
            ensure_ascii=True,
        ),
        flush=True,
    )

    try:
        for iteration in range(start_iteration + 1, start_iteration + args.iterations + 1):
            iteration_examples: list[tuple[np.ndarray, np.ndarray, float]] = []
            stats: list[dict[str, Any]] = []
            student_agent = MCTS(
                student_policy,
                simulations=args.mcts_sims,
                cpuct=args.cpuct,
                rng=rng,
                dirichlet_alpha=args.dirichlet_alpha,
                exploration_frac=args.exploration_frac,
            )
            teacher_agent = (
                MCTS(
                    teacher,
                    simulations=args.teacher_mcts_sims,
                    cpuct=args.teacher_cpuct,
                    rng=rng,
                    dirichlet_alpha=0.0,
                    exploration_frac=0.0,
                )
                if teacher is not None
                else None
            )

            for game_index in range(args.self_play_games):
                recorder = (
                    game_visualizer
                    if game_visualizer and game_visualizer.should_capture(iteration, game_index)
                    else None
                )
                examples, game_stats = play_game(
                    args=args,
                    rng=rng,
                    student_agent=student_agent,
                    teacher_agent=None,
                    mode="self",
                    student_red=True,
                    game_visualizer=recorder,
                    iteration=iteration,
                    game_index=game_index,
                )
                iteration_examples.extend(examples)
                stats.append(game_stats)
                print(
                    f"iter {iteration} self-play {game_index + 1}/{args.self_play_games} "
                    f"| plies {game_stats['plies']} | examples {len(examples)}",
                    flush=True,
                )

            for game_index in range(args.teacher_games):
                visual_index = args.self_play_games + game_index
                recorder = (
                    game_visualizer
                    if game_visualizer and game_visualizer.should_capture(iteration, visual_index)
                    else None
                )
                examples, game_stats = play_game(
                    args=args,
                    rng=rng,
                    student_agent=student_agent,
                    teacher_agent=teacher_agent,
                    mode="teacher",
                    student_red=teacher_student_red(args, iteration, game_index, rng),
                    game_visualizer=recorder,
                    iteration=iteration,
                    game_index=visual_index,
                )
                iteration_examples.extend(examples)
                stats.append(game_stats)
                print(
                    f"iter {iteration} teacher-play {game_index + 1}/{args.teacher_games} "
                    f"| student_result {game_stats['student_result']:+.0f} "
                    f"| plies {game_stats['plies']} | examples {len(examples)}",
                    flush=True,
                )

            replay.extend(iteration_examples)
            if args.save_examples and iteration_examples:
                save_examples(args.save_examples, iteration_examples)

            policy_loss, value_loss = train_epoch(student, list(replay), args, optimizer, device, rng)
            teacher_games = [row for row in stats if row["mode"] == "teacher"]
            teacher_student_wins = sum(1 for row in teacher_games if row["student_result"] > 0)
            teacher_student_losses = sum(1 for row in teacher_games if row["student_result"] < 0)
            teacher_student_draws = len(teacher_games) - teacher_student_wins - teacher_student_losses
            teacher_win_rate = teacher_student_wins / len(teacher_games) if teacher_games else 0.0
            metrics = {
                "iteration": iteration,
                "replay": len(replay),
                "new_examples": len(iteration_examples),
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "self_play_games": args.self_play_games,
                "teacher_games": args.teacher_games,
                "teacher_student_wins": teacher_student_wins,
                "teacher_student_losses": teacher_student_losses,
                "teacher_student_draws": teacher_student_draws,
                "teacher_student_win_rate": teacher_win_rate,
                "avg_plies": float(np.mean([row["plies"] for row in stats])) if stats else 0.0,
            }
            if metrics_visualizer is not None:
                metrics_visualizer.record(metrics)
            if not args.dry_run:
                save_checkpoint(
                    args.model,
                    student=student,
                    student_config=student_config,
                    optimizer=optimizer,
                    iteration=iteration,
                    replay_size=len(replay),
                    args=args,
                    metrics=metrics,
                    teacher=teacher,
                )
            print(
                f"iter {iteration}/{start_iteration + args.iterations} | replay {len(replay)} "
                f"| policy_loss {policy_loss:.4f} value_loss {value_loss:.4f} "
                f"| vs_teacher W/D/L {teacher_student_wins}/{teacher_student_draws}/{teacher_student_losses} "
                f"| saved {args.model if not args.dry_run else '[dry-run]'}",
                flush=True,
            )
    finally:
        if teacher is not None:
            teacher.close()


if __name__ == "__main__":
    main()
