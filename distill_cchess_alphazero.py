from __future__ import annotations

import argparse
import json
import random
import sys
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR / "ChineseChess-AlphaZero"
if not PROJECT_ROOT.exists():
    PROJECT_ROOT = SCRIPT_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cchess_alphazero.environment.static_env as senv
from cchess_alphazero.agent.model import CChessModel
from cchess_alphazero.config import Config
from cchess_alphazero.environment.lookup_tables import ActionLabelsRed, flip_move
from cchess_alphazero.lib.model_helper import load_model_weight


ACTION_SIZE = len(ActionLabelsRed)
BOARD_SHAPE = (14, 10, 9)
DEFAULT_MODEL_DIR = PROJECT_ROOT / "data/model"
DEFAULT_STUDENT = SCRIPT_DIR / "models/cchess_adversarial_resattn.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Distill ChineseChess-AlphaZero teacher-vs-teacher games into a "
            "modern PyTorch policy-value student."
        )
    )
    parser.add_argument("--teacher-a-config", type=Path, default=DEFAULT_MODEL_DIR / "model_best_config.json")
    parser.add_argument("--teacher-a-weight", type=Path, default=DEFAULT_MODEL_DIR / "model_best_weight.h5")
    parser.add_argument("--teacher-b-config", type=Path, default=None)
    parser.add_argument("--teacher-b-weight", type=Path, default=None)
    parser.add_argument(
        "--teacher-backend",
        choices=["original-pytorch-h5", "legacy-keras", "fallback"],
        default="original-pytorch-h5",
        help="teacher inference backend; original-pytorch-h5 loads the bundled Keras .h5 weights directly",
    )
    parser.add_argument("--student", type=Path, default=DEFAULT_STUDENT)
    parser.add_argument("--fresh", action="store_true", help="start a new student even if --student exists")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--require-cuda", action="store_true", help="fail instead of falling back to CPU")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--games", type=int, default=16, help="teacher-vs-teacher games per iteration")
    parser.add_argument("--max-game-length", type=int, default=160)
    parser.add_argument("--temperature-moves", type=int, default=18)
    parser.add_argument("--temperature", type=float, default=1.0, help="sampling temperature for the first --temperature-moves plies")
    parser.add_argument("--eval-temperature", type=float, default=0.0, help="sampling temperature after --temperature-moves")
    parser.add_argument("--learn-after-step", type=int, default=4)
    parser.add_argument("--replay-size", type=int, default=120000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--channels", type=int, default=192)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--attention-every", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--value-loss-weight", type=float, default=0.35)
    parser.add_argument("--mirror-augment", action="store_true", default=True)
    parser.add_argument("--no-mirror-augment", action="store_false", dest="mirror_augment")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--save-examples", type=Path, default=None, help="optional .npz path for latest examples")
    parser.add_argument("--visualize-games", action="store_true", help="save teacher game GIFs and final-board PNGs")
    parser.add_argument("--visual-games-dir", type=Path, default=SCRIPT_DIR / "runs/distill_games")
    parser.add_argument("--visual-games-every", type=int, default=1)
    parser.add_argument("--visual-games-max", type=int, default=1)
    parser.add_argument("--visual-gif-duration", type=int, default=420)
    parser.add_argument(
        "--visual-gif-max-frames",
        type=int,
        default=0,
        help="maximum frames saved into each GIF; 0 keeps every move frame",
    )
    parser.add_argument("--dry-run", action="store_true", help="build teachers/student and one tiny game, but do not save")
    args = parser.parse_args()
    if args.temperature_moves < 0:
        parser.error("--temperature-moves must be non-negative")
    if args.temperature < 0 or args.eval_temperature < 0:
        parser.error("temperatures must be non-negative")
    return args


@dataclass
class StudentConfig:
    channels: int = 192
    blocks: int = 10
    attention_every: int = 4
    dropout: float = 0.05


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 16)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.net(x).view(x.size(0), x.size(1), 1, 1)
        return x * scale


class SpatialSelfAttention(nn.Module):
    def __init__(self, channels: int, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, channels, height, width = x.shape
        seq = x.flatten(2).transpose(1, 2)
        attn_in = self.norm(seq)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        seq = seq + attn_out
        seq = seq + self.ffn(seq)
        return seq.transpose(1, 2).reshape(bsz, channels, height, width)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SqueezeExcite(channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.silu(self.bn1(self.conv1(x)), inplace=True)
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        x = self.se(x)
        return F.silu(x + residual, inplace=True)


class CChessDistillNet(nn.Module):
    def __init__(self, config: StudentConfig):
        super().__init__()
        self.config = config
        c = config.channels
        self.stem = nn.Sequential(
            nn.Conv2d(BOARD_SHAPE[0], c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )
        trunk: list[nn.Module] = []
        for i in range(config.blocks):
            trunk.append(ResidualBlock(c, dropout=config.dropout))
            if config.attention_every > 0 and (i + 1) % config.attention_every == 0:
                trunk.append(SpatialSelfAttention(c, dropout=config.dropout))
        self.trunk = nn.Sequential(*trunk)
        self.policy_head = nn.Sequential(
            nn.Conv2d(c, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64 * BOARD_SHAPE[1] * BOARD_SHAPE[2], ACTION_SIZE),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(c, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * BOARD_SHAPE[1] * BOARD_SHAPE[2], c),
            nn.SiLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(c, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.trunk(self.stem(x))
        return self.policy_head(x), self.value_head(x).squeeze(-1)


class OriginalResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels, eps=1e-3)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels, eps=1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class OriginalAlphaZeroNet(nn.Module):
    def __init__(self, channels: int, blocks: int, first_kernel: int):
        super().__init__()
        self.input_conv = nn.Conv2d(14, channels, first_kernel, padding=first_kernel // 2, bias=False)
        self.input_bn = nn.BatchNorm2d(channels, eps=1e-3)
        self.blocks = nn.ModuleList([OriginalResidualBlock(channels) for _ in range(blocks)])
        self.policy_conv = nn.Conv2d(channels, 4, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(4, eps=1e-3)
        self.policy_out = nn.Linear(4 * 10 * 9, ACTION_SIZE)
        self.value_conv = nn.Conv2d(channels, 2, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(2, eps=1e-3)
        self.value_dense = nn.Linear(2 * 10 * 9, 256)
        self.value_out = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.input_bn(self.input_conv(x)))
        for block in self.blocks:
            x = block(x)
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = self.policy_out(torch.flatten(p, 1))
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = F.relu(self.value_dense(torch.flatten(v, 1)))
        v = torch.tanh(self.value_out(v)).squeeze(-1)
        return p, v


class OriginalH5Teacher:
    def __init__(self, config_path: Path, weight_path: Path, *, name: str, device: torch.device):
        self.name = name
        self.config_path = config_path
        self.weight_path = weight_path
        self.device = device
        self.digest = CChessModel.fetch_digest(str(weight_path))
        self.model = self._load_model(weight_path).to(device)
        self.model.eval()

    @staticmethod
    def _dataset(h5, group: str, dataset: str) -> np.ndarray:
        return np.asarray(h5[f"{group}/{group}/{dataset}:0"])

    @staticmethod
    def _copy_conv(module: nn.Conv2d, h5, group: str) -> None:
        kernel = OriginalH5Teacher._dataset(h5, group, "kernel")
        module.weight.data.copy_(torch.tensor(np.transpose(kernel, (3, 2, 0, 1)), dtype=torch.float32))

    @staticmethod
    def _copy_bn(module: nn.BatchNorm2d, h5, group: str) -> None:
        module.weight.data.copy_(torch.tensor(OriginalH5Teacher._dataset(h5, group, "gamma"), dtype=torch.float32))
        module.bias.data.copy_(torch.tensor(OriginalH5Teacher._dataset(h5, group, "beta"), dtype=torch.float32))
        module.running_mean.data.copy_(torch.tensor(OriginalH5Teacher._dataset(h5, group, "moving_mean"), dtype=torch.float32))
        module.running_var.data.copy_(torch.tensor(OriginalH5Teacher._dataset(h5, group, "moving_variance"), dtype=torch.float32))

    @staticmethod
    def _copy_linear(module: nn.Linear, h5, group: str) -> None:
        kernel = OriginalH5Teacher._dataset(h5, group, "kernel")
        bias = OriginalH5Teacher._dataset(h5, group, "bias")
        module.weight.data.copy_(torch.tensor(kernel.T, dtype=torch.float32))
        module.bias.data.copy_(torch.tensor(bias, dtype=torch.float32))

    def _load_model(self, weight_path: Path) -> OriginalAlphaZeroNet:
        import h5py

        with h5py.File(weight_path, "r") as h5:
            input_group = next(name for name in h5.keys() if name.startswith("input_conv-"))
            input_kernel = self._dataset(h5, input_group, "kernel")
            first_kernel = int(input_kernel.shape[0])
            channels = int(input_kernel.shape[-1])
            block_ids = sorted(
                int(name[len("res") : name.index("_conv1")])
                for name in h5.keys()
                if name.startswith("res") and "_conv1-" in name
            )
            net = OriginalAlphaZeroNet(channels=channels, blocks=max(block_ids), first_kernel=first_kernel)
            self._copy_conv(net.input_conv, h5, input_group)
            self._copy_bn(net.input_bn, h5, "input_batchnorm")
            for idx, block in enumerate(net.blocks, start=1):
                self._copy_conv(block.conv1, h5, f"res{idx}_conv1-3-{channels}")
                self._copy_bn(block.bn1, h5, f"res{idx}_batchnorm1")
                self._copy_conv(block.conv2, h5, f"res{idx}_conv2-3-{channels}")
                self._copy_bn(block.bn2, h5, f"res{idx}_batchnorm2")
            self._copy_conv(net.policy_conv, h5, "policy_conv-1-2")
            self._copy_bn(net.policy_bn, h5, "policy_batchnorm")
            self._copy_linear(net.policy_out, h5, "policy_out")
            self._copy_conv(net.value_conv, h5, "value_conv-1-4")
            self._copy_bn(net.value_bn, h5, "value_batchnorm")
            self._copy_linear(net.value_dense, h5, "value_dense")
            self._copy_linear(net.value_out, h5, "value_out")
        return net

    @property
    def using_fallback(self) -> bool:
        return False

    @property
    def backend(self) -> str:
        return "original-pytorch-h5"

    def policy_value(self, state: str) -> tuple[np.ndarray, float]:
        planes = torch.tensor(np.asarray([senv.state_to_planes(state)], dtype=np.float32), device=self.device)
        with torch.inference_mode():
            logits, value = self.model(planes)
            policy = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        legal = senv.get_legal_moves(state)
        return mask_and_normalize_policy(policy, legal), float(value.detach().cpu().item())

    def close(self) -> None:
        pass


@dataclass
class CChessTrainingGameVisualizer:
    output_dir: Path
    every: int = 1
    max_games_per_iteration: int = 1
    cell: int = 58
    margin: int = 42
    bottom: int = 62
    duration: int = 420
    max_gif_frames: int = 0

    def should_capture(self, iteration: int, game_index: int) -> bool:
        if self.every <= 0:
            return False
        if iteration % self.every != 0:
            return False
        return game_index < self.max_games_per_iteration

    def render(
        self,
        frames: list[dict[str, Any]],
        *,
        iteration: int,
        phase: str,
        game_index: int,
        winner: int,
    ) -> None:
        if not frames:
            return
        try:
            from PIL import Image, ImageDraw, ImageFont
        except Exception as exc:
            print(f"[visualize-games] Pillow is required to write GIFs: {exc}", flush=True)
            return

        target_dir = self.output_dir / f"iter_{iteration:04d}"
        target_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{phase}_game_{game_index + 1:03d}"
        gif_frames = self._select_gif_frames(frames)
        images = [
            self._draw_frame(frame, winner=winner, phase=phase, pil=(Image, ImageDraw, ImageFont))
            for frame in gif_frames
        ]
        self._draw_frame(frames[-1], winner=winner, phase=phase, pil=(Image, ImageDraw, ImageFont)).save(
            target_dir / f"{stem}_final.png"
        )
        images[0].save(
            target_dir / f"{stem}.gif",
            save_all=True,
            append_images=images[1:],
            duration=self.duration,
            loop=0,
        )

    def _select_gif_frames(self, frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
        limit = max(0, int(self.max_gif_frames))
        if limit <= 0 or len(frames) <= limit:
            return frames
        if limit == 1:
            return [frames[-1]]
        indexes = [round(i * (len(frames) - 1) / (limit - 1)) for i in range(limit)]
        return [frames[index] for index in indexes]

    def _draw_frame(self, frame: dict[str, Any], *, winner: int, phase: str, pil: tuple[Any, Any, Any]):
        Image, ImageDraw, ImageFont = pil
        board = senv.state_to_board(frame["state"])
        board_w = self.cell * 8
        board_h = self.cell * 9
        width = self.margin * 2 + board_w
        height = self.margin * 2 + board_h + self.bottom
        image = Image.new("RGB", (width, height), "#d9b36d")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        line = "#4a2b12"
        x0 = self.margin
        y0 = self.margin
        x1 = self.margin + board_w
        y1 = self.margin + board_h

        for col in range(9):
            x = self.margin + col * self.cell
            draw.line((x, y0, x, y1), fill=line, width=2)
        for row in range(10):
            y = self.margin + row * self.cell
            draw.line((x0, y, x1, y), fill=line, width=2)

        river_y = self.margin + int(4.5 * self.cell)
        draw.rectangle((x0 + 2, river_y - 20, x1 - 2, river_y + 20), fill="#d9b36d")
        draw.text((x0 + board_w * 0.34, river_y - 6), "RIVER", fill=line, font=font)

        palace_lines = [
            ((3, 0), (5, 2)), ((5, 0), (3, 2)),
            ((3, 7), (5, 9)), ((5, 7), (3, 9)),
        ]
        for start, end in palace_lines:
            draw.line((*self._point_display(*start), *self._point_display(*end)), fill=line, width=2)

        move = frame.get("action")
        if move:
            sx, sy, dx, dy = [int(ch) for ch in move]
            sxp, syp = self._point_display(sx, sy)
            dxp, dyp = self._point_display(dx, dy)
            draw.line((sxp, syp, dxp, dyp), fill="#2563eb", width=4)
            draw.ellipse((dxp - 8, dyp - 8, dxp + 8, dyp + 8), outline="#2563eb", width=3)

        piece_names = {
            "r": "R", "k": "N", "e": "B", "m": "A", "s": "K", "c": "C", "p": "P",
            "R": "r", "K": "n", "E": "b", "M": "a", "S": "k", "C": "c", "P": "p",
        }
        for y in range(10):
            for x in range(9):
                piece = board[y][x]
                if piece == ".":
                    continue
                px, py = self._point_display(x, y)
                radius = int(self.cell * 0.38)
                fill = "#f8ead2" if piece.islower() else "#2b2b2b"
                outline = "#b91c1c" if piece.islower() else "#111827"
                text = piece_names.get(piece, piece)
                text_fill = "#b91c1c" if piece.islower() else "#f9fafb"
                draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=fill, outline=outline, width=3)
                bbox = draw.textbbox((0, 0), text, font=font)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
                draw.text((px - tw / 2, py - th / 2), text, fill=text_fill, font=font)

        footer_top = self.margin * 2 + board_h
        draw.rectangle((0, footer_top, width, height), fill="#20242a")
        winner_text = "ongoing" if winner == 0 else "current side wins" if winner > 0 else "current side loses"
        label = (
            f"{phase} | step {frame.get('step', 0)} | {frame.get('teacher', 'teacher')} "
            f"| {frame.get('side', '?')} moved | move {move or '-'} "
            f"| value {frame.get('value', 0.0):.3f} | {winner_text}"
        )
        draw.text((12, footer_top + 22), label[:95], fill="#f3f4f6", font=font)
        return image

    def _point_display(self, x: int, y: int) -> tuple[int, int]:
        return self.margin + x * self.cell, self.margin + (9 - y) * self.cell


def fixed_visual_state_action(state: str, action: str, step: int) -> tuple[str, str]:
    if step % 2 == 0:
        return senv.fliped_state(state), flip_move(action)
    return state, action


class AlphaZeroTeacher:
    def __init__(self, config_path: Path, weight_path: Path, *, name: str):
        self.name = name
        self.config_path = config_path
        self.weight_path = weight_path
        self.config = Config("mini")
        self.config.resource.create_directories()
        self.model = CChessModel(self.config)
        if not load_model_weight(self.model, str(config_path), str(weight_path), name=name):
            raise FileNotFoundError(f"{name} model files not found: {config_path}, {weight_path}")

    @property
    def digest(self) -> str | None:
        return self.model.digest

    @property
    def using_fallback(self) -> bool:
        return bool(getattr(self.model, "using_lightweight_fallback", False))

    @property
    def backend(self) -> str:
        return "fallback" if self.using_fallback else "legacy-keras"

    def policy_value(self, state: str) -> tuple[np.ndarray, float]:
        planes = np.asarray([senv.state_to_planes(state)], dtype=np.float32)
        with self.model.graph.as_default():
            policy, value = self.model.model.predict_on_batch(planes)
        legal = senv.get_legal_moves(state)
        return mask_and_normalize_policy(np.asarray(policy[0], dtype=np.float64), legal), float(np.ravel(value)[0])

    def close(self) -> None:
        self.model.close_pipes()


def create_teacher(
    backend: str,
    config_path: Path,
    weight_path: Path,
    *,
    name: str,
    device: torch.device,
):
    if backend == "original-pytorch-h5":
        return OriginalH5Teacher(config_path, weight_path, name=name, device=device)
    teacher = AlphaZeroTeacher(config_path, weight_path, name=name)
    if backend == "legacy-keras" and teacher.using_fallback:
        teacher.close()
        raise RuntimeError(
            f"{name} could not load through legacy Keras/TensorFlow; install a compatible TF/Keras "
            "environment or use --teacher-backend original-pytorch-h5."
        )
    return teacher


def mask_and_normalize_policy(policy: np.ndarray, legal_moves: Iterable[str]) -> np.ndarray:
    masked = np.zeros(ACTION_SIZE, dtype=np.float64)
    for move in legal_moves:
        masked[ACTION_TO_INDEX[move]] = max(float(policy[ACTION_TO_INDEX[move]]), 0.0)
    total = masked.sum()
    if total <= 0:
        legal = list(legal_moves)
        if not legal:
            return np.full(ACTION_SIZE, 1.0 / ACTION_SIZE, dtype=np.float64)
        for move in legal:
            masked[ACTION_TO_INDEX[move]] = 1.0 / len(legal)
        return masked
    return masked / total


def sample_action(policy: np.ndarray, rng: random.Random, temperature: float) -> str:
    if temperature <= 1e-6:
        return ActionLabelsRed[int(np.argmax(policy))]
    adjusted = np.power(policy, 1.0 / temperature)
    adjusted /= adjusted.sum()
    return ActionLabelsRed[int(rng.choices(range(ACTION_SIZE), weights=adjusted, k=1)[0])]


def move_temperature(args: argparse.Namespace, ply: int) -> float:
    if ply <= args.temperature_moves:
        return args.temperature
    return args.eval_temperature


def mirror_move(move: str) -> str:
    return f"{8 - int(move[0])}{move[1]}{8 - int(move[2])}{move[3]}"


def mirror_policy(policy: np.ndarray) -> np.ndarray:
    mirrored = np.zeros_like(policy)
    for idx, move in enumerate(ActionLabelsRed):
        mirrored[MIRROR_INDEX[idx]] = policy[idx]
    return mirrored


def augment_example(
    planes: np.ndarray,
    policy: np.ndarray,
    value: float,
    *,
    mirror: bool,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    examples = [(planes.astype(np.float32), policy.astype(np.float32), float(value))]
    if mirror:
        examples.append(
            (
                np.flip(planes, axis=2).copy().astype(np.float32),
                mirror_policy(policy).astype(np.float32),
                float(value),
            )
        )
    return examples


def teacher_game(
    teacher_a: AlphaZeroTeacher,
    teacher_b: AlphaZeroTeacher,
    args: argparse.Namespace,
    rng: random.Random,
    game_visualizer: CChessTrainingGameVisualizer | None = None,
    iteration: int = 0,
    game_index: int = 0,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    state = senv.INIT_STATE
    examples: list[tuple[np.ndarray, np.ndarray, float]] = []
    frames: list[dict[str, Any]] = []
    teachers = [teacher_a, teacher_b]
    final_value = 0

    for step in range(1, args.max_game_length + 1):
        done, final_value, _ = senv.done(state)
        if done:
            break

        teacher = teachers[(step - 1) % 2]
        policy, value = teacher.policy_value(state)
        if step > args.learn_after_step:
            planes = senv.state_to_planes(state)
            examples.extend(augment_example(planes, policy, value, mirror=args.mirror_augment))

        temperature = move_temperature(args, step)
        action = sample_action(policy, rng, temperature)
        if game_visualizer is not None:
            visual_state, visual_action = fixed_visual_state_action(state, action, step)
            frames.append(
                {
                    "state": visual_state,
                    "action": visual_action,
                    "step": step,
                    "side": "red" if step % 2 == 1 else "black",
                    "teacher": teacher.name,
                    "value": value,
                }
            )
        state = senv.step(state, action)

    if game_visualizer is not None:
        game_visualizer.render(
            frames,
            iteration=iteration,
            phase="teacher_distill",
            game_index=game_index,
            winner=final_value,
        )

    return examples


def batch_iter(
    replay: list[tuple[np.ndarray, np.ndarray, float]],
    batch_size: int,
    rng: random.Random,
):
    indices = list(range(len(replay)))
    rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch = [replay[i] for i in indices[start : start + batch_size]]
        states = torch.tensor(np.stack([item[0] for item in batch]), dtype=torch.float32)
        policies = torch.tensor(np.stack([item[1] for item in batch]), dtype=torch.float32)
        values = torch.tensor([item[2] for item in batch], dtype=torch.float32)
        yield states, policies, values


def train_student(
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
    for _ in range(args.epochs):
        for states, target_policy, target_value in batch_iter(replay, args.batch_size, rng):
            states = states.to(device)
            target_policy = target_policy.to(device)
            target_value = target_value.to(device)

            logits, value = student(states)
            policy_loss = -(target_policy * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
            value_loss = F.mse_loss(value, target_value)
            loss = policy_loss + args.value_loss_weight * value_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
            optimizer.step()

            policy_losses.append(float(policy_loss.detach().cpu()))
            value_losses.append(float(value_loss.detach().cpu()))

    return float(np.mean(policy_losses)), float(np.mean(value_losses))


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def load_or_create_student(args: argparse.Namespace, device: torch.device) -> tuple[CChessDistillNet, StudentConfig, int]:
    if args.student.exists() and not args.fresh:
        checkpoint = torch.load(args.student, map_location=device)
        config = StudentConfig(**checkpoint["student_config"])
        model = CChessDistillNet(config).to(device)
        model.load_state_dict(checkpoint["model"])
        iteration = int(checkpoint.get("iteration", 0))
        print(f"Loaded student {args.student} at iteration {iteration}")
        return model, config, iteration

    config = StudentConfig(
        channels=args.channels,
        blocks=args.blocks,
        attention_every=args.attention_every,
        dropout=args.dropout,
    )
    model = CChessDistillNet(config).to(device)
    print(f"Created student with {parameter_count(model):,} parameters on {device}")
    return model, config, 0


def save_student(
    path: Path,
    student: CChessDistillNet,
    student_config: StudentConfig,
    optimizer: optim.Optimizer,
    *,
    iteration: int,
    replay_size: int,
    teacher_a: AlphaZeroTeacher,
    teacher_b: AlphaZeroTeacher,
    policy_loss: float,
    value_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": student.state_dict(),
            "optimizer": optimizer.state_dict(),
            "student_config": asdict(student_config),
            "iteration": iteration,
            "replay_size": replay_size,
            "action_labels": ActionLabelsRed,
            "teacher": {
                "a_config": str(teacher_a.config_path),
                "a_weight": str(teacher_a.weight_path),
                "a_digest": teacher_a.digest,
                "a_fallback": teacher_a.using_fallback,
                "a_backend": teacher_a.backend,
                "b_config": str(teacher_b.config_path),
                "b_weight": str(teacher_b.weight_path),
                "b_digest": teacher_b.digest,
                "b_fallback": teacher_b.using_fallback,
                "b_backend": teacher_b.backend,
            },
            "metrics": {
                "policy_loss": policy_loss,
                "value_loss": value_loss,
            },
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


def main() -> None:
    args = parse_args()
    args.teacher_b_config = args.teacher_b_config or args.teacher_a_config
    args.teacher_b_weight = args.teacher_b_weight or args.teacher_a_weight
    if args.dry_run:
        args.iterations = 1
        args.games = 1
        args.max_game_length = min(args.max_game_length, 12)
        args.learn_after_step = 0
        args.epochs = 1
        args.batch_size = min(args.batch_size, 8)

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and not str(device_name).startswith("cuda"):
        raise RuntimeError("CUDA was required, but torch.cuda.is_available() is false. Check your PyTorch/CUDA install.")
    device = torch.device(device_name)

    student, student_config, start_iteration = load_or_create_student(args, device)
    teacher_a = create_teacher(
        args.teacher_backend,
        args.teacher_a_config,
        args.teacher_a_weight,
        name="teacher_a",
        device=device,
    )
    teacher_b = create_teacher(
        args.teacher_backend,
        args.teacher_b_config,
        args.teacher_b_weight,
        name="teacher_b",
        device=device,
    )
    print(
        json.dumps(
            {
                "teacher_a_fallback": teacher_a.using_fallback,
                "teacher_b_fallback": teacher_b.using_fallback,
                "teacher_a_backend": teacher_a.backend,
                "teacher_b_backend": teacher_b.backend,
                "student_parameters": parameter_count(student),
                "action_size": ACTION_SIZE,
                "device": str(device),
            },
            ensure_ascii=True,
        )
    )

    optimizer = optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    replay: deque[tuple[np.ndarray, np.ndarray, float]] = deque(maxlen=args.replay_size)
    game_visualizer = (
        CChessTrainingGameVisualizer(
            args.visual_games_dir,
            every=args.visual_games_every,
            max_games_per_iteration=args.visual_games_max,
            duration=args.visual_gif_duration,
            max_gif_frames=args.visual_gif_max_frames,
        )
        if args.visualize_games
        else None
    )

    try:
        for iteration in range(start_iteration + 1, start_iteration + args.iterations + 1):
            new_examples: list[tuple[np.ndarray, np.ndarray, float]] = []
            for game_index in range(args.games):
                recorder = (
                    game_visualizer
                    if game_visualizer and game_visualizer.should_capture(iteration, game_index)
                    else None
                )
                examples = teacher_game(
                    teacher_a,
                    teacher_b,
                    args,
                    rng,
                    game_visualizer=recorder,
                    iteration=iteration,
                    game_index=game_index,
                )
                new_examples.extend(examples)
                print(
                    f"iter {iteration} teacher-game {game_index + 1}/{args.games} "
                    f"| examples {len(examples)}",
                    flush=True,
                )
            replay.extend(new_examples)
            if args.save_examples:
                save_examples(args.save_examples, new_examples)

            policy_loss, value_loss = train_student(student, list(replay), args, optimizer, device, rng)
            if not args.dry_run:
                save_student(
                    args.student,
                    student,
                    student_config,
                    optimizer,
                    iteration=iteration,
                    replay_size=len(replay),
                    teacher_a=teacher_a,
                    teacher_b=teacher_b,
                    policy_loss=policy_loss,
                    value_loss=value_loss,
                )
            print(
                f"iter {iteration}/{start_iteration + args.iterations} | replay {len(replay)} "
                f"| policy_loss {policy_loss:.4f} value_loss {value_loss:.4f} "
                f"| saved {args.student if not args.dry_run else '[dry-run]'}",
                flush=True,
            )
    finally:
        teacher_a.close()
        teacher_b.close()


ACTION_TO_INDEX = {move: i for i, move in enumerate(ActionLabelsRed)}
MIRROR_INDEX = [ACTION_TO_INDEX[mirror_move(move)] for move in ActionLabelsRed]


if __name__ == "__main__":
    main()
