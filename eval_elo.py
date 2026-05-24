from __future__ import annotations

import argparse
import json
import math
import queue
import random
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR / "ChineseChess-AlphaZero"
DEFAULT_PIKAFISH_DIR = SCRIPT_DIR / "Pikafish/Pikafish.2026-01-02"
DEFAULT_PIKAFISH_ENGINE = DEFAULT_PIKAFISH_DIR / "Windows/pikafish-avx2.exe"
DEFAULT_PIKAFISH_NNUE = DEFAULT_PIKAFISH_DIR / "pikafish.nnue"
DEFAULT_STUDENT = SCRIPT_DIR / "models/cchess_dataset_midgame.pt"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import cchess_alphazero.environment.static_env as senv
from cchess_alphazero.environment.lookup_tables import flip_move
from distill_cchess_alphazero import CChessDistillNet, DEFAULT_MODEL_DIR, StudentConfig, create_teacher, parameter_count
from train_adversarial import MCTS, StudentPolicy


class EvalPlayer(Protocol):
    name: str
    kind: str

    def new_game(self) -> None:
        ...

    def select_action(self, state: str, turns: int, ply: int) -> str | None:
        ...

    def close(self) -> None:
        ...


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate Elo by playing any two supported Chinese chess engines/models.")
    parser.add_argument("--player-a", "--a-player", dest="player_a", choices=["student", "alphazero", "pikafish"], default="student")
    parser.add_argument("--player-b", "--b-player", dest="player_b", choices=["student", "alphazero", "pikafish"], default="pikafish")
    parser.add_argument(
        "--side",
        choices=["alternate", "a-first", "b-first", "student-first", "alphazero-first", "pikafish-first", "random"],
        default="alternate",
        help="which player is red first; alternate means A red in odd-numbered games",
    )

    parser.add_argument("--checkpoint", "--a-checkpoint", dest="a_checkpoint", type=Path, default=DEFAULT_STUDENT)
    parser.add_argument("--b-checkpoint", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--require-cuda", action="store_true")

    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--max-game-length", type=int, default=500)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--json-out", type=Path, default=None)

    parser.add_argument("--student-mcts-sims", "--a-mcts-sims", dest="a_mcts_sims", type=int, default=64)
    parser.add_argument("--b-mcts-sims", type=int, default=0)
    parser.add_argument("--cpuct", "--a-cpuct", dest="a_cpuct", type=float, default=1.5)
    parser.add_argument("--b-cpuct", type=float, default=1.5)
    parser.add_argument("--temperature-moves", type=int, default=18)
    parser.add_argument("--opening-temperature", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.0)

    parser.add_argument("--alphazero-config", type=Path, default=DEFAULT_MODEL_DIR / "model_best_config.json")
    parser.add_argument("--alphazero-weight", type=Path, default=DEFAULT_MODEL_DIR / "model_best_weight.h5")
    parser.add_argument(
        "--alphazero-backend",
        choices=["original-pytorch-h5", "legacy-keras", "fallback"],
        default="original-pytorch-h5",
    )

    parser.add_argument("--pikafish-path", type=Path, default=DEFAULT_PIKAFISH_ENGINE)
    parser.add_argument("--pikafish-nnue", type=Path, default=DEFAULT_PIKAFISH_NNUE)
    parser.add_argument("--pikafish-threads", type=int, default=1)
    parser.add_argument("--pikafish-hash", type=int, default=128)
    parser.add_argument("--pikafish-depth", type=int, default=4, help="Pikafish search depth; <=0 disables depth limit")
    parser.add_argument("--pikafish-movetime-ms", type=int, default=0, help="used when --pikafish-depth and --pikafish-nodes are <=0")
    parser.add_argument("--pikafish-nodes", type=int, default=0, help="Pikafish nodes per move; overrides movetime when >0")
    parser.add_argument("--pikafish-timeout", type=float, default=30.0)

    parser.add_argument("--anchor-player", choices=["a", "b"], default="b", help="which player's rating is supplied by --anchor-elo")
    parser.add_argument("--anchor-elo", type=float, default=None, help="known rating for the anchor player")
    parser.add_argument("--pikafish-elo", type=float, default=None, help="backward-compatible alias for --anchor-elo")
    args = parser.parse_args()
    if args.anchor_elo is None:
        args.anchor_elo = 0.0 if args.pikafish_elo is None else args.pikafish_elo
    return args


class PikafishUCI:
    def __init__(
        self,
        engine_path: Path,
        *,
        nnue_path: Path,
        threads: int,
        hash_mb: int,
        timeout: float,
    ) -> None:
        self.engine_path = engine_path.resolve()
        self.nnue_path = nnue_path.resolve()
        self.timeout = float(timeout)
        self.lines: queue.Queue[str] = queue.Queue()
        self.stderr_lines: queue.Queue[str] = queue.Queue()

        if not self.engine_path.exists():
            raise FileNotFoundError(f"Pikafish executable not found: {self.engine_path}")
        if not self.nnue_path.exists():
            raise FileNotFoundError(f"Pikafish NNUE file not found: {self.nnue_path}")

        creationflags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
        self.proc = subprocess.Popen(
            [str(self.engine_path)],
            cwd=str(self.engine_path.parent),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=creationflags,
        )
        self.stdout_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self.stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self.stdout_thread.start()
        self.stderr_thread.start()

        self.send("uci")
        self.wait_for("uciok")
        self.setoption("Threads", str(max(1, int(threads))))
        self.setoption("Hash", str(max(1, int(hash_mb))))
        self.setoption("EvalFile", str(self.nnue_path))
        self.isready()

    def _read_stdout(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self.lines.put(line.rstrip("\r\n"))

    def _read_stderr(self) -> None:
        assert self.proc.stderr is not None
        for line in self.proc.stderr:
            self.stderr_lines.put(line.rstrip("\r\n"))

    def send(self, command: str) -> None:
        if self.proc.poll() is not None:
            raise RuntimeError(f"Pikafish exited with code {self.proc.returncode}: {self.drain_stderr()}")
        assert self.proc.stdin is not None
        self.proc.stdin.write(command + "\n")
        self.proc.stdin.flush()

    def wait_for(self, expected: str, *, timeout: float | None = None) -> list[str]:
        deadline = time.monotonic() + (self.timeout if timeout is None else timeout)
        seen: list[str] = []
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"Pikafish exited with code {self.proc.returncode}: {self.drain_stderr()}")
            try:
                line = self.lines.get(timeout=0.05)
            except queue.Empty:
                continue
            seen.append(line)
            if line == expected or line.startswith(expected + " "):
                return seen
        raise TimeoutError(f"Timed out waiting for '{expected}'. Last lines: {seen[-8:]}; stderr: {self.drain_stderr()}")

    def setoption(self, name: str, value: str | None = None) -> None:
        self.send(f"setoption name {name}" if value is None else f"setoption name {name} value {value}")

    def isready(self) -> None:
        self.send("isready")
        self.wait_for("readyok")

    def new_game(self) -> None:
        self.send("ucinewgame")
        self.isready()

    def bestmove(self, fen: str, go_command: str) -> str | None:
        self.send(f"position fen {fen}")
        self.send(go_command)
        lines = self.wait_for("bestmove")
        best = lines[-1].split()
        if len(best) < 2 or best[1] in {"(none)", "none", "0000"}:
            return None
        return best[1]

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self.send("quit")
            except Exception:
                pass
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def drain_stderr(self) -> str:
        items: list[str] = []
        while True:
            try:
                items.append(self.stderr_lines.get_nowait())
            except queue.Empty:
                break
        return "\n".join(items[-8:])


class MCTSPlayer:
    def __init__(self, name: str, kind: str, agent: MCTS, args: argparse.Namespace):
        self.name = name
        self.kind = kind
        self.agent = agent
        self.args = args

    def new_game(self) -> None:
        return None

    def select_action(self, state: str, turns: int, ply: int) -> str | None:
        action, _, _ = self.agent.search(state, temperature=move_temperature(self.args, ply), add_noise=False)
        return action

    def close(self) -> None:
        close = getattr(self.agent.policy, "close", None)
        if callable(close):
            close()


class PikafishPlayer:
    def __init__(self, engine: PikafishUCI, limit: str):
        self.name = "pikafish"
        self.kind = "pikafish"
        self.engine = engine
        self.limit = limit

    def new_game(self) -> None:
        self.engine.new_game()

    def select_action(self, state: str, turns: int, ply: int) -> str | None:
        fen = senv.state_to_fen(state, turns)
        uci_move = self.engine.bestmove(fen, self.limit)
        if uci_move is None:
            return None
        action = senv.parse_ucci_move(uci_move[:4])
        if turns % 2 == 1:
            action = flip_move(action)
        return action

    def close(self) -> None:
        self.engine.close()


def load_student(checkpoint: Path, device: torch.device) -> CChessDistillNet:
    payload = torch.load(checkpoint, map_location=device)
    config = StudentConfig(**payload["student_config"])
    model = CChessDistillNet(config).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "iteration": int(payload.get("iteration", 0)),
                "parameters": parameter_count(model),
            },
            ensure_ascii=True,
        ),
        flush=True,
    )
    return model


def move_temperature(args: argparse.Namespace, ply: int) -> float:
    return args.opening_temperature if ply <= args.temperature_moves else args.temperature


def go_command(args: argparse.Namespace) -> str:
    if args.pikafish_depth > 0:
        return f"go depth {args.pikafish_depth}"
    if args.pikafish_nodes > 0:
        return f"go nodes {args.pikafish_nodes}"
    return f"go movetime {max(1, args.pikafish_movetime_ms)}"


def make_mcts_player(
    *,
    slot: str,
    kind: str,
    args: argparse.Namespace,
    device: torch.device,
    rng: random.Random,
) -> MCTSPlayer:
    sims = args.a_mcts_sims if slot == "a" else args.b_mcts_sims
    cpuct = args.a_cpuct if slot == "a" else args.b_cpuct
    if kind == "student":
        checkpoint = args.a_checkpoint if slot == "a" else (args.b_checkpoint or args.a_checkpoint)
        model = load_student(checkpoint, device)
        policy = StudentPolicy(model, device)
        name = f"{slot}:student:{checkpoint.name}"
    elif kind == "alphazero":
        policy = create_teacher(
            args.alphazero_backend,
            args.alphazero_config,
            args.alphazero_weight,
            name=f"{slot}_alphazero",
            device=device,
        )
        name = f"{slot}:alphazero:{args.alphazero_backend}"
    else:
        raise ValueError(f"{kind} is not an MCTS/policy player")
    agent = MCTS(policy, simulations=sims, cpuct=cpuct, rng=rng, dirichlet_alpha=0.0, exploration_frac=0.0)
    return MCTSPlayer(name, kind, agent, args)


def build_players(args: argparse.Namespace, device: torch.device, rng: random.Random) -> tuple[EvalPlayer, EvalPlayer, dict[str, Any]]:
    limit = go_command(args)
    pikafish_player: PikafishPlayer | None = None
    if args.player_a == "pikafish" or args.player_b == "pikafish":
        engine = PikafishUCI(
            args.pikafish_path,
            nnue_path=args.pikafish_nnue,
            threads=args.pikafish_threads,
            hash_mb=args.pikafish_hash,
            timeout=args.pikafish_timeout,
        )
        pikafish_player = PikafishPlayer(engine, limit)

    def make(slot: str, kind: str) -> EvalPlayer:
        if kind == "pikafish":
            assert pikafish_player is not None
            return pikafish_player
        return make_mcts_player(slot=slot, kind=kind, args=args, device=device, rng=rng)

    player_a = make("a", args.player_a)
    player_b = make("b", args.player_b)
    metadata = {
        "player_a": player_a.name,
        "player_b": player_b.name,
        "player_a_kind": player_a.kind,
        "player_b_kind": player_b.kind,
        "a_mcts_sims": args.a_mcts_sims if player_a.kind != "pikafish" else None,
        "b_mcts_sims": args.b_mcts_sims if player_b.kind != "pikafish" else None,
        "pikafish_path": str(args.pikafish_path) if pikafish_player else None,
        "pikafish_nnue": str(args.pikafish_nnue) if pikafish_player else None,
        "pikafish_threads": args.pikafish_threads if pikafish_player else None,
        "pikafish_hash": args.pikafish_hash if pikafish_player else None,
        "pikafish_go": limit if pikafish_player else None,
        "alphazero_backend": args.alphazero_backend if "alphazero" in {args.player_a, args.player_b} else None,
    }
    return player_a, player_b, metadata


def a_red_for_game(args: argparse.Namespace, player_a: EvalPlayer, player_b: EvalPlayer, game_index: int, rng: random.Random) -> bool:
    if args.side == "a-first":
        return True
    if args.side == "b-first":
        return False
    if args.side == "random":
        return bool(rng.getrandbits(1))
    if args.side == "alternate":
        return game_index % 2 == 0
    first_kind = args.side.removesuffix("-first")
    if player_a.kind == first_kind:
        return True
    if player_b.kind == first_kind:
        return False
    raise ValueError(f"--side {args.side} was requested, but neither player has kind {first_kind!r}")


def unique_players(*players: EvalPlayer) -> list[EvalPlayer]:
    seen: set[int] = set()
    unique: list[EvalPlayer] = []
    for player in players:
        ident = id(player)
        if ident not in seen:
            unique.append(player)
            seen.add(ident)
    return unique


def play_game(
    *,
    args: argparse.Namespace,
    player_a: EvalPlayer,
    player_b: EvalPlayer,
    a_red: bool,
) -> dict[str, Any]:
    for player in unique_players(player_a, player_b):
        player.new_game()

    state = senv.INIT_STATE
    final_value = 0
    last_ply = 0
    a_plies: list[int] = []

    for ply in range(1, args.max_game_length + 1):
        done, final_value, _ = senv.done(state)
        if done:
            break

        turns = ply - 1
        is_a = (ply % 2 == 1) == a_red
        actor = player_a if is_a else player_b
        action = actor.select_action(state, turns, ply)
        if action is None:
            return {"plies": last_ply, "a_result": -1.0 if is_a else 1.0, "a_red": a_red, "reason": "no_move"}

        legal_moves = set(senv.get_legal_moves(state))
        if action not in legal_moves:
            fen = senv.state_to_fen(state, turns)
            raise RuntimeError(f"{actor.name} returned illegal move {action}; fen={fen}; legal_count={len(legal_moves)}")

        if is_a:
            a_plies.append(ply)
        state = senv.step(state, action)
        last_ply = ply
    else:
        final_value = 0

    if final_value == 0 or not a_plies:
        a_result = 0.0
    else:
        last_a_ply = a_plies[-1]
        a_result = float(final_value * ((-1) ** (last_ply - last_a_ply + 1)))
    return {"plies": last_ply, "a_result": a_result, "a_red": a_red, "reason": "normal"}


def elo_diff(score_rate: float) -> float:
    if score_rate <= 0:
        return float("-inf")
    if score_rate >= 1:
        return float("inf")
    return -400.0 * math.log10(1.0 / score_rate - 1.0)


def anchored_elos(anchor_player: str, anchor_elo: float, diff_a_vs_b: float) -> dict[str, float]:
    if not math.isfinite(diff_a_vs_b):
        if anchor_player == "b":
            return {"estimated_a_elo": diff_a_vs_b, "estimated_b_elo": anchor_elo}
        return {"estimated_a_elo": anchor_elo, "estimated_b_elo": -diff_a_vs_b}
    if anchor_player == "b":
        return {"estimated_a_elo": anchor_elo + diff_a_vs_b, "estimated_b_elo": anchor_elo}
    return {"estimated_a_elo": anchor_elo, "estimated_b_elo": anchor_elo - diff_a_vs_b}


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and not str(device_name).startswith("cuda"):
        raise RuntimeError("CUDA was required, but torch.cuda.is_available() is false.")
    device = torch.device(device_name)

    player_a, player_b, metadata = build_players(args, device, rng)
    print(json.dumps(metadata, ensure_ascii=True), flush=True)

    try:
        results: list[dict[str, Any]] = []
        for game_index in range(args.games):
            result = play_game(
                args=args,
                player_a=player_a,
                player_b=player_b,
                a_red=a_red_for_game(args, player_a, player_b, game_index, rng),
            )
            results.append(result)
            print(
                f"game {game_index + 1}/{args.games} | a_result {result['a_result']:+.0f} "
                f"| a_red {int(result['a_red'])} | plies {result['plies']} | {result['reason']}",
                flush=True,
            )
    finally:
        for player in unique_players(player_a, player_b):
            player.close()

    a_wins = sum(1 for item in results if item["a_result"] > 0)
    b_wins = sum(1 for item in results if item["a_result"] < 0)
    draws = len(results) - a_wins - b_wins
    score = a_wins + 0.5 * draws
    score_rate = score / len(results) if results else 0.0
    diff = elo_diff(score_rate)
    estimated = anchored_elos(args.anchor_player, args.anchor_elo, diff)
    summary = {
        **metadata,
        "games": len(results),
        "a_wins": a_wins,
        "draws": draws,
        "b_wins": b_wins,
        "wins": a_wins,
        "losses": b_wins,
        "score_rate_a": score_rate,
        "elo_diff_a_vs_b": diff,
        "anchor_player": args.anchor_player,
        "anchor_elo": args.anchor_elo,
        **estimated,
        "avg_plies": float(np.mean([item["plies"] for item in results])) if results else 0.0,
    }
    if player_b.kind == "pikafish":
        summary["elo_diff_vs_pikafish"] = diff
        summary["estimated_student_elo"] = estimated["estimated_a_elo"] if player_a.kind == "student" else None
    print(json.dumps(summary, ensure_ascii=True, indent=2), flush=True)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
