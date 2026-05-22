from __future__ import annotations

import argparse
import copy
import sys
import math
from datetime import datetime
from pathlib import Path
from threading import Thread
from time import sleep, time

import numpy as np
import pygame
import torch
from pygame.locals import MOUSEBUTTONDOWN, QUIT, VIDEORESIZE

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR / "ChineseChess-AlphaZero"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import cchess_alphazero.environment.static_env as senv
from cchess_alphazero.config import Config
from cchess_alphazero.environment.env import CChessEnv
from cchess_alphazero.environment.lookup_tables import ActionLabelsRed, Winner, flip_move
from cchess_alphazero.play_games.play import (
    creat_sprite_group,
    get_font,
    select_sprite_from_group,
    translate_hit_area,
)
from distill_cchess_alphazero import CChessDistillNet, StudentConfig, parameter_count


DEFAULT_PLAY_CHECKPOINT = SCRIPT_DIR / "models/cchess_adversarial_resattn.pt"
PIECE_STYLES = ["WOOD", "POLISH", "DELICATE"]
BOARD_STYLES = ["CANVAS", "DROPS", "GREEN", "QIANHONG", "SHEET", "SKELETON", "WHITE", "WOOD"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play against the distilled PyTorch Chinese chess model.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_PLAY_CHECKPOINT)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--ai-move-first", action="store_true", help="AI plays red and moves first")
    parser.add_argument("--piece-style", choices=PIECE_STYLES, default="WOOD")
    parser.add_argument("--bg-style", choices=BOARD_STYLES, default="WOOD")
    parser.add_argument("--temperature", type=float, default=0.0, help="0 chooses the best move; >0 samples from policy")
    parser.add_argument("--top-k", type=int, default=5, help="show the top-k legal moves in the side panel")
    parser.add_argument("--mcts-sims", type=int, default=0, help="MCTS simulations per AI move; 0 uses direct policy")
    parser.add_argument("--cpuct", type=float, default=1.5)
    return parser.parse_args()


class DistilledPolicy:
    def __init__(self, checkpoint: Path, device: torch.device):
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Student checkpoint not found: {checkpoint}. Train one with distill_cchess_alphazero.py first."
            )
        self.checkpoint = checkpoint
        payload = torch.load(checkpoint, map_location=device)
        config = StudentConfig(**payload["student_config"])
        self.model = CChessDistillNet(config).to(device)
        self.model.load_state_dict(payload["model"])
        self.model.eval()
        self.device = device
        self.iteration = int(payload.get("iteration", 0))
        self.replay_size = int(payload.get("replay_size", 0))
        self.parameters = parameter_count(self.model)

    def policy_value(self, state: str) -> tuple[np.ndarray, float]:
        planes = torch.tensor(np.asarray([senv.state_to_planes(state)], dtype=np.float32), device=self.device)
        with torch.inference_mode():
            logits, value = self.model(planes)
            policy = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        legal_moves = senv.get_legal_moves(state)
        masked = np.zeros(len(ActionLabelsRed), dtype=np.float64)
        for move in legal_moves:
            idx = ActionLabelsRed.index(move)
            masked[idx] = max(float(policy[idx]), 0.0)
        total = masked.sum()
        if total <= 0 and legal_moves:
            for move in legal_moves:
                masked[ActionLabelsRed.index(move)] = 1.0 / len(legal_moves)
        elif total > 0:
            masked /= total
        return masked, float(value.detach().cpu().item())

    def choose_action(self, state: str, temperature: float = 0.0) -> tuple[str | None, np.ndarray, float]:
        policy, value = self.policy_value(state)
        legal_moves = senv.get_legal_moves(state)
        if not legal_moves:
            return None, policy, value
        legal_indexes = [ActionLabelsRed.index(move) for move in legal_moves]
        if temperature <= 1e-6:
            action_index = max(legal_indexes, key=lambda idx: policy[idx])
        else:
            probs = np.asarray([policy[idx] for idx in legal_indexes], dtype=np.float64)
            probs = np.power(probs, 1.0 / temperature)
            probs /= probs.sum()
            action_index = int(np.random.choice(legal_indexes, p=probs))
        return ActionLabelsRed[action_index], policy, value


class MCTSNode:
    def __init__(self, prior: float = 1.0):
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: dict[str, MCTSNode] = {}

    @property
    def value(self) -> float:
        return 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count

    def expanded(self) -> bool:
        return bool(self.children)


class StudentMCTS:
    def __init__(self, policy: DistilledPolicy, simulations: int, cpuct: float):
        self.policy = policy
        self.simulations = max(0, simulations)
        self.cpuct = cpuct

    def search(self, root_state: str, temperature: float = 0.0) -> tuple[str | None, np.ndarray, float, dict[str, tuple[int, float, float]]]:
        root = MCTSNode()
        policy, root_value = self.policy.policy_value(root_state)
        legal_moves = senv.get_legal_moves(root_state)
        if not legal_moves:
            return None, policy, root_value, {}
        for move in legal_moves:
            root.children[move] = MCTSNode(float(policy[ActionLabelsRed.index(move)]))

        for _ in range(self.simulations):
            self._simulate(root, root_state)

        search_policy = np.zeros(len(ActionLabelsRed), dtype=np.float64)
        debug: dict[str, tuple[int, float, float]] = {}
        for move, child in root.children.items():
            search_policy[ActionLabelsRed.index(move)] = child.visit_count
            debug[move] = (child.visit_count, child.value, child.prior)
        total = search_policy.sum()
        if total > 0:
            search_policy /= total
        else:
            search_policy = policy

        if temperature <= 1e-6:
            action = max(root.children, key=lambda move: root.children[move].visit_count)
        else:
            moves = list(root.children)
            probs = np.asarray([search_policy[ActionLabelsRed.index(move)] for move in moves], dtype=np.float64)
            probs = np.power(probs, 1.0 / temperature)
            probs /= probs.sum()
            action = str(np.random.choice(moves, p=probs))
        return action, search_policy, root_value, debug

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
            node.children[move] = MCTSNode(float(policy[ActionLabelsRed.index(move)]))
        self._backpropagate(path, value)
        return value

    def _select_child(self, node: MCTSNode) -> tuple[str, MCTSNode]:
        parent_sqrt = math.sqrt(max(1, node.visit_count))

        def score(item: tuple[str, MCTSNode]) -> float:
            _, child = item
            prior_score = self.cpuct * child.prior * parent_sqrt / (1 + child.visit_count)
            return -child.value + prior_score

        return max(node.children.items(), key=score)

    @staticmethod
    def _backpropagate(path: list[MCTSNode], value: float) -> None:
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            value = -value


class DistilledPlayWithHuman:
    def __init__(self, args: argparse.Namespace, device: torch.device):
        self.args = args
        self.config = Config("mini")
        self.config.opts.piece_style = args.piece_style
        self.config.opts.bg_style = args.bg_style
        self.config.resource.create_directories()
        self.env = CChessEnv()
        self.policy = DistilledPolicy(args.checkpoint, device)
        self.mcts = StudentMCTS(self.policy, args.mcts_sims, args.cpuct)
        self.screen_width = 720
        self.height = 577
        self.width = 521
        self.chessman_w = 57
        self.chessman_h = 57
        self.disp_record_num = 15
        self.rec_labels = [None] * self.disp_record_num
        self.nn_value = 0.0
        self.move_hints: dict[str, tuple[int, float, float]] = {}
        self.history: list[str] = []
        self.chessmans = None
        self.human_move_first = True
        if args.bg_style == "WOOD":
            self.chessman_w += 1
            self.chessman_h += 1

    def start(self) -> None:
        self.env.reset()
        self.human_move_first = not self.args.ai_move_first
        pygame.init()
        screen, board_background, widget_background = self.init_screen()
        framerate = pygame.time.Clock()
        current_chessman = None
        if self.human_move_first:
            self.env.board.calc_chessmans_moving_list()

        ai_worker = Thread(target=self.ai_move, name="distilled_ai_worker", daemon=True)
        ai_worker.start()

        while not self.env.board.is_end():
            for event in pygame.event.get():
                if event.type == QUIT:
                    self.save_record_and_exit()
                elif event.type == VIDEORESIZE:
                    pass
                elif event.type == MOUSEBUTTONDOWN and self.human_move_first == self.env.red_to_move:
                    current_chessman = self.handle_click(current_chessman)

            self.draw_widget(screen, widget_background)
            framerate.tick(30)
            self.chessmans.clear(screen, board_background)
            self.chessmans.update()
            self.chessmans.draw(screen)
            pygame.display.update()

        self.env.board.print_record()
        self.save_record()
        sleep(2)

    def init_screen(self):
        screen = pygame.display.set_mode([self.screen_width, self.height], 0, 32)
        pygame.display.set_caption("中国象棋 Distilled AI")
        board_image = pygame.image.load(
            PROJECT_ROOT / "cchess_alphazero" / "play_games" / "images" / f"{self.config.opts.bg_style}.GIF"
        ).convert()
        board_image = pygame.transform.scale(board_image, (self.width, self.height))
        board_background = pygame.Surface([self.width, self.height])
        board_background.blit(board_image, (0, 0))
        widget_background = pygame.Surface([self.screen_width - self.width, self.height])
        widget_background.fill((255, 255, 255))

        font = get_font(self.config.resource.font_path, 16)
        title = font.render("着法记录", True, (0, 0, 0), (255, 255, 255))
        widget_background.blit(title, (10, 10))

        screen.blit(board_background, (0, 0))
        screen.blit(widget_background, (self.width, 0))
        pygame.display.flip()
        self.chessmans = pygame.sprite.Group()
        # The imported sprite factory reads style from the original module-level variable.
        import cchess_alphazero.play_games.play as original_play

        original_play.PIECE_STYLE = self.config.opts.piece_style
        creat_sprite_group(self.chessmans, self.env.board.chessmans_hash, self.chessman_w, self.chessman_h)
        return screen, board_background, widget_background

    def handle_click(self, current_chessman):
        mouse_x, mouse_y = pygame.mouse.get_pos()
        col_num, row_num = translate_hit_area(mouse_x, mouse_y, self.chessman_w, self.chessman_h)
        chessman_sprite = select_sprite_from_group(self.chessmans, col_num, row_num)
        if current_chessman is None and chessman_sprite is not None:
            if chessman_sprite.chessman.is_red == self.env.red_to_move:
                chessman_sprite.is_selected = True
                return chessman_sprite
            return None

        if current_chessman is not None and chessman_sprite is not None:
            if chessman_sprite.chessman.is_red == self.env.red_to_move:
                current_chessman.is_selected = False
                chessman_sprite.is_selected = True
                return chessman_sprite
            return self.try_human_move(current_chessman, chessman_sprite, col_num, row_num)

        if current_chessman is not None:
            return self.try_human_move(current_chessman, None, col_num, row_num)
        return current_chessman

    def try_human_move(self, current_chessman, captured_sprite, col_num: int, row_num: int):
        move = (
            f"{current_chessman.chessman.col_num}{current_chessman.chessman.row_num}"
            f"{col_num}{row_num}"
        )
        success = current_chessman.move(col_num, row_num, self.chessman_w, self.chessman_h)
        self.history.append(move)
        if not success:
            return current_chessman
        if captured_sprite is not None:
            self.chessmans.remove(captured_sprite)
            captured_sprite.kill()
        current_chessman.is_selected = False
        self.history.append(self.env.get_state())
        return None

    def ai_move(self) -> None:
        ai_move_first = not self.human_move_first
        self.history = [self.env.get_state()]
        while not self.env.done:
            if ai_move_first != self.env.red_to_move:
                sleep(0.01)
                continue
            state = self.env.get_state()
            start_time = time()
            if self.args.mcts_sims > 0:
                action, policy, value, debug = self.mcts.search(state, self.args.temperature)
            else:
                action, policy, value = self.policy.choose_action(state, self.args.temperature)
                debug = {}
            if action is None:
                self.env.winner = Winner.draw
                self.env.board.winner = Winner.draw
                return
            self.history.append(action)
            board_action = flip_move(action) if not self.env.red_to_move else action
            self.nn_value = value
            self.update_move_hints(policy, value, debug)
            print(f"AI move {board_action} | sims {self.args.mcts_sims} | {time() - start_time:.2f}s", flush=True)
            x0, y0, x1, y1 = [int(ch) for ch in board_action]
            chessman_sprite = select_sprite_from_group(self.chessmans, x0, y0)
            captured_sprite = select_sprite_from_group(self.chessmans, x1, y1)
            if chessman_sprite is None:
                sleep(0.05)
                continue
            if captured_sprite:
                self.chessmans.remove(captured_sprite)
                captured_sprite.kill()
            chessman_sprite.move(x1, y1, self.chessman_w, self.chessman_h)
            self.history.append(self.env.get_state())

    def update_move_hints(self, policy: np.ndarray, value: float, debug: dict[str, tuple[int, float, float]] | None = None) -> None:
        self.move_hints = {}
        debug = debug or {}
        legal_moves = list(debug) if debug else senv.get_legal_moves(self.env.get_state())
        ranked = sorted(
            legal_moves,
            key=lambda move: debug[move][0] if move in debug else policy[ActionLabelsRed.index(move)],
            reverse=True,
        )[: self.args.top_k]
        for move in ranked:
            prior = float(policy[ActionLabelsRed.index(move)])
            visits, q_value, raw_prior = debug.get(move, (int(prior * 1000), value, prior))
            try:
                move_cn = self.env.board.make_single_record(int(move[0]), int(move[1]), int(move[2]), int(move[3]))
            except Exception:
                move_cn = move
            self.move_hints[move_cn] = (visits, q_value, raw_prior)

    def draw_widget(self, screen, widget_background) -> None:
        widget_background.fill((255, 255, 255))
        pygame.draw.line(widget_background, (255, 0, 0), (10, 285), (self.screen_width - self.width - 10, 285))
        screen.blit(widget_background, (self.width, 0))
        self.draw_records(screen, widget_background)
        self.draw_evaluation(screen, widget_background)

    def draw_records(self, screen, widget_background) -> None:
        self.draw_label(screen, widget_background, "着法记录", 10, 16, 10)
        records = self.env.board.record.split("\n")
        font = get_font(self.config.resource.font_path, 12)
        for i, record in enumerate(records[-self.disp_record_num :]):
            label = font.render(record, True, (0, 0, 0), (255, 255, 255))
            widget_background.blit(label, (10, 35 + i * 15))
        screen.blit(widget_background, (self.width, 0))

    def draw_evaluation(self, screen, widget_background) -> None:
        self.draw_label(screen, widget_background, "Distilled AI", 300, 16, 10)
        self.draw_label(screen, widget_background, f"iter: {self.policy.iteration}", 325, 13, 10)
        self.draw_label(screen, widget_background, f"examples: {self.policy.replay_size}", 345, 13, 10)
        self.draw_label(screen, widget_background, f"value: {self.nn_value:.3f}", 365, 13, 10)
        params_m = self.policy.parameters / 1_000_000
        self.draw_label(screen, widget_background, f"params: {params_m:.1f}M", 385, 13, 10)
        self.draw_label(screen, widget_background, f"mcts: {self.args.mcts_sims}", 405, 13, 10)
        self.draw_label(screen, widget_background, "候选走法:", 415, 12, 10)
        for i, (move, stats) in enumerate(copy.deepcopy(self.move_hints).items()):
            visits, value, prior = stats
            y = 435 + i * 20
            self.draw_label(screen, widget_background, move, y, 12, 10)
            self.draw_label(screen, widget_background, str(visits), y, 12, 70)
            self.draw_label(screen, widget_background, f"{value:.2f}", y, 12, 105)
            self.draw_label(screen, widget_background, f"{prior:.3f}", y, 12, 145)

    def draw_label(self, screen, widget_background, text: str, y: int, font_size: int, x: int | None = None) -> None:
        font = get_font(self.config.resource.font_path, font_size)
        label = font.render(text, True, (0, 0, 0), (255, 255, 255))
        rect = label.get_rect()
        rect.y = y
        rect.x = x if x is not None else 10
        widget_background.blit(label, rect)
        screen.blit(widget_background, (self.width, 0))

    def save_record(self) -> None:
        game_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = Path(self.config.resource.play_record_dir) / (self.config.resource.play_record_filename_tmpl % game_id)
        self.env.board.save_record(str(path))

    def save_record_and_exit(self) -> None:
        self.env.board.print_record()
        self.save_record()
        pygame.quit()
        sys.exit()


def main() -> None:
    args = parse_args()
    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and not str(device_name).startswith("cuda"):
        raise RuntimeError("CUDA was required, but torch.cuda.is_available() is false.")
    game = DistilledPlayWithHuman(args, torch.device(device_name))
    print(
        f"Loaded {args.checkpoint} on {device_name} | iter {game.policy.iteration} "
        f"| examples {game.policy.replay_size} | params {game.policy.parameters:,}"
    )
    game.start()


if __name__ == "__main__":
    main()
