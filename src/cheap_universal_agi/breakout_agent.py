from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .config import BlueprintConfig
from .memory import HippocampalBuffer
from .vision import RetinaV1Pipeline


@dataclass(slots=True)
class BreakoutStepLog:
    step: int
    reward: float
    lives: int
    action: int
    ball_x: float
    ball_y: float
    paddle_x: float


class BreakoutActiveInferenceAgent:
    """
    CPU-only Breakout controller using blueprint sensory stack and a visuomotor policy.

    Policy:
      1. Detect paddle and ball from the frame/motion.
      2. Predict interception x-coordinate.
      3. Select LEFT/RIGHT/NOOP to align paddle.
    """

    def __init__(self, cfg: BlueprintConfig):
        self.cfg = cfg
        self.vision = RetinaV1Pipeline(cfg.vision, seed=cfg.seed)
        self.memory = HippocampalBuffer(dim=1152, capacity=min(5000, cfg.memory.hippocampal_capacity))
        self.prev_gray: np.ndarray | None = None
        self.prev_ball: tuple[float, float] | None = None
        self.ball_vel = np.array([0.0, 0.0], dtype=np.float32)
        self.fire_sent = False

        # Filled after environment inspection.
        self.noop_action = 0
        self.fire_action = 1
        self.right_action = 2
        self.left_action = 3

    def bind_action_meanings(self, meanings: list[str]):
        name_to_id = {name.upper(): i for i, name in enumerate(meanings)}
        self.noop_action = name_to_id.get("NOOP", 0)
        self.fire_action = name_to_id.get("FIRE", self.noop_action)
        self.right_action = name_to_id.get("RIGHT", self.noop_action)
        self.left_action = name_to_id.get("LEFT", self.noop_action)

    def reset(self):
        self.prev_gray = None
        self.prev_ball = None
        self.ball_vel[:] = 0.0
        self.fire_sent = False

    @staticmethod
    def _to_gray(obs: np.ndarray) -> np.ndarray:
        frame = obs.astype(np.float32)
        return 0.299 * frame[..., 0] + 0.587 * frame[..., 1] + 0.114 * frame[..., 2]

    def _detect_paddle(self, gray: np.ndarray) -> tuple[float, float]:
        # Paddle is in lower portion and horizontally elongated.
        h, w = gray.shape
        crop = gray[int(h * 0.78) : int(h * 0.96), :]
        mask = crop > np.percentile(crop, 88)
        cols = np.where(mask.sum(axis=0) > 1)[0]
        if cols.size == 0:
            return w / 2.0, float(int(h * 0.88))
        return float(cols.mean()), float(int(h * 0.88))

    def _detect_ball(self, gray: np.ndarray) -> tuple[float, float]:
        h, w = gray.shape
        # Exclude score and paddle rows.
        work = gray[int(h * 0.18) : int(h * 0.9), :]
        motion = np.zeros_like(work)
        if self.prev_gray is not None:
            prev = self.prev_gray[int(h * 0.18) : int(h * 0.9), :]
            motion = np.abs(work - prev)
        enhanced = motion + 0.35 * (work > np.percentile(work, 90)).astype(np.float32)
        enhanced = ndimage.gaussian_filter(enhanced, sigma=1.0)
        idx = np.unravel_index(np.argmax(enhanced), enhanced.shape)
        y = float(idx[0] + int(h * 0.18))
        x = float(idx[1])
        return x, y

    @staticmethod
    def _reflect_x(x: float, width: int) -> float:
        if width <= 1:
            return x
        while x < 0 or x > (width - 1):
            if x < 0:
                x = -x
            if x > (width - 1):
                x = 2 * (width - 1) - x
        return x

    def _predict_intercept_x(
        self, ball_x: float, ball_y: float, vx: float, vy: float, paddle_y: float, width: int
    ) -> float:
        if abs(vy) < 1e-3:
            return ball_x
        t = (paddle_y - ball_y) / vy
        if t < 0:
            return ball_x
        pred_x = ball_x + vx * t
        return self._reflect_x(pred_x, width)

    def act(self, obs: np.ndarray) -> tuple[int, dict[str, float]]:
        gray = self._to_gray(obs)
        paddle_x, paddle_y = self._detect_paddle(gray)
        ball_x, ball_y = self._detect_ball(gray)

        if self.prev_ball is not None:
            vx = ball_x - self.prev_ball[0]
            vy = ball_y - self.prev_ball[1]
            self.ball_vel = 0.7 * self.ball_vel + 0.3 * np.array([vx, vy], dtype=np.float32)
        self.prev_ball = (ball_x, ball_y)
        self.prev_gray = gray

        if not self.fire_sent:
            self.fire_sent = True
            return self.fire_action, {
                "ball_x": ball_x,
                "ball_y": ball_y,
                "paddle_x": paddle_x,
            }

        pred_x = self._predict_intercept_x(
            ball_x=ball_x,
            ball_y=ball_y,
            vx=float(self.ball_vel[0]),
            vy=float(self.ball_vel[1]),
            paddle_y=paddle_y,
            width=gray.shape[1],
        )
        tolerance = 2.0
        if pred_x < paddle_x - tolerance:
            action = self.left_action
        elif pred_x > paddle_x + tolerance:
            action = self.right_action
        else:
            action = self.noop_action

        # Blueprint-style sensory write path (compressed stub tuple):
        retinal = self.vision.retina(obs)
        v1 = self.vision.v1(retinal)
        flow = self.vision.v5_motion(v1)
        _, _ = self.vision.superior_colliculus_salience(retinal.edge_density, flow, threat_scalar=0.0)
        flat = self.vision.flatten_v1(v1)
        tup = np.zeros(1152, dtype=np.float32)
        n = min(flat.shape[0], tup.shape[0])
        tup[:n] = flat[:n]
        novelty = float(np.clip(np.mean(np.abs(flow)), 0.0, 1.0))
        if novelty > self.cfg.memory.write_novelty_threshold:
            self.memory.write(tup, novelty=novelty, social=False)

        return action, {
            "ball_x": ball_x,
            "ball_y": ball_y,
            "paddle_x": paddle_x,
        }

    def run_episode(self, env, max_steps: int = 12000) -> dict[str, float | int | list[BreakoutStepLog]]:
        obs, info = env.reset()
        self.reset()
        done = False
        truncated = False
        step = 0
        score = 0.0
        logs: list[BreakoutStepLog] = []
        start_lives = int(info.get("lives", 5))
        min_lives = start_lives
        while not (done or truncated) and step < max_steps:
            action, meta = self.act(obs)
            obs, reward, done, truncated, info = env.step(action)
            lives = int(info.get("lives", min_lives))
            min_lives = min(min_lives, lives)
            score += float(reward)
            logs.append(
                BreakoutStepLog(
                    step=step,
                    reward=float(reward),
                    lives=lives,
                    action=int(action),
                    ball_x=float(meta["ball_x"]),
                    ball_y=float(meta["ball_y"]),
                    paddle_x=float(meta["paddle_x"]),
                )
            )
            step += 1
        return {
            "score": float(score),
            "steps": int(step),
            "start_lives": int(start_lives),
            "min_lives": int(min_lives),
            "end_lives": int(logs[-1].lives if logs else start_lives),
            "logs": logs,
        }
