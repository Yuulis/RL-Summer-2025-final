from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces


@dataclass
class ScoringRules:
    short_straight_points: int = 15
    big_straight_points: int = 30
    yacht_points: int = 50
    enable_one_roles_bonus: bool = True
    one_roles_bonus_threshold: int = 63
    one_roles_bonus_points: int = 35


class YachtEnv(gym.Env):
    """
    観測/マスク/報酬成形を内製したヨット環境
    - FullHouse はヨットでも成立、点は出目合計（(6,6,6,6,6)=30）
    - 報酬の基準スケールは /85（上段ボーナスの加点は最終手に加算）

    追加機能:
      - obs_augment: 既存の 9次元拡張（dice_counts6 + upper進捗3）
      - obs_add_scores: 各カテゴリの即時スコア12（正規化）の追加
      - potential shaping: 上段/下段の進捗に対する差分成形
    """
    NUM_DICE = 5
    DIE_FACES = 6
    SCORE_CATEGORIES_NUM = 12

    ACTION_REROLL_COUNT = 32
    ACTION_SCORE_BASE = 32

    REWARD_NORM_DENOMINATOR = 85.0

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        rules: Optional[ScoringRules] = None,
        seed: Optional[int] = None,
        obs_augment: bool = False,
        obs_add_scores: bool = False,
        disable_keep_all: bool = False,
        shaping_upper_eps: float = 0.0,
        shaping_lower_eps: float = 0.0,
    ):
        super().__init__()
        self.rules = rules or ScoringRules()
        self.obs_augment = bool(obs_augment)
        self.obs_add_scores = bool(obs_add_scores)
        self.disable_keep_all = bool(disable_keep_all)
        self.shaping_upper_eps = float(shaping_upper_eps)
        self.shaping_lower_eps = float(shaping_lower_eps)

        # RNG
        self._seed_value = seed
        self.np_random: np.random.Generator = np.random.default_rng(seed)

        # 状態
        self.dice: np.ndarray = np.zeros(self.NUM_DICE, dtype=np.int32)
        self.rolls_left: int = 2
        self.turns_done: int = 0
        self.categories_used: np.ndarray = np.zeros(self.SCORE_CATEGORIES_NUM, dtype=bool)
        self.scores_per_category: np.ndarray = np.zeros(self.SCORE_CATEGORIES_NUM, dtype=np.int32)
        self.total_score: int = 0
        self.one_roles_sum: int = 0  # 上段合計（ボーナス判定用）

        # ポテンシャル（成形用）
        self._phi_upper_last: float = 0.0
        self._phi_lower_last: float = 0.0

        # Spaces
        self.action_space = spaces.Discrete(self.ACTION_REROLL_COUNT + self.SCORE_CATEGORIES_NUM)
        base_obs_dim = self.NUM_DICE + self.SCORE_CATEGORIES_NUM + 2  # 5 + 12 + 2 = 19
        extra = (9 if self.obs_augment else 0) + (12 if self.obs_add_scores else 0)
        self._obs_dim = base_obs_dim + extra
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self._obs_dim,), dtype=np.float32)

    # Gym API
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            self._seed_value = seed
            self.np_random = np.random.default_rng(seed)

        self.turns_done = 0
        self.rolls_left = 2
        self.categories_used[:] = False
        self.scores_per_category[:] = 0
        self.total_score = 0
        self.one_roles_sum = 0

        self._roll_all_dice()

        # ポテンシャル初期化
        self._phi_upper_last = self._phi_upper()
        self._phi_lower_last = self._phi_lower()

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        terminated = False
        truncated = False
        reward = 0.0
        invalid = False

        if not self.action_space.contains(action):
            invalid = True
        else:
            if 0 <= action < self.ACTION_REROLL_COUNT:
                if self.rolls_left <= 0:
                    invalid = True
                else:
                    self._reroll_with_keep_mask(action)
                    reward = 0.0
            else:
                category = action - self.ACTION_SCORE_BASE
                if category < 0 or category >= self.SCORE_CATEGORIES_NUM or self.categories_used[category]:
                    invalid = True
                else:
                    raw_points = self._score_category(self.dice, category)
                    # スコア反映
                    self.scores_per_category[category] = raw_points
                    self.categories_used[category] = True
                    self.total_score += int(raw_points)
                    if 0 <= category <= 5:
                        self.one_roles_sum += int(raw_points)

                    # ターン進行
                    self.turns_done += 1

                    # 最終ターン: 上段ボーナス
                    bonus = 0
                    if self.turns_done >= self.SCORE_CATEGORIES_NUM:
                        if self.rules.enable_one_roles_bonus and self.one_roles_sum >= self.rules.one_roles_bonus_threshold:
                            bonus = int(self.rules.one_roles_bonus_points)
                            self.total_score += bonus
                        terminated = True
                    else:
                        self._start_next_turn()

                    reward = (raw_points + bonus) / self.REWARD_NORM_DENOMINATOR

        # ポテンシャル成形（差分）
        if not invalid:
            shaped = reward
            if self.shaping_upper_eps != 0.0 or self.shaping_lower_eps != 0.0:
                new_phi_u = self._phi_upper()
                new_phi_l = self._phi_lower()
                shaped += self.shaping_upper_eps * (new_phi_u - self._phi_upper_last)
                shaped += self.shaping_lower_eps * (new_phi_l - self._phi_lower_last)
                self._phi_upper_last = new_phi_u
                self._phi_lower_last = new_phi_l
            reward = shaped
        else:
            reward = 0.0

        obs = self._get_obs()
        info = self._get_info()
        if invalid:
            info["invalid_action"] = True
        return obs, float(reward), terminated, truncated, info

    # マスク（MaskablePPO が env.env_method("action_masks") で参照）
    def action_masks(self) -> np.ndarray:
        mask = np.zeros(self.action_space.n, dtype=bool)
        # リロール（残りがあれば許可）
        if self.rolls_left > 0:
            mask[0:self.ACTION_REROLL_COUNT] = True
            # keep-all禁止（任意）
            if self.disable_keep_all and mask.shape[0] > 31:
                mask[31] = False
        # スコア確定（未使用のみ）
        for c in range(self.SCORE_CATEGORIES_NUM):
            if not self.categories_used[c]:
                mask[self.ACTION_SCORE_BASE + c] = True
        return mask

    def get_action_mask(self) -> np.ndarray:
        return self.action_masks()

    def render(self):
        print(
            f"Turn {self.turns_done + 1}/12 | Dice: {self.dice.tolist()} | Rolls left: {self.rolls_left} | "
            f"Score so far: {self.total_score}"
        )
        cat_names = [
            "Aces", "Deuces", "Threes", "Fours", "Fives", "Sixes",
            "Choice", "FourDice", "FullHouse", "S.Straight", "B.Straight", "Yacht"
        ]
        used = [f"{cat_names[i]}:{'X' if self.categories_used[i] else '_'}({self.scores_per_category[i]})"
                for i in range(self.SCORE_CATEGORIES_NUM)]
        print(" | ".join(used))

    def close(self):
        pass

    # 内部ロジック
    def _roll_all_dice(self):
        self.dice = self.np_random.integers(1, self.DIE_FACES + 1, size=self.NUM_DICE, dtype=np.int32)

    def _reroll_with_keep_mask(self, keep_mask_bits: int):
        for i in range(self.NUM_DICE):
            keep = (keep_mask_bits >> i) & 1
            if keep == 0:
                self.dice[i] = int(self.np_random.integers(1, self.DIE_FACES + 1))
        self.rolls_left -= 1

    def _start_next_turn(self):
        self._roll_all_dice()
        self.rolls_left = 2

    @staticmethod
    def _counts(dice: np.ndarray) -> np.ndarray:
        return np.bincount(dice, minlength=7)

    def _score_category(self, dice: np.ndarray, category: int) -> int:
        counts = self._counts(dice)
        total = int(np.sum(dice))

        # 0..5: 上段（1〜6）
        if 0 <= category <= 5:
            face = category + 1
            return int(counts[face] * face)

        # 6: チョイス
        if category == 6:
            return total

        # 7: フォーダイス（>=4個同一で総和）
        if category == 7:
            if np.any(counts[1:] >= 4):
                return total
            return 0

        # 8: フルハウス（3+2 または 5個同一（ヨット）でも成立。点数は出目合計）
        if category == 8:
            has_three = np.any(counts[1:] == 3)
            has_two = np.any(counts[1:] == 2)
            is_yacht = np.any(counts[1:] == 5)
            if is_yacht or (has_three and has_two):
                return total
            return 0

        # 9: S.ストレート（4連続が含まれる）
        if category == 9:
            faces = set(int(x) for x in np.unique(dice))
            if ({1, 2, 3, 4}.issubset(faces)
                or {2, 3, 4, 5}.issubset(faces)
                or {3, 4, 5, 6}.issubset(faces)):
                return int(self.rules.short_straight_points)
            return 0

        # 10: B.ストレート（厳密に1-2-3-4-5 または 2-3-4-5-6）
        if category == 10:
            faces_sorted = sorted(int(x) for x in dice)
            if faces_sorted == [1, 2, 3, 4, 5] or faces_sorted == [2, 3, 4, 5, 6]:
                return int(self.rules.big_straight_points)
            return 0

        # 11: ヨット（5個同一で固定点）
        if category == 11:
            if np.any(counts[1:] == 5):
                return int(self.rules.yacht_points)
            return 0

        raise ValueError(f"Unknown category: {category}")

    def _now_scores_normalized(self) -> np.ndarray:
        """
        現在の出目で各カテゴリを即時確定した場合のスコア（12次元）を、カテゴリ毎の最大で正規化して返す。
        """
        raw = np.zeros(12, dtype=np.float32)
        for c in range(12):
            raw[c] = float(self._score_category(self.dice, c))

        # カテゴリ別の最大値
        upper_max = np.array([5, 10, 15, 20, 25, 30], dtype=np.float32)
        others_max = np.array([
            30,  # Choice
            30,  # FourDice
            30,  # FullHouse
            float(self.rules.short_straight_points),  # S
            30,  # B
            float(self.rules.yacht_points),  # Yacht
        ], dtype=np.float32)
        max_per_cat = np.concatenate([upper_max, others_max]).astype(np.float32)
        norm = raw / np.maximum(max_per_cat, 1e-6)
        return np.clip(norm, 0.0, 1.0)

    # 観測・情報
    def _get_obs(self) -> np.ndarray:
        dice_norm = (self.dice.astype(np.float32) - 1.0) / 5.0
        cat_free = (~self.categories_used).astype(np.float32)
        rolls_left_norm = np.array([self.rolls_left / 2.0], dtype=np.float32)
        turns_done_norm = np.array([self.turns_done / float(self.SCORE_CATEGORIES_NUM)], dtype=np.float32)
        obs = np.concatenate([dice_norm, cat_free, rolls_left_norm, turns_done_norm], dtype=np.float32)

        if self.obs_augment:
            counts = np.bincount(self.dice, minlength=7)[1:7].astype(np.float32) / 5.0
            upper_sum_norm = np.array([min(self.one_roles_sum, 63) / 63.0], dtype=np.float32)
            upper_slots_left = 6 - int(self.categories_used[:6].sum())
            lower_slots_left = 6 - int(self.categories_used[6:].sum())
            upper_slots_left_norm = np.array([upper_slots_left / 6.0], dtype=np.float32)
            lower_slots_left_norm = np.array([lower_slots_left / 6.0], dtype=np.float32)
            extra = np.concatenate([counts, upper_sum_norm, upper_slots_left_norm, lower_slots_left_norm], dtype=np.float32)
            obs = np.concatenate([obs, extra], dtype=np.float32)

        if self.obs_add_scores:
            obs = np.concatenate([obs, self._now_scores_normalized()], dtype=np.float32)

        return obs

    def _get_info(self) -> Dict[str, Any]:
        return {
            "dice": self.dice.copy(),
            "rolls_left": self.rolls_left,
            "turns_done": self.turns_done,
            "categories_used": self.categories_used.copy(),
            "scores_per_category": self.scores_per_category.copy(),
            "total_score": self.total_score,
            "one_roles_sum": self.one_roles_sum,
            "action_mask": self.action_masks(),
        }

    # ポテンシャル
    def _phi_upper(self) -> float:
        return min(float(self.one_roles_sum), 63.0) / 63.0

    def _phi_lower(self) -> float:
        filled_lower = int(self.categories_used[6:].sum())
        return float(filled_lower) / 6.0