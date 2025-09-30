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
    強化版ヨット環境（観測/行動マスク/報酬成形一体化）

    追加フラグ:
      - obs_augment: 既存拡張 (counts6 + upper進捗 + 残枠) = 9
      - obs_add_scores: 各カテゴリ即時スコア 12
      - obs_progress: 新規 11
          [needed_upper_faces(6), longest_run, missing_1_5, missing_2_6,
           max_count_norm, second_count_norm] (最後は 2 要素なので合計 11)
      - disable_keep_all: keep-all(=31) リロールをマスク
      - shaping_upper_eps / shaping_lower_eps: 既存ポテンシャル成形
      - shaping_bonus_eps: 上段ボーナス到達可能性ポテンシャル

    FullHouse: 5個同一（ヨット）も成立、合計点 (6,6,6,6,6)=30
    報酬: スコア確定時 (raw+bonus)/85 + 成形差分
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
        obs_progress: bool = False,
        disable_keep_all: bool = False,
        shaping_upper_eps: float = 0.0,
        shaping_lower_eps: float = 0.0,
        shaping_bonus_eps: float = 0.0,
    ):
        super().__init__()
        self.rules = rules or ScoringRules()
        self.obs_augment = bool(obs_augment)
        self.obs_add_scores = bool(obs_add_scores)
        self.obs_progress = bool(obs_progress)
        self.disable_keep_all = bool(disable_keep_all)
        self.shaping_upper_eps = float(shaping_upper_eps)
        self.shaping_lower_eps = float(shaping_lower_eps)
        self.shaping_bonus_eps = float(shaping_bonus_eps)

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
        self.one_roles_sum: int = 0  # 上段合計

        # ポテンシャル記録
        self._phi_upper_last = 0.0
        self._phi_lower_last = 0.0
        self._phi_bonus_last = 0.0

        # 観測次元計算
        base = 5 + 12 + 2  # 19
        extra = 0
        if self.obs_augment:
            extra += 9
        if self.obs_add_scores:
            extra += 12
        if self.obs_progress:
            extra += 11
        self._obs_dim = base + extra

        self.action_space = spaces.Discrete(self.ACTION_REROLL_COUNT + self.SCORE_CATEGORIES_NUM)
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
        self._phi_bonus_last = self._phi_bonus()

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
            else:
                category = action - self.ACTION_SCORE_BASE
                if category < 0 or category >= self.SCORE_CATEGORIES_NUM or self.categories_used[category]:
                    invalid = True
                else:
                    raw_points = self._score_category(self.dice, category)
                    self.scores_per_category[category] = raw_points
                    self.categories_used[category] = True
                    self.total_score += int(raw_points)
                    if 0 <= category <= 5:
                        self.one_roles_sum += int(raw_points)

                    self.turns_done += 1
                    bonus = 0
                    if self.turns_done >= self.SCORE_CATEGORIES_NUM:
                        if (self.rules.enable_one_roles_bonus
                                and self.one_roles_sum >= self.rules.one_roles_bonus_threshold):
                            bonus = int(self.rules.one_roles_bonus_points)
                            self.total_score += bonus
                        terminated = True
                    else:
                        self._start_next_turn()

                    reward = (raw_points + bonus) / self.REWARD_NORM_DENOMINATOR

        # 成形（差分）: upper / lower / bonus
        if not invalid:
            shaped = reward
            new_phi_upper = self._phi_upper()
            new_phi_lower = self._phi_lower()
            new_phi_bonus = self._phi_bonus()

            if self.shaping_upper_eps != 0.0:
                shaped += self.shaping_upper_eps * (new_phi_upper - self._phi_upper_last)
            if self.shaping_lower_eps != 0.0:
                shaped += self.shaping_lower_eps * (new_phi_lower - self._phi_lower_last)
            if self.shaping_bonus_eps != 0.0:
                shaped += self.shaping_bonus_eps * (new_phi_bonus - self._phi_bonus_last)

            self._phi_upper_last = new_phi_upper
            self._phi_lower_last = new_phi_lower
            self._phi_bonus_last = new_phi_bonus
            reward = shaped
        else:
            reward = 0.0

        obs = self._get_obs()
        info = self._get_info()
        if invalid:
            info["invalid_action"] = True
        return obs, float(reward), terminated, truncated, info

    # 行動マスク
    def action_masks(self) -> np.ndarray:
        mask = np.zeros(self.action_space.n, dtype=bool)
        if self.rolls_left > 0:
            mask[0:self.ACTION_REROLL_COUNT] = True
            if self.disable_keep_all and mask.shape[0] > 31:
                mask[31] = False
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

    # 内部ユーティリティ
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
        return np.bincount(dice, minlength=7)  # index 0 unused

    def _score_category(self, dice: np.ndarray, category: int) -> int:
        counts = self._counts(dice)
        total = int(np.sum(dice))

        if 0 <= category <= 5:
            face = category + 1
            return int(counts[face] * face)

        if category == 6:  # Choice
            return total

        if category == 7:  # FourDice
            if np.any(counts[1:] >= 4):
                return total
            return 0

        if category == 8:  # FullHouse (含むヨット)
            has_three = np.any(counts[1:] == 3)
            has_two = np.any(counts[1:] == 2)
            is_yacht = np.any(counts[1:] == 5)
            if is_yacht or (has_three and has_two):
                return total
            return 0

        if category == 9:  # S Straight
            faces = set(int(x) for x in np.unique(dice))
            if ({1, 2, 3, 4}.issubset(faces)
                or {2, 3, 4, 5}.issubset(faces)
                or {3, 4, 5, 6}.issubset(faces)):
                return int(self.rules.short_straight_points)
            return 0

        if category == 10:  # B Straight
            faces_sorted = sorted(int(x) for x in dice)
            if faces_sorted == [1, 2, 3, 4, 5] or faces_sorted == [2, 3, 4, 5, 6]:
                return int(self.rules.big_straight_points)
            return 0

        if category == 11:  # Yacht
            if np.any(counts[1:] == 5):
                return int(self.rules.yacht_points)
            return 0

        raise ValueError(f"Unknown category: {category}")

    def _now_scores_normalized(self) -> np.ndarray:
        raw = np.zeros(12, dtype=np.float32)
        for c in range(12):
            raw[c] = float(self._score_category(self.dice, c))
        upper_max = np.array([5, 10, 15, 20, 25, 30], dtype=np.float32)
        others_max = np.array([
            30,  # Choice
            30,  # FourDice
            30,  # FullHouse
            float(self.rules.short_straight_points),
            30,  # Big
            float(self.rules.yacht_points),
        ], dtype=np.float32)
        max_per_cat = np.concatenate([upper_max, others_max]).astype(np.float32)
        norm = raw / np.maximum(max_per_cat, 1e-6)
        return np.clip(norm, 0.0, 1.0)

    def _progress_features(self) -> np.ndarray:
        """
        追加 11 次元:
         needed_upper_faces(6),
         longest_run_norm,
         missing_1_5_norm,
         missing_2_6_norm,
         max_count_norm,
         second_count_norm
        """
        counts = self._counts(self.dice)[1:7]  # length 6
        faces_set = set(int(x) for x in self.dice)

        # needed_upper_faces: ボーナス到達残ポイントを各faceだけで埋めるなら何個必要か (dice数/5で正規化)
        remaining_needed = max(0, self.rules.one_roles_bonus_threshold - self.one_roles_sum)
        needed_list = []
        for face in range(1, 7):
            if self.categories_used[face - 1]:
                needed_list.append(0.0)
            else:
                if remaining_needed <= 0:
                    needed_list.append(0.0)
                else:
                    req_dice = remaining_needed / face  # その目のみで埋める場合必要個数
                    needed_list.append(float(min(1.0, req_dice / 5.0)))

        # 最長連続
        def longest_run():
            best = 1
            cur = 1
            in_set = faces_set
            for f in range(2, 7):
                if f in in_set and (f - 1) in in_set:
                    cur += 1
                    best = max(best, cur)
                else:
                    cur = 1
            return best

        lr_norm = longest_run() / 5.0
        missing_1_5 = (5 - len(faces_set.intersection({1, 2, 3, 4, 5}))) / 5.0
        missing_2_6 = (5 - len(faces_set.intersection({2, 3, 4, 5, 6}))) / 5.0

        sorted_counts = np.sort(counts)[::-1]
        maxc = sorted_counts[0] / 5.0
        secondc = (sorted_counts[1] / 5.0) if len(sorted_counts) > 1 else 0.0

        return np.array(
            needed_list + [lr_norm, missing_1_5, missing_2_6, maxc, secondc],
            dtype=np.float32
        )

    # 観測生成
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
            extra = np.concatenate(
                [counts, upper_sum_norm, upper_slots_left_norm, lower_slots_left_norm],
                dtype=np.float32
            )
            obs = np.concatenate([obs, extra], dtype=np.float32)

        if self.obs_add_scores:
            obs = np.concatenate([obs, self._now_scores_normalized()], dtype=np.float32)

        if self.obs_progress:
            obs = np.concatenate([obs, self._progress_features()], dtype=np.float32)

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

    def _phi_bonus(self) -> float:
        """
        楽観的上段最大 (現在上段 + 未使用face*5*face?) ではなく簡略:
        optimistic = one_roles_sum + Σ_{未使用 face} (5*face)
        """
        optimistic_remaining = 0
        for face in range(1, 7):
            if not self.categories_used[face - 1]:
                optimistic_remaining += 5 * face
        optimistic_total = self.one_roles_sum + optimistic_remaining
        if optimistic_total <= 0:
            return 0.0
        if optimistic_total >= self.rules.one_roles_bonus_threshold:
            return 1.0
        return optimistic_total / self.rules.one_roles_bonus_threshold