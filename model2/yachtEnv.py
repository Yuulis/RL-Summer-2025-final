from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces


@dataclass
class ScoringRules:
    """
    ヨットの得点ルールに関する設定. 任天堂の「世界の遊び大全51」で収録されているルールをベースにする.

      - フォーダイス : 同じ出目が4つ以上揃ったときに選択可. 点数はその時点の出目の総和.
      - フルハウス : 同じ出目のグループが2つあり, グループ内の個数が2個と3個であるときに選択可. 点数はそのじてんの出目の総和.
      - S.ストレート : 4つの出目が昇順につながっているときに選択すると, 15 点を得られる.
      - B.ストレート : 5つの出目が昇順にそろっているときに選択すると, 30 点を得られる.
      - ヨット  5つの出目が全て同じときに選択すると, 50 点を得られる.
      - チョイス : 選択すると, その時点の出目の総和を得点として得られる.
      - 一役割(one_roles) : 選択すると, その時点で該当の出目が出ている数だけ得点を得られる.
        - エース(1), デュース(2), トレイ(3), フォー(4), ファイブ(5), シックス(6)
        - 6種類の役の合計得点が 63 点を超えると, ボーナスとしてさらに 35 点を得られる.
    """

    short_straight_points: int = 30
    big_straight_points: int = 30
    yacht_points: int = 50
    enable_one_roles_bonus: bool = True
    one_roles_bonus_threshold: int = 63
    one_roles_bonus_points: int = 35


class YachtEnv(gym.Env):
    """
    Gymnasium環境下で, 1人でヨットをプレイできる環境.
    - Horizon = 36 (全 12 ターン * ロール最大 3 回)
    - 各ターンの初期状態では, 5つのサイコロは自動的に振られ, 残りロール回数(rolls_left)は 2 となる.

    - Agent は以下の行動を選択できる:
      1. キープするサイコロを任意個選択し, それ以外のサイコロについて, rolls_left を 1 消費して振り直す(rolls_left > 0 のときのみ選択可).
      2. スコアシート上の 12 個の枠のうちその時点で未確定のものを任意に選択し, その枠の得点を現時点のサイコロの出目の状態から計算される値で確定させる.
        - rolls_left が 0 でなくてもこの行動は選択可能であり, その場合は即座に現在のターンが終了する.
    - 実装上は以下のような行動空間 Discrete(44) を与える:
      - 0 ~ 31 : 5つのサイコロのうちどれをキープするかを表した 5 bits の2進数.
        - ex.) 0b01001(9) => 2つ目と5つ目のサイコロをキープする
      - 32 ~ 43 : スコアシートの 12 個の枠のうちどの枠の得点を確定させるかを表した整数.この値から 32 を引いた枠番号の得点が確定される.
        - 枠番号は以下の通り.
          - 0 : エース
          - 1 : デュース
          - 2 : トレイ
          - 3 : フォー
          - 4 : ファイブ
          - 5 : シックス
          - 6 : チョイス
          - 7 : フォーダイス
          - 8 : フルハウス
          - 9 : S.ストレート
          - 10 : B.ストレート
          - 11 : ヨット

    - Agent が得る報酬は以下の通り(正規化済):
      - 行動1 : + 0.0
      - 行動2 : + (確定した枠の得点) / 85

    - Agent に与えられる観測情報は 19 次元ベクトル. 内訳は以下の通り:
      - 5個のサイコロのそれぞれの出目((value-1)/5 として正規化)
      - スコアシートの 12 個の枠の空き状況(0 or 1 のフラグ)
      - 残りロール回数(rolls_left / 2 として正規化)
      - 経過ターン数(turns_done / 12 として正規化)
    - 実装上は Box(19, ) として与える.
    """

    NUM_DICE = 5  # サイコロの個数
    DIE_FACES = 6 # サイコロの面数
    SCORE_CATEGORIES_NUM = 12 # スコアシートの枠数
    ACTION_REROLL_COUNT = 32
    ACTION_SCORE_BASE = 32
    OBS_SIZE = NUM_DICE + SCORE_CATEGORIES_NUM + 2  # 観測空間のサイズ
    REWARD_NORM_DENOMINATOR = 85.0  # 報酬正規化のための定数

    metadata = {"render_modes": ["human"], "render_fps": 4}


    def __init__(self, rules: Optional[ScoringRules] = None, seed: Optional[int] = None):
        super().__init__()
        self.rules = rules or ScoringRules()

        # 観測空間と行動空間
        self.action_space = spaces.Discrete(self.ACTION_REROLL_COUNT + self.SCORE_CATEGORIES_NUM)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.OBS_SIZE,), dtype=np.float32)

        # 乱数
        self._seed_value = seed
        self.np_random: np.random.Generator = np.random.default_rng(seed)

        # 内部状態
        self.dice: np.ndarray = np.zeros(self.NUM_DICE, dtype=np.int32)
        self.rolls_left: int = 2
        self.turns_done: int = 0
        self.categories_used: np.ndarray = np.zeros(self.SCORE_CATEGORIES_NUM, dtype=bool)
        self.scores_per_category: np.ndarray = np.zeros(self.SCORE_CATEGORIES_NUM, dtype=np.int32)
        self.total_score: int = 0
        self.one_roles_sum: int = 0

    # ===== Gymnasium API =====
    # 環境の初期化
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
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    # 学習を 1 Step 進める
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        terminated = False
        truncated = False
        reward = 0.0
        invalid = False

        # 行動の正当性チェック
        if not self.action_space.contains(action):
            invalid = True
        else:
            if 0 <= action < self.ACTION_REROLL_COUNT:
                # リロール(rolls_left > 0 のときのみ)
                if self.rolls_left <= 0:
                    invalid = True
                else:
                    self._reroll_with_keep_mask(action)
                    reward = 0.0
            else:
                # スコアシートの空き枠を埋める
                category = action - self.ACTION_SCORE_BASE
                if category < 0 or category >= self.SCORE_CATEGORIES_NUM:
                    invalid = True
                elif self.categories_used[category]:
                    invalid = True
                else:
                    raw_points = self._score_category(self.dice, category)
                    # スコア反映
                    self.scores_per_category[category] = raw_points
                    self.categories_used[category] = True
                    self.total_score += int(raw_points)
                    if 0 <= category <= 5:
                        self.one_roles_sum += int(raw_points)

                    # ターン経過
                    self.turns_done += 1

                    # one_roles のボーナス判定
                    bonus = 0
                    if self.turns_done >= self.SCORE_CATEGORIES_NUM:
                        if self.rules.enable_one_roles_bonus and self.one_roles_sum >= self.rules.one_roles_bonus_threshold:
                            bonus = int(self.rules.one_roles_bonus_points)
                            self.total_score += bonus
                        terminated = True
                    else:
                        self._start_next_turn()

                    # 報酬
                    reward = (raw_points + bonus) / self.REWARD_NORM_DENOMINATOR

        # 無効アクションに対する報酬(実際は起こらない)
        if invalid:
            reward = 0.0

        obs = self._get_obs()
        info = self._get_info()
        if invalid:
            info["invalid_action"] = True
        return obs, float(reward), terminated, truncated, info

    # 環境の状態可視化
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

    # ===== ヘルパー関数 =====
    def get_action_mask(self) -> np.ndarray:
        """
        有効なアクションのみ True となるマスクを返す(長さ 44 の bool 配列)
        - リロール(0 ~ 31) : rolls_left > 0 のとき True
        - スコア確定(32 ~ 43) : 空き枠のみ True
        """

        mask = np.zeros(self.action_space.n, dtype=bool)

        # リロール
        if self.rolls_left > 0:
            mask[0:self.ACTION_REROLL_COUNT] = True
        # スコア確定
        for c in range(self.SCORE_CATEGORIES_NUM):
            if not self.categories_used[c]:
                mask[self.ACTION_SCORE_BASE + c] = True
        return mask

    # 内部ロジック
    def _roll_all_dice(self):
        self.dice = self.np_random.integers(1, self.DIE_FACES + 1, size=self.NUM_DICE, dtype=np.int32)

    def _reroll_with_keep_mask(self, keep_mask_bits: int):
        # 5ビットの各ビット（LSBがダイス0）で保持(1)/振り直し(0)を表す
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
        # counts[face] (face=1..6) を返す（index 0 は未使用）
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

        # 8: フルハウス（3+2のみ。5個同一は不可）
        if category == 8:
            has_three = np.any(counts[1:] == 3)
            has_two = np.any(counts[1:] == 2)
            if has_three and has_two:
                return total
            return 0

        # 9: S.ストレート（4連続が含まれる：1-2-3-4, 2-3-4-5, 3-4-5-6）
        if category == 9:
            faces = set(int(x) for x in np.unique(dice))
            if ({1, 2, 3, 4}.issubset(faces)
                or {2, 3, 4, 5}.issubset(faces)
                or {3, 4, 5, 6}.issubset(faces)):
                return int(self.rules.short_straight_points)
            return 0

        # 10: B.ストレート（厳密に1-2-3-4-5 または 2-3-4-5-6）
        if category == 10:
            faces = sorted(int(x) for x in dice)
            if faces == [1, 2, 3, 4, 5] or faces == [2, 3, 4, 5, 6]:
                return int(self.rules.big_straight_points)
            return 0

        # 11: ヨット（5個同一で固定点）
        if category == 11:
            if np.any(counts[1:] == 5):
                return int(self.rules.yacht_points)
            return 0

        # 想定外
        raise ValueError(f"Unknown category: {category}")

    # 観測・情報
    def _get_obs(self) -> np.ndarray:
        # ダイスの値を (value-1)/5 ∈ [0,1] に正規化
        dice_norm = (self.dice.astype(np.float32) - 1.0) / 5.0
        cat_free = (~self.categories_used).astype(np.float32)  # 空き状況：空き=1.0, 使用済み=0.0
        rolls_left_norm = np.array([self.rolls_left / 2.0], dtype=np.float32)
        turns_done_norm = np.array([self.turns_done / float(self.SCORE_CATEGORIES_NUM)], dtype=np.float32)
        obs = np.concatenate([dice_norm, cat_free, rolls_left_norm, turns_done_norm], dtype=np.float32)
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
            "action_mask": self.get_action_mask(),
        }