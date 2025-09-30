import argparse
from typing import Optional, List, Tuple

import numpy as np

from yachtEnv import YachtEnv, ScoringRules

# MaskablePPO を使う場合に備えて import（未インストールでも動くように try-import）
try:
    from sb3_contrib import MaskablePPO  # type: ignore
except Exception:
    MaskablePPO = None  # type: ignore


CATEGORY_NAMES: List[str] = [
    "Aces", "Deuces", "Threes", "Fours", "Fives", "Sixes",
    "Choice", "FourDice", "FullHouse", "S.Straight", "B.Straight", "Yacht"
]


def decode_action(action: int) -> str:
    """アクション整数を人間可読に整形."""
    if 0 <= action <= 31:
        bits = [(action >> i) & 1 for i in range(5)]
        keep = [i for i, b in enumerate(bits) if b == 1]
        return f"ReRoll keep_mask=0b{action:05b} keep={keep}"
    elif 32 <= action <= 43:
        cat = action - 32
        return f"Score category={cat}({CATEGORY_NAMES[cat]})"
    return f"Invalid({action})"


def pretty_score_sheet(categories_used: np.ndarray, scores_per_category: np.ndarray) -> str:
    """スコアシートの使用状況と確定スコアをテキストに整形."""
    rows = []
    for i, name in enumerate(CATEGORY_NAMES):
        used = "X" if categories_used[i] else "_"
        rows.append(f"{i:2d}:{name:<12} used={used} score={int(scores_per_category[i])}")
    return "\n".join(rows)


def choose_random_valid_action(info: dict, rng: np.random.Generator) -> int:
    """アクションマスクからランダムに有効行動を1つ選ぶ."""
    mask = np.asarray(info.get("action_mask", None))
    assert mask is not None and mask.dtype == bool and mask.shape[0] == 44, "action_mask が取得できません。"
    valid_idx = np.nonzero(mask)[0]
    return int(rng.choice(valid_idx))


def compute_step_scores(before_info: dict, after_info: dict) -> Tuple[int, int]:
    """
    ステップで新規に加算された「カテゴリ生点」と「ボーナス」を推定して返す.
    - 生点: ステップ前後の scores_per_category 合計の差
    - ボーナス: total_score(after) - sum(scores_per_category(after))
      （ボーナスは最終ステップでのみ非ゼロ）
    """
    pre_sum = int(np.sum(before_info["scores_per_category"]))
    post_sum = int(np.sum(after_info["scores_per_category"]))
    raw_points = post_sum - pre_sum
    # ボーナスは total_score - scores_sum（この環境では最終手のみ加算）
    bonus = int(after_info["total_score"]) - post_sum
    return raw_points, bonus


def log_episode(env: YachtEnv, model: Optional[object] = None, deterministic: bool = True, seed: int = 0):
    """
    1エピソードを実行し、各ステップで環境の変化をテキスト出力する。
    - model=None なら、アクションマスクにもとづくランダム方策で実行。
    - model が MaskablePPO の場合は action_masks を渡して推論する。
    """
    rng = np.random.default_rng(seed)

    obs, info = env.reset()
    step_idx = 0

    print("==== EPISODE START ====")
    print(f"Turn={info['turns_done']+1}/12, RollsLeft={info['rolls_left']}, Dice={info['dice'].tolist()}")
    print("Initial Score Sheet:\n" + pretty_score_sheet(info["categories_used"], info["scores_per_category"]))
    print("-" * 60)

    done = False
    truncated = False
    total_return = 0.0
    last_info = info

    while not (done or truncated):
        # ステップ前の情報をスナップショット（差分計算用）
        before = last_info

        # アクション選択
        if model is None:
            action = choose_random_valid_action(before, rng)
        else:
            if MaskablePPO is not None and isinstance(model, MaskablePPO):
                # sb3-contrib: action_masks を渡す
                action_masks = np.asarray(before["action_mask"], dtype=bool)
                action, _ = model.predict(obs, deterministic=deterministic, action_masks=action_masks)
            else:
                # 通常の SB3 モデル（PPO など）
                action, _ = model.predict(obs, deterministic=deterministic)

        # 実行
        step_idx += 1
        prev_dice = before["dice"].tolist()
        prev_rolls = before["rolls_left"]
        prev_turn = before["turns_done"] + 1

        obs, reward, done, truncated, info = env.step(int(action))
        total_return += float(reward)

        # 差分から「今回加算された生点」と「ボーナス」を推定
        raw_points, bonus = compute_step_scores(before, info)

        # ログ出力
        print(f"[Step {step_idx:03d}] Turn {prev_turn:02d} | {decode_action(int(action))}")
        print(f"  Dice: {prev_dice}  ->  {info['dice'].tolist()}")
        print(f"  RollsLeft: {prev_rolls} -> {info['rolls_left']}")
        if action >= 32:
            # スコア確定時のみスコア関連を詳しく表示
            cat = int(action) - 32
            print(f"  Scored: {CATEGORY_NAMES[cat]} | raw_points={raw_points} | bonus={bonus} "
                  f"| normalized_reward={reward:.3f} | total_score={info['total_score']}")
            print("  Score Sheet:\n" + pretty_score_sheet(info["categories_used"], info["scores_per_category"]))
        else:
            # リロール時は報酬は常に 0
            print(f"  Rerolled. reward={reward:.3f} (always 0 on reroll)")

        print("-" * 60)
        last_info = info

    print(f"==== EPISODE END ====")
    print(f"Total normalized return = {total_return:.3f}")
    print(f"Final total score (env) = {last_info.get('total_score', 0)}")
    print("Final Score Sheet:\n" + pretty_score_sheet(last_info["categories_used"], last_info["scores_per_category"]))


def make_env(seed: int = 0) -> YachtEnv:
    """検証用にシンプルな環境を生成（ActionMaskerは不要。info['action_mask']を直接利用します）"""
    return YachtEnv(rules=ScoringRules(), seed=seed)


def main():
    parser = argparse.ArgumentParser(description="Log one episode of YachtEnv step-by-step.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--model-path", type=str, default="", help="MaskablePPO model path (.zip). Empty for random.")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic policy at inference.")
    args = parser.parse_args()

    env = make_env(seed=args.seed)

    model = None
    if args.model_path:
        if MaskablePPO is None:
            raise RuntimeError("sb3-contrib is not installed, cannot load MaskablePPO model.")
        model = MaskablePPO.load(args.model_path, device="cpu")
        print(f"Loaded model: {args.model_path}")

    log_episode(env, model=model, deterministic=args.deterministic, seed=args.seed)


if __name__ == "__main__":
    main()