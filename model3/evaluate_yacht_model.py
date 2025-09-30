import os
import argparse
from typing import List, Tuple

import numpy as np

from yachtEnv import YachtEnv, ScoringRules

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.utils import get_action_masks


def mask_fn(env: YachtEnv):
    # 環境が提供するアクションマスク（True=選択可）を返す
    return env.get_action_mask()


def make_env(
    seed: int,
    short_straight_points: int,
    big_straight_points: int,
    yacht_points: int,
    enable_upper_bonus: bool,
    upper_bonus_threshold: int,
    upper_bonus_points: int,
) -> YachtEnv:
    """
    評価用の環境を生成（ActionMasker で無効行動を除外）。
    """
    rules = ScoringRules(
        short_straight_points=short_straight_points,
        big_straight_points=big_straight_points,
        yacht_points=yacht_points,
        enable_one_roles_bonus=enable_upper_bonus,
        one_roles_bonus_threshold=upper_bonus_threshold,
        one_roles_bonus_points=upper_bonus_points,
    )
    base_env = YachtEnv(rules=rules, seed=seed)
    return ActionMasker(base_env, mask_fn)


def run_one_episode(env: YachtEnv, model: MaskablePPO, deterministic: bool = True) -> Tuple[float, int]:
    """
    1エピソードを実行し、正規化リターン合計と最終 total_score を返す。
    """
    obs, info = env.reset()
    done = False
    truncated = False
    ep_return = 0.0
    last_info = info

    while not (done or truncated):
        # マスクを取得して predict に渡す（無効行動の除外）
        action_masks = get_action_masks(env)
        action, _ = model.predict(obs, deterministic=deterministic, action_masks=action_masks)
        obs, reward, done, truncated, info = env.step(int(action))
        ep_return += float(reward)
        last_info = info

    total_score = int(last_info.get("total_score", 0))
    return ep_return, total_score


def evaluate(
    model_path: str,
    episodes: int = 50,
    seed: int = 123,
    deterministic: bool = True,
    short_straight_points: int = 15,
    big_straight_points: int = 30,
    yacht_points: int = 50,
    enable_upper_bonus: bool = True,
    upper_bonus_threshold: int = 63,
    upper_bonus_points: int = 35,
    verbose: bool = True,
) -> None:
    """
    学習済みモデルを評価して指標を表示する。
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    # モデル読み込み
    model = MaskablePPO.load(model_path, device="auto")
    # 評価環境（毎エピソードごとにシードを少しずつずらす）
    returns: List[float] = []
    scores: List[int] = []
    bonus_hits = 0

    for ep in range(episodes):
        env = make_env(
            seed=seed + ep,
            short_straight_points=short_straight_points,
            big_straight_points=big_straight_points,
            yacht_points=yacht_points,
            enable_upper_bonus=enable_upper_bonus,
            upper_bonus_threshold=upper_bonus_threshold,
            upper_bonus_points=upper_bonus_points,
        )
        ep_ret, ep_score = run_one_episode(env, model, deterministic=deterministic)
        returns.append(ep_ret)
        scores.append(ep_score)

        # 上段ボーナス達成率（環境の info を直接参照できないため、しきい値推定）
        # 評価終了時点の one_roles_sum は env._get_info() の last_info で取っているが、
        # run_one_episode 内で last_info["one_roles_sum"] は最終時点のものが入っている。
        # ここでは ep_score と scores_per_category の内訳が必要だが簡便化のため推定不可。
        # 達成率を厳密に出したい場合は run_one_episode を拡張して last_info を返す。
        # ここでは「スコア >= 63(上段合計) + 35(ボーナス) の可能性」は判断できないため、
        # 達成率表示は省略するか、run_one_episode の戻り値を拡張してください。
        # bonus_hits += 1 if ... else 0

        if verbose:
            print(f"Episode {ep+1:03d}: normalized_return={ep_ret:.3f}, total_score={ep_score}")

        env.close()

    # 集計
    ret_mean = float(np.mean(returns)) if returns else 0.0
    ret_std = float(np.std(returns)) if returns else 0.0
    score_mean = float(np.mean(scores)) if scores else 0.0
    score_std = float(np.std(scores)) if scores else 0.0

    print("\n===== Evaluation Summary =====")
    print(f"Episodes            : {episodes}")
    print(f"Deterministic       : {deterministic}")
    print(f"Mean return (norm)  : {ret_mean:.4f} ± {ret_std:.4f}")
    print(f"Mean total score    : {score_mean:.2f} ± {score_std:.2f}")
    # print(f"Upper-bonus hit rate: {bonus_hits/episodes:.2%}")  # one_roles_sum を返すよう拡張したら有効化


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained MaskablePPO model on YachtEnv.")
    p.add_argument("--model-path", type=str, required=True, help="学習済みモデル(.zip)のパス")
    p.add_argument("--episodes", type=int, default=50, help="評価エピソード数")
    p.add_argument("--seed", type=int, default=123, help="評価用シード（各エピソードで +ep オフセット）")
    p.add_argument("--stochastic", action="store_true", help="確率的に行動（デフォルトは決定的）")

    # ルール設定（学習時と合わせる）
    p.add_argument("--short-straight-points", type=int, default=15)
    p.add_argument("--big-straight-points", type=int, default=30)
    p.add_argument("--yacht-points", type=int, default=50)
    p.add_argument("--disable-upper-bonus", action="store_true")
    p.add_argument("--upper-bonus-threshold", type=int, default=63)
    p.add_argument("--upper-bonus-points", type=int, default=35)

    p.add_argument("--quiet", action="store_true", help="各エピソードのログを抑制")
    return p.parse_args()


def main():
    args = parse_args()
    evaluate(
        model_path=args.model_path,
        episodes=args.episodes,
        seed=args.seed,
        deterministic=not args.stochastic,
        short_straight_points=args.short_straight_points,
        big_straight_points=args.big_straight_points,
        yacht_points=args.yacht_points,
        enable_upper_bonus=not args.disable_upper_bonus,
        upper_bonus_threshold=args.upper_bonus_threshold,
        upper_bonus_points=args.upper_bonus_points,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()