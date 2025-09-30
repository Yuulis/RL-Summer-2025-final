import os
import argparse
from typing import Tuple, List, Optional

import numpy as np
from sb3_contrib import MaskablePPO

from yachtenv import YachtEnv, ScoringRules


def theoretical_max_total(short_straight_points: int) -> int:
    """
    理論最大合計点（この実装のルールに基づく）
    - 上段（エース〜シックス）最大: 140（ボーナス込み）
    - Choice: 30
    - FourDice: 30
    - FullHouse: 30（ヨットも成立し合計点、(6,6,6,6,6)=30）
    - B.ストレート: 30
    - Yacht: 50
    - S.ストレート: 引数 short_straight_points（例: 15）
    合計: 310 + short_straight_points
    """
    return 310 + int(short_straight_points)


def _make_env_once(
    seed: int,
    rules: ScoringRules,
    obs_augment: bool,
    disable_keep_all: bool,
) -> YachtEnv:
    return YachtEnv(
        rules=rules,
        seed=seed,
        obs_augment=obs_augment,
        disable_keep_all=disable_keep_all,
    )


def build_env_for_model(
    model: MaskablePPO,
    seed: int,
    short_straight_points: int,
    big_straight_points: int,
    yacht_points: int,
    enable_upper_bonus: bool,
    upper_bonus_threshold: int,
    upper_bonus_points: int,
    obs_augment_mode: str = "auto",  # "auto" | "on" | "off"
    disable_keep_all: bool = False,
) -> Tuple[YachtEnv, bool]:
    """
    モデルが期待する観測次元（19 or 28）に合うように、自動で obs_augment を選んで環境を返す。
    戻り値: (env, used_obs_augment)
    """
    expected_dim = int(np.prod(model.observation_space.shape))
    if obs_augment_mode not in ("auto", "on", "off"):
        raise ValueError("--obs-augment は 'auto' | 'on' | 'off' のいずれかで指定してください。")

    # 希望値を決定（auto の場合は期待次元に合わせる）
    if obs_augment_mode == "on":
        want_aug = True
    elif obs_augment_mode == "off":
        want_aug = False
    else:
        want_aug = (expected_dim == 28)

    rules = ScoringRules(
        short_straight_points=short_straight_points,
        big_straight_points=big_straight_points,
        yacht_points=yacht_points,
        enable_one_roles_bonus=enable_upper_bonus,
        one_roles_bonus_threshold=upper_bonus_threshold,
        one_roles_bonus_points=upper_bonus_points,
    )

    # まず希望設定で作成
    env = _make_env_once(
        seed=seed,
        rules=rules,
        obs_augment=want_aug,
        disable_keep_all=disable_keep_all,
    )
    obs, _ = env.reset()

    if obs.shape[0] == expected_dim:
        return env, want_aug

    # 合わなければ一度だけ反転して再構築
    env.close()
    alt_aug = not want_aug
    env = _make_env_once(
        seed=seed,
        rules=rules,
        obs_augment=alt_aug,
        disable_keep_all=disable_keep_all,
    )
    obs, _ = env.reset()

    if obs.shape[0] != expected_dim:
        env.close()
        raise ValueError(
            f"観測次元の不一致: model expects {expected_dim}, env returned {obs.shape[0]}.\n"
            f"学習時の観測設定（obs_augmentの有無）と評価時が一致していない可能性があります。\n"
            f"評価をやり直す際は、--obs-augment on/off を明示的に指定してください。"
        )

    print(f"[Info] Auto-switched obs_augment to {alt_aug} to match model's expected dim {expected_dim}.")
    return env, alt_aug


def run_one_episode(
    env: YachtEnv,
    model: MaskablePPO,
    deterministic: bool = True,
) -> Tuple[float, int]:
    """
    1エピソード評価して
    - ep_return: 環境報酬合計（= total_score / 85 に一致）
    - total_score: 絶対合計点
    を返す
    """
    obs, info = env.reset()
    done = False
    truncated = False
    ep_return = 0.0
    last_info = info

    while not (done or truncated):
        action_masks = env.action_masks()  # 環境内マスク（ActionMasker不要）
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
    obs_augment_mode: str = "auto",  # auto/on/off
    disable_keep_all: bool = False,
    verbose: bool = True,
    save_csv: str = "",
) -> None:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = MaskablePPO.load(model_path, device="auto")

    returns: List[float] = []
    scores: List[int] = []
    details: List[Tuple[int, float, float]] = []  # (ep_idx, norm_by_85, norm_by_max)

    max_total = theoretical_max_total(short_straight_points)

    for ep in range(episodes):
        env, used_aug = build_env_for_model(
            model=model,
            seed=seed + ep,
            short_straight_points=short_straight_points,
            big_straight_points=big_straight_points,
            yacht_points=yacht_points,
            enable_upper_bonus=enable_upper_bonus,
            upper_bonus_threshold=upper_bonus_threshold,
            upper_bonus_points=upper_bonus_points,
            obs_augment_mode=obs_augment_mode,
            disable_keep_all=disable_keep_all,
        )

        ep_ret, ep_score = run_one_episode(env, model, deterministic=deterministic)
        env.close()

        returns.append(ep_ret)
        scores.append(ep_score)
        norm_by_85 = ep_ret
        norm_by_max = ep_score / max_total
        details.append((ep + 1, norm_by_85, norm_by_max))

        if verbose:
            print(
                f"Episode {ep+1:03d}: total_score={ep_score:3d} | "
                f"norm_by_85={norm_by_85:.3f} | norm_by_max({max_total})={norm_by_max:.3f}"
            )

    ret_mean = float(np.mean(returns)) if returns else 0.0
    ret_std = float(np.std(returns)) if returns else 0.0
    score_mean = float(np.mean(scores)) if scores else 0.0
    score_std = float(np.std(scores)) if scores else 0.0

    print("\n===== Evaluation Summary =====")
    print(f"Episodes                 : {episodes}")
    print(f"Deterministic            : {deterministic}")
    print(f"Mean return (norm/85)    : {ret_mean:.4f} ± {ret_std:.4f}")
    print(f"Mean total score         : {score_mean:.2f} ± {score_std:.2f}")
    print(f"Mean norm_by_max({max_total}): {score_mean/max_total:.4f}")
    print(f"Min/Max total score      : {min(scores)} / {max(scores)}")

    if save_csv:
        import csv
        os.makedirs(os.path.dirname(save_csv) or ".", exist_ok=True)
        with open(save_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "total_score", "norm_by_85", "norm_by_max"])
            for i, (ep_idx, norm85, normmax) in enumerate(details):
                writer.writerow([ep_idx, scores[i], f"{norm85:.6f}", f"{normmax:.6f}"])
        print(f"Saved per-episode results to: {save_csv}")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained MaskablePPO model on YachtEnv (auto-match obs dims).")
    p.add_argument("--model-path", type=str, required=True, help="Path to .zip saved by SB3/MaskablePPO")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--stochastic", action="store_true", help="Use stochastic actions (sampling) instead of deterministic")

    # ルール（学習時と一致させる）
    p.add_argument("--short-straight-points", type=int, default=15)
    p.add_argument("--big-straight-points", type=int, default=30)
    p.add_argument("--yacht-points", type=int, default=50)
    p.add_argument("--disable-upper-bonus", action="store_true")
    p.add_argument("--upper-bonus-threshold", type=int, default=63)
    p.add_argument("--upper-bonus-points", type=int, default=35)

    # 環境内オプション（学習時と一致させる）
    p.add_argument("--obs-augment", type=str, default="auto", choices=["auto", "on", "off"],
                   help="観測拡張を自動/有効/無効。autoはモデルの期待次元に合わせる")
    p.add_argument("--mask-keep-all", action="store_true", help="keep-all(31) を評価時も無効化")

    p.add_argument("--quiet", action="store_true", help="Per-episode printouts off")
    p.add_argument("--csv", type=str, default="", help="Optional path to save per-episode results as CSV")
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
        obs_augment_mode=args.obs_augment,
        disable_keep_all=args.mask_keep_all,
        verbose=not args.quiet,
        save_csv=args.csv,
    )


if __name__ == "__main__":
    main()