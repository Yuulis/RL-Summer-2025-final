import os
import argparse
import numpy as np
from typing import List, Dict, Tuple

from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecMonitor

from yachtenv import YachtEnv, ScoringRules

# --- 理論最大値（FullHouse=30 仕様） ---
def theoretical_max_total(short_straight_points: int) -> int:
    return 310 + int(short_straight_points)

# 観測次元 -> (obs_augment, obs_add_scores)
_OBS_DIM_MAP = {
    19: (False, False),
    28: (True, False),
    31: (False, True),
    40: (True, True),
}

def guess_obs_flags(model_obs_dim: int) -> Tuple[bool, bool]:
    if model_obs_dim not in _OBS_DIM_MAP:
        raise ValueError(
            f"未知の観測次元 {model_obs_dim}。想定: 19/28/31/40。"
            " 学習時と同じ観測拡張フラグを --obs-mode manual と併せて指定してください。"
        )
    return _OBS_DIM_MAP[model_obs_dim]


def make_env_single(seed: int,
                    short_straight_points: int,
                    big_straight_points: int,
                    yacht_points: int,
                    enable_upper_bonus: bool,
                    upper_bonus_threshold: int,
                    upper_bonus_points: int,
                    obs_augment: bool,
                    obs_add_scores: bool,
                    disable_keep_all: bool):
    def _init():
        rules = ScoringRules(
            short_straight_points=short_straight_points,
            big_straight_points=big_straight_points,
            yacht_points=yacht_points,
            enable_one_roles_bonus=enable_upper_bonus,
            one_roles_bonus_threshold=upper_bonus_threshold,
            one_roles_bonus_points=upper_bonus_points,
        )
        return YachtEnv(
            rules=rules,
            seed=seed,
            obs_augment=obs_augment,
            obs_add_scores=obs_add_scores,
            disable_keep_all=disable_keep_all,
            shaping_upper_eps=0.0,  # 評価時は shaping 無効
            shaping_lower_eps=0.0,
        )
    return _init


def build_eval_vec_env(
    n_envs: int,
    base_seed: int,
    short_straight_points: int,
    big_straight_points: int,
    yacht_points: int,
    enable_upper_bonus: bool,
    upper_bonus_threshold: int,
    upper_bonus_points: int,
    obs_augment: bool,
    obs_add_scores: bool,
    disable_keep_all: bool,
    parallel: bool,
):
    env_fns = [
        make_env_single(
            seed=base_seed + i,
            short_straight_points=short_straight_points,
            big_straight_points=big_straight_points,
            yacht_points=yacht_points,
            enable_upper_bonus=enable_upper_bonus,
            upper_bonus_threshold=upper_bonus_threshold,
            upper_bonus_points=upper_bonus_points,
            obs_augment=obs_augment,
            obs_add_scores=obs_add_scores,
            disable_keep_all=disable_keep_all,
        ) for i in range(n_envs)
    ]
    if parallel and n_envs > 1:
        venv = SubprocVecEnv(env_fns, start_method="spawn")
    else:
        venv = DummyVecEnv(env_fns)
    venv = VecMonitor(venv)
    return venv


def evaluate(
    model_path: str,
    episodes: int,
    seed: int,
    deterministic: bool,
    short_straight_points: int,
    big_straight_points: int,
    yacht_points: int,
    enable_upper_bonus: bool,
    upper_bonus_threshold: int,
    upper_bonus_points: int,
    obs_mode: str,          # auto | manual
    obs_augment_flag: bool,
    obs_add_scores_flag: bool,
    disable_keep_all: bool,
    vecnorm_path: str,
    csv_path: str,
    quiet: bool,
    n_eval_envs: int,
    parallel: bool,
):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    # まずモデルだけロード（env なし）し観測次元を知る
    tmp_model = MaskablePPO.load(model_path, device="auto")
    model_obs_dim = int(np.prod(tmp_model.observation_space.shape))

    if obs_mode == "auto":
        obs_augment, obs_add_scores = guess_obs_flags(model_obs_dim)
    else:
        obs_augment, obs_add_scores = obs_augment_flag, obs_add_scores_flag
        expected_dims = {
            (False, False): 19,
            (True, False): 28,
            (False, True): 31,
            (True, True): 40,
        }
        exp = expected_dims[(obs_augment, obs_add_scores)]
        if exp != model_obs_dim:
            print(
                f"[Warning] モデル期待次元={model_obs_dim} と手動指定構成({exp})が不一致です。"
                " 学習時のフラグを確認してください。"
            )

    max_total = theoretical_max_total(short_straight_points)

    # 評価用 VecEnv 構築（必要な本数）
    venv = build_eval_vec_env(
        n_envs=n_eval_envs,
        base_seed=seed,
        short_straight_points=short_straight_points,
        big_straight_points=big_straight_points,
        yacht_points=yacht_points,
        enable_upper_bonus=enable_upper_bonus,
        upper_bonus_threshold=upper_bonus_threshold,
        upper_bonus_points=upper_bonus_points,
        obs_augment=obs_augment,
        obs_add_scores=obs_add_scores,
        disable_keep_all=disable_keep_all,
        parallel=parallel,
    )

    # VecNormalize 統計を読み込む（学習で使っていた場合）
    if vecnorm_path and os.path.exists(vecnorm_path):
        venv = VecNormalize.load(vecnorm_path, venv)
        venv.training = False
        venv.norm_reward = False
        if not quiet:
            print(f"[Info] Loaded VecNormalize stats from: {vecnorm_path}")

    # ここで env を渡してロード → n_envs が学習時と異なってもOK
    model = MaskablePPO.load(model_path, env=venv, device="auto")

    # 評価ループ
    returns: List[float] = []
    scores: List[int] = []
    per_category_scores: List[np.ndarray] = []
    bonus_hits = 0

    # CSV 準備
    csv_writer = None
    csv_file = None
    if csv_path:
        import csv
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "episode", "total_score", "norm_by_85", f"norm_by_max_{max_total}",
            "aces","deuces","threes","fours","fives","sixes",
            "choice","fourdice","fullhouse","s_straight","b_straight","yacht",
            "upper_bonus_hit"
        ])

    # VecEnv 形式で一括して回す
    obs = venv.reset()
    episode_counts = np.zeros(n_eval_envs, dtype=int)
    ep_returns = np.zeros(n_eval_envs, dtype=float)

    # 各環境の乱数シード差を確かめたい場合：seed + i で構築済み
    while int(np.sum(episode_counts)) < episodes:
        # 各環境ごとの action mask を取得
        masks_list = venv.env_method("action_masks")  # list[n_envs] of np.ndarray
        action_masks = np.stack(masks_list, axis=0)  # shape (n_envs, action_dim)

        actions, _ = model.predict(obs, deterministic=deterministic, action_masks=action_masks)
        obs, rewards, dones, infos = venv.step(actions)

        ep_returns += rewards

        for i, done in enumerate(dones):
            if done:
                # エピソード完了
                total_score = int(infos[i].get("total_score", 0))
                returns.append(float(ep_returns[i]))
                scores.append(total_score)
                spc = infos[i].get("scores_per_category", None)
                if spc is not None:
                    per_category_scores.append(spc.astype(np.int32))
                    one_roles_sum = int(infos[i].get("one_roles_sum", 0))
                    if enable_upper_bonus and one_roles_sum >= upper_bonus_threshold:
                        bonus_hits += 1

                global_ep_idx = int(np.sum(episode_counts)) + 1  # 1-based
                if not quiet:
                    print(
                        f"Episode {global_ep_idx:03d}: total_score={total_score:3d} | "
                        f"norm_by_85={ep_returns[i]:.3f} | norm_by_max({max_total})={total_score/max_total:.3f}"
                    )

                if csv_writer is not None:
                    row = [
                        global_ep_idx,
                        total_score,
                        f"{ep_returns[i]:.6f}",
                        f"{total_score/max_total:.6f}",
                    ]
                    if spc is not None:
                        row.extend(list(spc.tolist()))
                        bonus_flag = 1 if (enable_upper_bonus and one_roles_sum >= upper_bonus_threshold) else 0
                    else:
                        row.extend([""] * 12)
                        bonus_flag = 0
                    row.append(bonus_flag)
                    csv_writer.writerow(row)

                episode_counts[i] += 1
                ep_returns[i] = 0.0  # リセット

                # 目標エピ数到達したら早期終了
                if int(np.sum(episode_counts)) >= episodes:
                    break

    if csv_writer is not None:
        csv_file.close()
        if not quiet:
            print(f"[Info] Saved per-episode CSV to: {csv_path}")

    # 集計
    if len(scores) == 0:
        print("No episodes evaluated.")
        return

    ret_arr = np.asarray(returns, dtype=np.float32)
    score_arr = np.asarray(scores, dtype=np.float32)

    def stats(a: np.ndarray) -> Dict[str, float]:
        return {
            "mean": float(a.mean()),
            "std": float(a.std()),
            "median": float(np.median(a)),
            "p05": float(np.percentile(a, 5)),
            "p25": float(np.percentile(a, 25)),
            "p75": float(np.percentile(a, 75)),
            "p95": float(np.percentile(a, 95)),
            "min": float(a.min()),
            "max": float(a.max()),
        }

    ret_stats = stats(ret_arr)
    score_stats = stats(score_arr)
    norm_max_mean = float(score_arr.mean() / max_total)

    # カテゴリ別集計
    cat_names = [
        "Aces","Deuces","Threes","Fours","Fives","Sixes",
        "Choice","FourDice","FullHouse","S.Straight","B.Straight","Yacht"
    ]
    category_summary = {}
    if per_category_scores:
        cat_matrix = np.vstack(per_category_scores)
        for i, name in enumerate(cat_names):
            col = cat_matrix[:, i]
            category_summary[name] = {
                "mean": float(col.mean()),
                "std": float(col.std()),
                "nonzero_rate": float(np.mean(col > 0.0)),
                "max": int(col.max()),
            }

    # 出力
    print("\n===== Evaluation Summary =====")
    print(f"Episodes                      : {len(scores)}")
    print(f"Deterministic                 : {deterministic}")
    print(f"Parallel eval envs            : {n_eval_envs}")
    print(f"Observation flags             : obs_augment={obs_augment}, obs_add_scores={obs_add_scores}")
    print(f"Mean return (norm/85)         : {ret_stats['mean']:.4f} ± {ret_stats['std']:.4f}")
    print(f"Median return                 : {ret_stats['median']:.4f} (p05={ret_stats['p05']:.3f}, p95={ret_stats['p95']:.3f})")
    print(f"Mean total score              : {score_stats['mean']:.2f} ± {score_stats['std']:.2f}")
    print(f"Median total score            : {score_stats['median']:.2f} (p05={score_stats['p05']:.1f}, p95={score_stats['p95']:.1f})")
    print(f"Min/Max total score           : {score_stats['min']:.0f} / {score_stats['max']:.0f}")
    print(f"Mean norm_by_max({max_total})  : {norm_max_mean:.4f}")
    if enable_upper_bonus:
        bonus_rate = bonus_hits / len(scores)
        print(f"Upper bonus hit rate          : {bonus_rate:.3f} ({bonus_hits}/{len(scores)})")

    print("\nCategory score summary:")
    for name in cat_names:
        if name in category_summary:
            cs = category_summary[name]
            print(
                f"  {name:<11} mean={cs['mean']:.2f} std={cs['std']:.2f} "
                f"nonzero_rate={cs['nonzero_rate']:.3f} max={cs['max']}"
            )
    print("\nDone.")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MaskablePPO on YachtEnv (supports different n_envs from training).")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--stochastic", action="store_true")

    # ルール
    parser.add_argument("--short-straight-points", type=int, default=15)
    parser.add_argument("--big-straight-points", type=int, default=30)
    parser.add_argument("--yacht-points", type=int, default=50)
    parser.add_argument("--disable-upper-bonus", action="store_true")
    parser.add_argument("--upper-bonus-threshold", type=int, default=63)
    parser.add_argument("--upper-bonus-points", type=int, default=35)

    # 観測
    parser.add_argument("--obs-mode", type=str, choices=["auto", "manual"], default="auto")
    parser.add_argument("--obs-augment", action="store_true", help="manual の時のみ有効")
    parser.add_argument("--obs-add-scores", action="store_true", help="manual の時のみ有効")

    # マスク設定
    parser.add_argument("--mask-keep-all", action="store_true")

    # VecNormalize
    parser.add_argument("--vecnorm-path", type=str, default="")

    # 出力
    parser.add_argument("--csv", type=str, default="")
    parser.add_argument("--quiet", action="store_true")

    # 並列評価
    parser.add_argument("--n-eval-envs", type=int, default=1, help="評価並列環境数（学習時と異なってOK）")
    parser.add_argument("--parallel", action="store_true", help="n-eval-envs>1 のとき SubprocVecEnv を使用")

    return parser.parse_args()


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
        obs_mode=args.obs_mode,
        obs_augment_flag=args.obs_augment,
        obs_add_scores_flag=args.obs_add_scores,
        disable_keep_all=args.mask_keep_all,
        vecnorm_path=args.vecnorm_path,
        csv_path=args.csv,
        quiet=args.quiet,
        n_eval_envs=args.n_eval_envs,
        parallel=args.parallel,
    )


if __name__ == "__main__":
    main()