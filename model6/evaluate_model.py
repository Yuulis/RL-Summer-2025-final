import os
import argparse
import numpy as np
from typing import List, Dict, Tuple

from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecMonitor

from yachtenv import YachtEnv, ScoringRules

# 観測サイズマッピング (base=19)
OBS_COMBO_MAP = {
    19:  (False, False, False),
    28:  (True,  False, False),   # augment
    31:  (False, True,  False),   # scores
    30:  (False, False, True),    # progress
    40:  (True,  True,  False),   # augment+scores
    39:  (True,  False, True),    # augment+progress
    42:  (False, True,  True),    # scores+progress
    51:  (True,  True,  True),    # augment+scores+progress
}

def theoretical_max_total(short_straight_points: int) -> int:
    return 310 + int(short_straight_points)


def guess_flags(dim: int) -> Tuple[bool, bool, bool]:
    if dim not in OBS_COMBO_MAP:
        raise ValueError(
            f"未知の観測次元 {dim}. 想定: {sorted(OBS_COMBO_MAP.keys())}. "
            "学習時と同じフラグを --mode manual で指定してください。"
        )
    return OBS_COMBO_MAP[dim]


def make_env(seed: int,
             short_straight_points: int,
             big_straight_points: int,
             yacht_points: int,
             enable_upper_bonus: bool,
             upper_bonus_threshold: int,
             upper_bonus_points: int,
             obs_augment: bool,
             obs_add_scores: bool,
             obs_progress: bool,
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
            obs_progress=obs_progress,
            disable_keep_all=disable_keep_all,
            shaping_upper_eps=0.0,
            shaping_lower_eps=0.0,
            shaping_bonus_eps=0.0,
        )
    return _init


def build_eval_env(n_envs: int,
                   base_seed: int,
                   parallel: bool,
                   short_straight_points: int,
                   big_straight_points: int,
                   yacht_points: int,
                   enable_upper_bonus: bool,
                   upper_bonus_threshold: int,
                   upper_bonus_points: int,
                   obs_augment: bool,
                   obs_add_scores: bool,
                   obs_progress: bool,
                   disable_keep_all: bool):
    fns = [
        make_env(
            seed=base_seed + i,
            short_straight_points=short_straight_points,
            big_straight_points=big_straight_points,
            yacht_points=yacht_points,
            enable_upper_bonus=enable_upper_bonus,
            upper_bonus_threshold=upper_bonus_threshold,
            upper_bonus_points=upper_bonus_points,
            obs_augment=obs_augment,
            obs_add_scores=obs_add_scores,
            obs_progress=obs_progress,
            disable_keep_all=disable_keep_all,
        )
        for i in range(n_envs)
    ]
    if parallel and n_envs > 1:
        venv = SubprocVecEnv(fns, start_method="spawn")
    else:
        venv = DummyVecEnv(fns)
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
    mode: str,  # auto | manual
    obs_augment_flag: bool,
    obs_add_scores_flag: bool,
    obs_progress_flag: bool,
    disable_keep_all: bool,
    vecnorm_path: str,
    csv_path: str,
    quiet: bool,
    n_eval_envs: int,
    parallel: bool,
):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model not found: {model_path}")

    # 一旦モデルのみロードして観測サイズ判別
    tmp = MaskablePPO.load(model_path, device="auto")
    model_obs_dim = int(np.prod(tmp.observation_space.shape))

    if mode == "auto":
        obs_augment, obs_add_scores, obs_progress = guess_flags(model_obs_dim)
    else:
        obs_augment, obs_add_scores, obs_progress = obs_augment_flag, obs_add_scores_flag, obs_progress_flag
        # 簡易一致チェック
        try:
            exp_dim = None
            for dim, combo in OBS_COMBO_MAP.items():
                if combo == (obs_augment, obs_add_scores, obs_progress):
                    exp_dim = dim
                    break
            if exp_dim is not None and exp_dim != model_obs_dim:
                print(f"[Warning] モデル期待次元={model_obs_dim} と手動指定組合せの次元={exp_dim} が不一致です。")
        except Exception:
            pass

    max_total = theoretical_max_total(short_straight_points)

    venv = build_eval_env(
        n_envs=n_eval_envs,
        base_seed=seed,
        parallel=parallel,
        short_straight_points=short_straight_points,
        big_straight_points=big_straight_points,
        yacht_points=yacht_points,
        enable_upper_bonus=enable_upper_bonus,
        upper_bonus_threshold=upper_bonus_threshold,
        upper_bonus_points=upper_bonus_points,
        obs_augment=obs_augment,
        obs_add_scores=obs_add_scores,
        obs_progress=obs_progress,
        disable_keep_all=disable_keep_all,
    )

    if vecnorm_path and os.path.exists(vecnorm_path):
        venv = VecNormalize.load(vecnorm_path, venv)
        venv.training = False
        venv.norm_reward = False
        if not quiet:
            print(f"[Info] Loaded VecNormalize stats from: {vecnorm_path}")

    # 環境付きで再ロード（n_envs が異なっても OK）
    model = MaskablePPO.load(model_path, env=venv, device="auto")

    returns: List[float] = []
    scores: List[int] = []
    per_category_scores: List[np.ndarray] = []
    bonus_hits = 0

    # CSV
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

    obs = venv.reset()
    ep_returns = np.zeros(n_eval_envs, dtype=float)
    finished = 0

    while finished < episodes:
        masks_list = venv.env_method("action_masks")
        action_masks = np.stack(masks_list, axis=0)
        actions, _ = model.predict(obs, deterministic=deterministic, action_masks=action_masks)
        obs, rewards, dones, infos = venv.step(actions)
        ep_returns += rewards

        for i, d in enumerate(dones):
            if d and finished < episodes:
                info = infos[i]
                total_score = int(info.get("total_score", 0))
                scores.append(total_score)
                returns.append(float(ep_returns[i]))
                spc = info.get("scores_per_category", None)
                if spc is not None:
                    per_category_scores.append(spc.astype(np.int32))
                    one_sum = int(info.get("one_roles_sum", 0))
                    if enable_upper_bonus and one_sum >= upper_bonus_threshold:
                        bonus_hits += 1

                finished += 1
                if not quiet:
                    print(
                        f"Episode {finished:03d}: total_score={total_score:3d} | "
                        f"norm_by_85={ep_returns[i]:.3f} | norm_by_max({max_total})={total_score/max_total:.3f}"
                    )

                if csv_writer is not None:
                    if spc is not None:
                        one_sum = int(info.get("one_roles_sum", 0))
                        bonus_flag = 1 if (enable_upper_bonus and one_sum >= upper_bonus_threshold) else 0
                    else:
                        bonus_flag = 0
                    row = [
                        finished,
                        total_score,
                        f"{ep_returns[i]:.6f}",
                        f"{total_score/max_total:.6f}",
                    ]
                    if spc is not None:
                        row.extend(list(spc.tolist()))
                    else:
                        row.extend([""] * 12)
                    row.append(bonus_flag)
                    csv_writer.writerow(row)

                ep_returns[i] = 0.0  # reset

                if finished >= episodes:
                    break

    if csv_writer:
        csv_file.close()
        if not quiet:
            print(f"[Info] Saved CSV to: {csv_path}")

    if not scores:
        print("No episodes evaluated.")
        return

    score_arr = np.asarray(scores, dtype=np.float32)
    return_arr = np.asarray(returns, dtype=np.float32)

    def stats(a: np.ndarray) -> Dict[str, float]:
        return dict(
            mean=float(a.mean()),
            std=float(a.std()),
            median=float(np.median(a)),
            p05=float(np.percentile(a, 5)),
            p95=float(np.percentile(a, 95)),
            min=float(a.min()),
            max=float(a.max()),
        )

    s_stats = stats(score_arr)
    r_stats = stats(return_arr)
    norm_max_mean = float(score_arr.mean() / max_total)

    cat_names = [
        "Aces","Deuces","Threes","Fours","Fives","Sixes",
        "Choice","FourDice","FullHouse","S.Straight","B.Straight","Yacht"
    ]
    category_summary = {}
    if per_category_scores:
        mat = np.vstack(per_category_scores)
        for i, name in enumerate(cat_names):
            col = mat[:, i]
            category_summary[name] = dict(
                mean=float(col.mean()),
                std=float(col.std()),
                nonzero_rate=float(np.mean(col > 0)),
                max=int(col.max()),
            )

    print("\n===== Evaluation Summary =====")
    print(f"Episodes                  : {episodes}")
    print(f"Deterministic             : {deterministic}")
    print(f"Eval envs (parallel)      : {n_eval_envs}")
    print(f"Observation flags         : augment={obs_augment}, scores={obs_add_scores}, progress={obs_progress}")
    print(f"Mean return (norm/85)     : {r_stats['mean']:.4f} ± {r_stats['std']:.4f}")
    print(f"Median return             : {r_stats['median']:.4f} (p05={r_stats['p05']:.3f}, p95={r_stats['p95']:.3f})")
    print(f"Mean total score          : {s_stats['mean']:.2f} ± {s_stats['std']:.2f}")
    print(f"Median total score        : {s_stats['median']:.2f} (p05={s_stats['p05']:.1f}, p95={s_stats['p95']:.1f})")
    print(f"Min/Max total score       : {s_stats['min']:.0f} / {s_stats['max']:.0f}")
    print(f"Mean norm_by_max({max_total}): {norm_max_mean:.4f}")
    if enable_upper_bonus:
        print(f"Upper bonus hit rate      : {bonus_hits/episodes:.3f} ({bonus_hits}/{episodes})")

    print("\nCategory score summary:")
    for n in cat_names:
        if n in category_summary:
            cs = category_summary[n]
            print(
                f"  {n:<11} mean={cs['mean']:.2f} std={cs['std']:.2f} "
                f"nonzero={cs['nonzero_rate']:.3f} max={cs['max']}"
            )
    print("\nDone.")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate MaskablePPO YachtEnv (supports new progress features)")
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stochastic", action="store_true")

    # ルール
    p.add_argument("--short-straight-points", type=int, default=15)
    p.add_argument("--big-straight-points", type=int, default=30)
    p.add_argument("--yacht-points", type=int, default=50)
    p.add_argument("--disable-upper-bonus", action="store_true")
    p.add_argument("--upper-bonus-threshold", type=int, default=63)
    p.add_argument("--upper-bonus-points", type=int, default=35)

    # 観測モード
    p.add_argument("--mode", type=str, choices=["auto", "manual"], default="auto")
    p.add_argument("--obs-augment", action="store_true")
    p.add_argument("--obs-add-scores", action="store_true")
    p.add_argument("--obs-progress", action="store_true")

    p.add_argument("--mask-keep-all", action="store_true")
    p.add_argument("--vecnorm-path", type=str, default="")
    p.add_argument("--csv", type=str, default="")
    p.add_argument("--quiet", action="store_true")

    p.add_argument("--n-eval-envs", type=int, default=1)
    p.add_argument("--parallel", action="store_true")

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
        mode=args.mode,
        obs_augment_flag=args.obs_augment,
        obs_add_scores_flag=args.obs_add_scores,
        obs_progress_flag=args.obs_progress,
        disable_keep_all=args.mask_keep_all,
        vecnorm_path=args.vecnorm_path,
        csv_path=args.csv,
        quiet=args.quiet,
        n_eval_envs=args.n_eval_envs,
        parallel=args.parallel,
    )


if __name__ == "__main__":
    main()