import os
import argparse
from datetime import datetime

import numpy as np
import gymnasium as gym

from yachtenv import YachtEnv, ScoringRules

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed, get_linear_fn
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement


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
):
    """
    学習・評価で使う環境ファクトリ。
    - 無効行動は ActionMasker により事前に除外されます（MaskablePPO 前提）。
    - 環境側で報酬は (点数+ボーナス)/85 に正規化済み。
    """
    def _init():
        rules = ScoringRules(
            short_straight_points=short_straight_points,
            big_straight_points=big_straight_points,
            yacht_points=yacht_points,
            enable_one_roles_bonus=enable_upper_bonus,
            one_roles_bonus_threshold=upper_bonus_threshold,
            one_roles_bonus_points=upper_bonus_points,
        )
        base_env = YachtEnv(rules=rules, seed=seed)
        masked_env = ActionMasker(base_env, mask_fn)
        return masked_env
    return _init


def parse_args():
    parser = argparse.ArgumentParser(description="Train MaskablePPO on YachtEnv with TensorBoard logging and eval callback.")
    # 環境・ルール
    parser.add_argument("--short-straight-points", type=int, default=15, help="S.ストレートの点数（一般的には15）")
    parser.add_argument("--big-straight-points", type=int, default=30, help="B.ストレートの点数（一般的には30）")
    parser.add_argument("--yacht-points", type=int, default=50, help="ヨットの点数（一般的には50）")
    parser.add_argument("--disable-upper-bonus", action="store_true", help="上段ボーナスを無効化する")
    parser.add_argument("--upper-bonus-threshold", type=int, default=63, help="上段ボーナスのしきい値（一般的には63）")
    parser.add_argument("--upper-bonus-points", type=int, default=35, help="上段ボーナス点（一般的には35）")

    # 学習設定
    parser.add_argument("--total-timesteps", type=int, default=10_000_000)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--policy", type=str, default="MlpPolicy")

    # PPOハイパラ
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--lr-final-ratio", type=float, default=0.3, help="学習率の線形スケジュール最終比（初期lr * ratio）")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--clip-range", type=float, default=0.1)

    # ログ・保存
    parser.add_argument("--log-dir", type=str, default="./logs/yacht_ppo")
    parser.add_argument("--save-path", type=str, default="./models/yacht_maskable_ppo_last.zip")

    # 評価
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--eval-freq", type=int, default=50_000, help="評価間隔（総環境ステップの目安）")
    parser.add_argument("--early-stop-no-improve", type=int, default=0,
                        help="早期停止: 改善なし評価回数。0なら無効。例: 10")

    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    set_random_seed(args.seed)

    # 並列環境の構築
    env_fns = [
        make_env(
            seed=args.seed + i,
            short_straight_points=args.short_straight_points,
            big_straight_points=args.big_straight_points,
            yacht_points=args.yacht_points,
            enable_upper_bonus=not args.disable_upper_bonus,
            upper_bonus_threshold=args.upper_bonus_threshold,
            upper_bonus_points=args.upper_bonus_points,
        ) for i in range(args.n_envs)
    ]
    venv = DummyVecEnv(env_fns)
    venv = VecMonitor(venv, filename=None)

    # 学習率スケジュール（線形）
    lr_schedule = get_linear_fn(args.learning_rate, args.learning_rate * args.lr_final_ratio, 1.0)

    run_name = f"MaskablePPO_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    model = MaskablePPO(
        args.policy,
        venv,
        verbose=1,
        seed=args.seed,
        device=args.device,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coefficient if hasattr(args, "ent_coefficient") else args.ent_coef,  # 互換性のため
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=lr_schedule,
        clip_range=args.clip_range,
        tensorboard_log=args.log_dir,
        policy_kwargs=dict(net_arch=[256, 256, 128], ortho_init=True),
    )

    # 評価環境
    eval_env_fn = make_env(
        seed=args.seed + 10_000,
        short_straight_points=args.short_straight_points,
        big_straight_points=args.big_straight_points,
        yacht_points=args.yacht_points,
        enable_upper_bonus=not args.disable_upper_bonus,
        upper_bonus_threshold=args.upper_bonus_threshold,
        upper_bonus_points=args.upper_bonus_points,
    )
    eval_env = DummyVecEnv([eval_env_fn])
    eval_env = VecMonitor(eval_env)

    # 早期停止（任意）
    callback_after_eval = None
    if args.early_stop_no_improve and args.early_stop_no_improve > 0:
        callback_after_eval = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=args.early_stop_no_improve,
            min_evals=args.early_stop_no_improve,  # 最低これだけ評価してから判定
            verbose=1,
        )

    # 評価コールバック（最良モデル保存＋TensorBoardへ評価結果を記録）
    # eval_freq は「全体の環境ステップ」を目安で指定。VecEnvの場合は内部でカウントされるため、
    # 近似として eval_freq // n_envs を使うと、"総ステップ"相当の間隔になります。
    per_env_eval_freq = max(1, args.eval_freq // max(1, args.n_envs))
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(args.log_dir, "best"),
        log_path=os.path.join(args.log_dir, "eval"),
        eval_freq=per_env_eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        callback_after_eval=callback_after_eval,
        warn=True,
    )

    # 学習
    model.learn(total_timesteps=args.total_timesteps, tb_log_name=run_name, callback=eval_cb)

    # 保存
    model.save(args.save_path)
    print(f"Model saved to: {args.save_path}")
    print(f"TensorBoard logs in: {args.log_dir} (run={run_name})")
    print(f"Best model (if saved) under: {os.path.join(args.log_dir, 'best')}")


if __name__ == "__main__":
    main()