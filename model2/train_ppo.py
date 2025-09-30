import os
import argparse
from datetime import datetime

import numpy as np
import gymnasium as gym

from yachtEnv import YachtEnv, ScoringRules

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.utils import get_action_masks

from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed


def mask_fn(env: YachtEnv):
    # 環境が提供するアクションマスク（True=選択可）を返す
    return env.get_action_mask()


def make_env(seed: int = 0):
    """
    学習・評価で使う環境ファクトリ。
    - 無効行動は ActionMasker により事前に除外されます（MaskablePPO 前提）。
    - yachtenv 側で報酬は (点数+ボーナス)/85 に正規化済み。
    """
    def _init():
        rules = ScoringRules()  # ルールは必要に応じて調整可
        base_env = YachtEnv(rules=rules, seed=seed)
        masked_env = ActionMasker(base_env, mask_fn)
        return masked_env
    return _init


def parse_args():
    parser = argparse.ArgumentParser(description="Train MaskablePPO on YachtEnv with TensorBoard logging.")
    parser.add_argument("--total-timesteps", type=int, default=2_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", type=str, default="./logs/yacht_ppo")
    parser.add_argument("--save-path", type=str, default="./models/yacht_maskable_ppo.zip")
    parser.add_argument("--policy", type=str, default="MlpPolicy")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--clip-range", type=float, default=0.2)
    return parser.parse_args()


def main():
    args = parse_args()

    # ディレクトリ作成
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    # 乱数シード
    set_random_seed(args.seed)

    # ベクトル化環境（Monitorでエピソード統計をTensorBoardに記録）
    env_fns = [make_env(args.seed + i) for i in range(args.n_envs)]
    venv = DummyVecEnv(env_fns)
    venv = VecMonitor(venv, filename=None)

    # 実験名（TensorBoard上のラン名）
    run_name = f"MaskablePPO_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    # モデル作成（TensorBoardログを有効化）
    model = MaskablePPO(
        args.policy,
        venv,
        verbose=1,
        seed=args.seed,
        device=args.device,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        clip_range=args.clip_range,
        tensorboard_log=args.log_dir,
        policy_kwargs=dict(net_arch=[256, 256]),
    )

    # 学習
    model.learn(total_timesteps=args.total_timesteps, tb_log_name=run_name)

    # 保存
    model.save(args.save_path)
    print(f"Model saved to: {args.save_path}")
    print(f"TensorBoard logs in: {args.log_dir} (run={run_name})")


if __name__ == "__main__":
    main()