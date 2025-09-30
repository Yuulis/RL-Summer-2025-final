import os
import glob
import argparse
from datetime import datetime

from yachtenv import YachtEnv, ScoringRules

from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed, get_linear_fn
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement, CheckpointCallback, CallbackList


def make_env(
    seed: int,
    short_straight_points: int,
    big_straight_points: int,
    yacht_points: int,
    enable_upper_bonus: bool,
    upper_bonus_threshold: int,
    upper_bonus_points: int,
    obs_augment: bool,
    disable_keep_all: bool,
):
    def _init():
        rules = ScoringRules(
            short_straight_points=short_straight_points,
            big_straight_points=big_straight_points,
            yacht_points=yacht_points,
            enable_one_roles_bonus=enable_upper_bonus,
            one_roles_bonus_threshold=upper_bonus_threshold,
            one_roles_bonus_points=upper_bonus_points,
        )
        env = YachtEnv(
            rules=rules,
            seed=seed,
            obs_augment=obs_augment,
            disable_keep_all=disable_keep_all,
        )
        return env
    return _init


def parse_args():
    p = argparse.ArgumentParser(description="Train MaskablePPO on YachtEnv with periodic checkpoints and resume.")
    # ルール
    p.add_argument("--short-straight-points", type=int, default=15)
    p.add_argument("--big-straight-points", type=int, default=30)
    p.add_argument("--yacht-points", type=int, default=50)
    p.add_argument("--disable-upper-bonus", action="store_true")
    p.add_argument("--upper-bonus-threshold", type=int, default=63)
    p.add_argument("--upper-bonus-points", type=int, default=35)

    # 環境内オプション
    p.add_argument("--obs-augment", action="store_true", help="観測を環境内で拡張（+9次元）")
    p.add_argument("--mask-keep-all", action="store_true", help="リロール時の keep-all(31) を常に無効にする")

    # 学習設定
    p.add_argument("--total-timesteps", type=int, default=20_000_000)
    p.add_argument("--n-envs", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--policy", type=str, default="MlpPolicy")
    p.add_argument("--vec", type=str, choices=["dummy", "subproc"], default="subproc")

    # PPOハイパラ
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--lr-final-ratio", type=float, default=0.2)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--n-steps", type=int, default=512)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--ent-coef", type=float, default=0.02)
    p.add_argument("--clip-range", type=float, default=0.1)

    # ログ・保存・評価
    p.add_argument("--log-dir", type=str, default="./logs/yacht_ppo")
    p.add_argument("--save-path", type=str, default="./models/yacht_maskable_ppo_v5.zip")
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--eval-freq", type=int, default=500_000)  # 合計ステップ間隔
    p.add_argument("--no-eval", action="store_true")
    p.add_argument("--early-stop-no-improve", type=int, default=0)

    # チェックポイント/再開
    p.add_argument("--checkpoint-freq", type=int, default=200_000, help="この合計ステップ間隔ごとに ckpt を保存")
    p.add_argument("--checkpoint-dir", type=str, default="", help="未指定なら {log_dir}/ckpt")
    p.add_argument("--resume-from", type=str, default="", help="再開元の .zip（モデル）へのパス")
    p.add_argument("--resume-latest", action="store_true", help="checkpoint-dir や best から最新版を自動探索して再開")
    p.add_argument("--reset-num-timesteps", action="store_true", help="再開時もタイムステップカウンタを0からにする（通常は指定しない）")

    return p.parse_args()


def build_vec_env(args, for_eval: bool = False):
    env_fns = [
        make_env(
            seed=args.seed + (10_000 if for_eval else 0) + i,
            short_straight_points=args.short_straight_points,
            big_straight_points=args.big_straight_points,
            yacht_points=args.yacht_points,
            enable_upper_bonus=not args.disable_upper_bonus,
            upper_bonus_threshold=args.upper_bonus_threshold,
            upper_bonus_points=args.upper_bonus_points,
            obs_augment=args.obs_augment,
            disable_keep_all=args.mask_keep_all,
        )
        for i in range(1 if for_eval else args.n_envs)
    ]
    if args.vec == "subproc" and (args.n_envs > 1 or for_eval):
        venv = SubprocVecEnv(env_fns, start_method="forkserver")
    else:
        venv = DummyVecEnv(env_fns)
    venv = VecMonitor(venv, filename=None)
    return venv


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


def _find_latest_checkpoint(ckpt_dir: str, best_dir: str) -> str:
    # ckpt/*.zip の中から最終更新が新しいもの。なければ best/best_model.zip を返す。
    candidates = []
    if os.path.isdir(ckpt_dir):
        candidates += glob.glob(os.path.join(ckpt_dir, "*.zip"))
    if os.path.isdir(best_dir):
        best = os.path.join(best_dir, "best_model.zip")
        if os.path.exists(best):
            candidates.append(best)
    if not candidates:
        return ""
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def main():
    args = parse_args()
    _ensure_dir(args.log_dir)
    _ensure_dir(os.path.dirname(args.save_path))
    set_random_seed(args.seed)

    ckpt_dir = args.checkpoint_dir or os.path.join(args.log_dir, "ckpt")
    best_dir = os.path.join(args.log_dir, "best")
    _ensure_dir(ckpt_dir)
    _ensure_dir(best_dir)

    # 学習 VecEnv
    venv = build_vec_env(args, for_eval=False)

    # 学習率スケジュール
    lr_schedule = get_linear_fn(args.learning_rate, args.learning_rate * args.lr_final_ratio, 1.0)
    run_name = f"MaskablePPO_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    # 再開元の決定
    resume_path = args.resume_from
    if args.resume_latest and not resume_path:
        resume_path = _find_latest_checkpoint(ckpt_dir, best_dir)
        if resume_path:
            print(f"[Resume] Auto-selected latest checkpoint: {resume_path}")
        else:
            print("[Resume] No checkpoint/best model found. Starting fresh.")

    # モデルの用意（新規 or 再開）
    if resume_path and os.path.exists(resume_path):
        print(f"[Resume] Loading model from: {resume_path}")
        model = MaskablePPO.load(resume_path, device=args.device)
        model.set_env(venv)
        reset_flag = args.reset_num_timesteps is True
    else:
        print("[Start] Training a new model")
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
            learning_rate=lr_schedule,
            clip_range=args.clip_range,
            tensorboard_log=args.log_dir,
            policy_kwargs=dict(net_arch=[256, 256, 128], ortho_init=True),
        )
        reset_flag = True  # 新規学習は 0 から

    # コールバック（eval + ckpt）
    callback_list = []

    if not args.no_eval:
        eval_env = build_vec_env(args, for_eval=True)
        callback_after_eval = None
        if args.early_stop_no_improve and args.early_stop_no_improve > 0:
            callback_after_eval = StopTrainingOnNoModelImprovement(
                max_no_improvement_evals=args.early_stop_no_improve,
                min_evals=args.early_stop_no_improve,
                verbose=1,
            )
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=best_dir,
            log_path=os.path.join(args.log_dir, "eval"),
            eval_freq=args.eval_freq,  # 合計ステップベース
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
            callback_after_eval=callback_after_eval,
            warn=True,
        )
        callback_list.append(eval_cb)

    if args.checkpoint_freq and args.checkpoint_freq > 0:
        ckpt_cb = CheckpointCallback(
            save_freq=args.checkpoint_freq,
            save_path=ckpt_dir,
            name_prefix="ppo_maskable",
            save_replay_buffer=False,  # PPOでは不要
        )
        callback_list.append(ckpt_cb)

    callbacks = None
    if len(callback_list) == 1:
        callbacks = callback_list[0]
    elif len(callback_list) > 1:
        callbacks = CallbackList(callback_list)

    # 学習（中断対応）
    interrupted_save = os.path.join(args.log_dir, "interrupt", f"ppo_maskable_interrupt_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.zip")
    _ensure_dir(os.path.dirname(interrupted_save))

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            tb_log_name=run_name,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=reset_flag,
        )
    except KeyboardInterrupt:
        # 割り込み保存
        model.save(interrupted_save)
        print(f"\n[Interrupt] Saved interrupted model to: {interrupted_save}")
        raise
    finally:
        # 参考情報の出力のみ。正常終了時の最終保存はこの後で実施。
        print(f"[Info] Checkpoints: {ckpt_dir} (e.g., ppo_maskable_step-*.zip)")
        print(f"[Info] Best model: {os.path.join(best_dir, 'best_model.zip')}")

    # 正常終了時の最終保存
    model.save(args.save_path)
    print(f"[Done] Final model saved to: {args.save_path}")
    print(f"[TB] TensorBoard logs in: {args.log_dir} (run={run_name})")


if __name__ == "__main__":
    main()