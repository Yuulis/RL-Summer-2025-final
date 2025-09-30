import os
# 可能なら実行の最初に入れてください（TF/JAXの冗長ログ抑制）
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import glob
import time
import argparse
from datetime import datetime

from yachtenv import YachtEnv, ScoringRules

from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.utils import set_random_seed, get_linear_fn
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
    CheckpointCallback,
    CallbackList,
)


def make_env(
    seed: int,
    short_straight_points: int,
    big_straight_points: int,
    yacht_points: int,
    enable_upper_bonus: bool,
    upper_bonus_threshold: int,
    upper_bonus_points: int,
    obs_augment: bool,
    obs_add_scores: bool,
    disable_keep_all: bool,
    shaping_upper_eps: float,
    shaping_lower_eps: float,
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
            obs_add_scores=obs_add_scores,
            disable_keep_all=disable_keep_all,
            shaping_upper_eps=shaping_upper_eps,
            shaping_lower_eps=shaping_lower_eps,
        )
        return env
    return _init


class SaveLatestCallback(BaseCallback):
    """
    一定ステップ/一定秒数ごとに「最新版」を単一ファイルへ上書き保存するコールバック。
    - latest_path: 保存先ファイル（例: ./models/yacht_maskable_ppo_latest.zip）
    - save_every_steps: この学習ステップ間隔ごとに保存（モデルの num_timesteps 基準、VecEnvでは合計）
    - save_every_seconds: この秒数経過ごとに保存（実時間）
    - vecnorm_path: VecNormalize 利用時に統計も保存するパス（例: ./logs/yacht_ppo/vecnorm_latest.pkl）
    """
    def __init__(
        self,
        latest_path: str,
        save_every_steps: int = 100_000,
        save_every_seconds: float = 300.0,
        vecnorm_path: str = "",
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.latest_path = latest_path
        self.save_every_steps = int(save_every_steps) if save_every_steps and save_every_steps > 0 else 0
        self.save_every_seconds = float(save_every_seconds) if save_every_seconds and save_every_seconds > 0 else 0.0
        self.vecnorm_path = vecnorm_path
        self._last_save_step = 0
        self._last_save_time = 0.0

    def _on_training_start(self) -> None:
        os.makedirs(os.path.dirname(self.latest_path) or ".", exist_ok=True)
        if self.vecnorm_path:
            os.makedirs(os.path.dirname(self.vecnorm_path) or ".", exist_ok=True)
        self._last_save_step = int(self.model.num_timesteps)
        self._last_save_time = time.time()
        return None

    def _save_now(self):
        # モデル本体
        self.model.save(self.latest_path)
        # VecNormalize の統計（使っている場合のみ）
        try:
            env = self.model.get_env()
            if isinstance(env, VecNormalize):
                env.save(self.vecnorm_path or (os.path.join(os.path.dirname(self.latest_path) or ".", "vecnorm_latest.pkl")))
        except Exception:
            # 正規化なし、または取得失敗時は無視
            pass
        if self.verbose:
            print(f"[SaveLatest] Saved latest model to: {self.latest_path}")

    def _on_step(self) -> bool:
        now = time.time()
        step = int(self.model.num_timesteps)

        by_step = self.save_every_steps > 0 and (step - self._last_save_step) >= self.save_every_steps
        by_time = self.save_every_seconds > 0 and (now - self._last_save_time) >= self.save_every_seconds

        if by_step or by_time:
            self._save_now()
            self._last_save_step = step
            self._last_save_time = now
        return True


def parse_args():
    p = argparse.ArgumentParser(description="Train MaskablePPO on YachtEnv (periodic checkpoints + latest save + resume).")
    # ルール
    p.add_argument("--short-straight-points", type=int, default=15)
    p.add_argument("--big-straight-points", type=int, default=30)
    p.add_argument("--yacht-points", type=int, default=50)
    p.add_argument("--disable-upper-bonus", action="store_true")
    p.add_argument("--upper-bonus-threshold", type=int, default=63)
    p.add_argument("--upper-bonus-points", type=int, default=35)

    # 環境
    p.add_argument("--obs-augment", action="store_true")
    p.add_argument("--obs-add-scores", action="store_true")
    p.add_argument("--mask-keep-all", action="store_true")
    p.add_argument("--shape-upper-eps", type=float, default=0.02)
    p.add_argument("--shape-lower-eps", type=float, default=0.01)

    # 学習設定
    p.add_argument("--total-timesteps", type=int, default=30_000_000)
    p.add_argument("--n-envs", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--policy", type=str, default="MlpPolicy")
    p.add_argument("--vec", type=str, choices=["dummy", "subproc"], default="subproc")
    p.add_argument("--mp-start", type=str, choices=["spawn", "forkserver", "fork"], default="spawn",
                   help="SubprocVecEnv の start_method（混在環境では spawn 推奨）")

    # PPOハイパラ（ent/clip は float 固定：sb3-contribの仕様に合わせる）
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--lr-final-ratio", type=float, default=0.2)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--n-steps", type=int, default=1024)
    p.add_argument("--gamma", type=float, default=0.997)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--ent-coef", type=float, default=0.02)
    p.add_argument("--clip-range", type=float, default=0.2)

    # ログ・保存・評価
    p.add_argument("--log-dir", type=str, default="./logs/yacht_ppo")
    p.add_argument("--save-path", type=str, default="./models/yacht_maskable_ppo_last.zip")
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--eval-freq", type=int, default=600_000)
    p.add_argument("--no-eval", action="store_true")
    p.add_argument("--early-stop-no-improve", type=int, default=0)

    # チェックポイント/再開（スナップショット）
    p.add_argument("--checkpoint-freq", type=int, default=200_000, help="合計タイムステップ間隔ごとに ckpt を保存")
    p.add_argument("--checkpoint-dir", type=str, default="", help="未指定なら {log_dir}/ckpt")

    # 最新版の上書き保存
    p.add_argument("--latest-path", type=str, default="./models/yacht_maskable_ppo_latest.zip")
    p.add_argument("--save-latest-every-steps", type=int, default=100_000, help="このステップ間隔ごとに latest を保存（0で無効）")
    p.add_argument("--save-latest-every-seconds", type=float, default=300.0, help="この秒数ごとに latest を保存（0で無効）")

    # 再開
    p.add_argument("--resume-from", type=str, default="", help="再開元 .zip（モデル）へのパス")
    p.add_argument("--resume-latest", action="store_true", help="checkpoint-dir や best から最新版を自動探索して再開")
    p.add_argument("--reset-num-timesteps", action="store_true", help="再開時にタイムステップを0からにする場合のみ指定")

    # 観測正規化
    p.add_argument("--norm-obs", action="store_true")
    p.add_argument("--norm-rew", action="store_true")
    p.add_argument("--vecnorm-path", type=str, default="", help="VecNormalize 統計の保存/読込パス（未指定は {log_dir}/vecnorm.pkl）")

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
            obs_add_scores=args.obs_add_scores,
            disable_keep_all=args.mask_keep_all,
            shaping_upper_eps=args.shape_upper_eps,
            shaping_lower_eps=args.shape_lower_eps,
        )
        for i in range(1 if for_eval else args.n_envs)
    ]
    if args.vec == "subproc" and (args.n_envs > 1 or for_eval):
        venv = SubprocVecEnv(env_fns, start_method=args.mp_start)
    else:
        venv = DummyVecEnv(env_fns)
    venv = VecMonitor(venv, filename=None)
    return venv


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


def _find_latest_checkpoint(ckpt_dir: str, best_dir: str) -> str:
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
    _ensure_dir(os.path.dirname(args.save_path) or ".")
    _ensure_dir(os.path.dirname(args.latest_path) or ".")
    set_random_seed(args.seed)

    ckpt_dir = args.checkpoint_dir or os.path.join(args.log_dir, "ckpt")
    best_dir = os.path.join(args.log_dir, "best")
    _ensure_dir(ckpt_dir)
    _ensure_dir(best_dir)
    vecnorm_path = args.vecnorm_path or os.path.join(args.log_dir, "vecnorm.pkl")

    # VecEnv 構築
    venv = build_vec_env(args, for_eval=False)

    # VecNormalize（必要なら）
    if args.norm_obs or args.norm_rew:
        venv = VecNormalize(venv, norm_obs=args.norm_obs, norm_reward=args.norm_rew, clip_obs=5.0, gamma=args.gamma)

    # 学習率のみスケジュール（ent/clipはfloat固定）
    lr_schedule = get_linear_fn(args.learning_rate, args.learning_rate * args.lr_final_ratio, 1.0)
    run_name = f"MaskablePPO_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    # 再開判定
    resume_path = args.resume_from
    if args.resume_latest and not resume_path:
        resume_path = _find_latest_checkpoint(ckpt_dir, best_dir)
        if resume_path:
            print(f"[Resume] Auto-selected latest checkpoint: {resume_path}")
        else:
            print("[Resume] No checkpoint/best model found. Starting fresh.")

    # モデル準備
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
            ent_coef=float(args.ent_coef),          # 関数ではなく float
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            learning_rate=lr_schedule,              # これはスケジュール可
            clip_range=float(args.clip_range),      # 関数ではなく float
            tensorboard_log=args.log_dir,
            policy_kwargs=dict(net_arch=[512, 512, 256], ortho_init=True),
        )
        reset_flag = True

    # コールバック（評価 + スナップショット + 最新版）
    callback_list = []

    if not args.no_eval:
        eval_env = build_vec_env(args, for_eval=True)
        if args.norm_obs or args.norm_rew:
            eval_env = VecNormalize(eval_env, norm_obs=args.norm_obs, norm_reward=False, clip_obs=5.0, gamma=args.gamma)
            eval_env.training = False

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
            eval_freq=args.eval_freq,  # 合計タイムステップ基準
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
            save_replay_buffer=False,
        )
        callback_list.append(ckpt_cb)

    # 最新版を上書き保存（スナップショットと併用推奨）
    latest_cb = SaveLatestCallback(
        latest_path=args.latest_path,
        save_every_steps=args.save_latest_every_steps,
        save_every_seconds=args.save_latest_every_seconds,
        vecnorm_path=os.path.join(args.log_dir, "vecnorm_latest.pkl") if (args.norm_obs or args.norm_rew) else "",
        verbose=1,
    )
    callback_list.append(latest_cb)

    callbacks = None
    if len(callback_list) == 1:
        callbacks = callback_list[0]
    elif len(callback_list) > 1:
        callbacks = CallbackList(callback_list)

    # 中断時の割り込み保存先
    interrupt_dir = _ensure_dir(os.path.join(args.log_dir, "interrupt"))
    interrupt_path = os.path.join(interrupt_dir, f"ppo_maskable_interrupt_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.zip")

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            tb_log_name=run_name,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=reset_flag,
        )
    except KeyboardInterrupt:
        # 割り込み保存（latestにも上書きしてから保存）
        try:
            model.save(args.latest_path)
            print(f"[Interrupt] Also saved latest model to: {args.latest_path}")
        except Exception:
            pass
        model.save(interrupt_path)
        print(f"\n[Interrupt] Saved interrupted model to: {interrupt_path}")
        raise
    finally:
        # VecNormalize の統計保存（使っていれば）
        if isinstance(model.get_env(), VecNormalize):
            model.get_env().save(vecnorm_path)
            print(f"[Info] Saved VecNormalize stats to: {vecnorm_path}")
        print(f"[Info] Checkpoints dir: {ckpt_dir}")
        print(f"[Info] Best model: {os.path.join(best_dir, 'best_model.zip')}")
        print(f"[Info] Latest model: {args.latest_path}")

    # 正常終了時の最終保存
    model.save(args.save_path)
    print(f"[Done] Final model saved to: {args.save_path}")
    print(f"[TB] TensorBoard logs in: {args.log_dir} (run={run_name})")


if __name__ == "__main__":
    main()