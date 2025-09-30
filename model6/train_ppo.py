import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import glob
import argparse
from datetime import datetime

from yachtenv import YachtEnv, ScoringRules
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.utils import set_random_seed, get_linear_fn
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnNoModelImprovement,
    CheckpointCallback,
    CallbackList,
    BaseCallback
)


# ---- 最新モデル上書きコールバック ----
class SaveLatestCallback(BaseCallback):
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
        self.save_every_steps = save_every_steps
        self.save_every_seconds = save_every_seconds
        self.vecnorm_path = vecnorm_path
        self._last_save_step = 0
        self._last_save_time = 0.0

    def _on_training_start(self) -> None:
        import time
        os.makedirs(os.path.dirname(self.latest_path) or ".", exist_ok=True)
        if self.vecnorm_path:
            os.makedirs(os.path.dirname(self.vecnorm_path) or ".", exist_ok=True)
        self._last_save_step = int(self.model.num_timesteps)
        self._last_save_time = time.time()

    def _save(self):
        import time
        self.model.save(self.latest_path)
        env = self.model.get_env()
        try:
            if isinstance(env, VecNormalize) and self.vecnorm_path:
                env.save(self.vecnorm_path)
        except Exception:
            pass
        if self.verbose:
            print(f"[SaveLatest] step={self.model.num_timesteps} time={time.time():.0f} saved {self.latest_path}")

    def _on_step(self) -> bool:
        import time
        step = int(self.model.num_timesteps)
        t = time.time()
        cond_step = self.save_every_steps > 0 and (step - self._last_save_step) >= self.save_every_steps
        cond_time = self.save_every_seconds > 0 and (t - self._last_save_time) >= self.save_every_seconds
        if cond_step or cond_time:
            self._save()
            self._last_save_step = step
            self._last_save_time = t
        return True


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
    obs_progress: bool,
    disable_keep_all: bool,
    shaping_upper_eps: float,
    shaping_lower_eps: float,
    shaping_bonus_eps: float,
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
            obs_progress=obs_progress,
            disable_keep_all=disable_keep_all,
            shaping_upper_eps=shaping_upper_eps,
            shaping_lower_eps=shaping_lower_eps,
            shaping_bonus_eps=shaping_bonus_eps,
        )
        return env
    return _init


def parse_args():
    p = argparse.ArgumentParser(description="Train MaskablePPO on YachtEnv with progress features & checkpoints")
    # ルール
    p.add_argument("--short-straight-points", type=int, default=15)
    p.add_argument("--big-straight-points", type=int, default=30)
    p.add_argument("--yacht-points", type=int, default=50)
    p.add_argument("--disable-upper-bonus", action="store_true")
    p.add_argument("--upper-bonus-threshold", type=int, default=63)
    p.add_argument("--upper-bonus-points", type=int, default=35)

    # 環境観測フラグ
    p.add_argument("--obs-augment", action="store_true")
    p.add_argument("--obs-add-scores", action="store_true")
    p.add_argument("--obs-progress", action="store_true", help="新規 progress 特徴 11次元を追加")
    p.add_argument("--mask-keep-all", action="store_true")

    # 成形係数
    p.add_argument("--shape-upper-eps", type=float, default=0.02)
    p.add_argument("--shape-lower-eps", type=float, default=0.01)
    p.add_argument("--shape-bonus-eps", type=float, default=0.01, help="ボーナス楽観 φ への shaping 係数 (0 で無効)")

    # 学習設定
    p.add_argument("--total-timesteps", type=int, default=50_000_000)
    p.add_argument("--n-envs", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--policy", type=str, default="MlpPolicy")
    p.add_argument("--vec", type=str, choices=["dummy", "subproc"], default="subproc")
    p.add_argument("--mp-start", type=str, choices=["spawn", "forkserver", "fork"], default="spawn")

    # PPO ハイパラ
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--lr-final-ratio", type=float, default=0.2)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--n-steps", type=int, default=1024)
    p.add_argument("--gamma", type=float, default=0.997)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--ent-coef", type=float, default=0.02)
    p.add_argument("--clip-range", type=float, default=0.2)

    # ログ/評価
    p.add_argument("--log-dir", type=str, default="./logs/yacht_ppo")
    p.add_argument("--save-path", type=str, default="./models/yacht_maskable_ppo_v8.zip")
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--eval-freq", type=int, default=800_000)
    p.add_argument("--no-eval", action="store_true")
    p.add_argument("--early-stop-no-improve", type=int, default=0)

    # チェックポイント & 最新
    p.add_argument("--checkpoint-freq", type=int, default=500_000)
    p.add_argument("--checkpoint-dir", type=str, default="")
    p.add_argument("--latest-path", type=str, default="./models/yacht_maskable_ppo_latest.zip")
    p.add_argument("--save-latest-every-steps", type=int, default=200_000)
    p.add_argument("--save-latest-every-seconds", type=float, default=300.0)

    # 再開
    p.add_argument("--resume-from", type=str, default="")
    p.add_argument("--resume-latest", action="store_true")
    p.add_argument("--reset-num-timesteps", action="store_true")

    # 観測正規化
    p.add_argument("--norm-obs", action="store_true")
    p.add_argument("--norm-rew", action="store_true")
    p.add_argument("--vecnorm-path", type=str, default="")

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
            obs_progress=args.obs_progress,
            disable_keep_all=args.mask_keep_all,
            shaping_upper_eps=args.shape_upper_eps if not for_eval else 0.0,
            shaping_lower_eps=args.shape_lower_eps if not for_eval else 0.0,
            shaping_bonus_eps=args.shape_bonus_eps if not for_eval else 0.0,
        )
        for i in range(1 if for_eval else args.n_envs)
    ]
    if args.vec == "subproc" and (args.n_envs > 1 or for_eval):
        venv = SubprocVecEnv(env_fns, start_method=args.mp_start)
    else:
        venv = DummyVecEnv(env_fns)
    venv = VecMonitor(venv)
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

    # 環境
    venv = build_vec_env(args, for_eval=False)
    if args.norm_obs or args.norm_rew:
        venv = VecNormalize(venv, norm_obs=args.norm_obs, norm_reward=args.norm_rew, clip_obs=5.0, gamma=args.gamma)

    # 学習率スケジュール（ent/clip は float 固定）
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

    # モデル生成/読み込み
    if resume_path and os.path.exists(resume_path):
        print(f"[Resume] Loading model from: {resume_path}")
        model = MaskablePPO.load(resume_path, device=args.device)
        # 既存モデルと新環境の観測次元が異なる場合は再学習が必要
        old_dim = int(np.prod(model.observation_space.shape))
        new_dim = venv.observation_space.shape[0]
        if old_dim != new_dim:
            raise ValueError(
                f"観測次元不一致: 既存モデル={old_dim}, 新環境={new_dim}。\n"
                "このまま継続したい場合は以前と同じ観測フラグで起動してください。\n"
                "新しい progress 特徴を使う場合は新規学習が必要です。"
            )
        model.set_env(venv)
        reset_flag = args.reset_num_timesteps is True
    else:
        print("[Start] Training NEW model (fresh weights)")
        model = MaskablePPO(
            args.policy,
            venv,
            verbose=1,
            seed=args.seed,
            device=args.device,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            ent_coef=float(args.ent_coef),
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            learning_rate=lr_schedule,
            clip_range=float(args.clip_range),
            tensorboard_log=args.log_dir,
            policy_kwargs=dict(net_arch=[512, 512, 256], ortho_init=True),
        )
        reset_flag = True

    # コールバック構築
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
            eval_freq=args.eval_freq,
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

    latest_cb = SaveLatestCallback(
        latest_path=args.latest_path,
        save_every_steps=args.save_latest_every_steps,
        save_every_seconds=args.save_latest_every_seconds,
        vecnorm_path=(os.path.join(args.log_dir, "vecnorm_latest.pkl")
                      if (args.norm_obs or args.norm_rew) else ""),
        verbose=1,
    )
    callback_list.append(latest_cb)

    if len(callback_list) == 1:
        callbacks = callback_list[0]
    else:
        callbacks = CallbackList(callback_list)

    # 割り込み保存
    interrupt_dir = _ensure_dir(os.path.join(args.log_dir, "interrupt"))
    interrupt_path = os.path.join(interrupt_dir, f"ppo_maskable_interrupt_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.zip")

    try:
        model.learn(
            total_timesteps=args.total_timestamps if hasattr(args, "total_timestamps") else args.total_timesteps,
            tb_log_name=run_name,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=reset_flag,
        )
    except KeyboardInterrupt:
        model.save(interrupt_path)
        print(f"\n[Interrupt] Saved interrupted model to: {interrupt_path}")
        raise
    finally:
        if isinstance(model.get_env(), VecNormalize):
            model.get_env().save(vecnorm_path)
            print(f"[Info] Saved VecNormalize stats to: {vecnorm_path}")
        print(f"[Info] Checkpoints dir: {ckpt_dir}")
        print(f"[Info] Best model: {os.path.join(best_dir, 'best_model.zip')}")
        print(f"[Info] Latest model: {args.latest_path}")

    model.save(args.save_path)
    print(f"[Done] Final model saved to: {args.save_path}")
    print(f"[TB] TensorBoard logs in: {args.log_dir} (run={run_name})")


if __name__ == "__main__":
    main()