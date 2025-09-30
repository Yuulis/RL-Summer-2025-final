import os
import argparse
from typing import List, Tuple, Optional
import numpy as np
import cloudpickle

from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

from yachtenv import YachtEnv, ScoringRules

# 観測次元 -> (augment, scores, progress)
OBS_COMBO_MAP = {
    19:  (False, False, False),
    28:  (True,  False, False),
    31:  (False, True,  False),
    30:  (False, False, True),
    40:  (True,  True,  False),
    39:  (True,  False, True),
    42:  (False, True,  True),
    51:  (True,  True,  True),
}

CAT_NAMES = [
    "Aces","Deuces","Threes","Fours","Fives","Sixes",
    "Choice","FourDice","FullHouse","S.Straight","B.Straight","Yacht"
]

def guess_flags_from_dim(dim: int) -> Tuple[bool, bool, bool]:
    if dim not in OBS_COMBO_MAP:
        raise ValueError(
            f"未知の観測次元 {dim}. 想定: {sorted(OBS_COMBO_MAP.keys())}. "
            "学習時と同じ観測構成で評価してください。"
        )
    return OBS_COMBO_MAP[dim]

def score_category_now(dice: np.ndarray, cat: int, rules: ScoringRules) -> int:
    counts = np.bincount(dice, minlength=7)
    total = int(np.sum(dice))
    if 0 <= cat <= 5:
        face = cat + 1
        return int(counts[face] * face)
    if cat == 6:  # Choice
        return total
    if cat == 7:  # FourDice
        if np.any(counts[1:] >= 4):
            return total
        return 0
    if cat == 8:  # FullHouse（ヨットも成立）
        has_three = np.any(counts[1:] == 3)
        has_two = np.any(counts[1:] == 2)
        is_yacht = np.any(counts[1:] == 5)
        if is_yacht or (has_three and has_two):
            return total
        return 0
    if cat == 9:  # Small Straight
        faces = set(int(x) for x in np.unique(dice))
        if ({1, 2, 3, 4}.issubset(faces)
            or {2, 3, 4, 5}.issubset(faces)
            or {3, 4, 5, 6}.issubset(faces)):
            return int(rules.short_straight_points)
        return 0
    if cat == 10:  # Big Straight
        faces_sorted = sorted(int(x) for x in dice)
        if faces_sorted == [1, 2, 3, 4, 5] or faces_sorted == [2, 3, 4, 5, 6]:
            return int(rules.big_straight_points)
        return 0
    if cat == 11:  # Yacht
        if np.any(counts[1:] == 5):
            return int(rules.yacht_points)
        return 0
    raise ValueError(cat)

def format_keep_mask(bits: int, num_dice: int = 5) -> str:
    # 下位ビット i=0..4 がダイス index（1=keep）。人間向けに左を die0 として並べ替え。
    s = "".join("1" if (bits >> i) & 1 else "0" for i in range(num_dice))
    return s[::-1]

def try_load_vecnorm(vecnorm_path: str, venv, quiet: bool = False):
    # shape 一致時のみ安全に VecNormalize を適用
    try:
        with open(vecnorm_path, "rb") as f:
            obj = cloudpickle.load(f)
        saved_shape = obj.observation_space.shape
        env_shape = venv.observation_space.shape
        if saved_shape != env_shape:
            if not quiet:
                print(f"[Warn] VecNormalize shape mismatch: saved={saved_shape} vs env={env_shape}. Skip.")
            return venv
        obj.set_venv(venv)
        obj.training = False
        obj.norm_reward = False
        if not quiet:
            print(f"[Info] Loaded VecNormalize stats from: {vecnorm_path}")
        return obj
    except FileNotFoundError:
        if not quiet:
            print(f"[Info] VecNormalize file not found: {vecnorm_path} (skip)")
    except Exception as e:
        if not quiet:
            print(f"[Warn] Failed to load VecNormalize: {e} (skip)")
    return venv

def make_env_single(
    seed: int,
    rules: ScoringRules,
    obs_augment: bool,
    obs_add_scores: bool,
    obs_progress: bool,
    disable_keep_all: bool,
):
    def _init():
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

def draw_table(headers: List[str], rows: List[List[str]]) -> str:
    # シンプルなASCIIテーブル描画
    col_w = [len(h) for h in headers]
    for r in rows:
        for j, cell in enumerate(r):
            col_w[j] = max(col_w[j], len(str(cell)))
    sep = "+".join("-" * (w + 2) for w in col_w)
    sep = f"+{sep}+"
    def fmt_row(vals):
        return "|" + "|".join(f" {str(v).ljust(w)} " for v, w in zip(vals, col_w)) + "|"
    out = [sep, fmt_row(headers), sep]
    for r in rows:
        out.append(fmt_row(r))
    out.append(sep)
    return "\n".join(out)

def render_scoresheet_table(
    dice: np.ndarray,
    categories_used: np.ndarray,
    scores_per_category: np.ndarray,
    rules: ScoringRules,
    mask: np.ndarray,
) -> str:
    headers = ["Category", "Used", "Scored", "Immediate", "Allowed"]
    rows: List[List[str]] = []
    for i, name in enumerate(CAT_NAMES):
        used = bool(categories_used[i])
        scored = str(int(scores_per_category[i])) if used else ""
        imm = str(score_category_now(dice, i, rules))
        allowed = "✓" if bool(mask[32 + i]) else ""
        rows.append([name, "X" if used else "", scored, imm, allowed])
    return draw_table(headers, rows)

def run(
    model_path: str,
    vecnorm_path: str,
    episodes: int,
    seed: int,
    deterministic: bool,
    mode: str,
    obs_augment_flag: bool,
    obs_add_scores_flag: bool,
    obs_progress_flag: bool,
    disable_keep_all: bool,
    short_straight_points: int,
    big_straight_points: int,
    yacht_points: int,
    enable_upper_bonus: bool,
    upper_bonus_threshold: int,
    upper_bonus_points: int,
    save_path: Optional[str] = None,
):
    # モデルのみ先に読み、観測次元からフラグ決定
    tmp = MaskablePPO.load(model_path, device="cpu")
    model_obs_dim = int(np.prod(tmp.observation_space.shape))
    if mode == "auto":
        obs_augment, obs_add_scores, obs_progress = guess_flags_from_dim(model_obs_dim)
    else:
        obs_augment, obs_add_scores, obs_progress = obs_augment_flag, obs_add_scores_flag, obs_progress_flag

    rules = ScoringRules(
        short_straight_points=short_straight_points,
        big_straight_points=big_straight_points,
        yacht_points=yacht_points,
        enable_one_roles_bonus=enable_upper_bonus,
        one_roles_bonus_threshold=upper_bonus_threshold,
        one_roles_bonus_points=upper_bonus_points,
    )

    # 環境を 1 本だけ作る
    venv = DummyVecEnv([
        make_env_single(
            seed=seed,
            rules=rules,
            obs_augment=obs_augment,
            obs_add_scores=obs_add_scores,
            obs_progress=obs_progress,
            disable_keep_all=disable_keep_all,
        )
    ])
    venv = VecMonitor(venv)
    if vecnorm_path:
        venv = try_load_vecnorm(vecnorm_path, venv, quiet=False)

    # n_envs 不一致を避けるため、env を渡してロード
    model = MaskablePPO.load(model_path, env=venv, device="auto")

    lines: List[str] = []
    def emit(s: str = ""):
        print(s)
        lines.append(s)

    emit("========== Yacht Agent Play (Tabular per Roll) ==========")
    emit(f"Model: {os.path.basename(model_path)} | Deterministic={deterministic}")
    emit(f"Obs flags: augment={obs_augment}, scores={obs_add_scores}, progress={obs_progress}")
    emit(f"Rules: S={rules.short_straight_points}, B={rules.big_straight_points}, Y={rules.yacht_points}, UpperBonus={'ON' if rules.enable_one_roles_bonus else 'OFF'}")

    episodes_done = 0
    obs = venv.reset()

    while episodes_done < episodes:
        emit(f"\n--- Episode {episodes_done+1:03d} START ---")
        ep_return = 0.0
        while True:
            dice = venv.get_attr("dice")[0].copy()
            rolls_left = int(venv.get_attr("rolls_left")[0])     # 2,1,0
            turns_done = int(venv.get_attr("turns_done")[0])     # 0..11 (確定済数)
            categories_used = venv.get_attr("categories_used")[0].copy()
            scores_per_category = venv.get_attr("scores_per_category")[0].copy()
            one_roles_sum = int(venv.get_attr("one_roles_sum")[0])

            # 行動マスク
            masks_list = venv.env_method("action_masks")
            mask = masks_list[0]

            # ロール番号（1..3）
            roll_idx = 3 - rolls_left

            # 表示ヘッダ
            emit(f"\n[Turn {turns_done+1:02d} | Roll {roll_idx}/3] Dice = {dice.tolist()} | UpperSum={one_roles_sum}/{rules.one_roles_bonus_threshold}")
            # スコアシート表
            table = render_scoresheet_table(dice, categories_used, scores_per_category, rules, mask)
            emit(table)

            # 推論（action_masks を渡す）
            action, _ = model.predict(obs, deterministic=deterministic, action_masks=np.asarray([mask], dtype=bool))
            action = int(action[0])

            # 行動説明
            if action < 32:
                keep_bits = action
                emit(f"Action: REROLL      keep_bits={format_keep_mask(keep_bits)} ({keep_bits})")
            else:
                cat = action - 32
                raw = score_category_now(dice, cat, rules)
                emit(f"Action: SCORE       -> {CAT_NAMES[cat]}  raw={raw}")

            # 1 ステップ進める
            obs, rewards, dones, infos = venv.step([action])
            ep_return += float(rewards[0])

            # 進捗表示
            if action < 32:
                dice_after = venv.get_attr("dice")[0].copy()
                rolls_left_after = int(venv.get_attr("rolls_left")[0])
                emit(f" -> After reroll: Dice={dice_after.tolist()} RollsLeft={rolls_left_after}")
            else:
                total_score = int(infos[0].get("total_score", 0))
                one_roles_sum = int(infos[0].get("one_roles_sum", 0))
                emit(f" -> Scored. TotalScore so far: {total_score} | UpperSum={one_roles_sum}/{rules.one_roles_bonus_threshold}")

            if dones[0]:
                total_score = int(infos[0].get("total_score", 0))
                bonus_hit = (rules.enable_one_roles_bonus and int(infos[0].get("one_roles_sum", 0)) >= rules.one_roles_bonus_threshold)
                emit(f"\n=== Episode {episodes_done+1:03d} END ===  TotalScore={total_score}  Return~{ep_return:.3f}  UpperBonus={'HIT' if bonus_hit else 'NO'}")
                episodes_done += 1
                if episodes_done < episodes:
                    obs = venv.reset()
                break

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"\n[Saved] Transcript written to: {save_path}")

def parse_args():
    p = argparse.ArgumentParser(description="Visualize a learned Yacht agent per roll with tabular score sheet.")
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--vecnorm-path", type=str, default="", help="Matching VecNormalize stats (optional)")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--deterministic", action="store_true")

    # 観測モード
    p.add_argument("--mode", type=str, choices=["auto", "manual"], default="auto")
    p.add_argument("--obs-augment", action="store_true")
    p.add_argument("--obs-add-scores", action="store_true")
    p.add_argument("--obs-progress", action="store_true")

    p.add_argument("--mask-keep-all", action="store_true")

    # ルール
    p.add_argument("--short-straight-points", type=int, default=15)
    p.add_argument("--big-straight-points", type=int, default=30)
    p.add_argument("--yacht-points", type=int, default=50)
    p.add_argument("--disable-upper-bonus", action="store_true")
    p.add_argument("--upper-bonus-threshold", type=int, default=63)
    p.add_argument("--upper-bonus-points", type=int, default=35)

    p.add_argument("--save-path", type=str, default="", help="Transcript output path (optional)")

    return p.parse_args()

def main():
    args = parse_args()
    run(
        model_path=args.model_path,
        vecnorm_path=args.vecnorm_path,
        episodes=args.episodes,
        seed=args.seed,
        deterministic=args.deterministic,
        mode=args.mode,
        obs_augment_flag=args.obs_augment,
        obs_add_scores_flag=args.obs_add_scores,
        obs_progress_flag=args.obs_progress,
        disable_keep_all=args.mask_keep_all,
        short_straight_points=args.short_straight_points,
        big_straight_points=args.big_straight_points,
        yacht_points=args.yacht_points,
        enable_upper_bonus=not args.disable_upper_bonus,
        upper_bonus_threshold=args.upper_bonus_threshold,
        upper_bonus_points=args.upper_bonus_points,
        save_path=(args.save_path or None),
    )

if __name__ == "__main__":
    main()