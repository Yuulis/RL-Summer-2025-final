import os
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from yachtEnv import YachtEnv, ScoringRules

# sb3-contrib（MaskablePPO）関連は任意。モデルがない場合や未インストールでもランダムで可視化可能。
_USE_MASKABLE = True
try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from sb3_contrib.common.maskable.utils import get_action_masks
except Exception:
    _USE_MASKABLE = False
    MaskablePPO = None
    ActionMasker = None
    get_action_masks = None


CAT_NAMES = [
    "Aces", "Deuces", "Threes", "Fours", "Fives", "Sixes",
    "Choice", "FourDice", "FullHouse", "S.Straight", "B.Straight", "Yacht"
]

def mask_fn(env: YachtEnv):
    return env.get_action_mask()


def make_env(seed: int = 123) -> YachtEnv:
    rules = ScoringRules()
    env = YachtEnv(rules=rules, seed=seed)
    # 推論時も無効アクションを選ばないためにラップ（インストール時のみ）
    if _USE_MASKABLE and ActionMasker is not None:
        env = ActionMasker(env, mask_fn)
    return env


def decode_action(action: int) -> Dict[str, Any]:
    if 0 <= action <= 31:
        bits = [(action >> i) & 1 for i in range(5)]
        keep_indices = [i for i, b in enumerate(bits) if b == 1]
        return {"type": "reroll", "keep_bits": bits, "keep_indices": keep_indices}
    elif 32 <= action <= 43:
        category = action - 32
        return {"type": "score", "category": category, "category_name": CAT_NAMES[category]}
    return {"type": "invalid"}


def run_episode(model_path: Optional[str] = None,
                seed: int = 123,
                deterministic: bool = True) -> Dict[str, List[Any]]:
    """
    1エピソードを実行して履歴を返す。
    model_path が指定され、MaskablePPO が使える場合はモデルで行動、それ以外はマスク内ランダム。
    """
    env = make_env(seed=seed)

    model = None
    if model_path is not None and os.path.exists(model_path):
        if not _USE_MASKABLE:
            raise RuntimeError("sb3-contrib (MaskablePPO) がインストールされていないため、モデル推論は使えません。pip install sb3-contrib を実行してください。")
        model = MaskablePPO.load(model_path, device="cpu")
        print(f"Loaded model: {model_path}")

    obs, info = env.reset()
    history: Dict[str, List[Any]] = {
        "step": [], "turn": [], "rolls_left": [], "dice": [],
        "action": [], "action_decoded": [], "reward": [],
        "total_score": [], "points_added": [], "action_mask_true": [],
    }

    step_idx = 0
    done = False
    truncated = False
    prev_total = int(info.get("total_score", 0))

    # 初期ステップのマスク数
    mask_count = int(np.sum(info.get("action_mask"))) if "action_mask" in info else int(np.sum(env.get_action_mask()))
    while not (done or truncated):
        if model is not None:
            # モデル推論（マスクを渡す）
            if _USE_MASKABLE and get_action_masks is not None:
                action_masks = get_action_masks(env)
                action, _ = model.predict(obs, deterministic=deterministic, action_masks=action_masks)
            else:
                # フォールバック（理論上ここには来ない）
                action, _ = model.predict(obs, deterministic=deterministic)
        else:
            # マスク内ランダム
            action_mask = info.get("action_mask")
            if action_mask is None:
                action_mask = env.get_action_mask()
            valid_actions = np.flatnonzero(action_mask)
            action = int(np.random.choice(valid_actions))

        decoded = decode_action(int(action))
        obs, reward, done, truncated, info = env.step(int(action))

        # 追加情報
        total = int(info.get("total_score", 0))
        points_added = total - prev_total  # 最後のステップではボーナスも含む可能性あり
        prev_total = total
        mask_count = int(np.sum(info.get("action_mask"))) if "action_mask" in info else mask_count

        # ログ
        history["step"].append(step_idx)
        history["turn"].append(int(info.get("turns_done", 0)))  # 確定後に進むことに注意
        history["rolls_left"].append(int(info.get("rolls_left", 0)))
        history["dice"].append(np.array(info.get("dice"), dtype=int))
        history["action"].append(int(action))
        history["action_decoded"].append(decoded)
        history["reward"].append(float(reward))
        history["total_score"].append(total)
        history["points_added"].append(int(points_added))
        history["action_mask_true"].append(mask_count)

        step_idx += 1

    try:
        env.close()
    except Exception:
        pass

    return history


def plot_episode(history: Dict[str, List[Any]], save_path: Optional[str] = None, figsize: Tuple[int, int] = (14, 12)):
    steps = np.array(history["step"])
    total_score = np.array(history["total_score"])
    rewards = np.array(history["reward"])
    rolls_left = np.array(history["rolls_left"])
    points_added = np.array(history["points_added"])
    dice_arr = np.vstack(history["dice"]) if len(history["dice"]) > 0 else np.zeros((0, 5), dtype=int)
    mask_true = np.array(history["action_mask_true"])
    actions_decoded = history["action_decoded"]

    # スコア確定が行われたステップ（points_added > 0 or カテゴリゼロ確定時でも True にしたい場合は decoded を見る）
    score_steps = [i for i, d in enumerate(actions_decoded) if d.get("type") == "score"]

    fig, axes = plt.subplots(3, 2, figsize=figsize)
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.ravel()

    # 1) 総合スコアの推移
    ax1.plot(steps, total_score, marker="o")
    ax1.set_title("Total Score over Steps")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Total Score")
    ax1.grid(True, alpha=0.3)
    for si in score_steps:
        ax1.axvline(steps[si], color="orange", alpha=0.2, linestyle="--")

    # 2) ステップ報酬（正規化）
    # Matplotlib 3.8+ では use_line_collection が廃止されているため渡さない
    ax2.stem(steps, rewards, basefmt=" ")
    ax2.set_title("Per-step Reward (normalized)")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Reward (0..1)")
    ax2.grid(True, alpha=0.3)

    # 3) 残りロール回数
    ax3.step(steps, rolls_left, where="post")
    ax3.set_title("Rolls Left per Step")
    ax3.set_xlabel("Step")
    ax3.set_ylabel("rolls_left")
    ax3.set_yticks([0, 1, 2])
    ax3.grid(True, alpha=0.3)

    # 4) ダイス値の推移（各ダイスを散布）
    if dice_arr.shape[0] > 0:
        for d in range(dice_arr.shape[1]):
            ax4.scatter(steps, dice_arr[:, d], label=f"Die{d}", s=20)
        ax4.set_ylim(0.5, 6.5)
        ax4.set_yticks([1, 2, 3, 4, 5, 6])
    ax4.set_title("Dice values per Step")
    ax4.set_xlabel("Step")
    ax4.set_ylabel("Face")
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc="upper right", fontsize=8)

    # 5) カテゴリ確定タイミング（どのカテゴリを確定したか）
    xs, ys, labels = [], [], []
    for i, dec in enumerate(actions_decoded):
        if dec.get("type") == "score":
            xs.append(steps[i])
            cat_id = int(dec["category"])
            ys.append(cat_id)
            labels.append(CAT_NAMES[cat_id])
    if xs:
        ax5.scatter(xs, ys, marker="s", s=80, c="tab:green")
        for x, y, lab in zip(xs, ys, labels):
            ax5.text(x, y + 0.15, lab, fontsize=8, rotation=30)
    ax5.set_title("Scored Category per Step")
    ax5.set_xlabel("Step")
    ax5.set_ylabel("Category ID")
    ax5.set_yticks(list(range(len(CAT_NAMES))))
    ax5.set_yticklabels(CAT_NAMES)
    ax5.grid(True, alpha=0.3)

    # 6) 有効アクション数（マスクのTrue数）
    ax6.plot(steps, mask_true, marker=".")
    ax6.set_title("Number of Valid Actions (mask True count)")
    ax6.set_xlabel("Step")
    ax6.set_ylabel("#valid actions")
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved figure to {save_path}")
    plt.show()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Visualize one episode of YachtEnv.")
    parser.add_argument("--model-path", type=str, default="", help="Path to a trained MaskablePPO model (.zip). Optional.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--save-path", type=str, default="")
    args = parser.parse_args()

    model_path = args.model_path if args.model_path else None
    history = run_episode(model_path=model_path, seed=args.seed, deterministic=args.deterministic)
    plot_episode(history, save_path=args.save_path if args.save_path else None)


if __name__ == "__main__":
    main()