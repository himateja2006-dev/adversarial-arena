from __future__ import annotations

import json
from pathlib import Path

from training.train_local import evaluate
from utils.io import save_json
from utils.plotting import save_baseline_vs_trained, save_reward_curve

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


def main() -> None:
    plots_dir = Path("plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    baseline_avg, baseline_series, baseline_consistency = evaluate("random", episodes=30)
    trained_avg, trained_series, trained_consistency = evaluate("trained", episodes=30)

    if plt is not None:
        # Reward vs episode
        plt.figure(figsize=(8, 4))
        plt.plot(baseline_series, label="before training baseline (random)", alpha=0.7)
        plt.plot(trained_series, label="after training policy", alpha=0.9)
        plt.xlabel("Episode")
        plt.ylabel("Score")
        plt.title("Reward vs Episode")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "reward_curve.png", dpi=150)
        plt.close()

        # Baseline vs trained
        plt.figure(figsize=(5, 4))
        plt.bar(
            ["before baseline avg reward", "after training avg reward"],
            [baseline_avg, trained_avg],
            color=["#7a7aee", "#22aa66"],
        )
        plt.ylim(0, 1)
        plt.ylabel("Average score")
        plt.title("Before vs After Training Comparison")
        plt.tight_layout()
        plt.savefig(plots_dir / "comparison.png", dpi=150)
        plt.close()
    else:
        # Fallback for minimal environments that cannot install matplotlib.
        save_reward_curve(trained_series, str(plots_dir / "reward_curve.png"))
        save_baseline_vs_trained(baseline_avg, trained_avg, str(plots_dir / "comparison.png"))

    result = {
        "before_baseline_avg_reward": baseline_avg,
        "after_training_avg_reward": trained_avg,
        "improvement": trained_avg - baseline_avg,
        "before_baseline_consistency": baseline_consistency,
        "after_training_consistency": trained_consistency,
        "training_note": "After-training policy learned from high-reward filtered trajectories (PPO-style objective proxy).",
    }
    save_json("training/artifacts/evaluation_report.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
