from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


def plot_training_progress(log_dir, out_path):
    log_dir = Path(log_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ea = event_accumulator.EventAccumulator(str(log_dir))
    ea.Reload()

    tags = {
        "rollout/ep_rew_mean": "Episode Reward Mean",
        "train/policy_loss": "Policy Loss",
        "train/value_loss": "Value Loss",
        "train/explained_variance": "Explained Variance",
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, (tag, title) in zip(axes, tags.items()):
        if tag not in ea.Tags().get("scalars", []):
            ax.set_title(f"{title} (missing)")
            ax.axis("off")
            continue
        scalars = ea.Scalars(tag)
        steps = [s.step for s in scalars]
        values = [s.value for s in scalars]
        ax.plot(steps, values)
        ax.set_title(title)
        ax.set_xlabel("Steps")
        ax.grid(True)

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
