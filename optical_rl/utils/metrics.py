import csv
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


class MetricsLogger:
    def __init__(self, log_dir, csv_path=None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        self.csv_path = Path(csv_path) if csv_path else None
        self._csv_file = None
        self._csv_writer = None

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def log_episode(self, episode_idx, total_reward, episode_length):
        self.log_scalar("custom/episode_reward", total_reward, episode_idx)
        self.log_scalar("custom/episode_length", episode_length, episode_idx)

        if self.csv_path:
            if self._csv_writer is None:
                self.csv_path.parent.mkdir(parents=True, exist_ok=True)
                self._csv_file = self.csv_path.open("a", newline="", encoding="utf-8")
                self._csv_writer = csv.writer(self._csv_file)
                if self._csv_file.tell() == 0:
                    self._csv_writer.writerow(
                        ["episode", "total_reward", "episode_length"]
                    )
            self._csv_writer.writerow([episode_idx, total_reward, episode_length])
            self._csv_file.flush()

    def close(self):
        self.writer.flush()
        self.writer.close()
        if self._csv_file:
            self._csv_file.close()
