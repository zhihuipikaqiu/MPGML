"""Write training configuration, task sampling records, and evaluation metrics."""

import csv
import json
import os
import platform
import sys
from datetime import datetime


def _to_csv_value(value):
    if value is None:
        return ""
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _append_csv(path, fieldnames, row):
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({key: _to_csv_value(row.get(key)) for key in fieldnames})


class ExperimentRecorder:
    """Append per-run records and aggregate records in the parent sweep directory."""

    def __init__(self, dump_dir, args):
        self.dump_dir = dump_dir
        self.sweep_dir = os.path.dirname(dump_dir)
        self.args = args
        self.started_at = datetime.now().isoformat(timespec="seconds")
        self.metrics_path = os.path.join(dump_dir, "metrics.csv")
        self.train_task_samples_path = os.path.join(dump_dir, "train_task_samples.csv")
        self.eval_task_scores_path = os.path.join(dump_dir, "eval_task_scores.csv")
        self.summary_csv_path = os.path.join(dump_dir, "summary.csv")
        self.summary_txt_path = os.path.join(dump_dir, "summary.txt")
        self.run_command_txt_path = os.path.join(dump_dir, "run_command.txt")
        self.run_command_csv_path = os.path.join(dump_dir, "run_command.csv")
        self.sweep_commands_csv_path = os.path.join(self.sweep_dir, "run_commands.csv")
        self.sweep_summaries_csv_path = os.path.join(self.sweep_dir, "run_summaries.csv")
        self.run_config_csv_path = os.path.join(dump_dir, "run_config.csv")

    def write_initial_files(self):
        self._write_run_command()
        self._write_run_config()

    def _base_context(self):
        return {
            "started_at": self.started_at,
            "dataset": self.args.dataset,
            "random_seed": self.args.random_seed,
            "exp_name": self.args.exp_name,
            "exp_id": self.args.exp_id,
            "n_support": self.args.n_support,
            "n_query": self.args.n_query,
            "episode": self.args.episode,
            "eval_step": self.args.eval_step,
            "inner_lr": self.args.inner_lr,
            "meta_lr": self.args.meta_lr,
            "rel_ratio": self.args.rel_ratio,
            "ib_ratio": self.args.ib_ratio,
            "contr_w": self.args.contr_w,
            "gpu": self.args.gpu,
        }

    def _write_run_command(self):
        command = getattr(self.args, "command", "")
        argv = getattr(self.args, "argv", sys.argv)
        metadata = {
            **self._base_context(),
            "command": command,
            "dump_dir": self.dump_dir,
            "python_executable": sys.executable,
            "argv": argv,
            "working_directory": os.getcwd(),
            "platform": platform.platform(),
        }

        with open(self.run_command_txt_path, "w", encoding="utf-8") as fout:
            fout.write(f"started_at: {self.started_at}\n")
            fout.write(f"working_directory: {metadata['working_directory']}\n")
            fout.write(f"python_executable: {metadata['python_executable']}\n")
            fout.write(f"command: {command}\n")
            fout.write("argv_json: ")
            fout.write(json.dumps(argv, ensure_ascii=False))
            fout.write("\n")

        _append_csv(
            self.run_command_csv_path,
            [
                "started_at",
                "dataset",
                "random_seed",
                "exp_name",
                "exp_id",
                "dump_dir",
                "command",
                "python_executable",
                "argv",
                "working_directory",
                "platform",
            ],
            metadata,
        )
        _append_csv(
            self.sweep_commands_csv_path,
            [
                "started_at",
                "dataset",
                "random_seed",
                "exp_name",
                "exp_id",
                "dump_dir",
                "command",
                "python_executable",
                "argv",
                "working_directory",
                "platform",
            ],
            metadata,
        )

    def _write_run_config(self):
        path = self.run_config_csv_path
        with open(path, "w", newline="", encoding="utf-8") as fout:
            writer = csv.DictWriter(fout, fieldnames=["key", "value"])
            writer.writeheader()
            for key, value in sorted(vars(self.args).items()):
                writer.writerow({"key": key, "value": _to_csv_value(value)})

    def record_epoch(self, epoch, loss_cls, epoch_time_sec, elapsed_time_sec,
                     loss_contr=None, reg_loss=None, loss_total=None,
                     score=None, best_score=None):
        row = {
            **self._base_context(),
            "epoch": epoch,
            "loss_cls": loss_cls,
            "loss_contr": loss_contr,
            "reg_loss": reg_loss,
            "loss_total": loss_total,
            "epoch_time_sec": epoch_time_sec,
            "elapsed_time_sec": elapsed_time_sec,
            "score": score,
            "best_score": best_score,
        }
        _append_csv(
            self.metrics_path,
            [
                "started_at",
                "dataset",
                "random_seed",
                "exp_name",
                "exp_id",
                "n_support",
                "n_query",
                "episode",
                "eval_step",
                "inner_lr",
                "meta_lr",
                "rel_ratio",
                "ib_ratio",
                "contr_w",
                "gpu",
                "epoch",
                "loss_cls",
                "loss_contr",
                "reg_loss",
                "loss_total",
                "epoch_time_sec",
                "elapsed_time_sec",
                "score",
                "best_score",
            ],
            row,
        )

    def record_eval_task_scores(self, epoch, task_scores):
        for item in task_scores:
            row = {
                **self._base_context(),
                "epoch": epoch,
                "task_id": item["task_id"],
                "roc_auc": item["roc_auc"],
            }
            _append_csv(
                self.eval_task_scores_path,
                [
                    "started_at",
                    "dataset",
                    "random_seed",
                    "exp_name",
                    "exp_id",
                    "n_support",
                    "n_query",
                    "inner_lr",
                    "meta_lr",
                    "rel_ratio",
                    "ib_ratio",
                    "gpu",
                    "epoch",
                    "task_id",
                    "roc_auc",
                ],
                row,
            )

    def record_train_task_samples(self, epoch, task_records):
        for item in task_records:
            row = {
                **self._base_context(),
                "epoch": epoch,
                "task_id": item["task_id"],
                "auxi_tasks": item["auxi_tasks"],
            }
            _append_csv(
                self.train_task_samples_path,
                [
                    "started_at",
                    "dataset",
                    "random_seed",
                    "exp_name",
                    "exp_id",
                    "n_support",
                    "n_query",
                    "inner_lr",
                    "meta_lr",
                    "rel_ratio",
                    "ib_ratio",
                    "gpu",
                    "epoch",
                    "task_id",
                    "auxi_tasks",
                ],
                row,
            )

    def record_summary(self, best_score, mean_epoch_time_sec,
                       total_time_sec, completed_epochs):
        row = {
            **self._base_context(),
            "completed_epochs": completed_epochs,
            "best_score": best_score,
            "mean_epoch_time_sec": mean_epoch_time_sec,
            "total_time_sec": total_time_sec,
            "dump_dir": self.dump_dir,
            "command": getattr(self.args, "command", ""),
        }
        _append_csv(
            self.summary_csv_path,
            [
                "started_at",
                "dataset",
                "random_seed",
                "exp_name",
                "exp_id",
                "n_support",
                "n_query",
                "episode",
                "eval_step",
                "inner_lr",
                "meta_lr",
                "rel_ratio",
                "ib_ratio",
                "contr_w",
                "gpu",
                "dump_dir",
                "completed_epochs",
                "best_score",
                "mean_epoch_time_sec",
                "total_time_sec",
                "command",
            ],
            row,
        )
        _append_csv(
            self.sweep_summaries_csv_path,
            [
                "started_at",
                "dataset",
                "random_seed",
                "exp_name",
                "exp_id",
                "n_support",
                "n_query",
                "episode",
                "eval_step",
                "inner_lr",
                "meta_lr",
                "rel_ratio",
                "ib_ratio",
                "contr_w",
                "gpu",
                "dump_dir",
                "completed_epochs",
                "best_score",
                "mean_epoch_time_sec",
                "total_time_sec",
                "command",
            ],
            row,
        )
        with open(self.summary_txt_path, "w", encoding="utf-8") as fout:
            fout.write(f"dataset: {self.args.dataset}\n")
            fout.write(f"random_seed: {self.args.random_seed}\n")
            fout.write(f"exp_id: {self.args.exp_id}\n")
            fout.write(f"completed_epochs: {completed_epochs}\n")
            fout.write(f"best_score: {best_score:.5f}\n")
            fout.write(f"mean_epoch_time_sec: {mean_epoch_time_sec:.5f}\n")
            fout.write(f"total_time_sec: {total_time_sec:.5f}\n")
            fout.write(f"command: {getattr(self.args, 'command', '')}\n")
