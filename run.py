"""Train MPGML and record periodic held-out-task evaluation."""

import logging
import os
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from args_parser import args_parser
from experiment import describe_model, get_dump_path, initialize_exp, save_model, set_seed
from meta_learner import MetaLearner
from result_recorder import ExperimentRecorder

logger = logging.getLogger(__name__)


class Runner:
    """Coordinate meta-training, evaluation, checkpoints, and run records."""

    def __init__(self, args, logger_path, writer, recorder=None):
        self.args = args
        self.writer = writer
        self.recorder = recorder
        self.meta_learner = MetaLearner(args)
        describe_model(self.meta_learner.maml.module, logger_path, name='model')
        self.logger_path = logger_path

    def run(self):
        best_score = -1
        pbar = tqdm(range(1, self.args.episode + 1))
        cost_time_ls = []
        run_start = time.time()

        for epoch in pbar:
            start = time.time()
            train_metrics = self.meta_learner.train_step(return_metrics=True)
            loss_cls = train_metrics["loss_cls"]
            cost_time = time.time() - start
            cost_time_ls.append(cost_time)
            self.writer.add_scalar('loss-cls', loss_cls, epoch)
            self.writer.add_scalar('loss-contr', train_metrics["loss_contr"], epoch)
            self.writer.add_scalar('loss-total', train_metrics["loss_total"], epoch)

            pbar.set_description(f"loss={loss_cls:.4f}")
            score = None
            if epoch % self.args.eval_step == 0:
                score, task_scores = self.meta_learner.test_step(return_task_scores=True)
                if score > best_score:
                    best_score = score
                    if self.args.save_best_model:
                        save_model(self.meta_learner.maml.module, self.logger_path, model_name='best_model')
                logger.info(f"{epoch} | score: {score:.5f}, best_score: {best_score:.5f}")
                self.writer.add_scalars('score', {'score': score, 'best': best_score}, epoch)
                if self.recorder is not None:
                    self.recorder.record_eval_task_scores(epoch, task_scores)
            if self.recorder is not None:
                self.recorder.record_epoch(
                    epoch=epoch,
                    loss_cls=loss_cls,
                    loss_contr=train_metrics["loss_contr"],
                    reg_loss=train_metrics["reg_loss"],
                    loss_total=train_metrics["loss_total"],
                    epoch_time_sec=cost_time,
                    elapsed_time_sec=time.time() - run_start,
                    score=score,
                    best_score=best_score if score is not None else None,
                )
                self.recorder.record_train_task_samples(
                    epoch=epoch,
                    task_records=train_metrics["train_task_records"],
                )
        mean_time = float(np.mean(cost_time_ls)) if len(cost_time_ls) > 0 else 0
        total_time = time.time() - run_start
        logger.info(f"best score: {best_score:.5f}")
        logger.info(f"time cost: {mean_time:.5f}s")
        if self.args.save_last_model:
            save_model(self.meta_learner.maml.module, self.logger_path, model_name='last_model')
        if self.recorder is not None:
            self.recorder.record_summary(
                best_score=best_score,
                mean_epoch_time_sec=mean_time,
                total_time_sec=total_time,
                completed_epochs=self.args.episode,
            )

def main():
    args = args_parser()
    if int(args.gpu) >= 0:
        torch.cuda.set_device(int(args.gpu))
    set_seed(args.random_seed)
    initialize_exp(args)
    logger_path = get_dump_path(args)
    recorder = ExperimentRecorder(logger_path, args)
    recorder.write_initial_files()
    writer = SummaryWriter(log_dir=os.path.join(logger_path, 'tensorboard'))

    try:
        runner = Runner(args, logger_path, writer, recorder)
        runner.run()
    finally:
        writer.close()


if __name__ == '__main__':
    main()
