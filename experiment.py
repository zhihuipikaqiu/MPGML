"""Experiment directories, logging, random seeds, and state-dict checkpoints."""

import json
import logging
import os
import random
import shlex
import subprocess
import sys
import time
from datetime import date, datetime, timedelta

import numpy as np
import torch


class LogFormatter:
    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime('%x %X'),
            timedelta(seconds=elapsed_seconds)
        )
        message = record.getMessage()
        message = message.replace('\n', '\n' + ' ' * (len(prefix) + 3))
        return "%s - %s" % (prefix, message) if message else ''


def create_logger(filepath, rank):
    """
    Create a logger.
    Use a different log file for each process.
    """
    # create log formatter
    log_formatter = LogFormatter()

    # create file handler and set level to debug
    if filepath is not None:
        if rank > 0:
            filepath = '%s-%i' % (filepath, rank)
        file_handler = logging.FileHandler(filepath, "a", encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_formatter)

    # create console handler and set level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    # create logger and set level to debug
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if filepath is not None:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # reset logger elapsed time
    def reset_time():
        log_formatter.start_time = time.time()

    logger.reset_time = reset_time

    return logger


def initialize_exp(params):
    """Create the run directory, save JSON configuration, and configure logging."""
    exp_folder = get_dump_path(params)

    # get running command
    argv = [sys.executable] + sys.argv
    if os.name == 'nt':
        command = subprocess.list2cmdline(argv)
    else:
        command = shlex.join(argv)
    params.command = command
    params.argv = sys.argv[:]

    with open(os.path.join(exp_folder, 'config.json'), 'w', encoding='utf-8') as fout:
        json.dump(vars(params), fout, indent=4, ensure_ascii=False)

    # check experiment name
    assert len(params.exp_name.strip()) > 0

    # create a logger
    logger = create_logger(os.path.join(exp_folder, 'train.log'), rank=getattr(params, 'global_rank', 0))
    logger.info("============ Initialized logger ============")
    logger.info("\n".join("%s: %s" % (k, str(v))
                          for k, v in sorted(dict(vars(params)).items())))

    logger.info("The experiment will be stored in %s\n" % exp_folder)
    logger.info("Running command: %s" % command)
    return logger


def get_dump_path(params):
    """
    Create a directory to store the experiment.
    """
    assert len(params.exp_name) > 0
    assert not params.dump_path in ('', None), \
        'Choose a non-empty experiment output directory.'
    dump_path = params.dump_path

    # create the sweep path if it does not exist
    when = date.today().strftime('%m%d-')
    temp_exp_name = 'rel_ratio_{}_ib_ratio_{}'.format(params.rel_ratio, params.ib_ratio)

    if params.lr_check_flag:
        temp_exp_name = temp_exp_name + '_meta_lr_{}_inner_lr_{}'.format(params.meta_lr, params.inner_lr)

    if params.auxi_task_check_flag:
        sweep_path = os.path.join(dump_path, params.dataset, str(params.n_support), str(params.train_auxi_task_num),
                                  temp_exp_name, when + params.exp_name)
    else:
        sweep_path = os.path.join(dump_path, params.dataset, str(params.n_support), temp_exp_name,
                                  when + params.exp_name)

    os.makedirs(sweep_path, exist_ok=True)

    # Use a timestamp when no explicit experiment ID is supplied.
    if params.exp_id == '':
        exp_id = datetime.now().strftime('%H-%M-%S.%f')[:-3]
        params.exp_id = exp_id

    # Create the run directory and preserve existing run-path conventions.
    exp_folder = os.path.join(sweep_path, params.exp_id)
    os.makedirs(exp_folder, exist_ok=True)
    return exp_folder


def describe_model(model, path, name='model'):
    file_path = os.path.join(path, f'{name}.describe')
    with open(file_path, 'w', encoding='utf-8') as fout:
        print(model, file=fout)


def set_seed(seed):
    """
    Freeze every seed for reproducibility.
    torch.cuda.manual_seed_all is useful when using random generation on GPUs.
    e.g. torch.cuda.FloatTensor(100).uniform_()
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_model(model, save_dir, epoch=None, model_name='model'):
    model_to_save = model.module if hasattr(model, "module") else model
    if epoch is None:
        save_path = os.path.join(save_dir, f'{model_name}.pkl')
    else:
        save_path = os.path.join(save_dir, f'{model_name}-{epoch}.pkl')
    torch.save(model_to_save.state_dict(), save_path)

