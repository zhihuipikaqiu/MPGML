"""Prototype-guided episodic training and held-out assay evaluation."""

import logging

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import optim
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from dataset import FewshotMolDataset, dataset_sampler
from dataset.sampler import episode_is_possible
from models import MPGML, focal_loss

logger = logging.getLogger(__name__)


class MetaLearner:
    """Adapt only the episode head and meta-optimize the shared encoder."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if int(args.gpu) >= 0 else "cpu")
        self.dataset = FewshotMolDataset(
            root=args.data_root,
            name=args.dataset,
            use_fingerprint=args.use_fingerprint,
            use_descriptor=args.use_descriptor,
            fp_type=args.fp_type,
            rebuild_feature_cache=args.rebuild_feature_cache,
        )
        self.train_task_range = self._eligible_tasks(self.dataset.train_task_range, "meta-training")
        self.test_task_range = self._eligible_tasks(self.dataset.test_task_range, "meta-test")
        self.model = MPGML(args).to(self.device)
        self.opt = optim.AdamW(
            self.model.parameters(), lr=args.meta_lr, weight_decay=args.weight_decay,
        )

    def _eligible_tasks(self, tasks, split):
        eligible = [
            task_id for task_id in tasks
            if episode_is_possible(self.dataset, self.args.n_support, task_id)
        ]
        omitted = sorted(set(tasks) - set(eligible))
        if omitted:
            logger.warning(
                "%s tasks without a disjoint two-class support and query set: %s", split, omitted,
            )
        if not eligible:
            raise ValueError(f"No {split} assay can form the requested episodes.")
        return eligible

    def train_step(self, return_metrics=False):
        """Perform one outer update using the mean focal loss over sampled tasks."""
        self.model.train()
        self.opt.zero_grad(set_to_none=True)
        task_ids = np.random.choice(
            self.train_task_range, min(self.args.pool_num, len(self.train_task_range)), replace=False,
        )
        query_losses = []
        for task_id in task_ids:
            support, query = dataset_sampler(
                self.dataset, self.args.n_support, self.args.n_query, task_id,
            )
            support = Batch.from_data_list(support).to(self.device)
            query = Batch.from_data_list(query).to(self.device)
            head = self.model.adapt_head(
                self.model(support), support.y[:, task_id].long(), meta_training=True,
            )
            query_logits = F.linear(self.model(query), *head)
            query_losses.append(focal_loss(query_logits, query.y[:, task_id].long(), self.args.focal_gamma))

        loss_cls = torch.stack(query_losses).mean()
        reg_loss = self.model.get_regularization_loss()
        loss_total = loss_cls + reg_loss
        loss_total.backward()
        self.opt.step()

        if return_metrics:
            return {
                "loss_cls": loss_cls.item(),
                "reg_loss": reg_loss.item(),
                "loss_total": loss_total.item(),
                "train_task_records": [{"task_id": int(task_id)} for task_id in task_ids],
            }
        return loss_cls.item()

    def test_step(self, return_task_scores=False):
        """Adapt on support and evaluate each held-out assay on its full query set."""
        self.model.eval()
        task_scores = []
        for task_id in tqdm(self.test_task_range, desc="eval"):
            support, query = dataset_sampler(
                self.dataset, self.args.n_support, self.args.n_query, task_id, inductive=True,
            )
            support = Batch.from_data_list(support).to(self.device)
            with torch.no_grad():
                support_features = self.model(support)
            head = self.model.adapt_head(support_features, support.y[:, task_id].long())

            loader = DataLoader(
                query,
                batch_size=self.args.test_batch_size,
                num_workers=self.args.test_num_workers,
                shuffle=False,
                pin_memory=self.device.type == "cuda",
            )
            labels, probabilities = [], []
            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(self.device)
                    logits = F.linear(self.model(batch), *head)
                    probabilities.append(logits.softmax(dim=-1)[:, 1].cpu())
                    labels.append(batch.y[:, task_id].cpu())
            score = roc_auc_score(torch.cat(labels).numpy(), torch.cat(probabilities).numpy())
            task_scores.append({"task_id": int(task_id), "roc_auc": float(score)})

        mean_score = float(np.mean([row["roc_auc"] for row in task_scores]))
        if return_task_scores:
            return mean_score, task_scores
        return mean_score
