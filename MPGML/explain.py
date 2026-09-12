"""Explain trained MPGML checkpoints with modality ablation and projections."""

import csv
import json
import math
import os
import re
import sys
from collections import defaultdict
from copy import deepcopy

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch_geometric.data import Batch

from args_parser import build_parser, finalize_args
from dataset import FewshotMolDataset, dataset_sampler
from dataset.molecular_features import get_descriptor_names
from experiment import set_seed
from models import MAML, MPGML


EXPLAIN_ARG_KEYS = {
    "run_dir",
    "checkpoint",
    "output_dir",
    "tasks",
    "split",
    "max_tasks",
    "task_sample_mode",
    "max_query_per_task",
    "contribution_batch_size",
    "representation_batch_size",
    "projection_method",
    "projection_reps",
    "explain_top_k",
    "analysis_seed",
    "allow_random_init",
}


def explicit_cli_keys(argv, parser):
    """Resolve option aliases to destinations before merging saved settings."""
    keys = set()
    for arg in argv:
        if arg.startswith("--"):
            option = arg.split("=", 1)[0]
            action = parser._option_string_actions.get(option)
            if action is not None:
                keys.add(action.dest)
    return keys


def load_run_config(run_dir):
    if not run_dir:
        return {}
    config_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(config_path):
        # Older runs stored JSON under a misleading .pkl extension.
        config_path = os.path.join(run_dir, "params.pkl")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Cannot find config.json or legacy params.pkl in {run_dir}")
    with open(config_path, "r", encoding="utf-8") as fin:
        return json.load(fin)


def apply_run_config(args, explicit_keys):
    if not args.run_dir:
        return args

    config = load_run_config(args.run_dir)
    for key, value in config.items():
        if key in EXPLAIN_ARG_KEYS:
            continue
        if hasattr(args, key) and key not in explicit_keys:
            setattr(args, key, value)
    return args


def build_explain_parser():
    parser = build_parser()
    parser.description = "Explain trained MPGML predictions with modality ablation and 2-D projections."
    parser.add_argument("--run_dir", default=None, type=str,
                        help="Training run directory containing config.json and best_model.pkl.")
    parser.add_argument("--checkpoint", default=None, type=str,
                        help="Checkpoint path. Relative paths are resolved under --run_dir first.")
    parser.add_argument("--output_dir", default=None, type=str,
                        help="Directory for explainability outputs. Defaults to <run_dir>/explainability.")
    parser.add_argument("--tasks", default=None, type=str,
                        help="Comma-separated task ids or exact task names. Defaults to tasks from --split.")
    parser.add_argument("--split", default="test", choices=["train", "test", "all"],
                        help="Task split to explain when --tasks is not provided.")
    parser.add_argument("--max_tasks", default=3, type=int,
                        help="Maximum number of tasks to explain. Use 0 for all selected tasks.")
    parser.add_argument("--task_sample_mode", default="first", choices=["first", "random"],
                        help="How to choose tasks when --max_tasks limits the selected split.")
    parser.add_argument("--max_query_per_task", default=200, type=int,
                        help="Maximum query molecules per task. Use 0 for all query molecules.")
    parser.add_argument("--contribution_batch_size", default=1, type=int,
                        help="Query batch size for modality ablation. Keep 1 for molecule-level isolation.")
    parser.add_argument("--representation_batch_size", default=64, type=int,
                        help="Batch size for extracting projection representations.")
    parser.add_argument("--projection_method", default="pca", choices=["pca", "tsne", "both"],
                        help="2-D projection method.")
    parser.add_argument("--projection_reps", default="fused,graph,fingerprint,descriptor", type=str,
                        help="Comma-separated representations to project.")
    parser.add_argument("--explain_top_k", default=30, type=int,
                        help="Number of largest per-molecule modality changes to save.")
    parser.add_argument("--analysis_seed", default=None, type=int,
                        help="Seed used for task/query subsampling. Defaults to --random_seed.")
    parser.add_argument("--allow_random_init", action="store_true", default=False,
                        help="Allow running without a checkpoint. Use only for smoke tests.")
    return parser


def parse_args(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = build_explain_parser()
    args = parser.parse_args(argv)
    args = apply_run_config(args, explicit_cli_keys(argv, parser))
    args = finalize_args(args, parser)
    if args.analysis_seed is None:
        args.analysis_seed = int(args.random_seed)
    return args


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def default_output_dir(args):
    if args.output_dir:
        return args.output_dir
    if args.run_dir:
        return os.path.join(args.run_dir, "explainability")
    return os.path.join(
        args.dump_path,
        "explainability",
        args.dataset,
        f"seed_{args.random_seed}",
    )


def resolve_checkpoint(args):
    candidates = []
    if args.checkpoint:
        candidates.append(args.checkpoint)
        if args.run_dir and not os.path.isabs(args.checkpoint):
            candidates.insert(0, os.path.join(args.run_dir, args.checkpoint))
    elif args.run_dir:
        candidates.extend([
            os.path.join(args.run_dir, "best_model.pkl"),
            os.path.join(args.run_dir, "last_model.pkl"),
            os.path.join(args.run_dir, "model.pkl"),
        ])

    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def safe_name(value, max_len=80):
    value = str(value)
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value[:max_len].strip("_") or "item"


def write_csv(path, rows, fieldnames):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path, payload):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as fout:
        json.dump(payload, fout, indent=2, ensure_ascii=False)


def get_device(args):
    use_cuda = int(args.gpu) >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(int(args.gpu))
        return torch.device("cuda")
    return torch.device("cpu")


def load_task_names(dataset):
    with open(dataset.raw_paths[0], "r", newline="", encoding="utf-8") as fin:
        header = next(csv.reader(fin))
    y_idx = dataset.names[dataset.name][4]
    names = header[y_idx] if isinstance(y_idx, slice) else [header[y_idx]]
    return list(names)


def parse_task_selector(task_text, task_names):
    selected = []
    name_to_id = {name: idx for idx, name in enumerate(task_names)}
    for item in task_text.split(","):
        item = item.strip()
        if not item:
            continue
        if item.isdigit():
            task_id = int(item)
        elif item in name_to_id:
            task_id = name_to_id[item]
        else:
            raise ValueError(f"Unknown task selector {item!r}. Use a task id or exact task name.")
        if task_id < 0 or task_id >= len(task_names):
            raise ValueError(f"Task id {task_id} is out of range [0, {len(task_names) - 1}].")
        selected.append(task_id)
    return selected


def select_tasks(args, dataset, task_names, rng):
    if args.tasks:
        tasks = parse_task_selector(args.tasks, task_names)
    elif args.split == "train":
        tasks = list(dataset.train_task_range)
    elif args.split == "test":
        tasks = list(dataset.test_task_range)
    else:
        tasks = list(range(dataset.total_tasks))

    if args.max_tasks and args.max_tasks > 0 and len(tasks) > args.max_tasks:
        if args.task_sample_mode == "random":
            tasks = rng.choice(tasks, size=args.max_tasks, replace=False).tolist()
            tasks = sorted(int(task_id) for task_id in tasks)
        else:
            tasks = tasks[:args.max_tasks]
    return tasks


def data_mol_id(data):
    value = getattr(data, "id", None)
    if value is None:
        return None
    if torch.is_tensor(value):
        return int(value.view(-1)[0].item())
    return int(value)


def data_label(data, task_id):
    value = data.y.view(-1)[task_id].item()
    return float(value)


def data_smiles(data):
    value = getattr(data, "smiles", "")
    if isinstance(value, (list, tuple)):
        return value[0] if value else ""
    return str(value)


def finite_label(data, task_id):
    label = data_label(data, task_id)
    return not math.isnan(label)


def limit_query_list(query_list, task_id, max_query, rng):
    query_list = [data for data in query_list if finite_label(data, task_id)]
    if not max_query or max_query <= 0 or len(query_list) <= max_query:
        return query_list

    class0 = [data for data in query_list if data_label(data, task_id) < 0.5]
    class1 = [data for data in query_list if data_label(data, task_id) > 0.5]
    rng.shuffle(class0)
    rng.shuffle(class1)

    half = max_query // 2
    selected = class0[:half] + class1[:half]
    remaining = class0[half:] + class1[half:]
    rng.shuffle(remaining)
    selected += remaining[:max_query - len(selected)]
    rng.shuffle(selected)
    return selected


def iter_chunks(items, batch_size):
    batch_size = max(1, int(batch_size))
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def sample_auxiliary(tgt_task_id, auxi_task_range, auxi_task_num, rng):
    candidates = list(auxi_task_range)
    if tgt_task_id in candidates:
        candidates.remove(tgt_task_id)
    if auxi_task_num is None:
        auxi_task_num = len(candidates)
    auxi_task_num = min(int(auxi_task_num), len(candidates))
    if auxi_task_num <= 0:
        return []
    return rng.choice(candidates, auxi_task_num, replace=False).astype(int).tolist()


def build_model(args, dataset, device, load_pretrained=True):
    model_args = deepcopy(args)
    if not load_pretrained:
        # A complete experiment checkpoint includes the graph encoder weights.
        model_args.mol_pretrain_load_path = None
    model = MPGML(
        task_num=dataset.total_tasks,
        train_task_num=dataset.n_task_train,
        args=model_args,
    ).to(device)
    return model


def load_checkpoint(model, checkpoint_path, device):
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=True)
    return {
        "path": checkpoint_path,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }


def prepare_task_episode(args, dataset, maml, task_id, rng, device, cls_criterion):
    support_list, query_list = dataset_sampler(
        dataset,
        args.n_support,
        args.n_query,
        tgt_id=task_id,
        inductive=True,
    )
    query_list = limit_query_list(query_list, task_id, args.max_query_per_task, rng)

    support_batch = Batch.from_data_list(support_list).to(device)
    auxi_num = args.test_auxi_task_num
    if auxi_num is None:
        auxi_num = len(dataset.train_task_range)
    auxi_tasks = sample_auxiliary(task_id, dataset.train_task_range, auxi_num, rng)
    sampled_task = torch.tensor([task_id] + auxi_tasks, dtype=torch.long, device=device)
    support_y = support_batch.y[:, sampled_task]

    task_model = maml.clone()
    task_model.train()

    for _ in range(int(args.inner_update_step)):
        if len(query_list) == 0:
            break
        batch_size = min(max(1, int(args.n_query)), len(query_list))
        selected = rng.choice(len(query_list), size=batch_size, replace=False).tolist()
        adapt_query = [query_list[idx] for idx in selected]
        adapt_query_batch = Batch.from_data_list(adapt_query).to(device)
        adapt_query_y = adapt_query_batch.y[:, sampled_task]
        pred, _, _, _, _ = task_model(
            support_batch,
            adapt_query_batch,
            support_y,
            adapt_query_y,
            sampled_task,
        )
        support_logit, _, support_label, _ = pred
        inner_loss = cls_criterion(support_logit, support_label)
        task_model.adapt(inner_loss)

    task_model.eval()
    return {
        "model": task_model,
        "support_list": support_list,
        "query_list": query_list,
        "support_batch": support_batch,
        "support_y": support_y,
        "sampled_task": sampled_task,
        "auxi_tasks": auxi_tasks,
    }


def predict_query(task_model, support_batch, support_y, sampled_task, query_list,
                  device, batch_size, q_ablate_modalities=None):
    rows = []
    with torch.no_grad():
        for chunk in iter_chunks(query_list, batch_size):
            query_batch = Batch.from_data_list(chunk).to(device)
            query_y = query_batch.y[:, sampled_task]
            pred, _, _, _, _ = task_model(
                support_batch,
                query_batch,
                support_y,
                query_y,
                sampled_task,
                test=True,
                q_ablate_modalities=q_ablate_modalities,
            )
            _, query_logit, _, query_label = pred
            logits = query_logit.detach().cpu().view(-1).numpy()
            probs = torch.sigmoid(query_logit).detach().cpu().view(-1).numpy()
            labels = query_label.detach().cpu().view(-1).numpy()
            for data, logit, prob, label in zip(chunk, logits, probs, labels):
                rows.append({
                    "mol_id": data_mol_id(data),
                    "smiles": data_smiles(data),
                    "label": float(label),
                    "logit": float(logit),
                    "prob": float(prob),
                })
    return rows


def safe_auc(labels, scores):
    labels = np.asarray(labels, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    valid = np.isfinite(labels) & np.isfinite(scores)
    labels = labels[valid]
    scores = scores[valid]
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def signed_projection_auc(labels, coords):
    auc = safe_auc(labels, coords[:, 0])
    if math.isnan(auc):
        return auc
    return float(max(auc, 1.0 - auc))


def collect_modality_contributions(args, task_id, task_name, episode, modalities, device):
    task_model = episode["model"]
    support_batch = episode["support_batch"]
    support_y = episode["support_y"]
    sampled_task = episode["sampled_task"]
    query_list = episode["query_list"]

    base_rows = predict_query(
        task_model,
        support_batch,
        support_y,
        sampled_task,
        query_list,
        device,
        args.contribution_batch_size,
    )
    base_by_id = {row["mol_id"]: row for row in base_rows}
    labels = [row["label"] for row in base_rows]
    base_probs = [row["prob"] for row in base_rows]
    auc_full = safe_auc(labels, base_probs)

    molecule_rows = []
    summary_rows = []
    prediction_rows = []
    for row in base_rows:
        prediction_rows.append({
            "task_id": task_id,
            "task_name": task_name,
            "mol_id": row["mol_id"],
            "smiles": row["smiles"],
            "label": row["label"],
            "baseline_logit": row["logit"],
            "baseline_prob": row["prob"],
            "baseline_pred_label": int(row["prob"] >= 0.5),
            "baseline_correct": int((row["prob"] >= 0.5) == (row["label"] >= 0.5)),
        })

    for modality in modalities:
        masked_rows = predict_query(
            task_model,
            support_batch,
            support_y,
            sampled_task,
            query_list,
            device,
            args.contribution_batch_size,
            q_ablate_modalities=[modality],
        )
        masked_probs = [row["prob"] for row in masked_rows]
        auc_masked = safe_auc(labels, masked_probs)
        prob_deltas = []
        logit_deltas = []

        for masked in masked_rows:
            base = base_by_id[masked["mol_id"]]
            prob_delta = base["prob"] - masked["prob"]
            logit_delta = base["logit"] - masked["logit"]
            prob_deltas.append(prob_delta)
            logit_deltas.append(logit_delta)
            molecule_rows.append({
                "task_id": task_id,
                "task_name": task_name,
                "mol_id": masked["mol_id"],
                "smiles": masked["smiles"],
                "label": masked["label"],
                "modality": modality,
                "baseline_logit": base["logit"],
                "baseline_prob": base["prob"],
                "masked_logit": masked["logit"],
                "masked_prob": masked["prob"],
                "logit_delta": float(logit_delta),
                "prob_delta": float(prob_delta),
                "abs_logit_delta": float(abs(logit_delta)),
                "abs_prob_delta": float(abs(prob_delta)),
                "baseline_pred_label": int(base["prob"] >= 0.5),
                "masked_pred_label": int(masked["prob"] >= 0.5),
                "prediction_flipped": int((base["prob"] >= 0.5) != (masked["prob"] >= 0.5)),
            })

        summary_rows.append({
            "task_id": task_id,
            "task_name": task_name,
            "modality": modality,
            "n_query": len(base_rows),
            "auc_full": auc_full,
            "auc_masked": auc_masked,
            "auc_drop": float(auc_full - auc_masked) if not (math.isnan(auc_full) or math.isnan(auc_masked)) else float("nan"),
            "mean_prob_delta": float(np.mean(prob_deltas)) if prob_deltas else float("nan"),
            "mean_abs_prob_delta": float(np.mean(np.abs(prob_deltas))) if prob_deltas else float("nan"),
            "mean_logit_delta": float(np.mean(logit_deltas)) if logit_deltas else float("nan"),
            "mean_abs_logit_delta": float(np.mean(np.abs(logit_deltas))) if logit_deltas else float("nan"),
            "flip_rate": float(np.mean([row["prediction_flipped"] for row in molecule_rows if row["task_id"] == task_id and row["modality"] == modality])) if molecule_rows else 0.0,
        })

    prob_total_by_task = sum(row["mean_abs_prob_delta"] for row in summary_rows if np.isfinite(row["mean_abs_prob_delta"]))
    logit_total_by_task = sum(row["mean_abs_logit_delta"] for row in summary_rows if np.isfinite(row["mean_abs_logit_delta"]))
    for row in summary_rows:
        row["normalized_importance_prob"] = (
            float(row["mean_abs_prob_delta"] / prob_total_by_task)
            if prob_total_by_task > 0 and np.isfinite(row["mean_abs_prob_delta"])
            else float("nan")
        )
        row["normalized_importance_logit"] = (
            float(row["mean_abs_logit_delta"] / logit_total_by_task)
            if logit_total_by_task > 0 and np.isfinite(row["mean_abs_logit_delta"])
            else float("nan")
        )
        row["normalized_importance"] = (
            row["normalized_importance_prob"]
            if np.isfinite(row["normalized_importance_prob"])
            else row["normalized_importance_logit"]
        )

    return prediction_rows, molecule_rows, summary_rows


def collect_gate_rows(model):
    rows = []
    module = model.module if hasattr(model, "module") else model

    if getattr(module, "fusion_residual_gate", None) is not None:
        rows.append({
            "component": "fusion",
            "feature": "fusion_residual_gate",
            "index": "",
            "value": float(torch.sigmoid(module.fusion_residual_gate).detach().cpu().item()),
        })

    if getattr(module, "fp_encoder", None) is not None and getattr(module.fp_encoder, "gate", None) is not None:
        values = torch.sigmoid(module.fp_encoder.gate.logits).detach().cpu().numpy()
        for idx, value in enumerate(values):
            rows.append({
                "component": "fingerprint",
                "feature": f"fp_{idx}",
                "index": idx,
                "value": float(value),
            })

    if getattr(module, "desc_encoder", None) is not None and getattr(module.desc_encoder, "gate", None) is not None:
        values = torch.sigmoid(module.desc_encoder.gate.logits).detach().cpu().numpy()
        names = list(get_descriptor_names())
        for idx, value in enumerate(values):
            rows.append({
                "component": "descriptor",
                "feature": names[idx] if idx < len(names) else f"desc_{idx}",
                "index": idx,
                "value": float(value),
            })
    return rows


def extract_representation_rows(task_model, query_list, task_id, task_name, reps, device, batch_size):
    rows_by_rep = defaultdict(list)
    with torch.no_grad():
        for chunk in iter_chunks(query_list, batch_size):
            batch = Batch.from_data_list(chunk).to(device)
            feature_dict = task_model.module.extract_modal_features(batch)
            for rep in reps:
                if rep not in feature_dict:
                    continue
                values = feature_dict[rep].detach().cpu().numpy()
                for data, vector in zip(chunk, values):
                    rows_by_rep[rep].append({
                        "task_id": task_id,
                        "task_name": task_name,
                        "mol_id": data_mol_id(data),
                        "smiles": data_smiles(data),
                        "label": data_label(data, task_id),
                        "vector": vector.astype(np.float32),
                    })
    return rows_by_rep


def project_vectors(vectors, method, seed):
    x = np.asarray(vectors, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if x.shape[0] < 2:
        return None, {}
    x = StandardScaler().fit_transform(x)

    if method == "pca":
        reducer = PCA(n_components=2, random_state=seed)
        coords = reducer.fit_transform(x)
        return coords, {
            "explained_variance_1": float(reducer.explained_variance_ratio_[0]),
            "explained_variance_2": float(reducer.explained_variance_ratio_[1]),
        }

    perplexity = min(30, max(2, (x.shape[0] - 1) // 3))
    if perplexity >= x.shape[0]:
        perplexity = max(1, x.shape[0] - 1)
    reducer = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
        random_state=seed,
    )
    coords = reducer.fit_transform(x)
    return coords, {
        "perplexity": float(perplexity),
    }


def projection_metrics(coords, labels):
    labels = np.asarray(labels, dtype=np.float32)
    valid = np.isfinite(labels)
    coords = coords[valid]
    labels = labels[valid]
    if coords.shape[0] == 0:
        return {
            "projection_auc_axis1": float("nan"),
            "silhouette": float("nan"),
            "centroid_distance": float("nan"),
        }

    classes = np.unique(labels)
    if len(classes) < 2:
        return {
            "projection_auc_axis1": float("nan"),
            "silhouette": float("nan"),
            "centroid_distance": float("nan"),
        }

    class0 = coords[labels < 0.5]
    class1 = coords[labels > 0.5]
    centroid_distance = float(np.linalg.norm(class0.mean(axis=0) - class1.mean(axis=0))) if len(class0) and len(class1) else float("nan")
    try:
        sil = float(silhouette_score(coords, labels.astype(int)))
    except Exception:
        sil = float("nan")

    return {
        "projection_auc_axis1": signed_projection_auc(labels, coords),
        "silhouette": sil,
        "centroid_distance": centroid_distance,
    }


def plot_projection(path, coords, labels, title, method):
    labels = np.asarray(labels, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(7, 5), dpi=160)
    for label, color, marker in [(0.0, "#2F6BFF", "o"), (1.0, "#E56B2F", "^")]:
        mask = labels == label
        if np.any(mask):
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=28,
                c=color,
                marker=marker,
                alpha=0.78,
                edgecolors="none",
                label=f"label {int(label)}",
            )
    ax.set_title(title[:120])
    ax.set_xlabel(f"{method.upper()}-1")
    ax.set_ylabel(f"{method.upper()}-2")
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_projection_outputs(args, output_dir, task_id, task_name, rows_by_rep):
    projection_methods = ["pca", "tsne"] if args.projection_method == "both" else [args.projection_method]
    projection_rows = []
    projection_summary_rows = []

    for rep, rows in rows_by_rep.items():
        if not rows:
            continue
        vectors = [row["vector"] for row in rows]
        labels = np.asarray([row["label"] for row in rows], dtype=np.float32)

        for method in projection_methods:
            coords, reducer_info = project_vectors(vectors, method, args.analysis_seed)
            if coords is None:
                continue
            metrics = projection_metrics(coords, labels)
            for row, xy in zip(rows, coords):
                projection_rows.append({
                    "task_id": task_id,
                    "task_name": task_name,
                    "mol_id": row["mol_id"],
                    "smiles": row["smiles"],
                    "label": row["label"],
                    "representation": rep,
                    "method": method,
                    "x": float(xy[0]),
                    "y": float(xy[1]),
                })

            summary = {
                "task_id": task_id,
                "task_name": task_name,
                "representation": rep,
                "method": method,
                "n_query": len(rows),
                **metrics,
                **reducer_info,
            }
            projection_summary_rows.append(summary)

            plot_name = f"projection_task_{task_id}_{safe_name(task_name)}_{rep}_{method}.png"
            plot_path = os.path.join(output_dir, "plots", plot_name)
            ensure_dir(os.path.dirname(plot_path))
            title = (
                f"Task {task_id}: {task_name} | {rep} | {method.upper()} "
                f"| silhouette={metrics['silhouette']:.3f}"
            )
            plot_projection(plot_path, coords, labels, title, method)

    return projection_rows, projection_summary_rows


def global_modality_summary(summary_rows):
    grouped = defaultdict(list)
    for row in summary_rows:
        grouped[row["modality"]].append(row)

    def safe_nanmean(values):
        values = [value for value in values if np.isfinite(value)]
        return float(np.mean(values)) if values else float("nan")

    output = []
    for modality, rows in grouped.items():
        output.append({
            "modality": modality,
            "n_tasks": len(rows),
            "mean_auc_drop": safe_nanmean([row["auc_drop"] for row in rows]),
            "mean_abs_prob_delta": safe_nanmean([row["mean_abs_prob_delta"] for row in rows]),
            "mean_abs_logit_delta": safe_nanmean([row["mean_abs_logit_delta"] for row in rows]),
            "mean_normalized_importance": safe_nanmean([row["normalized_importance"] for row in rows]),
            "mean_normalized_importance_prob": safe_nanmean([row["normalized_importance_prob"] for row in rows]),
            "mean_normalized_importance_logit": safe_nanmean([row["normalized_importance_logit"] for row in rows]),
            "mean_flip_rate": safe_nanmean([row["flip_rate"] for row in rows]),
        })
    return sorted(output, key=lambda row: row["mean_abs_prob_delta"], reverse=True)


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.analysis_seed)
    rng = np.random.default_rng(args.analysis_seed)
    output_dir = ensure_dir(default_output_dir(args))
    device = get_device(args)

    dataset = FewshotMolDataset(
        root=args.data_root,
        name=args.dataset,
        use_fingerprint=args.use_fingerprint,
        use_descriptor=args.use_descriptor,
        fp_type=args.fp_type,
        rebuild_feature_cache=args.rebuild_feature_cache,
    )
    task_names = load_task_names(dataset)
    selected_tasks = select_tasks(args, dataset, task_names, rng)

    checkpoint_path = resolve_checkpoint(args)
    checkpoint_info = {"path": None, "missing_keys": [], "unexpected_keys": []}
    if checkpoint_path is None:
        if not args.allow_random_init:
            raise FileNotFoundError(
                "No checkpoint found. Pass --checkpoint, use --run_dir with best_model.pkl, "
                "or pass --allow_random_init for a smoke test only."
            )
    model = build_model(args, dataset, device, load_pretrained=checkpoint_path is None)
    if checkpoint_path is not None:
        checkpoint_info = load_checkpoint(model, checkpoint_path, device)

    maml = MAML(
        model,
        lr=args.inner_lr,
        first_order=not args.second_order,
        anil=False,
        allow_unused=True,
    )
    cls_criterion = nn.BCEWithLogitsLoss()
    modalities = maml.module.available_modalities()
    projection_reps = [rep.strip().lower() for rep in args.projection_reps.split(",") if rep.strip()]

    all_prediction_rows = []
    all_molecule_rows = []
    all_summary_rows = []
    all_projection_rows = []
    all_projection_summary_rows = []

    for task_id in selected_tasks:
        task_name = task_names[task_id]
        episode = prepare_task_episode(args, dataset, maml, task_id, rng, device, cls_criterion)
        if len(episode["query_list"]) == 0:
            continue

        prediction_rows, molecule_rows, summary_rows = collect_modality_contributions(
            args,
            task_id,
            task_name,
            episode,
            modalities,
            device,
        )
        all_prediction_rows.extend(prediction_rows)
        all_molecule_rows.extend(molecule_rows)
        all_summary_rows.extend(summary_rows)

        rows_by_rep = extract_representation_rows(
            episode["model"],
            episode["query_list"],
            task_id,
            task_name,
            projection_reps,
            device,
            args.representation_batch_size,
        )
        projection_rows, projection_summary_rows = write_projection_outputs(
            args,
            output_dir,
            task_id,
            task_name,
            rows_by_rep,
        )
        all_projection_rows.extend(projection_rows)
        all_projection_summary_rows.extend(projection_summary_rows)

    top_rows = sorted(
        all_molecule_rows,
        key=lambda row: (row["abs_prob_delta"], row["abs_logit_delta"]),
        reverse=True,
    )[:max(0, int(args.explain_top_k))]

    write_csv(
        os.path.join(output_dir, "baseline_predictions.csv"),
        all_prediction_rows,
        [
            "task_id", "task_name", "mol_id", "smiles", "label",
            "baseline_logit", "baseline_prob", "baseline_pred_label", "baseline_correct",
        ],
    )
    write_csv(
        os.path.join(output_dir, "molecule_contributions.csv"),
        all_molecule_rows,
        [
            "task_id", "task_name", "mol_id", "smiles", "label", "modality",
            "baseline_logit", "baseline_prob", "masked_logit", "masked_prob",
            "logit_delta", "prob_delta", "abs_logit_delta", "abs_prob_delta",
            "baseline_pred_label", "masked_pred_label", "prediction_flipped",
        ],
    )
    write_csv(
        os.path.join(output_dir, "top_molecule_cases.csv"),
        top_rows,
        [
            "task_id", "task_name", "mol_id", "smiles", "label", "modality",
            "baseline_logit", "baseline_prob", "masked_logit", "masked_prob",
            "logit_delta", "prob_delta", "abs_logit_delta", "abs_prob_delta",
            "baseline_pred_label", "masked_pred_label", "prediction_flipped",
        ],
    )
    write_csv(
        os.path.join(output_dir, "modality_contribution_summary.csv"),
        all_summary_rows,
        [
            "task_id", "task_name", "modality", "n_query", "auc_full", "auc_masked",
            "auc_drop", "mean_prob_delta", "mean_abs_prob_delta", "mean_logit_delta",
            "mean_abs_logit_delta", "flip_rate", "normalized_importance",
            "normalized_importance_prob", "normalized_importance_logit",
        ],
    )
    write_csv(
        os.path.join(output_dir, "global_modality_summary.csv"),
        global_modality_summary(all_summary_rows),
        [
            "modality", "n_tasks", "mean_auc_drop", "mean_abs_prob_delta",
            "mean_abs_logit_delta", "mean_normalized_importance",
            "mean_normalized_importance_prob", "mean_normalized_importance_logit",
            "mean_flip_rate",
        ],
    )
    write_csv(
        os.path.join(output_dir, "projection.csv"),
        all_projection_rows,
        [
            "task_id", "task_name", "mol_id", "smiles", "label",
            "representation", "method", "x", "y",
        ],
    )
    write_csv(
        os.path.join(output_dir, "projection_summary.csv"),
        all_projection_summary_rows,
        [
            "task_id", "task_name", "representation", "method", "n_query",
            "projection_auc_axis1", "silhouette", "centroid_distance",
            "explained_variance_1", "explained_variance_2", "perplexity",
        ],
    )
    write_csv(
        os.path.join(output_dir, "feature_gates.csv"),
        collect_gate_rows(maml.module),
        ["component", "feature", "index", "value"],
    )
    write_json(
        os.path.join(output_dir, "explain_config.json"),
        {
            "dataset": args.dataset,
            "random_seed": args.random_seed,
            "analysis_seed": args.analysis_seed,
            "run_dir": args.run_dir,
            "checkpoint": checkpoint_info,
            "output_dir": output_dir,
            "selected_tasks": selected_tasks,
            "selected_task_names": {str(task_id): task_names[task_id] for task_id in selected_tasks},
            "modalities": modalities,
            "projection_reps": projection_reps,
            "device": str(device),
            "args": vars(args),
        },
    )

    print(f"Explainability outputs written to: {output_dir}")
    print(f"Tasks: {selected_tasks}")
    print(f"Modalities: {modalities}")


if __name__ == "__main__":
    main()
