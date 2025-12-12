import argparse
import math
import os
import random
import sys

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure local packages (model/, flow_matching/) are importable when running from the repo root.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
if MODEL_DIR not in sys.path:
    sys.path.append(MODEL_DIR)

from model.dataset.smd import SMDSegLoader
from model.dataset.ucr import UCRSegLoader
from model.dataset.gesture2d import Gesture2DSegLoader
from model.dataset.pd import PDSegLoader
from model.dataset.ecg import ECGSegLoader
from model.metric import binary_classification_metrics
from model.model_flow import FlowMatchingTransformer

from flow_matching.path import CondOTProbPath
from flow_matching.solver import ODESolver


LOG_2PI = math.log(2 * math.pi)


def compute_anomaly_scores(
    terminal: torch.Tensor,
    method: str = "r2_z",
    k: int = 3,
    latent_stats: dict | None = None,
) -> torch.Tensor:
    """
    Compute anomaly scores from terminal latent states.

    terminal: [B, L, D] or [B, L] or [B, D] or [L, D]
    method:
        - "nll_mean":     mean NLL over dims (original default)
        - "r2":           squared L2 norm ||z||^2
        - "r2_z":         dimension-normalized radius ((||z||^2 - D) / sqrt(2D))
        - "mahalanobis":  Mahalanobis distance squared using latent_stats["mean"] and latent_stats["cov_inv"]
        - "max_nll":      max per-dim NLL (channel-sensitive)
        - "topk_nll":     mean NLL of top-k dims (channel-sensitive)

    Returns: scores of shape [B, L]
    """

    # --------- Normalize shape to [B, L, D] ---------
    if terminal.dim() == 1:
        terminal = terminal.view(1, 1, -1)          # [D] -> [1, 1, D]
    elif terminal.dim() == 2:
        terminal = terminal.unsqueeze(-1)           # [B, L] -> [B, L, 1]
    elif terminal.dim() == 3:
        pass
    else:
        raise ValueError(f"Unexpected terminal shape: {terminal.shape}")

    B, L, D = terminal.shape

    # Common NLL tensor for N(0,1): 0.5 * (z^2 + log 2π), shape [B, L, D]
    nll = 0.5 * (terminal ** 2 + LOG_2PI)

    if method == "nll_mean":
        scores = nll.mean(dim=-1)  # [B, L]

    elif method == "r2":
        scores = (terminal ** 2).sum(dim=-1)  # [B, L]

    elif method == "r2_z":
        r2 = (terminal ** 2).sum(dim=-1)
        mean_r2 = float(D)
        std_r2 = math.sqrt(2.0 * D)
        scores = (r2 - mean_r2) / std_r2

    elif method == "mahalanobis":
        if latent_stats is None:
            raise ValueError("latent_stats must be provided for method='mahalanobis'.")
        mu = latent_stats.get("mean", None)
        cov_inv = latent_stats.get("cov_inv", None)
        if mu is None or cov_inv is None:
            raise ValueError("latent_stats['mean'] and latent_stats['cov_inv'] must be set.")
        mu = mu.to(terminal.device)
        cov_inv = cov_inv.to(terminal.device)
        delta = terminal - mu.view(1, 1, D)
        tmp = torch.matmul(delta, cov_inv)  # [B, L, D]
        scores = (tmp * delta).sum(dim=-1)

    elif method == "max_nll":
        scores = nll.max(dim=-1).values

    elif method == "topk_nll":
        k = max(1, min(k, D))
        topk_vals, _ = nll.topk(k, dim=-1)
        scores = topk_vals.mean(dim=-1)

    else:
        raise ValueError(f"Unknown scoring method: {method}")

    return scores


def set_seed(seed: int = 64, deterministic: bool = False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Flow matching training for anomaly detection"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="smd",
        choices=["smd", "ucr", "gesture2d", "pd", "ecg"],
        help="Dataset to use: smd, ucr, gesture2d, pd, or ecg",
    )
    parser.add_argument(
        "--ucr-id",
        type=int,
        default=135,
        help="UCR subset id (e.g., 135/136/137/138) when using --dataset ucr",
    )
    parser.add_argument(
        "--ecg-id",
        type=str,
        default="F",
        help="ECG subset id (e.g., A/B/C/D/E/F) when using --dataset ecg",
    )
    parser.add_argument(
        "--allow-ucr-train-without-labels",
        action="store_true",
        help="Allow UCR training when UCR_train_label.npy is missing (assumes all train samples are normal).",
    )
    parser.add_argument(
        "--data-root", type=str, default="data", help="Root directory for datasets"
    )

    parser.add_argument("--win-size", type=int, default=100, help="Sliding window size")
    parser.add_argument("--step", type=int, default=1, help="Stride for sliding window")

    parser.add_argument("--batch-size", type=int, default=32, help="Train batch size")
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=None,
        help="Test batch size (defaults to train batch size)",
    )

    parser.add_argument("--d-model", type=int, default=256, help="Transformer hidden dimension")
    parser.add_argument(
        "--dim-feedforward",
        type=int,
        default=512,
        help="Feedforward dimension inside transformer blocks",
    )
    parser.add_argument("--nhead", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=6, help="Number of transformer layers")
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=None,
        help="Optional latent dimension for model output; defaults to input_dim when None.",
    )

    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--eval-every",
        type=int,
        default=1,
        help="Evaluate on test set every N epochs",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="./checkpoints", help="Checkpoint directory")
    parser.add_argument(
        "--save-name",
        type=str,
        default="flow_transformer.pth",
        help="Checkpoint filename",
    )
    parser.add_argument(
        "--ode-step-size",
        type=float,
        default=1.0,
        help="Step size for ODE solver during evaluation (1.0 means a single Euler step from t=0 to t=1).",
    )
    parser.add_argument(
        "--ode-method",
        type=str,
        default="euler",
        help="torchdiffeq ODE solver method (default: euler).",
    )
    parser.add_argument(
        "--score-method",
        type=str,
        default="mahalanobis",
        choices=["nll_mean", "r2", "r2_z", "mahalanobis", "max_nll", "topk_nll"],
        help="Anomaly score aggregation over channels.",
    )
    parser.add_argument(
        "--score-topk",
        type=int,
        default=3,
        help="k for topk_nll scoring (ignored otherwise).",
    )
    return parser.parse_args()


def apply_point_adjustment(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """
    Point adjustment for time-series anomaly detection:
    mark an entire ground-truth segment as predicted positive if any point inside is predicted positive.
    """
    assert pred.shape == gt.shape
    adjusted = pred.copy()
    T = len(gt)

    in_segment = False
    seg_start = 0

    for t in range(T):
        if gt[t] == 1 and not in_segment:
            in_segment = True
            seg_start = t
        elif gt[t] == 0 and in_segment:
            seg_end = t - 1
            if pred[seg_start : seg_end + 1].any():
                adjusted[seg_start : seg_end + 1] = 1
            in_segment = False

    if in_segment:
        seg_end = T - 1
        if pred[seg_start : seg_end + 1].any():
            adjusted[seg_start : seg_end + 1] = 1

    return adjusted


def sweep_threshold(scores: np.ndarray, gt: np.ndarray, use_point_adjustment: bool):
    """
    Sweep thresholds over scores to find the best F1.
    """
    max_score = scores.max()
    thresholds = np.linspace(0.0, max_score if max_score > 0 else 1.0, 500)

    best = None
    for thr in thresholds:
        preds = (scores > thr).astype(int)
        preds_eval = apply_point_adjustment(preds, gt) if use_point_adjustment else preds

        acc, pre, rec, f1, conf = binary_classification_metrics(gt, preds_eval)
        if best is None or f1 > best["f1"]:
            best = {
                "f1": f1,
                "acc": acc,
                "pre": pre,
                "rec": rec,
                "conf": conf,
                "thr": thr,
                "pred_sum": int(preds_eval.sum()),
            }
    return best


def build_dataloaders(args):
    if args.dataset == "smd":
        data_source = os.path.join(args.data_root, "smd_1_1")
        dataset_cls = SMDSegLoader
    elif args.dataset == "ucr":
        data_source = os.path.join(args.data_root, f"UCR_{args.ucr_id}")
        dataset_cls = UCRSegLoader
    elif args.dataset == "gesture2d":
        data_source = os.path.join(args.data_root, "2DGesture")
        dataset_cls = Gesture2DSegLoader
    elif args.dataset == "pd":
        data_source = os.path.join(args.data_root, "pd", "labeled")
        dataset_cls = PDSegLoader
    elif args.dataset == "ecg":
        data_source = os.path.join(args.data_root, f"ECG_{args.ecg_id}")
        dataset_cls = ECGSegLoader
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    common_kwargs = {
        "data_path": data_source,
        "win_size": args.win_size,
        "step": args.step,
    }
    train_kwargs = {**common_kwargs, "mode": "train"}
    test_kwargs = {**common_kwargs, "mode": "test"}

    if args.dataset == "ucr":
        train_label_path = os.path.join(data_source, "UCR_train_label.npy")
        require_labels = not args.allow_ucr_train_without_labels
        if require_labels and not os.path.exists(train_label_path):
            print(
                f"Warning: {train_label_path} not found. "
                "Assuming all UCR train samples are normal. "
                "Pass --allow-ucr-train-without-labels to suppress this warning."
            )
            require_labels = False

        train_kwargs["train_only_normal"] = True
        test_kwargs["train_only_normal"] = True
        train_kwargs["require_train_labels"] = require_labels
        test_kwargs["require_train_labels"] = require_labels

    train_dataset = dataset_cls(**train_kwargs)
    test_dataset = dataset_cls(**test_kwargs)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size or args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    sample_x, _ = train_dataset[0]  # shape (L, D) or (L,)
    input_dim = 1 if sample_x.ndim == 1 else sample_x.shape[-1]

    return train_dataset, test_dataset, train_loader, test_loader, data_source, input_dim


def train_epoch_flow(model, prob_path, dataloader, optimizer, device, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(
        dataloader,
        desc=f"TrainFM {epoch}/{total_epochs}",
        leave=False,
        dynamic_ncols=True,
    )

    for windows, _ in pbar:
        windows = windows.to(device)
        if windows.dim() == 2:
            windows = windows.unsqueeze(-1)

        target_normal = torch.randn_like(windows)
        t = torch.rand(windows.size(0), device=device)

        path_sample = prob_path.sample(x_0=windows, x_1=target_normal, t=t)
        pred_velocity = model(path_sample.x_t, t)

        loss = torch.mean((pred_velocity - path_sample.dx_t) ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({"fm_loss": f"{loss.item():.4f}"})

    pbar.close()
    return total_loss / max(num_batches, 1)


def evaluate_flow(model, test_loader, test_dataset, device, ode_step_size, ode_method, score_method, score_topk):
    """
    Evaluate via ODE integration from data (t=0) to the normal target (t=1).
    Anomaly score: mean negative log-likelihood under N(0,1) of the terminal state.
    """
    model.eval()
    solver = ODESolver(model)

    T = test_dataset.test.shape[0]
    win_size = test_dataset.win_size
    step = test_dataset.step

    time_grid = torch.tensor([0.0, 1.0], device=device)

    # First pass: collect terminal latents on CPU
    terminals = []
    with torch.no_grad():
        for windows, _ in tqdm(test_loader, desc="Testing (collect)", leave=False, dynamic_ncols=True):
            windows = windows.to(device)
            if windows.dim() == 2:
                windows = windows.unsqueeze(-1)
            terminal = solver.sample(
                x_init=windows,
                step_size=ode_step_size,
                method=ode_method,
                time_grid=time_grid,
                return_intermediates=False,
            )
            terminals.append(terminal.cpu())

    latent_stats = None
    if score_method == "mahalanobis":
        flat = torch.cat([t.reshape(-1, t.shape[-1]) for t in terminals], dim=0)
        mu = flat.mean(dim=0)
        xm = flat - mu
        cov = (xm.T @ xm) / max(flat.shape[0] - 1, 1)
        cov = cov + 1e-6 * torch.eye(cov.shape[0])
        cov_inv = torch.linalg.pinv(cov)
        latent_stats = {"mean": mu, "cov_inv": cov_inv}

    point_err_sum = np.zeros(T, dtype=np.float64)
    point_err_count = np.zeros(T, dtype=np.float64)
    window_idx = 0

    # Second pass: score and aggregate
    for terminal in tqdm(terminals, desc="Testing (score)", leave=False, dynamic_ncols=True):
        scores = compute_anomaly_scores(
            terminal=terminal,
            method=score_method,
            k=score_topk,
            latent_stats=latent_stats,
        )
        scores_np = scores.numpy()
        batch_size = scores_np.shape[0]

        for b in range(batch_size):
            idx = window_idx + b
            start = idx * step
            end = start + win_size

            if start >= T:
                continue

            if end > T:
                end = T
                err_window = scores_np[b, : end - start]
            else:
                err_window = scores_np[b]

            point_err_sum[start:end] += err_window
            point_err_count[start:end] += 1

        window_idx += batch_size

    point_scores = np.zeros(T, dtype=np.float64)
    valid_mask = point_err_count > 0
    point_scores[valid_mask] = point_err_sum[valid_mask] / point_err_count[valid_mask]

    raw_labels = test_dataset.test_labels
    if raw_labels.ndim == 2:
        gt = (raw_labels > 0).any(axis=-1).astype(int)
    elif raw_labels.ndim == 1:
        gt = raw_labels.astype(int)
    else:
        raise ValueError(f"Unexpected label shape: {raw_labels.shape}")

    best_pointwise = sweep_threshold(point_scores, gt, use_point_adjustment=False)
    best_pa = sweep_threshold(point_scores, gt, use_point_adjustment=True)

    return best_pointwise, best_pa


def main():
    args = parse_args()
    set_seed(args.seed, deterministic=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    (
        train_dataset,
        test_dataset,
        train_loader,
        test_loader,
        data_source,
        input_dim,
    ) = build_dataloaders(args)

    print(
        f"Dataset '{args.dataset}' loaded from {data_source}. "
        f"Train windows: {len(train_dataset)}, Test windows: {len(test_dataset)} "
        f"| Timeline length: {test_dataset.test.shape[0]}"
    )
    # Persist input_dim so checkpoints can rebuild the model correctly.
    args.input_dim = input_dim
    if args.dataset == "ucr":
        print(
            f"UCR id: {args.ucr_id} | train=UCR_train (normal only if train labels exist) | test=UCR_test"
        )

    model = FlowMatchingTransformer(
        input_dim=input_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        max_len=args.win_size,
        latent_dim=args.latent_dim,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    prob_path = CondOTProbPath()

    if args.dataset == "smd":
        subdir = "smd"
    elif args.dataset == "ucr":
        subdir = f"UCR_{args.ucr_id}"
    elif args.dataset == "gesture2d":
        subdir = "gesture2d"
    elif args.dataset == "pd":
        subdir = "pd"
    elif args.dataset == "ecg":
        subdir = f"ECG_{args.ecg_id}"
    else:
        subdir = args.dataset
    os.makedirs(os.path.join(args.output_dir, subdir), exist_ok=True)
    base_name, ext = os.path.splitext(args.save_name)
    if not ext:
        ext = ".pth"
    save_path = os.path.join(args.output_dir, subdir, base_name + ext)

    print("Starting flow matching training...")
    for epoch in range(1, args.epochs + 1):
        avg_loss = train_epoch_flow(
            model, prob_path, train_loader, optimizer, device, epoch, args.epochs
        )
        print(f"Epoch {epoch}/{args.epochs} | Train FM loss: {avg_loss:.4f}")

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            best_point, best_pa = evaluate_flow(
                model,
                test_loader,
                test_dataset,
                device,
                args.ode_step_size,
                args.ode_method,
                args.score_method,
                args.score_topk,
            )

            if best_point:
                acc, pre, rec, f1, conf = (
                    best_point["acc"],
                    best_point["pre"],
                    best_point["rec"],
                    best_point["f1"],
                    best_point["conf"],
                )
                tp, fp, fn, tn = conf
                print(
                    "[Flow-Eval-Pointwise] "
                    f"F1={f1:.4f}, Acc={acc:.4f}, Pre={pre:.4f}, Rec={rec:.4f} "
                    f"@ thr={best_point['thr']:.6f} | "
                    f"TP={tp}, FP={fp}, FN={fn}, TN={tn} | "
                    f"GT pos: {tp+fn} | Pred pos: {best_point['pred_sum']}"
                )

            if best_pa:
                acc, pre, rec, f1, conf = (
                    best_pa["acc"],
                    best_pa["pre"],
                    best_pa["rec"],
                    best_pa["f1"],
                    best_pa["conf"],
                )
                tp, fp, fn, tn = conf
                print(
                    "[Flow-Eval-PA]       "
                    f"F1={f1:.4f}, Acc={acc:.4f}, Pre={pre:.4f}, Rec={rec:.4f} "
                    f"@ thr={best_pa['thr']:.6f} | "
                    f"TP={tp}, FP={fp}, FN={fn}, TN={tn} | "
                    f"GT pos: {tp+fn} | Pred pos: {best_pa['pred_sum']}"
                )

        checkpoint_path = os.path.join(
            args.output_dir, subdir, f"{base_name}_epoch{epoch}{ext}"
        )
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
            },
            checkpoint_path,
        )
        print(f"Saved checkpoint: {checkpoint_path}")

    torch.save(
        {
            "model": model.state_dict(),
            "args": vars(args),
        },
        save_path,
    )
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
