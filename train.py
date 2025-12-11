import argparse
import os
import random

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.dataset.smd import SMDSegLoader
from model.dataset.ucr import UCRSegLoader
from model.dataset.gesture2d import Gesture2DSegLoader
from model.dataset.pd import PDSegLoader
from model.dataset.ecg import ECGSegLoader
from model.metric import binary_classification_metrics
from model.model import Transformer


# ------------------------------
# Utils
# ------------------------------


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
        description="Simple reconstruction training for anomaly detection (SMD/UCR)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ecg",
        choices=["smd", "ucr", "gesture2d", "pd", "ecg"],
        help="Dataset to use: smd, ucr, gesture2d, pd, or ecg",
    )
    parser.add_argument(
        "--ucr-id",
        type=int,
        default=138,
        help="UCR subset id (e.g., 135/136/137/138) when using --dataset ucr",
    )
    parser.add_argument(
        "--ecg-id",
        type=str,
        default="D",
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

    parser.add_argument("--batch-size", type=int, default=64, help="Train batch size")
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

    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
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
        default="tiny_transformer_recon.pth",
        help="Checkpoint filename",
    )
    return parser.parse_args()


# ------------------------------
# Point-adjustment
# ------------------------------


def apply_point_adjustment(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """
    Point-adjustment cho time-series anomaly detection.

    pred, gt: 1D numpy array (0/1), length = T.
    Nếu bất kỳ điểm dự đoán dương trong một anomaly segment (liên tiếp gt == 1),
    ta cho cả segment đó thành 1.
    """
    assert pred.shape == gt.shape
    pred_pa = pred.copy()
    T = len(gt)

    in_seg = False
    seg_start = 0

    for t in range(T):
        if gt[t] == 1 and not in_seg:
            # bắt đầu một anomaly segment mới
            in_seg = True
            seg_start = t
        elif gt[t] == 0 and in_seg:
            # kết thúc segment tại t - 1
            seg_end = t - 1
            if pred[seg_start : seg_end + 1].any():
                pred_pa[seg_start : seg_end + 1] = 1
            in_seg = False

    # Nếu kết thúc sequence mà vẫn đang trong segment
    if in_seg:
        seg_end = T - 1
        if pred[seg_start : seg_end + 1].any():
            pred_pa[seg_start : seg_end + 1] = 1

    return pred_pa


def sweep_threshold(scores: np.ndarray, gt: np.ndarray, use_point_adjustment: bool):
    """
    Sweep threshold trên scores để tìm F1 tốt nhất.
    - scores: anomaly score per timestep, shape (T,)
    - gt: ground-truth labels (0/1), shape (T,)
    - use_point_adjustment: nếu True thì áp dụng PA trước khi tính F1.
    """
    max_score = scores.max()
    thresholds = np.linspace(0.0, max_score if max_score > 0 else 1.0, 500)

    best = None
    for thr in thresholds:
        preds = (scores > thr).astype(int)

        if use_point_adjustment:
            preds_eval = apply_point_adjustment(preds, gt)
        else:
            preds_eval = preds

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


# ------------------------------
# Training & Evaluation
# ------------------------------


def train_epoch(model, dataloader, optimizer, device, epoch, total_epochs):
    """
    Train the transformer to reconstruct the full window directly from the input.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(
        dataloader,
        desc=f"Train Epoch {epoch}/{total_epochs}",
        leave=False,
        dynamic_ncols=True,
    )

    for windows, _ in pbar:
        windows = windows.to(device)

        preds = model(windows)
        loss = torch.mean((preds - windows) ** 2)  # MSE over all timesteps & features

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({"mse": f"{loss.item():.4f}"})

    pbar.close()
    return total_loss / max(num_batches, 1)


def evaluate(model, test_loader, test_dataset, device):
    """
    - Chạy reconstruction trên toàn bộ test windows.
    - Aggregate reconstruction error về timeline gốc (point_scores, length = T).
    - Sweep threshold để lấy best F1:
        + point-wise (no PA)
        + với point-adjustment (PA)
    """
    model.eval()

    T = test_dataset.test.shape[0]        # số timestep gốc
    win_size = test_dataset.win_size
    step = test_dataset.step

    point_err_sum = np.zeros(T, dtype=np.float64)
    point_err_count = np.zeros(T, dtype=np.float64)

    window_idx = 0  # index window trong dataset (0..len(test_dataset)-1)

    with torch.no_grad():
        for windows, _ in tqdm(
            test_loader, desc="Testing", leave=False, dynamic_ncols=True
        ):
            batch_size = windows.size(0)
            windows = windows.to(device)

            preds = model(windows)
            err = (preds - windows).abs()  # (B, L, D) hoặc (B, L)

            # Reduce feature dim → (B, L)
            if err.dim() == 3:
                err = err.mean(dim=-1)

            err_np = err.cpu().numpy()  # (B, L)

            for b in range(batch_size):
                idx = window_idx + b
                start = idx * step
                end = start + win_size  # exclusive

                if start >= T:
                    continue

                if end > T:
                    end = T
                    err_window = err_np[b, : end - start]
                else:
                    err_window = err_np[b]

                point_err_sum[start:end] += err_window
                point_err_count[start:end] += 1

            window_idx += batch_size

    # Tránh chia cho 0 (nếu có timestep không nằm trong window nào)
    point_scores = np.zeros(T, dtype=np.float64)
    valid_mask = point_err_count > 0
    point_scores[valid_mask] = point_err_sum[valid_mask] / point_err_count[valid_mask]

    # Ground-truth labels per timestep (0/1)
    raw_labels = test_dataset.test_labels
    if raw_labels.ndim == 2:
        # (T, D) -> (T,), timestep là anomaly nếu bất kỳ channel nào là anomaly
        gt = (raw_labels > 0).any(axis=-1).astype(int)
    elif raw_labels.ndim == 1:
        gt = raw_labels.astype(int)
    else:
        raise ValueError(f"Unexpected label shape: {raw_labels.shape}")

    # Best point-wise (no PA)
    best_pointwise = sweep_threshold(point_scores, gt, use_point_adjustment=False)
    # Best with Point Adjustment
    best_pa = sweep_threshold(point_scores, gt, use_point_adjustment=True)

    return best_pointwise, best_pa


# ------------------------------
# Data
# ------------------------------


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

    # Lấy input_dim từ 1 sample
    sample_x, _ = train_dataset[0]  # shape (L, D)
    if sample_x.ndim == 1:
        input_dim = 1
    else:
        input_dim = sample_x.shape[-1]

    return train_dataset, test_dataset, train_loader, test_loader, data_source, input_dim


# ------------------------------
# Main
# ------------------------------


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
    if args.dataset == "ucr":
        print(
            f"UCR id: {args.ucr_id} | train=UCR_train (normal only if train labels exist) | test=UCR_test"
        )

    model = Transformer(
        input_dim=input_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        max_len=args.win_size,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # lưu ra folder theo dataset
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
    save_path = os.path.join(args.output_dir, subdir, args.save_name)

    print("Starting reconstruction training...")
    for epoch in range(1, args.epochs + 1):
        avg_loss = train_epoch(model, train_loader, optimizer, device, epoch, args.epochs)
        print(f"Epoch {epoch}/{args.epochs} | Train MSE: {avg_loss:.4f}")

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            best_point, best_pa = evaluate(model, test_loader, test_dataset, device)

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
                    "[Eval-Pointwise] "
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
                    "[Eval-PA]       "
                    f"F1={f1:.4f}, Acc={acc:.4f}, Pre={pre:.4f}, Rec={rec:.4f} "
                    f"@ thr={best_pa['thr']:.6f} | "
                    f"TP={tp}, FP={fp}, FN={fn}, TN={tn} | "
                    f"GT pos: {tp+fn} | Pred pos: {best_pa['pred_sum']}"
                )

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
