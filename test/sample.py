import argparse
import torch
import matplotlib.pyplot as plt
import os
import sys
import numpy as np

# Add parent directory to path to import model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model.model import Transformer
from model.dataset.air_passengers import AirPassengersDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Flow Matching Time Series Forecast")
    # Default checkpoint path relative to project root
    default_ckpt = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'air_passengers', 'tiny_transformer_fm.pth')
    parser.add_argument("--checkpoint", type=str, default=default_ckpt, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="forecast_plot.png", help="Output image path")
    parser.add_argument("--context-len", type=int, default=24, help="Context length")
    parser.add_argument("--forecast-len", type=int, default=24, help="Number of steps to forecast autoregressively")
    parser.add_argument("--d-model", type=int, default=64, help="Model dimension")
    parser.add_argument("--nhead", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--steps", type=int, default=100, help="Euler integration steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dataset-name", type=str, default="air_passengers", help="Name of the dataset")
    return parser.parse_args()

def generate_one_step(model, context, device, steps=100):
    # context: (1, L, 1)
    # x_0 ~ N(0, 1)
    x_t = torch.randn(1, 1, device=device)
    dt = 1.0 / steps
    
    model.eval()
    with torch.no_grad():
        for i in range(steps):
            t = torch.tensor([i * dt], device=device)
            # v = model(x_t, t, context)
            v = model(x_t, t, context)
            x_t = x_t + v * dt
            
    return x_t

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Dataset (Test set)
    # We need the full data to pick a sequence
    # Point to project root data directory
    data_root = os.path.join(os.path.dirname(__file__), '..', 'data')
    # Use full dataset (train=None) to ensure we have enough data for context + forecast
    dataset = AirPassengersDataset(context_len=args.context_len, train=None, data_root=data_root, name=args.dataset_name)
    
    # Load Model
    model = Transformer(
        input_dim=1, 
        d_model=args.d_model, 
        nhead=args.nhead,
        num_layers=args.num_layers,
        context_len=args.context_len
    ).to(device)
    
    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded model from {args.checkpoint}")
    else:
        print(f"Checkpoint {args.checkpoint} not found! Using random weights (garbage output expected).")

    # Pick a sample
    # Let's pick the first sample from the test set
    # The dataset returns (context, target)
    # But we want to forecast multiple steps.
    # So we need access to the underlying data.
    
    # dataset.data is the full time series (normalized)
    # Test split starts at split_idx
    # Let's just grab a sequence from the dataset object
    
    # We want a start index
    start_idx = 0
    initial_context, _ = dataset[start_idx] # (L, 1)
    
    # Ground truth for the next forecast_len steps
    ground_truth = []
    for i in range(args.forecast_len):
        _, target = dataset[start_idx + i]
        ground_truth.append(target.item())
        
    # Autoregressive Generation
    current_context = initial_context.unsqueeze(0).to(device) # (1, L, 1)
    predictions = []
    
    print(f"Generating {args.forecast_len} steps...")
    for i in range(args.forecast_len):
        # Generate next step
        pred = generate_one_step(model, current_context, device, steps=args.steps)
        pred_val = pred.item()
        predictions.append(pred_val)
        
        # Update context
        # Remove first element, add prediction
        # current_context: (1, L, 1)
        # pred: (1, 1) -> (1, 1, 1)
        new_val = pred.view(1, 1, 1)
        current_context = torch.cat([current_context[:, 1:, :], new_val], dim=1)
        
    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Plot Context
    context_vals = initial_context.flatten().numpy()
    time_ctx = np.arange(len(context_vals))
    plt.plot(time_ctx, context_vals, 'k-', label='Context')
    
    # Connect lines: Prepend last context point to future data
    last_ctx_val = context_vals[-1]
    last_ctx_time = time_ctx[-1]
    
    # Ground Truth
    gt_plot = [last_ctx_val] + ground_truth
    time_future = np.arange(len(context_vals) - 1, len(context_vals) + len(ground_truth))
    plt.plot(time_future, gt_plot, 'g-', label='Ground Truth')
    
    # Predictions
    pred_plot = [last_ctx_val] + predictions
    plt.plot(time_future, pred_plot, 'r--', label='Forecast (Flow Matching)')
    
    plt.title("Flow Matching Time Series Forecast")
    plt.xlabel("Time Step")
    plt.ylabel("Normalized Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(args.output)
    print(f"Plot saved to {args.output}")

if __name__ == "__main__":
    main()
