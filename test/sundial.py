import argparse
import torch
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
from transformers import AutoModelForCausalLM

# Add parent directory to path to import model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model.dataset.air_passengers import AirPassengersDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Sundial Time Series Forecast")
    parser.add_argument("--output", type=str, default="sundial_forecast.png", help="Output image path")
    parser.add_argument("--context-len", type=int, default=128, help="Context length (lookback)")
    parser.add_argument("--forecast-len", type=int, default=24, help="Forecast length")
    parser.add_argument("--dataset-name", type=str, default="air_passengers", help="Name of the dataset")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Dataset
    data_root = os.path.join(os.path.dirname(__file__), '..', 'data')
    # Use full dataset to get enough context
    dataset = AirPassengersDataset(context_len=args.context_len, train=None, data_root=data_root, name=args.dataset_name)
    
    # Get the last sequence for context
    # The dataset returns (context, target), but here we want a longer context if possible
    # Let's just grab the raw data from the dataset object
    full_data = dataset.data.numpy().flatten() # (N,)
    
    # Ensure we have enough data
    total_len = len(full_data)
    if total_len < args.context_len + args.forecast_len:
        print(f"Warning: Dataset length ({total_len}) is shorter than context + forecast ({args.context_len + args.forecast_len}).")
        # Just use what we have, might crash if too short
    
    # Select context: use the data up to the last 'forecast_len' points as context (or as much as possible)
    # We want to predict the last 'forecast_len' points to compare with ground truth
    split_idx = total_len - args.forecast_len
    
    # If context_len is larger than available history, clip it
    actual_context_len = min(args.context_len, split_idx)
    start_idx = split_idx - actual_context_len
    
    context_vals = full_data[start_idx:split_idx]
    ground_truth = full_data[split_idx:]
    
    print(f"Context length: {len(context_vals)}")
    print(f"Ground truth length: {len(ground_truth)}")
    
    # Prepare input for Sundial
    # Model expects (batch_size, lookback_length)
    # We need to reshape to (1, lookback_length)
    seqs = torch.tensor(context_vals, dtype=torch.float32).unsqueeze(0).to(device) # (1, L)
    
    # Load Model
    print("Loading Sundial model...")
    model = AutoModelForCausalLM.from_pretrained('thuml/sundial-base-128m', trust_remote_code=True).to(device)
    
    # Generate
    print(f"Generating {args.forecast_len} steps with {args.num_samples} samples...")
    # output shape: (batch_size, num_samples, forecast_length) ? or (batch_size, forecast_length, num_samples)?
    # The snippet says: output = model.generate(seqs, max_new_tokens=forecast_length, num_samples=num_samples)
    # Let's assume the output is the forecast.
    
    with torch.no_grad():
        output = model.generate(seqs, max_new_tokens=args.forecast_len, num_samples=args.num_samples, use_cache=False)
    
    print(f"Output shape: {output.shape}")
    
    # Process output
    # Assuming output is (batch_size, num_samples, forecast_length) based on typical probabilistic forecasting
    # Or maybe (batch_size, forecast_length) if num_samples=1?
    # Snippet says: "use raw predictions for mean/quantiles/confidence-interval estimation"
    
    predictions = output.cpu().numpy() # (1, num_samples, forecast_len)
    
    # Calculate mean and confidence intervals
    pred_mean = np.mean(predictions, axis=1).flatten() # (forecast_len,)
    pred_std = np.std(predictions, axis=1).flatten()
    
    # Visualization
    plt.figure(figsize=(10, 6))
    
    # Time steps
    time_ctx = np.arange(len(context_vals))
    time_future = np.arange(len(context_vals) - 1, len(context_vals) + len(ground_truth))
    
    # Plot Context
    plt.plot(time_ctx, context_vals, 'k-', label='Context')
    
    # Connect lines
    last_ctx_val = context_vals[-1]
    
    # Plot Ground Truth
    gt_plot = [last_ctx_val] + ground_truth.tolist()
    plt.plot(time_future, gt_plot, 'g-', label='Ground Truth')
    
    # Plot Forecast (Mean)
    pred_plot = [last_ctx_val] + pred_mean.tolist()
    plt.plot(time_future, pred_plot, 'b--', label='Forecast (Sundial Mean)')
    
    # Plot Uncertainty (if num_samples > 1)
    if args.num_samples > 1:
        # We need to align uncertainty with the plot. 
        # The fill_between should start from the last context point (uncertainty 0) or just the future.
        # Let's plot it for the future steps.
        time_pred = np.arange(len(context_vals), len(context_vals) + len(ground_truth))
        plt.fill_between(time_pred, pred_mean - 2*pred_std, pred_mean + 2*pred_std, color='b', alpha=0.2, label='95% Confidence')

    plt.title("Sundial Time Series Forecast")
    plt.xlabel("Time Step")
    plt.ylabel("Normalized Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(args.output)
    print(f"Plot saved to {args.output}")

if __name__ == "__main__":
    main()
