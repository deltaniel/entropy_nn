"""Test script to demonstrate EntropyAwareMLP timing debugging."""

import torch
from entropy_nn.data.mnist import make_loaders
from entropy_nn.models import MLP, MNIST_DIMS
from entropy_nn.models.mlp import EntropyAwareMLP

# Load checkpoint
print("Loading checkpoint...")
ckpt_path = "checkpoints/mnist_mlp_experiment/model_state.pt"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

baseline_model = MLP(dims=MNIST_DIMS, flatten_input=True)
baseline_model.load_state_dict(ckpt["model_state"], strict=True)

# Create EntropyAwareMLP with ANS
print("Creating EntropyAwareMLP (ANS, 8-bit)...")
entropy_ans = EntropyAwareMLP.from_trained_mlp(
    baseline_model,
    bits=8,
    coder="ans",
    dropout=0.0,
)

# Create EntropyAwareMLP with Huffman
print("Creating EntropyAwareMLP (Huffman, 8-bit)...")
entropy_huffman = EntropyAwareMLP.from_trained_mlp(
    baseline_model,
    bits=8,
    coder="huffman",
    dropout=0.0,
)

# Load test data
print("\nLoading test data...")
_, test_loader = make_loaders(data_root="./data", batch_size=256, num_workers=0)
sample_batch, _ = next(iter(test_loader))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Move models to device
entropy_ans = entropy_ans.to(device)
entropy_huffman = entropy_huffman.to(device)
sample_batch = sample_batch.to(device)

# Warmup
print("\nWarming up...")
for _ in range(5):
    _ = entropy_ans(sample_batch)
    _ = entropy_huffman(sample_batch)

# Profile ANS
print("\n" + "="*80)
print("PROFILING ANS CODER")
print("="*80)
entropy_ans.enable_debug_timing()
for _ in range(10):
    _ = entropy_ans(sample_batch)
entropy_ans.print_timing_stats()

# Profile Huffman
print("\n" + "="*80)
print("PROFILING HUFFMAN CODER")
print("="*80)
entropy_huffman.enable_debug_timing()
for _ in range(10):
    _ = entropy_huffman(sample_batch)
entropy_huffman.print_timing_stats()

# Compare summary
print("\n" + "="*80)
print("SUMMARY COMPARISON")
print("="*80)
ans_stats = entropy_ans.get_timing_stats()
huffman_stats = entropy_huffman.get_timing_stats()

ans_decode_time = sum(s["total_ms"] for k, s in ans_stats.items() if "decode" in k)
huffman_decode_time = sum(s["total_ms"] for k, s in huffman_stats.items() if "decode" in k)

ans_forward_time = sum(s["total_ms"] for k, s in ans_stats.items() if "forward" in k and "total" not in k)
huffman_forward_time = sum(s["total_ms"] for k, s in huffman_stats.items() if "forward" in k and "total" not in k)

print(f"{'Operation':<20} {'ANS (ms)':<15} {'Huffman (ms)':<15} {'Ratio':<10}")
print("-"*60)
print(f"{'Total Decode':<20} {ans_decode_time:>14.1f} {huffman_decode_time:>14.1f} {huffman_decode_time/ans_decode_time:>9.1f}x")
print(f"{'Total Forward':<20} {ans_forward_time:>14.1f} {huffman_forward_time:>14.1f} {huffman_forward_time/ans_forward_time:>9.1f}x")
print(f"{'Per-Batch (mean)':<20} {ans_stats['forward_total']['mean_ms']:>14.3f} {huffman_stats['forward_total']['mean_ms']:>14.3f} {huffman_stats['forward_total']['mean_ms']/ans_stats['forward_total']['mean_ms']:>9.1f}x")
print("="*80)
