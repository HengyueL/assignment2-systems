"""
Benchmark the (single-head) scaled dot-product attention forward + backward pass.

Sweeps the cartesian product of d_model x sequence_length at a fixed batch size,
timing the forward and backward passes and recording how much CUDA memory is in
use right before the backward pass starts. Each config is benchmarked both in
eager mode and with torch.compile so the two can be compared side by side. Large
configurations are expected to run out of memory; those are caught and reported
rather than crashing the sweep.

Note: Single GPU, single-head attention (no multi-head reshaping).
"""
import argparse
import itertools
import random
import timeit

import numpy as np
import torch

from cs336_basics.model import scaled_dot_product_attention

BATCH_SIZE = 8
D_MODELS = [16, 32, 64, 128]
SEQ_LENS = [256, 1024, 4096, 8192, 16384]


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--benchmark_steps", type=int, default=100)
    parser.add_argument(
        "--mem-snapshot",
        action="store_true",
        help="Record a PyTorch CUDA memory history over one forward+backward of "
             "a single config (set via --d_model/--seq_len) and dump a pickle for "
             "https://pytorch.org/memory_viz. Skips the timing sweep.",
    )
    parser.add_argument("--d_model", type=int, default=64,
                        help="d_model to profile when --mem-snapshot is set.")
    parser.add_argument("--seq_len", type=int, default=4096,
                        help="Sequence length to profile when --mem-snapshot is set.")
    parser.add_argument(
        "--compile",
        action="store_true",
        help="In --mem-snapshot mode, profile the torch.compile'd attention "
             "instead of eager. The timing sweep always reports both.",
    )
    return parser.parse_args()


def record_mem_snapshot(attn_fn, d_model, seq_len, device, out_path):
    """Record a CUDA memory history over one forward+backward of a single
    attention config and dump a pickle for https://pytorch.org/memory_viz."""
    Q = torch.randn(BATCH_SIZE, seq_len, d_model, device=device, requires_grad=True)
    K = torch.randn(BATCH_SIZE, seq_len, d_model, device=device, requires_grad=True)
    V = torch.randn(BATCH_SIZE, seq_len, d_model, device=device, requires_grad=True)

    # Warmup once so allocator caches (and any torch.compile artifacts) are
    # established before we start recording.
    o = attn_fn(Q, K, V)
    o.sum().backward()
    Q.grad = K.grad = V.grad = None
    torch.cuda.synchronize()

    torch.cuda.memory._record_memory_history(max_entries=1_000_000)
    o = attn_fn(Q, K, V)
    o.sum().backward()
    torch.cuda.synchronize()
    torch.cuda.memory._dump_snapshot(out_path)
    torch.cuda.memory._record_memory_history(enabled=None)
    print(f"Saved {out_path}. Load it at https://pytorch.org/memory_viz")


def benchmark_one(attn_fn, d_model, seq_len, device, warmup_steps, benchmark_steps):
    """Benchmark a single (d_model, seq_len) configuration with `attn_fn`.

    `attn_fn` is the attention callable (eager or torch.compile'd). Returns a
    dict with forward/backward timings (seconds) and the memory in use (bytes)
    before the backward pass.
    """
    # Fresh random inputs of shape (batch, seq_len, d_model). d_model doubles as
    # the head dimension here since we use plain single-head attention.
    Q = torch.randn(BATCH_SIZE, seq_len, d_model, device=device, requires_grad=True)
    K = torch.randn(BATCH_SIZE, seq_len, d_model, device=device, requires_grad=True)
    V = torch.randn(BATCH_SIZE, seq_len, d_model, device=device, requires_grad=True)

    # Warmup: full forward + backward so both paths are compiled/cached. For the
    # torch.compile'd fn this is where compilation actually happens.
    for _ in range(warmup_steps):
        o = attn_fn(Q, K, V)
        o.sum().backward()
        Q.grad = K.grad = V.grad = None
    torch.cuda.synchronize()

    # --- Time forward passes (same input each time) ---------------------------
    fwd_times = []
    for _ in range(benchmark_steps):
        t0 = timeit.default_timer()
        o = attn_fn(Q, K, V)
        torch.cuda.synchronize()
        fwd_times.append(timeit.default_timer() - t0)

    # --- Memory in use right before the backward pass starts ------------------
    # `o` above still holds the forward graph (saved tensors), so this captures
    # the memory footprint going into backward.
    mem_before_backward = torch.cuda.memory_allocated(device)

    # --- Time backward passes ------------------------------------------------
    # Each backward consumes the graph, so we rebuild it with an (untimed)
    # forward and time only the backward call.
    bwd_times = []
    for _ in range(benchmark_steps):
        o = attn_fn(Q, K, V)
        loss = o.sum()
        torch.cuda.synchronize()
        t0 = timeit.default_timer()
        loss.backward()
        torch.cuda.synchronize()
        bwd_times.append(timeit.default_timer() - t0)
        Q.grad = K.grad = V.grad = None

    return {
        "oom": False,
        "fwd": np.array(fwd_times),
        "bwd": np.array(bwd_times),
        "mem": mem_before_backward,
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA device.")
    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Batch size: {BATCH_SIZE} | warmup: {args.warmup_steps} | "
          f"benchmark steps: {args.benchmark_steps}\n")

    # Memory-snapshot mode: profile a single config and skip the timing sweep.
    if args.mem_snapshot:
        if args.compile:
            torch._dynamo.reset()
            attn_fn = torch.compile(scaled_dot_product_attention)
            tag = "compiled"
        else:
            attn_fn = scaled_dot_product_attention
            tag = "eager"
        out_path = f"attn_mem_d{args.d_model}_s{args.seq_len}_{tag}.pickle"
        print(f"Recording memory history ({tag}) for d_model={args.d_model}, "
              f"seq_len={args.seq_len} -> {out_path}")
        record_mem_snapshot(attn_fn, args.d_model, args.seq_len, device, out_path)
        return

    header = (f"{'d_model':>8} {'seq_len':>8} {'mode':>9} "
              f"{'fwd (ms)':>12} {'bwd (ms)':>12} {'mem (MB)':>12}")
    print(header)
    print("-" * len(header))

    for d_model, seq_len in itertools.product(D_MODELS, SEQ_LENS):
        # Eager baseline plus a freshly-compiled variant. Reset Dynamo before
        # compiling so each config gets a clean compile rather than counting
        # against the per-code-object recompile cache limit (default 8), which
        # 20 distinct shapes would otherwise blow past and silently fall back.
        torch._dynamo.reset()
        modes = (
            ("eager", scaled_dot_product_attention),
            ("compiled", torch.compile(scaled_dot_product_attention)),
        )
        for label, attn_fn in modes:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            try:
                r = benchmark_one(attn_fn, d_model, seq_len, device,
                                  args.warmup_steps, args.benchmark_steps)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"{d_model:>8} {seq_len:>8} {label:>9} "
                      f"{'OOM':>12} {'OOM':>12} {'OOM':>12}")
                continue

            fwd_ms = r["fwd"].mean() * 1e3
            bwd_ms = r["bwd"].mean() * 1e3
            mem_mb = r["mem"] / (1024 ** 2)
            print(f"{d_model:>8} {seq_len:>8} {label:>9} "
                  f"{fwd_ms:>12.3f} {bwd_ms:>12.3f} {mem_mb:>12.1f}")


if __name__ == "__main__":
    main()
    print("\nCompleted!")
