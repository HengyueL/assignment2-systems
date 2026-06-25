"""
Perform basic end2end benchmarking of the forward pass, backward pass, and optimizer step.

Note: Single GPU

"""
import argparse
import contextlib
import torch
import numpy as np
import random
import timeit

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def sync_clock(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=10_000)
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--d_ff", type=int, default=3072)
    parser.add_argument("--rope_theta", type=float, default=10_000.0)
    parser.add_argument("--batch_size", type=int, default=4)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta_1", type=float, default=0.9)
    parser.add_argument("--beta_2", type=float, default=0.99)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=1e-2)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--benchmark_steps", type=int, default=10)
    parser.add_argument(
        "--mem-snapshot",
        action="store_true",
        help="Record a PyTorch CUDA memory history for one step and dump a "
             "pickle for https://pytorch.org/memory_viz. Skips the timing loop.",
    )
    parser.add_argument(
        "--nvtx-profile",
        action="store_true",
        help="Run one step wrapped in torch.autograd.profiler.emit_nvtx() and "
             "cudaProfilerApi start/stop so Nsight Systems can capture per-op "
             "NVTX labels + memory. Pair with `nsys profile "
             "--capture-range=cudaProfilerApi`. Skips the timing loop.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp32",
        choices=["fp32", "bf16", "fp16"],
        help="Precision used for the forward pass. 'fp32' disables autocast; "
             "'bf16'/'fp16' wrap the forward pass in torch.autocast.",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    set_seed(args.seed)

    # Assign device intelligently
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Resolve the autocast dtype. fp32 => autocast disabled (no-op).
    autocast_dtype = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[args.dtype]
    use_amp = args.dtype in ("bf16", "fp16")
    print(f"Device: {device.type}  |  precision: {args.dtype}")

    # Generate a random batch of data (batch, n_seq)
    test_batch = torch.randint(
        low=0, high=args.vocab_size,
        size=(args.batch_size, args.context_length)
    )

    # Move Data to device
    test_batch = test_batch.to(device=device)

    # Instantiate a model
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    )

    # Move model to device
    model = model.to(device=device)

    optimizer = AdamW(
        params=model.parameters(),
        lr=args.lr,
        betas=(args.beta_1, args.beta_2),
        eps=args.eps,
        weight_decay=args.weight_decay
    )

    # GradScaler only matters for fp16 (to avoid gradient underflow); it is a
    # no-op when disabled, so fp32/bf16 paths flow through unchanged.
    scaler = torch.amp.GradScaler(device.type, enabled=(args.dtype == "fp16"))

    def nvtx_range(label):
        """Context manager for NVTX ranges (no-op on non-CUDA devices)."""
        import contextlib
        if device.type == "cuda":
            return torch.cuda.nvtx.range(label)
        return contextlib.nullcontext()

    def run_step(tag="step"):
        with nvtx_range(tag):
            optimizer.zero_grad()

            with nvtx_range(f"{tag}/forward"):
                sync_clock(device)
                t0 = timeit.default_timer()
                with torch.autocast(
                    device_type=device.type,
                    dtype=autocast_dtype,
                    enabled=use_amp,
                ):
                    out = model(test_batch)
                    loss = out.mean()
                sync_clock(device)
                t_fwd = timeit.default_timer() - t0

            with nvtx_range(f"{tag}/backward"):
                sync_clock(device)
                t1 = timeit.default_timer()
                scaler.scale(loss).backward()
                sync_clock(device)
                t_bwd = timeit.default_timer() - t1

            with nvtx_range(f"{tag}/optimizer"):
                sync_clock(device)
                t2 = timeit.default_timer()
                scaler.step(optimizer)
                scaler.update()
                sync_clock(device)
                t_opt = timeit.default_timer() - t2

        return t_fwd, t_bwd, t_opt

    is_cuda = device.type == "cuda"

    # Warmup
    print(f"Running {args.warmup_steps} warmup step(s)...")
    for i in range(args.warmup_steps):
        run_step(tag=f"warmup/{i}")

    # --- Dedicated profiling runs (skip the timing loop) -----------------------
    # PyTorch native memory snapshot over ONE full step (fwd+bwd+opt) recorded
    # continuously, then dumped as a single pickle for https://pytorch.org/memory_viz.
    # This is a *different* tool from Nsight/NVTX.
    if args.mem_snapshot:
        if not is_cuda:
            raise RuntimeError("--mem-snapshot requires a CUDA device.")
        out_path = f"memory_snapshot_{args.dtype}.pickle"
        print(f"Recording memory history for one step -> {out_path}")
        torch.cuda.memory._record_memory_history(max_entries=1_000_000)
        run_step(tag="mem_snapshot")
        torch.cuda.synchronize()
        torch.cuda.memory._dump_snapshot(out_path)
        torch.cuda.memory._record_memory_history(enabled=None)
        print(f"Saved {out_path}. Load it at https://pytorch.org/memory_viz")
        return

    # Single step annotated for Nsight Systems. emit_nvtx() adds PyTorch's
    # per-op NVTX labels (backward ops are tagged with the forward op that
    # produced them), and cudaProfilerApi start/stop bounds the capture so
    # `nsys profile --capture-range=cudaProfilerApi` records exactly this step.
    if args.nvtx_profile:
        print("Running one NVTX-annotated step for nsys capture...")
        sync_clock(device)
        nvtx_ctx = (
            torch.autograd.profiler.emit_nvtx() if is_cuda else contextlib.nullcontext()
        )
        with nvtx_ctx:
            if is_cuda:
                torch.cuda.profiler.start()
            run_step(tag="profile")
            if is_cuda:
                torch.cuda.profiler.stop()
        sync_clock(device)
        print("Done.")
        return

    # --- Timing benchmark ------------------------------------------------------
    print(f"Running {args.benchmark_steps} benchmark step(s)...")
    fwd_times, bwd_times, opt_times = [], [], []
    
    for i in range(args.benchmark_steps):
        t_fwd, t_bwd, t_opt = run_step(tag=f"benchmark/{i}")
        fwd_times.append(t_fwd)
        bwd_times.append(t_bwd)
        opt_times.append(t_opt)

    

    fwd = np.array(fwd_times)
    bwd = np.array(bwd_times)
    opt = np.array(opt_times)
    print(f"Forward pass:    mean={fwd.mean():.4f}s  std={fwd.std():.4f}s")
    print(f"Backward pass:   mean={bwd.mean():.4f}s  std={bwd.std():.4f}s")
    print(f"Optimizer step:  mean={opt.mean():.4f}s  std={opt.std():.4f}s")



if __name__ == "__main__":
    main()

    print("Completed!")