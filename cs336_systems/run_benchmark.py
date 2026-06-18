"""
Perform basic end2end benchmarking of the forward pass, backward pass, and optimizer step.

Note: Single GPU

"""
import argparse
import torch
import numpy as np
import random

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=10_000)
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--rope_theta", type=float, default=10_000.0)
    parser.add_argument("--batch_size", type=int, default=4)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta_1", type=float, default=0.9)
    parser.add_argument("--beta_2", type=float, default=0.99)
    parser.add_argument("--eps", type=float, default=1e-8)

    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    set_seed(args.seed)

    # Assign device intelligently
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

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

    # Forward path
    test_out = model(test_batch)


if __name__ == "__main__":
    main()

    print("Completed!")