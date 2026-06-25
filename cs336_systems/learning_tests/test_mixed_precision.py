import torch
import torch.nn as nn


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()
    def forward(self, x):
        print("Input: ", x.dtype)

        x = self.fc1(x)
        print("FC1 output: ", x.dtype)

        x = self.relu(x)
        print("Relu: ", x.dtype)

        x = self.ln(x)
        print("LN: ", x.dtype)

        x = self.fc2(x)
        print("FC2: ", x.dtype)
        return x


if __name__ == "__main__":
    assert torch.cuda.is_available(), "Cuda is required for this test"

    device = torch.device("cuda")
    default_dtype = torch.float32
    in_feature, out_feature = 10, 3
    model = ToyModel(
        in_features=in_feature, out_features=out_feature
    ).to(device=device, dtype=default_dtype)

    test_input = torch.randn(
        size=(5, in_feature)
    ).to(device=device, dtype=default_dtype)

    test_output = torch.randn(
        size=(5, out_feature)
    ).to(device=device, dtype=default_dtype)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda")
    cast_dtype = torch.bfloat16

    with torch.autocast(device_type="cuda", dtype=cast_dtype):
        y = model(test_input)
        loss = ((y - test_output) ** 2).mean()
        # Loss dtype is decided here (note LayerNorm/reductions stay in fp32).
        print("\nLoss: ", loss.dtype)

    # Scale the loss to avoid fp16 gradient underflow, then backpropagate.
    scaler.scale(loss).backward()

    print("\nParameters dtype: ")
    for name, param in model.named_parameters():
        print(f"{name}: {param.dtype}")

    print("\nGradient dtype: ")
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name}: {param.grad.dtype}")

    # Unscale + step + update completes one mixed-precision optimizer iteration.
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
