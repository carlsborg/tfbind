import torch  # Core PyTorch library for tensor operations, autograd, and GPU support
import torch.nn as nn  # Neural network module providing layers, loss functions, and model base classes


# ---------------------------------------------------------------------------
# VARIATION: Activation — LeakyReLU instead of GELU
# ---------------------------------------------------------------------------
class ConvModelLeakyReLU(nn.Module):
    """LeakyReLU activation variant.

    Like ReLU but allows a small negative slope (default 0.01) for negative inputs.
    This prevents the "dying neuron" problem where units permanently output zero.
    Still cheap to compute like ReLU.
    """

    def __init__(self, conv_filters: int = 64, kernel_size: int = 10, dense_units: int = 128):
        super().__init__()
        self.conv1 = nn.Conv1d(4, conv_filters, kernel_size=kernel_size, padding="same")
        self.conv2 = nn.Conv1d(conv_filters, conv_filters, kernel_size=kernel_size, padding="same")
        self.pool = nn.MaxPool1d(kernel_size=2)
        # LeakyReLU with default negative_slope=0.01 replaces GELU.
        self.act = nn.LeakyReLU()
        flat_dim = conv_filters * 50
        self.fc1 = nn.Linear(flat_dim, dense_units)
        self.fc2 = nn.Linear(dense_units, dense_units // 2)
        self.fc3 = nn.Linear(dense_units // 2, 1)

    def model_id(self) -> str:
        return "conv2_k10_leakyrelu"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1).float()
        x = self.pool(self.act(self.conv1(x)))
        x = self.pool(self.act(self.conv2(x)))
        x = x.flatten(start_dim=1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.fc3(x)
