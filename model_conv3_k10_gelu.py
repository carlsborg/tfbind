import torch  # Core PyTorch library for tensor operations, autograd, and GPU support
import torch.nn as nn  # Neural network module providing layers, loss functions, and model base classes


# ---------------------------------------------------------------------------
# VARIATION: Layer count — three conv layers instead of two
# ---------------------------------------------------------------------------
class ConvModel3Layer(nn.Module):
    """Three conv+pool block variant.

    Three stacked conv layers can learn hierarchical motif combinations
    at the cost of more parameters and a smaller spatial dimension before flattening.
    Shape: 200 -> pool -> 100 -> pool -> 50 -> pool -> 25; flatten -> (batch, 64*25=1600).
    """

    def __init__(self, conv_filters: int = 64, kernel_size: int = 10, dense_units: int = 128):
        super().__init__()
        self.conv1 = nn.Conv1d(4, conv_filters, kernel_size=kernel_size, padding="same")
        self.conv2 = nn.Conv1d(conv_filters, conv_filters, kernel_size=kernel_size, padding="same")
        # Third convolutional layer: captures even higher-order motif interactions.
        self.conv3 = nn.Conv1d(conv_filters, conv_filters, kernel_size=kernel_size, padding="same")
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.gelu = nn.GELU()
        # Three pool steps: 200 -> 100 -> 50 -> 25; flattened: 64 * 25 = 1600.
        flat_dim = conv_filters * 25
        self.fc1 = nn.Linear(flat_dim, dense_units)
        self.fc2 = nn.Linear(dense_units, dense_units // 2)
        self.fc3 = nn.Linear(dense_units // 2, 1)

    def model_id(self) -> str:
        return "conv3_k10_gelu"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1).float()
        x = self.pool(self.gelu(self.conv1(x)))
        x = self.pool(self.gelu(self.conv2(x)))
        # Third conv+pool block: (batch, 64, 50) -> (batch, 64, 25).
        x = self.pool(self.gelu(self.conv3(x)))
        x = x.flatten(start_dim=1)
        x = self.gelu(self.fc1(x))
        x = self.gelu(self.fc2(x))
        return self.fc3(x)
