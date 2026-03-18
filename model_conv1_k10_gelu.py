import torch  # Core PyTorch library for tensor operations, autograd, and GPU support
import torch.nn as nn  # Neural network module providing layers, loss functions, and model base classes


# ---------------------------------------------------------------------------
# VARIATION: Layer count — single conv layer instead of two
# ---------------------------------------------------------------------------
class ConvModel1Layer(nn.Module):
    """Single conv+pool block variant.

    One conv layer means fewer parameters and faster training,
    but can only detect simple, short-range motifs directly.
    Shape: (batch, 4, 200) -> conv+pool -> (batch, 64, 100) -> flatten -> (batch, 6400).
    """

    def __init__(self, conv_filters: int = 64, kernel_size: int = 10, dense_units: int = 128):
        super().__init__()
        # Single convolutional layer: 4 input channels -> conv_filters output channels.
        self.conv1 = nn.Conv1d(4, conv_filters, kernel_size=kernel_size, padding="same")
        # Only one pool step: 200 -> 100.
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.gelu = nn.GELU()
        # Flattened dimension: 64 channels * 100 positions = 6400.
        flat_dim = conv_filters * 100
        self.fc1 = nn.Linear(flat_dim, dense_units)
        self.fc2 = nn.Linear(dense_units, dense_units // 2)
        self.fc3 = nn.Linear(dense_units // 2, 1)

    def model_id(self) -> str:
        return "conv1_k10_gelu"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1).float()
        # Single conv+pool block: (batch, 4, 200) -> (batch, 64, 100).
        x = self.pool(self.gelu(self.conv1(x)))
        x = x.flatten(start_dim=1)
        x = self.gelu(self.fc1(x))
        x = self.gelu(self.fc2(x))
        return self.fc3(x)
