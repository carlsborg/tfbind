import torch  # Core PyTorch library for tensor operations, autograd, and GPU support
import torch.nn as nn  # Neural network module providing layers, loss functions, and model base classes


# ---------------------------------------------------------------------------
# VARIATION: Kernel size — large kernel (20 instead of 10)
# ---------------------------------------------------------------------------
class ConvModelLargeKernel(nn.Module):
    """Large kernel (k=20) variant.

    A wider receptive field per conv layer. Can directly detect longer motifs
    and spacing patterns between sub-motifs without needing many layers.
    More parameters per conv layer than the base model.
    """

    def __init__(self, conv_filters: int = 64, kernel_size: int = 20, dense_units: int = 128):
        super().__init__()
        self.conv1 = nn.Conv1d(4, conv_filters, kernel_size=kernel_size, padding="same")
        self.conv2 = nn.Conv1d(conv_filters, conv_filters, kernel_size=kernel_size, padding="same")
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.gelu = nn.GELU()
        flat_dim = conv_filters * 50
        self.fc1 = nn.Linear(flat_dim, dense_units)
        self.fc2 = nn.Linear(dense_units, dense_units // 2)
        self.fc3 = nn.Linear(dense_units // 2, 1)

    def model_id(self) -> str:
        return "conv2_k20_gelu"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1).float()
        x = self.pool(self.gelu(self.conv1(x)))
        x = self.pool(self.gelu(self.conv2(x)))
        x = x.flatten(start_dim=1)
        x = self.gelu(self.fc1(x))
        x = self.gelu(self.fc2(x))
        return self.fc3(x)
