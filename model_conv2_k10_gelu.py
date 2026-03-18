import torch  # Core PyTorch library for tensor operations, autograd, and GPU support
import torch.nn as nn  # Neural network module providing layers, loss functions, and model base classes


class ConvModel(nn.Module):
    """PyTorch equivalent of the Flax ConvModel in chapter3.py.

    Input shape:  (batch, seq_len=200, channels=4)  — same as the Flax version.
    The sequences are transposed internally to (batch, 4, 200) for Conv1d.
    After two conv+pool blocks: (batch, 64, 50) -> flatten -> (batch, 3200).
    """

    def __init__(self, conv_filters: int = 64, kernel_size: int = 10, dense_units: int = 128):
        # conv_filters: number of output channels (feature maps) for each convolutional layer (default 64)
        # kernel_size: width of the 1D convolutional sliding window in base pairs (default 10)
        # dense_units: number of neurons in the first fully connected hidden layer (default 128)

        super().__init__()  # Initialize the parent nn.Module class (required for proper PyTorch model setup)

        # First 1D convolution: takes 4 input channels (one per nucleotide: A, C, G, T)
        # and produces conv_filters (64) output feature maps.
        # padding="same" keeps the sequence length unchanged after convolution (200 -> 200).
        self.conv1 = nn.Conv1d(4, conv_filters, kernel_size=kernel_size, padding="same")

        # Second 1D convolution: takes conv_filters (64) channels from the first conv layer
        # and outputs the same number of feature maps (64). Detects higher-level motif patterns.
        # padding="same" again preserves sequence length (100 -> 100 at this point, after first pool).
        self.conv2 = nn.Conv1d(conv_filters, conv_filters, kernel_size=kernel_size, padding="same")

        # Max pooling layer: reduces sequence length by half by taking the max value
        # in each non-overlapping window of size 2. This downsamples the spatial dimension,
        # retaining the strongest activations and providing translational invariance.
        self.pool = nn.MaxPool1d(kernel_size=2)

        # GELU (Gaussian Error Linear Unit) activation function.
        # A smooth, non-monotonic activation that outperforms ReLU in many tasks.
        # It gently suppresses small negative values instead of zeroing them out.
        self.gelu = nn.GELU()

        # Calculate flattened feature dimension after the two conv+pool blocks:
        # Starting seq_len=200 -> pool1 -> 100 -> pool2 -> 50
        # With conv_filters=64 channels: 64 * 50 = 3200 total features after flattening.
        flat_dim = conv_filters * 50

        # First fully connected (dense) layer: maps the 3200-dim flattened feature vector
        # down to dense_units (128) neurons. This layer learns global patterns across
        # all positions and feature maps combined.
        self.fc1 = nn.Linear(flat_dim, dense_units)

        # Second fully connected layer: further reduces dimensionality from 128 to 64 neurons.
        # Acts as a bottleneck, forcing the network to learn a compact representation.
        self.fc2 = nn.Linear(dense_units, dense_units // 2)

        # Output layer: maps from 64 neurons to a single scalar output.
        # No activation is applied here — raw logit output suitable for
        # binary classification with BCEWithLogitsLoss (which applies sigmoid internally).
        self.fc3 = nn.Linear(dense_units // 2, 1)

    def model_id(self) -> str:
        # Returns a unique string identifier for this model variant.
        return "conv2_k10_gelu"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x arrives as (batch, seq_len=200, 4) — each row is a one-hot encoded DNA sequence
        # where the 4 channels represent the nucleotides A, C, G, T.

        # Transpose to (batch, 4, seq_len=200) because PyTorch Conv1d expects
        # the channel dimension before the spatial (sequence) dimension.
        # .float() ensures the tensor is float32 (needed if input was int or float64).
        x = x.permute(0, 2, 1).float()

        # Block 1: Apply first convolution (detects low-level sequence motifs like
        # transcription factor binding sites), then GELU activation for non-linearity,
        # then max pool to halve the sequence: (batch, 64, 200) -> (batch, 64, 100).
        x = self.pool(self.gelu(self.conv1(x)))

        # Block 2: Apply second convolution (detects higher-order combinations of motifs),
        # GELU activation, then max pool again: (batch, 64, 100) -> (batch, 64, 50).
        x = self.pool(self.gelu(self.conv2(x)))

        # Flatten all dimensions except the batch dimension into a single vector:
        # (batch, 64, 50) -> (batch, 3200). This converts the spatial feature maps
        # into a flat vector that can be fed into fully connected layers.
        x = x.flatten(start_dim=1)

        # Pass through first dense layer + GELU activation:
        # (batch, 3200) -> (batch, 128). Learns non-linear combinations of all features.
        x = self.gelu(self.fc1(x))

        # Pass through second dense layer + GELU activation:
        # (batch, 128) -> (batch, 64). Further compresses the learned representation.
        x = self.gelu(self.fc2(x))

        # Final linear projection to a single output logit per sample:
        # (batch, 64) -> (batch, 1). Returns raw logits (not probabilities).
        # Apply sigmoid externally or use BCEWithLogitsLoss for training.
        return self.fc3(x)
