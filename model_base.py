"""Re-exports all model variants from their individual files.

Each model lives in its own module named model_{model_id}.py.
Import from here for convenience or import directly from the individual files.
"""

from model_conv2_k10_gelu import ConvModel
from model_conv1_k10_gelu import ConvModel1Layer
from model_conv3_k10_gelu import ConvModel3Layer
from model_conv2_k5_gelu import ConvModelSmallKernel
from model_conv2_k20_gelu import ConvModelLargeKernel
from model_conv2_k10_relu import ConvModelReLU
from model_conv2_k10_silu import ConvModelSiLU
from model_conv2_k10_leakyrelu import ConvModelLeakyReLU

# Registry: all model classes for easy iteration during experiments
ALL_MODELS = [
    ConvModel,             # conv2_k10_gelu      (base)
    ConvModel1Layer,       # conv1_k10_gelu      (layer count: 1)
    ConvModel3Layer,       # conv3_k10_gelu      (layer count: 3)
    ConvModelSmallKernel,  # conv2_k5_gelu       (kernel size: 5)
    ConvModelLargeKernel,  # conv2_k20_gelu      (kernel size: 20)
    ConvModelReLU,         # conv2_k10_relu      (activation: ReLU)
    ConvModelSiLU,         # conv2_k10_silu      (activation: SiLU)
    ConvModelLeakyReLU,    # conv2_k10_leakyrelu (activation: LeakyReLU)
]
