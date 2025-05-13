import scipy as sp
import torch
import torch.nn as nn
from models.quantized_linear import QuantizeLinear
import math

class LoRAModule(nn.Module):
    """Low-Rank Adaptation module."""
    def __init__(self, in_features, out_features, r=8, alpha=16):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # Low-rank matrices A and B
        self.lora_A = nn.Parameter(torch.zeros((r, in_features)))
        self.lora_B = nn.Parameter(torch.zeros((out_features, r)))

        # Initialize A with Gaussian and B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # (batch, in_features) @ (r, in_features)^T -> (batch, r)
        # -> (batch, r) @ (out_features, r)^T -> (batch, out_features)
        return (x @ self.lora_A.T @ self.lora_B.T) * self.scaling


# cpt training
class AdaptiveLoRALinear(QuantizeLinear):
    """Linear layer with multiple LoRA modules for different quantization settings."""
    def __init__(self, in_features, out_features, bit_widths=[4, 8], r=8, alpha=16, bias=False):
        super().__init__(in_features, out_features, bias=True)
        self.bit_widths = bit_widths
        self.lora_module = LoRAModule(in_features, out_features, r, alpha)
    
    def forward(self, x, num_bits=8, sp_config=None):
        """Forward pass with quantization and LoRA."""        
        # Base forward pass with quantization
        out = super().forward(x, w_bits=num_bits, a_bits=num_bits)
        lora_output = self.lora_module(x)
        out = out + lora_output
        
        return out
