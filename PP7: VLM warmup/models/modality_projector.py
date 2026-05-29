"""Exercise stub for the PP7 modality projector."""

import torch.nn as nn

class ModalityProjector(nn.Module):    
    
    """Student implementation target for the modality projector exercise."""

    def __init__(self, input_dim, output_dim):
        """Placeholder initializer for the exercise implementation.

        Args:
            *args: Positional arguments the student-defined projector may need.
            **kwargs: Keyword arguments the student-defined projector may need.
        """
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x):
        return self.proj(x)