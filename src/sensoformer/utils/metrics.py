import torch
import torch.nn as nn
from typing import Literal

class FocalLossForRegression(nn.Module):
    """
    A regression-appropriate Focal Loss implementation.
    
    This loss function dynamically re-weights the contribution of each sample.
    It gives higher weight to "hard" samples (those with large errors) and
    lower weight to "easy" samples (those with small errors).
    
    Formula: Loss = (1 - exp(-beta * |y - y_pred|))^gamma * |y - y_pred|
    
    Attributes:
        gamma (float): The focusing parameter. Higher values give more weight
                       to hard examples. Default: 1.0
        beta (float):  A smoothing parameter to control the exponential decay
                       of the score. Default: 1.0
        reduction (str): 'mean', 'sum', or 'none'. Default: 'mean'
    """
    def __init__(
        self, 
        gamma: float = 1.0, 
        beta: float = 1.0, 
        reduction: Literal['mean', 'sum', 'none'] = 'mean'
    ):
        super(FocalLossForRegression, self).__init__()
        self.gamma = gamma
        self.beta = beta
        self.reduction = reduction
        # Use L1 Loss (MAE) as the base error
        self.mae = nn.L1Loss(reduction='none')

    def forward(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_tensor: Predicted values.
            target_tensor: Ground truth values.
        """
        # 1. Calculate the base error (MAE) for each sample
        error = self.mae(input_tensor.squeeze(), target_tensor.squeeze())

        # 2. Calculate a "score" (s) between 0 and 1.
        #    High error -> low score. Low error -> high score.
        score = torch.exp(-self.beta * error)

        # 3. Calculate the focal weight.
        #    The modulating factor is (1 - score)^gamma.
        focal_weight = torch.pow(1 - score, self.gamma)

        # 4. The final loss is the focal weight multiplied by the base error
        focal_loss = focal_weight * error

        # 5. Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
