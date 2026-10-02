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


class OrientationCosineLoss(nn.Module):
    """
    Scale-invariant orientation loss on the deviatoric moment tensor.

    Inputs are 5-component vectors [Mxx, Myy, Mxy, Mxz, Myz] with the
    zero-trace constraint Mzz = -(Mxx + Myy) implied. The loss is
        L = 1 - <P, T>_F / (||P||_F ||T||_F)
    where <.,.>_F is the Frobenius inner product of the reconstructed
    symmetric 3x3 tensors,
        <A, B>_F = Axx Bxx + Ayy Byy + Azz Bzz
                   + 2 (Axy Bxy + Axz Bxz + Ayz Byz).
    For double-couple sources this cosine is a smooth, monotone surrogate of
    the tensor rotation (Kagan) angle: L = 0 iff the mechanisms share
    orientation (regardless of scalar moment), L = 2 for anti-aligned tensors.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    @staticmethod
    def _frobenius_inner(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # a, b: (..., 5) = [Mxx, Myy, Mxy, Mxz, Myz]; Mzz = -(Mxx + Myy)
        azz = -(a[..., 0] + a[..., 1])
        bzz = -(b[..., 0] + b[..., 1])
        diag = a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1] + azz * bzz
        offd = (a[..., 2] * b[..., 2] + a[..., 3] * b[..., 3]
                + a[..., 4] * b[..., 4])
        return diag + 2.0 * offd

    def forward(self, pred_mt: torch.Tensor, target_mt: torch.Tensor) -> torch.Tensor:
        inner = self._frobenius_inner(pred_mt, target_mt)
        norm_p = torch.sqrt(self._frobenius_inner(pred_mt, pred_mt).clamp(min=self.eps))
        norm_t = torch.sqrt(self._frobenius_inner(target_mt, target_mt).clamp(min=self.eps))
        cosine = inner / (norm_p * norm_t)
        return (1.0 - cosine).mean()
