import torch
import torch.nn as nn
from fastmri.models import Unet

class NormUnet(nn.Module):
    """
    Normalized U-Net model.
    This is the same as a regular U-Net, but with normalization applied to the
    input before the U-Net. This keeps the values more numerically stable
    during training. 
    IMPORTANT: This NormUnet is not the same as NormUnet from fastMRI's VarNet block. 
    """

    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 1,
        out_chans: int = 1,
        drop_prob: float = 0.0,
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.unet = Unet(
            in_chans=in_chans,
            out_chans=out_chans,
            chans=chans,
            num_pool_layers=num_pools,
            drop_prob=drop_prob,
        )

    def norm(self, x):
        mean = x.mean(dim=(-2,-1), keepdim=True)
        std = x.std(dim=(-2,-1), keepdim=True)
        x = (x-mean)/std

        return x, mean, std

    def unnorm(self, x, mean, std):
        
        return x * std + mean

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, mean, std = self.norm(x)
        x = self.unet(x)
        x = self.unnorm(x, mean, std)

        return x