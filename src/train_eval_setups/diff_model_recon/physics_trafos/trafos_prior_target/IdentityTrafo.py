from torch import Tensor
from src.train_eval_setups.diff_model_recon.physics_trafos.trafos_prior_target.BasePriorTrafo import BasePriorTrafo


class IdentityTrafo(BasePriorTrafo):
    """
    Complex to real transform.
    """
    def __init__(self):
        """
        Parameters
        ----------
        im_shape : 2-tuple of int
            Image shape.
        crop_shape : 2-tuple of int
            Crop shape.
        """
        super().__init__()

    def __call__(self, x: Tensor) -> Tensor:
        """
        Apply the forward projection.

        Parameters
        ----------
        x : :class:`torch.Tensor`
            Image of attenuation.
        """
        return x

    def trafo_inv(self, x: Tensor) -> Tensor:
        """
        Apply the forward projection.

        Parameters
        ----------
        x : :class:`torch.Tensor`
            Image of attenuation.
        """
        return x