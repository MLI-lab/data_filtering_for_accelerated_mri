from typing import Optional, Dict

import torch


from src.train_eval_setups.diff_model_recon.diffmodels.networks import UNetModel, ExponentialMovingAverage

def save_model(score: UNetModel, epoch: int, max_epochs : int, ema: Optional[ExponentialMovingAverage] = None) -> None:

        model_filename = f'model_{epoch}.pt'
        torch.save(score.state_dict(), model_filename)
        if ema is not None:
            ema_filename = f'ema_model_{epoch}.pt'
            torch.save(ema.state_dict(), ema_filename)
            
