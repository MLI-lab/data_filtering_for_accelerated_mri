from typing import Dict
from collections import OrderedDict

import torch
import logging

from .archs import UNetModel
from .ema import ExponentialMovingAverage

from src.train_eval_setups.diff_model_recon.utils.utils import get_path_by_cluster_name


def create_model(
    num_channels: int,
    in_channels: int,
    out_channels: int,
    num_res_blocks: int,
    channel_mult: str = '',
    use_checkpoint: bool = False,
    attention_resolutions: str = '16',
    num_heads: int = 1,
    num_head_channels: int = -1,
    num_heads_upsample: int = -1,
    use_scale_shift_norm: bool = False,
    dropout: float = 0.,
    resblock_updown: bool = False,
    use_fp16: bool = False,
    use_new_attention_order: bool = False,
    resamp_with_conv : bool = True,
    learn_sigma : bool = False,
    **kwargs
):

    logging.info(f"Unused kwargs: {kwargs}")
    
    attention_ds = []
    for res in attention_resolutions.split(","):
        #attention_ds.append(image_size // int(res))
        attention_ds.append(int(res)) # this is different now

    return UNetModel(
        in_channels=in_channels,
        model_channels=num_channels,
        out_channels=(in_channels if not learn_sigma else in_channels * 2),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=tuple(channel_mult),
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
        conv_resample=resamp_with_conv
    )

def load_score_model(cfg: Dict, device) -> UNetModel:

    kwargs_score = dict(cfg.diffmodels.arch)
    #kwargs_score['image_size'] = cfg.trafo_dataset.im_size

    score = create_model(**kwargs_score).to(device)

    # load_params_from_path = get_path_by_cluster_name(cfg.load_params_from_path, cfg)
    load_params_from_path = cfg.reconstruct.load_params_from_path
    if load_params_from_path is not None:
        try: 
            score.load_state_dict(
                torch.load(load_params_from_path, map_location=device)
            )
        except: 
            state_dict = torch.load(load_params_from_path, map_location=device)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[k.replace('module.', '')] = v # remove 'module.' of DataParallel/DistributedDataParallel
            score.load_state_dict(new_state_dict)
        logging.info(f'model ckpt loaded from: {load_params_from_path}')

    if cfg.reconstruct.load_ema_params_from_path is not None:
        ema = ExponentialMovingAverage(score.parameters(), decay=0.999)
        ema.load_state_dict(torch.load(cfg.reconstruct.load_ema_params_from_path, map_location=device))
        ema.copy_to(score.parameters())
        print("Model loaded")
        logging.info(f'model ema ckpt loaded from: {cfg.reconstruct.load_ema_params_from_path}')

    score.convert_to_fp32()
    score.dtype = torch.float32

    return score