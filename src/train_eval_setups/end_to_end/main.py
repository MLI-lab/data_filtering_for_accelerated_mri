import os
import yaml
import json
from glob import glob
from utils import natural_sort
from pathlib import Path
import train
import evaluate
import argparse


if __name__ == '__main__':
    parser = train.build_parser()

    # had to copy this from evaluate.py
    parser.add_argument(
        "--eval_setup", 
        type=str,
        required=True,
        help="path to eval config"
    )
    
    args = parser.parse_args()

    with open(args.train_setup) as f:
        train_config = yaml.safe_load(f)

    exp_name = train_config['name'] + '_' + Path(args.model_setup).stem 

    args.outfile = os.path.join('exp_data/models/', exp_name + '.json')

    train.main(args)

    with open(args.outfile) as f:
        model_summary = json.load(f)

    checkpoints = model_summary['checkpoints']
    for name in natural_sort(checkpoints.keys())[::-1]:
        model_path = checkpoints[name]
        args.outfile = os.path.join('exp_data/evals/', exp_name + '_' + Path(model_path).stem + '.json')
        args.model_path = model_path
        evaluate.main(args)
