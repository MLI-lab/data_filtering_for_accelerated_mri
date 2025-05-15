
# %%
from src.interfaces import config_factory
from src.train_eval_setups.train_eval_setups import list_train_eval_setups_str, list_train_eval_setups, get_train_eval_setup
import logging
import argparse
import os
from src.train_eval_setups.common_utils.process_eval_results import print_eval_results_on_metric, print_single_eval_results

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--path_config", 
        type=str,
        required=True,
        help="Configuration of the base paths (where to store train and eval output)."
    )
    parser.add_argument(
        "-s",
        "--setup_configs", 
        type=str,
        required=True,
        help="Path to the config files of the setups (can also be comma-separated list)."
    )
    parser.add_argument(
        "-t",
        "--train_dataset_config", 
        type=str,
        required=True,
        help="Path to the train dataset config."
    )
    parser.add_argument(
        "-e",
        "--eval_dataset_config", 
        type=str,
        required=True,
        help="Path to the eval dataset config."
    ) 
    parser.add_argument(
        "-T",
        "--train", 
        action="store_true",
        help="Training is skipped if set to False."
    ) 
    parser.add_argument(
        "-E",
        "--eval", 
        action="store_true",
        help="Evaluation is skipped if not set."
    )
    parser.add_argument(
        "-v",
        "--verbose", 
        action="store_true",
        help="Verbose logging (if set -> DEBUG level, otherwise WARNING)."
    )
    parser.add_argument(
        "-O",
        "--setup_overwrites",
        type=str,
        default="",
        help="Overwrite setup config values."
    )
    parser.add_argument(
        "--resume", 
        action="store_true",
        default=False,
        help="If set, resume training from last checkpoint"
    )
    parser.add_argument(
        "-f",
        "--finetune", 
        type=str,
        required=False,
        default=None,
        help="Path to checkpoint for finetuning"
    )
    return parser

def main(args):
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    print(f"Validating base path setup: {args.path_config}")
    base_path_config = config_factory.base_path_config_by_file(args.path_config)

    full_path_train_dataset_config = os.path.join(base_path_config.base_path_train_dataset_configs, args.train_dataset_config)
    print(f"Validating train dataset config: {full_path_train_dataset_config}")
    train_dataset_config = config_factory.train_dataset_config_by_file(full_path_train_dataset_config)

    full_path_eval_dataset_config = os.path.join(base_path_config.base_path_evals_dataset_configs, args.eval_dataset_config)
    print(f"Validating eval dataset config: {full_path_eval_dataset_config}")
    eval_dataset_config = config_factory.eval_dataset_config_by_file(full_path_eval_dataset_config)

    print(f"Given setups: {args.setup_configs}")

    setup_configs = args.setup_configs.split(",")
    results = []
    for setup_config in setup_configs:
        
        setup_name, config_name = setup_config.split("/")[-2:] # take last two elements
        setup_config_path = os.path.join(base_path_config.base_path_train_eval_setup_configs, setup_config) #setup_name, config_name)

        print("-" * 50)
        print(f"Loading setup: {setup_name}")
        print("-" * 50)
        setup = get_train_eval_setup(setup_name)
        print(f"Initializing with config: {setup_config_path}")
        setup.initialize(
            base_path_config = base_path_config, setup_config_path = setup_config_path,
            setup_overwrites=args.setup_overwrites, train_dataset_config = train_dataset_config, eval_dataset_config = eval_dataset_config, resume=args.resume, finetune=args.finetune
        )

        if args.train:
            print(f"Start training.")
            setup.train()
        else:
            print(f"Skipping training.")

        if args.eval:
            print(f"Start evaluation.")
            eval_result = setup.eval()
            print(f"Results:")
            print_single_eval_results(eval_result)
            results.append(eval_result)
        else:
            eval_result = setup.load_eval_summary()
            if eval_result:
                print(f"Results:")
                print_single_eval_results(eval_result)
                results.append(eval_result)
            else:
                print(f"No evaluation summary found.")
        print("-" * 50)

    if len(results) > 0:
        print_eval_results_on_metric(results, setup_configs)

    return results

if __name__ == '__main__':
    """
        Example use:
            python main.py -p assets/configs/path_config.yml -s setup1.yml,setup2.yml -t train.yml -e eval.yml
    """
    parser = build_parser()
    args = parser.parse_args()
    main(args)

# %%
