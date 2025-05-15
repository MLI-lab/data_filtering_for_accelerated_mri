from src.interfaces.config_models import *
import pandas as pd
import numpy

def eval_output_to_df(eval_outputs : EvalOutputModel) -> pd.DataFrame:
    """
        Converts the evaluation output into one DataFrame, whoose columns are the metrics in the EvalSingleMetricsModel, as follows:

        Dataset | type |  SSIM | SSIM_normal | PSNR | PSNR_normal | LPIPS | LPIPS_normal | DISTS | DISTS_normal
        ---------------------------------------------------------------------------------------------------
        dataset_1 | classic |  0.99 | 0.99        | 30.0 | 30.0        | 0.01  | 0.01        | 0.01  | 0.01
        dataset_2 | classic | 0.99 | 0.99        | 30.0 | 30.0        | 0.01  | 0.01        | 0.01  | 0.01
    """
    eval_results = []

    for dataset_name, eval_metrics in eval_outputs.eval_metrics.classic.items():
        eval_results.append({
            "Dataset" : dataset_name,
            "Dataset_type" : "classic",
            "SSIM" : eval_metrics.SSIM,
            "SSIM_normal" : eval_metrics.SSIM_normal,
            "PSNR" : eval_metrics.PSNR,
            "PSNR_normal" : eval_metrics.PSNR_normal,
            "LPIPS" : eval_metrics.LPIPS,
            "LPIPS_normal" : eval_metrics.LPIPS_normal,
            "DISTS" : eval_metrics.DISTS,
            "DISTS_normal" : eval_metrics.DISTS_normal
        })

    for dataset_name, eval_metrics in eval_outputs.eval_metrics.pathology.items():
        eval_results.append({
            "Dataset" : dataset_name,
            "Dataset_type" : "pathology",
            "SSIM" : eval_metrics.SSIM,
            "SSIM_normal" : eval_metrics.SSIM_normal,
            "PSNR" : eval_metrics.PSNR,
            "PSNR_normal" : eval_metrics.PSNR_normal,
            "LPIPS" : eval_metrics.LPIPS,
            "LPIPS_normal" : eval_metrics.LPIPS_normal,
            "DISTS" : eval_metrics.DISTS,
            "DISTS_normal" : eval_metrics.DISTS_normal
        })
        
    return pd.DataFrame(eval_results)

def print_single_eval_results(eval_output : EvalOutputModel):
    """
        Prints the evaluation results in a human readable format.
    """
    print(eval_output_to_df(eval_output))

def print_eval_results_on_metric(eval_outputs: List[EvalOutputModel], setups : List[str], metric : str = "PSNR"):
    """
        Prints the evaluation results in a human readable format.
    """
    assert len(setups) == len(eval_outputs), "The number of setups and evaluation outputs must be the"

    if len(setups) == 0:
        return

    print(f"Summary results (on {metric})")
    dfs = [eval_output_to_df(eval_output) for eval_output in eval_outputs]

    # only take the SSIM columns
    ssim_dfs = [df[["Dataset", metric]] for df in dfs]

    # join the dataframes on the column Dataset and rename the SSIM to the setup name
    ssim_df = ssim_dfs[0]
    for i, df in enumerate(ssim_dfs[1:]):
        ssim_df = ssim_df.merge(df, on="Dataset", suffixes=("", f"_{setups[i+1]}"))
    ssim_df = ssim_df.rename(columns={metric : f"{metric}_{setups[0]}"})

    print(ssim_df)



