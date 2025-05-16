# Data filtering for accelerated MRI
### Summary
This repository contains
- code to download some of the datasets used in our work. Some datasets like the fastMRI or CMRxRecon dataset need to requested from the source.
- code to convert the datasets into a consistent format used in our work.
- code for training and evaluating models.
- exact information on the training and evaluation configurations used in our work. For example, the folder `datasets/train` contains slice-level information about our unfiltered and filtered datasets.
- the raw experimental data used in our work, which can be found in the foler `exp_data`.

### How to use
1. Run `bash setup.sh` to install all necessary packages.
2. Create an output directory `/path/to/output/directory` for saving created resources, such as datasets, model checkpoints, etc. The available disk space must be at least 130GB for running the demo.
3. Adjust the absolute paths in `configs/paths/output_dirs.yml`. In that file, substitue `/path/to/output/directory` with the path you created in 2., and substitue `/path/to/this/repository` with the absolute path to this repository.
4. Run `bash run_demo.sh </path/to/output/directory>` for a demo. This will do the following:
    - Downloads two datasets which will be saved in `/path/to/output/directory` created in point 2. 
    - Converts the two datasets and saves them as HDF5 files. One dataset will serve as training set and the other as test set.
    - MVUE reconstructions will be computed and added to those HDF5 files.
    - Apply retrieval filtering on the training set based on the test set.
    - Train one model on the unfiltered training set for two epochs and one model on the retrieval filtered dataset for two epochs.
    - Store the checkpoints, scores, and other outputs in the paths provided in `configs/paths/output_dirs.yml`.


You can use `main_train_eval.py` to run any of the train_eval setups. For example, if you have downloaded and converted all datasets used in our work, to reproduce our results for a VarNet trained for 4-fold accerated MRI on the unfiltered datasets of 120k slices with 1\% in-distribution data, run:

```sh
python main_train_eval.py -p debug/configs/paths/output_dirs.yml -s End2EndSetup/varnet-large_c8.yml -t data_pool_random_volume_subset_120k_epochs=20.yml -e eval_2d_curated_v2.yml -T -E -v
```
You can also skip the `-T` or `-E` option to not rerun training/evaluation again.

### Project structure
**Working directory**
```
 ├───configs                            # human-readable configuration files
 │   ├─── paths                         # base paths (yaml)
 │   ├─── training                      # epoch-wise training dataset configs
 │   ├─── evals                         # list of datasets for evaluation
 │   └─── train/eval setups             # setup-specific configurations, contains predefined model configurations.
 ├───datasets                           # human-readable dataset files 
 │   ├─── train                         # slice-level information on the training datasets (storage location etc)
 │   └─── evals                         # slice-level information on the evaluation datasets (storage location etc)
 ├───exp_data                           # lightweight outputs (model and eval summary files)
 ├───setup                              # some installation code (e.g. bart, python dependencies)
 ├───src                                # data conversion, training and evaluation code
 │   ├─── data                          # data convesion code
 │   ├─── interfaces                    # 
 │   │    ├─── base_train_eval_setup.py # defines common behavior of each setup in an abstract class
 │   │    ├─── config_models.py         # formalizes the data setups for train/eval
 │   │    ├─── config_factory.py        # code for validating json files against the defined models
 │   │    └─── path_setups.py           # defining where to store output data and summary files. 
 │   └─── train_eval_setups             # contains the various train/eval setups
 │        ├─── train_eval_setups.py     # 
 │        ├─── common_utils             # processing results etc.
 │        ├─── end_to_end               
 │        ├─── diff_model_recon
 │        └─── (other setups..)
 ├───main_train_eval.py                 # for running the train_eval setups
 ├───download_datasets.py               # for downloading datasets that do not need manual access request
 ├───convert_datasets.py                # for downloading datasets that do not need manual access request
 ├───create_dataset_json.py             # creates slice-information on training and evaluation datasets, e.g. which slices within a volume to access for training
 ├───add_mvue_smaps_to_h5.py            # computes and stores MVUE reconstructions and sensitivty maps
 ├───compute_embeddings.py              # computes dreamsim embeddings which are used for retrieval filtering
 ├───retrieval_filter.py                # applies retreival filtering on a dataset given a reference set
```
**Note:** The folder `datasets` contains dataset files saved as `.json` files referencing the stored k-space data, not the k-space data itself. These files are mainly used to specfiy which k-space data to use for training and evaluation. Each file specfies the paths to k-space volumes (stored as `.h5` files following the fastMRI convention) and indicates which slices within a volume are used.

