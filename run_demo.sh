#!/bin/bash
save_path="$1"
python download_datasets.py -s "${save_path}/datasets/"
python convert_datasets.py -s "${save_path}/datasets/"
python create_dataset_json.py -s "${save_path}/datasets/"
python demo_compute_mvue_and_smaps.py
python compute_embeddings.py -s "${save_path}/embeddings/" -d datasets/train/accel_whole-heart_3D_T2_mapping.json
python compute_embeddings.py -s "${save_path}/embeddings/" -d datasets/evals/classic/smurf.json
python retrieval_filter.py -d datasets/train/accel_whole-heart_3D_T2_mapping.json -f "${save_path}/embeddings/dreamsim_ensemble_emb_p128_accel_whole-heart_3D_T2_mapping.pt" -r "${save_path}/embeddings/dreamsim_ensemble_emb_p128_smurf.pt" -k 1
python main_train_eval.py -p configs/paths/output_dirs.yml -s End2EndSetup/varnet-large_c8.yml -t demo_accel_whole-heart_3D_T2_mapping_epochs=2.yml -e demo_eval_2d_smurf.yml -T -E -v
python main_train_eval.py -p configs/paths/output_dirs.yml -s End2EndSetup/varnet-large_c8.yml -t demo_dreamsim_ensemble_emb_p128_accel_whole-heart_3D_T2_mapping_epochs=2.yml -e demo_eval_2d_smurf.yml -T -E -v
