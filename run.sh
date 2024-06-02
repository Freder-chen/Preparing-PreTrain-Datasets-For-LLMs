MODEL_NAME="{your_model_name_or_path}"  # e.g. Qwen/Qwen1.5-7B-Chat
DATASET_NAME="{your_dataset_name_or_path}" # e.g. .../your_data/
BASE_DIR="{your_base_dir}" # e.g. ./tmp/

DISK_PATH=${BASE_DIR}/tokenized_data
PPL_DATA_DIR=${BASE_DIR}/data_with_ppl

python 1_tokenize_dataset.py \
    --model_name ${MODEL_NAME} \
    --dataset_name ${DATASET_NAME} \
    --dist_path ${DISK_PATH}
    # --samples 100

mkdir -p ${PPL_DATA_DIR}
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
accelerate launch --main_process_port 65500 2_calculate_dataset_ppl.py \
    --model_name ${MODEL_NAME} \
    --dataset_name ${DISK_PATH} \
    --save_dir ${PPL_DATA_DIR} \
    --use_dist

# log & sample dataset
python 3_log_and_sample_dataset.py \
    --dataset_dir ${PPL_DATA_DIR} \
    --save_dir ${BASE_DIR} \
    --draw_distribution
