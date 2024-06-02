# Preparing Pre-Train Datasets for LLMs


## Introduction

This guide provides instructions on preparing datasets for pre-training Large Language Models (LLMs). The process includes setting up the necessary environment, understanding the required data formats, and organizing the output.

## 1. Requirements

Ensure you have the following dependencies installed:
- Python 3.7+
- [pytorch](https://github.com/pytorch/pytorch)
- [accelerate](https://github.com/huggingface/accelerate)
- [transformers](https://github.com/huggingface/transformers)
- [datasets](https://github.com/huggingface/datasets)
- [flash-attention](https://github.com/HazyResearch/flash-attention)
- [tqdm](https://github.com/tqdm/tqdm)

### 1.1. Installing the Dependencies

To install the required dependencies, run:

```bash
pip install torch
pip install accelerate transformers datasets flash-attention tqdm
```

## 2. Preparing the Dataset

### 2.1 Input Format

Any format readable by `datasets.load_dataset` can be used for our pre-training dataset. Each item should contain a `text` field, with its value being a string. Example:

```json
{"text": "This is a Text.", "meta": {"source": "..."}, ...}
```

### 2.2 Output Format

After preprocessing, the dataset will have the following structure:

- A `text` field with a string value.
- A `meta` field which is a dictionary containing:
    - `"source"`: A string indicating the source of the dataset.
    - `SEQLEN_NAME`: A string indicating the name of the sequence length field, defaults to `"qwen_tokenized_len"`.
    - `PPL_NAME`: A string indicating the name of the perplexity field, defaults to `"qwen_ppl"`.

Example:
```json
{"text": "This is a Text.", "meta": {"source": "...", SEQLEN_NAME: ..., PPL_NAME: ...}}
```

## 3. Usage Instructions

After preparing your data, refer to `run.sh` to execute the preprocessing scripts. Here are the detailed steps:

### 3.1. Tokenizing Data

Optional: For large datasets, we recommend tokenizing the data in advance. For small datasets, you can directly use the raw data for pre-training.

```shell
python 1_tokenize_dataset.py \
    --model_name ${MODEL_NAME} \
    --dataset_name ${DATASET_NAME} \
    --dist_path ${DISK_PATH}
```

You can also use the `--samples <num>` option to test the data.

### 3.2. Calculating Perplexity


```shell
python 2_calculate_perplexity.py \
    --model_name ${MODEL_NAME} \
    --dataset_name ${DATASET_NAME} \
    --dist_path ${DISK_PATH}
```

### 3.3. Sampling Dataset

```shell
python 3_log_and_sample_dataset.py \
    --dataset_dir ${PPL_DATA_DIR} \
    --save_dir ${BASE_DIR} \
    --draw_distribution
```
