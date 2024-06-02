import os
import sys
import torch
import datasets

from transformers import AutoModelForCausalLM, AutoTokenizer


def load_tokenizer(model_name, model_max_length=sys.maxsize):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=model_max_length,
        trust_remote_code=False,
        add_bos_token=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_name, device_map):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2",
        device_map=device_map,
    )
    model.eval()
    return model


def load_dataset(dataset_name, split='train', streaming=False):
    if os.path.isdir(dataset_name):
        dataset = datasets.load_dataset(path=dataset_name, split=split, streaming=streaming)
    elif os.path.isfile(dataset_name) and dataset_name.endswith(('.jsonl', '.json', '.jsonl.zst', '.json.zst')):
        dataset = datasets.load_dataset(path='json', data_files=dataset_name, split=split, streaming=streaming)
    else:
        raise NotImplementedError(f"Unsupported dataset: {dataset_name}")
    return dataset
