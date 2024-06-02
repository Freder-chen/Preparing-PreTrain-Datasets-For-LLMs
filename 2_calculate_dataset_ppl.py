import os
import sys
import json
import datasets

import torch

from tqdm import tqdm
from functools import partial
from collections.abc import Mapping

from accelerate import Accelerator
from transformers import set_seed
from torch.utils.data import DataLoader
from flash_attn.losses.cross_entropy import CrossEntropyLoss

from utils import load_tokenizer, load_model, load_dataset

os.environ["TOKENIZERS_PARALLELISM"] = 'false'
IGNORE_INDEX = -100
SEQLEN_NAME = 'qwen_tokenized_len'
PPL_NAME = 'qwen_ppl'


def calculate_batch_perplexity(data, model, loss_func, sliding_window=1024*7, max_length=1024*8):
    max_length = max_length or sliding_window
    
    labels = data["input_ids"]
    seq_len = labels.size(1)

    nlls = list()
    prev_end_loc = 0
    with torch.inference_mode():
        for begin_loc in range(0, seq_len, sliding_window):
            torch.cuda.empty_cache()

            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            input_ids = labels[:, begin_loc:end_loc]

            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = IGNORE_INDEX
            # move target to the left by one (remember to add one new -100)
            target_ids = target_ids.roll(-1, dims=1)
            target_ids[:, -1] = IGNORE_INDEX

            position_ids = (
                torch.arange(target_ids.shape[1])
                .unsqueeze(0)
                .expand(input_ids.shape[0], -1)
            )

            logits = model(
                input_ids=input_ids,
                position_ids=position_ids,
            ).logits
            neg_log_likelihood = loss_func(
                logits.reshape(-1, logits.shape[-1]), target_ids.reshape(-1)
            )
            nlls.append(neg_log_likelihood)
            
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        ppl = float(torch.exp(torch.stack(nlls).mean()).float().cpu())
    return ppl


def tokenize_dataset(examples, tokenizer, feature):
    text_examples = [texts + tokenizer.eos_token for texts in examples[feature]]
    tokenized = tokenizer(
        text_examples,
        add_special_tokens=False,
        padding=False,
        truncation=True,
        max_length=sys.maxsize,
        return_attention_mask=True,
    )
    return tokenized


def _data_collator(features):
    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    first = features[0]
    # Tensor keys
    batch = {
        "input_ids": torch.tensor([f["input_ids"] for f in features]),
        "attention_mask": torch.tensor([f["attention_mask"] for f in features]),
    }
    # Other keys
    for k, v in first.items():
        if k not in ("input_ids", "attention_mask") and v is not None:
            batch[k] = [f[k] for f in features]

    return batch


def main(dataset_name, model_name, save_dir, use_dist=True):
    set_seed(42)

    accelerator = Accelerator(mixed_precision="bf16")
    accelerator.print(f"Total GPUs: {accelerator.num_processes}")
    rank = accelerator.process_index

    # Load dataset
    if use_dist:
        assert os.path.exists(dataset_name), f"Dataset not found: {dataset_name}"
        dataset = datasets.load_from_disk(dataset_name)
    else:
        dataset = load_dataset(dataset_name, split='train', streaming=False)
        # dataset = dataset.select(range(10000)) # Sample data
        tokenizer = load_tokenizer(model_name)
        dataset = dataset.map(partial(tokenize_dataset, tokenizer=tokenizer, feature='text'), batched=True)

    if os.path.exists(save_dir) and bool(os.listdir(save_dir)):
        accelerator.print("Origin Dataset Size:", len(dataset))
        tmp_dataset = load_dataset(save_dir, split='train', streaming=False)
        tmp_texts = set(example['text'] for example in tmp_dataset)
        dataset = dataset.filter(lambda example: example['text'] not in tmp_texts)
    
    accelerator.print("Dataset Size:", len(dataset))

    # Load model & dataloader
    model = load_model(model_name, device_map=accelerator.device)

    batch_size = 1
    assert batch_size == 1, "batch_size must be 1."
    dataloader = DataLoader(
        dataset,
        collate_fn=_data_collator,
        shuffle=False,
        batch_size=batch_size,
    )
    model, dataloader = accelerator.prepare(model, dataloader)

    loss_func = CrossEntropyLoss(inplace_backward=True)
    pbar = tqdm(dataloader, disable=not accelerator.is_local_main_process)
    for step, batch in enumerate(pbar):
        if 'meta' not in batch.keys():
            batch['meta'] = [{} for _ in range(batch.shape[0])]

        ppl = calculate_batch_perplexity(batch, model, loss_func)
        for item_idx in range(len(batch['input_ids'])):
            seq_len = batch['input_ids'].shape[1] # NOTE: assert batch size is 1
            text = batch['text'][item_idx]
            with open(os.path.join(save_dir, f'rank{rank}.jsonl'), 'a', encoding="utf-8") as f:
                dict_ = {
                    'meta': {
                        SEQLEN_NAME: seq_len,
                        PPL_NAME: ppl
                    },
                    'text': text,
                }
                if 'meta' in batch.keys():
                    dict_['meta'].update(**batch['meta'][item_idx])
                json.dump(dict_, f, ensure_ascii=False)
                f.write('\n')
        pbar.set_postfix(ppl=ppl)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default='./tmp/data_ppl/')
    parser.add_argument("--use_dist", action='store_true')
    args = parser.parse_args()

    main(
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        save_dir=args.save_dir,
        use_dist=args.use_dist,
    )
