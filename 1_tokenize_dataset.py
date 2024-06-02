import sys

from functools import partial
from utils import load_tokenizer, load_dataset


def tokenize_dataset(examples, tokenizer, feature):
    text_examples = [texts + tokenizer.eos_token for texts in examples[feature]]
    tokenized = tokenizer(
        text_examples,
        add_special_tokens=False,
        padding=False,
        truncation=True,
        max_length=sys.maxsize,
        return_attention_mask=False,
    )
    return tokenized


def main(dataset_name, model_name, dist_path, samples=None):
    dataset = load_dataset(dataset_name, split='train', streaming=False)
    if isinstance(samples, int):
        dataset = dataset.select(range(samples))

    # Load tokenizer
    tokenizer = load_tokenizer(model_name)
    dataset = dataset.map(partial(tokenize_dataset, tokenizer=tokenizer, feature='text'), batched=True)

    dataset.save_to_disk(dist_path)
    print(f"Dataset cache saved at {dist_path}.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dist_path", type=str, required=True)
    parser.add_argument("--samples", type=int, default=None)
    args = parser.parse_args()

    main(
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        dist_path=args.dist_path,
        samples=args.samples
    )

