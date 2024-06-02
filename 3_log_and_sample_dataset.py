import os
import glob

from tqdm import tqdm
from utils import (
    load_dataset,
    draw_data,
    read_jsonl,
    save_json_line,
    
)


def draw_ppl_distribution(dataset, output_filename, start_x=0, end_x=200):
    ppls = list()
    for d in tqdm(dataset, desc='Process PPL'):
        ppl = d['meta']['qwen_ppl']
        ppls.append(ppl)

    # print('Min PPL Data:')
    # print(min(dataset, key=lambda x: x['meta']['qwen_ppl'])['meta'])
    # print('Max PPL Data:')
    # print(max(dataset, key=lambda x: x['meta']['qwen_ppl'])['meta'])
    print(f'Min PPL: {min(ppls)}, Max PPL: {max(ppls)}, Average PPL: {sum(ppls) / len(ppls)}')
    draw_data(ppls, 'PPL', output_filename, start_x=start_x, end_x=end_x, count=True)


def draw_seq_len_distribution(dataset, output_filename, start_x=0, end_x=128*1024):
    seq_lens = list()
    for d in tqdm(dataset, desc='Process Seq Length'):
        seq_len = d['meta']['qwen_tokenized_len']
        seq_lens.append(seq_len)
    
    # print('Min Seq Length Data:')
    # print(min(dataset, key=lambda x: x['meta']['qwen_tokenized_len']))
    # print('Max Seq Length Data:')
    # print(max(dataset, key=lambda x: x['meta']['qwen_tokenized_len']))
    print(f'Min Seq Length: {min(seq_lens)}, Max Seq Length: {max(seq_lens)}, Average Seq Length: {sum(seq_lens) / len(seq_lens)}')
    print(f'Total number of sequences: {len(seq_lens)}, tokenized length: {sum(seq_lens)}')
    draw_data(seq_lens, 'Seq_Length', output_filename, start_x=start_x, end_x=end_x, count=True)


def draw_source_distribution(dataset, output_filename):
    sources = list()
    for d in tqdm(dataset, desc='Process Source'):
        source = d['meta']['source']
        sources.append(source)
    draw_data(sources, 'Source', output_filename, count=True)


def save_dataset(dataset_dir, output_filename, ppl_range=(1.1, 200), seqlen_range=(1024*8, 512*1024)):
    min_ppl, max_ppl = ppl_range
    min_seqlen, max_seqlen = seqlen_range
    for filename in tqdm(glob.glob(os.path.join(dataset_dir, '*.jsonl'))):
        data = read_jsonl(filename)
        for i, d in enumerate(data):
            ppl = d['meta']['qwen_ppl']
            length = d['meta']['qwen_tokenized_len']
            if min_ppl < ppl < max_ppl and min_seqlen < length < max_seqlen:
                save_json_line(d, output_filename)


def main(dataset_dir, output_dir, draw_distribution=True):
    os.makedirs(output_dir, exist_ok=True)

    # Log Dataset
    if draw_distribution:
        dataset = load_dataset(dataset_dir, split='train')
        draw_ppl_distribution(dataset, os.path.join(output_dir, 'ppl_distribution.png'), start_x=0, end_x=200)
        draw_seq_len_distribution(dataset, os.path.join(output_dir, 'seq_len_distribution.png'), start_x=0, end_x=64*1024)
        draw_source_distribution(dataset, os.path.join(output_dir, 'source_distribution.png'))

    # Save File
    output_filename = os.path.join(output_dir, 'data_with_ppl.jsonl')
    save_dataset(dataset_dir, output_filename)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default='./tmp/')
    parser.add_argument("--draw_distribution", action='store_true')
    args = parser.parse_args()

    main(
        dataset_dir=args.dataset_dir,
        output_dir=args.save_dir,
        draw_distribution=args.draw_distribution,
    )
