import json


def read_jsonl(filename, encoding='utf-8'):
    with open(filename, 'r', encoding=encoding) as f:
        data = [json.loads(line) for line in f]
    return data


def read_json(filename, encoding='utf-8'):
    with open(filename, 'r', encoding=encoding) as f:
        data = json.load(f)
    return data


def save_json(data, filename, ensure_ascii=False, indent=4):
    with open(filename, 'w') as f:
        json.dump(data, f, ensure_ascii=ensure_ascii, indent=indent)


def save_json_line(data, filename, ensure_ascii=False):
    with open(filename, 'a') as f:
        json.dump(data, f, ensure_ascii=ensure_ascii)
        f.write('\n')

