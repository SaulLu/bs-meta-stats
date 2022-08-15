from collections import Counter, defaultdict
from load_dataset import load_dataset_by_files, all_files

dataset_path = '/home/lucile_huggingface_co/data/c4-en-html-with-metadata'

files = [
    "c4-en-html_cc-main-2019-18_pq00-000.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-001.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-002.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-003.jsonl",
    "c4-en-html_cc-main-2019-18_pq00-004.jsonl",
]

ds = load_dataset_by_files(files=all_files, dataset_name_or_path=dataset_path)

print(ds)