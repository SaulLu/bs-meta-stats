import argparse
from collections import Counter, defaultdict
from load_dataset import load_dataset_by_files, all_files

def main(args):
    dataset_path = args.dataset_path 
    files = [file[:-len(".gz")] for file in all_files if file.endswith(".jsonl.gz") and file not in args.files_to_exclude]
    print(f"{len(files)} files are included")

    ds = load_dataset_by_files(files=files, dataset_name_or_path=dataset_path)

    print(ds)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load a dataset.')
    parser.add_argument('--dataset_path', type=str, default='c4-en-html-with-metadata')
    parser.add_argument('--files_to_exclude', nargs='+', default=[])
    args = parser.parse_args()

    main(args)