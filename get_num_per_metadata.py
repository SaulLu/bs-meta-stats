import argparse
import logging
from collections import Counter, defaultdict
from pathlib import Path
from load_dataset import load_dataset_by_files
from datasets.utils.logging import set_verbosity_info


set_verbosity_info()
logger = logging.getLogger(__name__)


def is_not_empty_list(values):
    return {"not_none": list(filter(lambda v: v != [], values))}


def get_num(ds, metadata_col_name, num_proc):
    column_names = list(ds.features.keys())
    sub_ds = ds.map(
        is_not_empty_list,
        batched=True,
        num_proc=num_proc,
        remove_columns=column_names,
        input_columns=[metadata_col_name],
    )
    return len(sub_ds)


def main(args):
    logger.info(f"{len(args.files)} files are included")
    assert len(args.files) == 1

    ds = load_dataset_by_files(files=args.files, dataset_name_or_path=args.dataset_path)
    logger.info("Dataset successfully loaded")

    for name in args.metadata_names:
        num = get_num(ds, metadata_col_name=name, num_proc=args.num_proc)
        filename = f"{name}___{args.files[0][:-len('.jsonl')]}.txt"
        with open(args.save_dir / filename, "w") as f:
            f.write(str(num))
        logger.info(f"The total number of metadata of type | {name} | {num}")

    logger.info(ds)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser(description="Load a dataset.")
    parser.add_argument("--save_dir", type=Path)
    parser.add_argument("--dataset_path", type=str, default="c4-en-html-with-metadata")
    parser.add_argument("--num_proc", type=int, default=40)
    parser.add_argument('--files', nargs='+', default=[])
    parser.add_argument(
        "--metadata_names",
        nargs="+",
        default=[
            "metadata_generation_datasource",
            "metadata_generation_length_sentence",
            "metadata_generation_length_text",
            "metadata_html",
            "metadata_paragraph",
            "metadata_timestamp",
            "metadata_title",
            "metadata_url",
            "metadata_website_desc",
            "metadata_entity",
            "metadata_entity_paragraph",
        ],
    )

    args = parser.parse_args()

    main(args)
