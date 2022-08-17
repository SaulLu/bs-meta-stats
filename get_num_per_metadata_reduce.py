import argparse
import logging
from collections import Counter, defaultdict
from pathlib import Path
from load_dataset import load_dataset_by_files
from datasets.utils.logging import set_verbosity_info


set_verbosity_info()
logger = logging.getLogger(__name__)


def main(args):

    for name in args.metadata_names:
        total = 0
        for file in args.files:
            filename = f"{name}___{file[:-len('.jsonl')]}.txt"
            with open(args.save_dir / filename, "r") as f:
                num = f.read()
                total += int(num)
        logger.info(f"The total number of metadata of type | {name} | {total}")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser(description="Load a dataset.")
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
