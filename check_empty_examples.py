import argparse
import logging
from collections import Counter, defaultdict
from pathlib import Path
from load_dataset import load_dataset_by_files
from datasets.utils.logging import set_verbosity_info


set_verbosity_info()
logger = logging.getLogger(__name__)



def main(args):
    logger.info(f"{len(args.files)} files are included")
    assert len(args.files) == 1

    ds = load_dataset_by_files(files=args.files, dataset_name_or_path=args.dataset_path)

    len_before = len(ds)
    logger.info(f"Filtering out empty examples | len before {len(ds)}")
    ds = ds.filter(
        lambda x: [text is None or len(text) == 0 for text in x["text"]],
        batched=True
    )
    len_after = len(ds)
    logger.info(f"Dataset successfully loaded | len after {len(ds)}")
    if len_after != 0:
        logger.warning(f"There are empty examples: {len_after} ")

    logger.info(ds)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser(description="Load a dataset.")
    parser.add_argument("--dataset_path", type=str, default="c4-en-html-with-metadata")
    parser.add_argument("--num_proc", type=int, default=40)
    parser.add_argument('--files', nargs='+', default=[])


    args = parser.parse_args()

    main(args)
