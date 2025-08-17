import logging
import os
from pathlib import Path

from datasets import concatenate_datasets, load_dataset, load_from_disk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def make_dir(dir_path):
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def push_to_hub(dataset, repo_id: str, private: bool = False):
    logger.info(f"Pushing dataset to Huggingface Hub at: {repo_id}")
    dataset.push_to_hub(repo_id, private=private)
    logger.info("Push successful!")


def load_all_chunk(src_dir: str):
    save_path = Path(src_dir)
    chunk_dirs = sorted(
        [p for p in save_path.iterdir() if p.is_dir() and "chunk_" in p.name]
    )
    all_datasets = []

    for chunk_dir in chunk_dirs:
        try:
            ds = load_from_disk(chunk_dir)
            all_datasets.append(ds)
            logger.info(f"Loaded {chunk_dir}")
        except Exception as e:
            logger.warning(f" Failed to load {chunk_dir}: {e}")
    return concatenate_datasets(all_datasets)


def download_dataset_from_huggingface(
    dataset_name: str, save_dir: Path, subset: str = "default"
) -> None:
    """
    Download a dataset form the Hugginface Hub and save it to disk.

    Args:
        dataset_name (str): Name of the dataset on Hugging Face (e.g., "open-r1/OpenR1-Math-Raw").
        save_dir (Path): Directory where the dataset should be saved.
        subset (str, optional): Subset configuration of the dataset. Defaults to "default".

    Raises:
        Exception: If the dataset fails to download or save.

    """

    try:
        # 1. Create folder to save dataset
        make_dir(save_dir)
        save_dataset_path = os.path.join(save_dir)
        logger.info(f"Downloading dataset: {dataset_name} (subset: {subset})")
        # 2. Load dataset

        dataset = load_dataset(dataset_name, subset)
        logger.info(f" Saving dataset to: {save_dataset_path}")

        # 3. Save dataset
        dataset.save_to_disk(str(save_dataset_path))

        logger.info("Dataset successfully downloaded and saved.")

    except Exception as e:
        logger.error(f"Failed to download or save dataset: {e}")
        raise


def apply_llamaX_template(query: str, src_language: str, trg_language: str):
    instruction = (
        f"Translate the following sentences from {src_language} to {trg_language}."
    )
    prompt = (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n"
        f"### Instruction:\n{instruction}\n"
        f'### Input:\n"""{query}"""\n### Response:'
    )
    return prompt
