import os
from pathlib import Path
from typing import List, Dict, Optional
import copy
from tqdm.auto import tqdm
from datasets import load_dataset, load_from_disk
import argparse

from src.model import TranslateModel
from utils.utils import download_dataset_from_huggingface, push_to_hub, make_dir, apply_llamaX_template, load_all_chunk

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AbtractTranslateProcessor:

    """
    Initialize the translation processor.

    Args:
        model_id (str): The identifier of the translation model to be used.
                        Example: 'LLaMAX/LLaMAX3-8B-Alpaca'.

        repo_id (str) The identifier of the huggingface repo id to push dataset when translated.

        src_language (str): The source language to translate from.
                            Example: 'English'.

        trg_language (str): The target language to translate to.
                            Example: 'Vietnamese'.

        max_length_token (int): The maximum number of tokens allowed in the input prompt.
                                Inputs longer than this may be truncated depending on the model configuration.

        dataset_name (str): The name of the HuggingFace dataset to be downloaded and translated.

        translated_dataset_dir (str): The directory where translated dataset chunks will be saved.

        subset (str, optional): The subset of the dataset to use (if applicable).
                                Default is 'default'.

        start_inter (int, optional): The index of the chunk to start processing from.
                                     Useful for resuming translation from a specific chunk.
                                     Default is 0.

        batch_size (int, optional): The batch size passed to `.map()` for processing individual records.
                                    Since translation is typically done per-record, this is usually 1.
                                    Default is 1.

        download_dataset_dir (str, optional): Directory to download the raw dataset to.
                                              If None, the dataset will be loaded from HuggingFace cache.
                                              Default is None.

        writer_batch_size (int, optional): The number of records to process and save as one chunk.
                                           Each chunk will be saved to disk after translation.
                                           Default is 20.
    """


    def __init__(self,
                 prompt_template_fn,
                 model_id:str,
                 repo_id:str, 
                 src_language:str,
                 trg_language:str,
                 max_length_token:int,
                 dataset_name:str,
                 translated_dataset_dir:str,
                 subset:str = 'default',
                 start_inter:int = 0,
                 batch_size:int= 1,
                 download_dataset_dir:str= None,
                 writer_batch_size:int=20):
        
        self.prompt_template_fn = prompt_template_fn
        self.model_id = model_id
        self.repo_id = repo_id, 
        self.dataset_name= dataset_name
        self.subset = subset
        self.start_inter = start_inter
        self.batch_size = batch_size
        self.writer_batch_size = writer_batch_size
        self.src_language = src_language
        self.trg_language = trg_language
        self.download_dataset_dir= download_dataset_dir
        self.translated_dataset_dir = translated_dataset_dir
        self.translate_model = TranslateModel(prompt_template_fn= prompt_template_fn,
                                              model_id= self.model_id,
                                              max_length_token= max_length_token,
                                              )
        
        # Apply src_language and trg_language for config transalte model

    
    def postprocess_llm_output(self, input_prompt: str, output_llm_raw: str) -> str:

            if input_prompt in output_llm_raw:
                result = output_llm_raw.split(input_prompt, 1)[1]
            else:
                result = output_llm_raw  
            return result.replace("\"\"\"", "").strip()


    def dataset_process_mapping(self, sample):
         
        problem = sample['problem']
        solution = sample['solution']

        # Create two key to save 
        problem_column_translated = f'problem_{self.trg_language}'
        solution_column_translated = f'solution{self.trg_language}'

        # Translate sentences with llm model
        problem_translated = self.translate_model.translate(query= problem, 
                                                            src_language= self.src_language, 
                                                            trg_language= self.trg_language)
        solution_translated = self.translate_model.translate(query = solution,
                                                             src_language= self.src_language,
                                                             trg_language= self.trg_language)

        # Post process after translate
        problem_postprocessed= self.postprocess_llm_output(problem_translated)
        solution_translated = self.postprocess_llm_output(solution_translated)

        return {
             
             problem_column_translated: problem_postprocessed,
             solution_column_translated: solution_column_translated
        }
    
    def __call__(self):
    # Create folder to save translated data
        if self.download_dataset_dir is not None:
            dataset_path = Path(os.path.join(self.download_dataset_dir, self.dataset_name))
            make_dir(dataset_path)

            # Download dataset from HuggingFace and save to local disk
            download_dataset_from_huggingface(
                dataset_name=self.dataset_name, 
                save_dir=dataset_path,
            )
            dataset = load_from_disk(dataset_path=dataset_path)
            dataset = dataset[f'{self.subset}']
        else:
            # Load from HuggingFace cache
            dataset = load_dataset(self.dataset_name)
            dataset = dataset[f'{self.subset}']
  
        save_translated_path = Path(os.path.join(self.translated_dataset_dir, self.dataset_name))
        make_dir(save_translated_path)
        total = len(dataset)
        total_chunks = (total + self.writer_batch_size - 1) // self.writer_batch_size

        for chunk_id in tqdm(range(self.start_inter, total_chunks), desc="Translating and saving chunks"):
            start = chunk_id * self.writer_batch_size
            end = min(start + self.writer_batch_size, total)  # Ensure end does not exceed dataset length

            if start >= total:
                logger.warning(f"Start index {start} exceeds dataset length {total}, skipping.")
                break

            chunk_save_path = save_translated_path / f'chunk_{chunk_id:05d}'
            if chunk_save_path.exists():
                logger.info(f"Skipping existing chunk {chunk_id} → {chunk_save_path}")
                continue
          
            # Select the current chunk safely
            try:
                split_dataset = dataset.select(range(start, end))  # Safe even for last incomplete chunk
            except Exception as e:
                logger.error(f"Failed to select range({start}, {end}): {e}")
                continue

            # Translate the chunk
            try:
                dataset_processed = split_dataset.map(
                    self.dataset_process_mapping,
                    batched=False,
                    batch_size=self.batch_size
                )
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_id}: {e}")
                continue

            # Save translated chunk to disk
            dataset_processed.save_to_disk(chunk_save_path)
            logger.info(f"Saved chunk {chunk_id} → {chunk_save_path}")

        logger.info(f"Finished processing from chunk {self.start_inter} to {total_chunks - 1}")

        # Merge dataset and push to hub
        merged_dataset = load_all_chunk(src_dir=save_translated_path)
        push_to_hub(dataset= merged_dataset, repo_id= self.repo_id )

def get_args():
    parser = argparse.ArgumentParser(description="Translation Processor Arguments")

    parser.add_argument('--model_id', type=str, required=True,
                        help="The identifier of the translation model (e.g., 'facebook/nllb-200-distilled-600M')")

    parser.add_argument('--repo_id', type=str, required=True,
                    help="The repo to push dataset")
    
    parser.add_argument('--src_language', type=str, required=True,
                        help="Source language (e.g., 'English')")

    parser.add_argument('--trg_language', type=str, required=True,
                        help="Target language (e.g., 'Vietnamese')")

    parser.add_argument('--max_length_token', type=int, default=512,
                        help="Maximum number of tokens allowed in the input")

    parser.add_argument('--dataset_name', type=str, required=True,
                        help="Name of the HuggingFace dataset")

    parser.add_argument('--translated_dataset_dir', type=str, required=True,
                        help="Directory to save translated dataset")

    parser.add_argument('--subset', type=str, default='train',
                        help="Subset of the dataset to use")

    parser.add_argument('--start_inter', type=int, default=0,
                        help="Chunk index to start processing from")

    parser.add_argument('--batch_size', type=int, default=1,
                        help="Batch size for mapping translation function")

    parser.add_argument('--download_dataset_dir', type=str, default=None,
                        help="Directory to save downloaded dataset (optional)")

    parser.add_argument('--writer_batch_size', type=int, default=20,
                        help="Number of records per translated chunk")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    
    processor = AbtractTranslateProcessor(
        prompt_template_fn= apply_llamaX_template,
        model_id=args.model_id,
        repo_id= args.repo_id, 
        src_language=args.src_language,
        trg_language=args.trg_language,
        max_length_token=args.max_length_token,
        dataset_name=args.dataset_name,
        translated_dataset_dir=args.translated_dataset_dir,
        subset=args.subset,
        start_inter=args.start_inter,
        batch_size=args.batch_size,
        download_dataset_dir=args.download_dataset_dir,
        writer_batch_size=args.writer_batch_size,
    )

    processor()  # run the translation process