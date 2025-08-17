#!/usr/bin/env python3
"""
HuggingFace Model Test Suite
Tests LLaMAX models on GPU without CPU offloading
"""

import asyncio
import sys
import time
from pathlib import Path
import os
from torch import device

# Add the project root and src directory to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from utils.utils import load_all_chunk

try:
    from src.orchestrator import DynamicTranslator
except ImportError:
    # Try alternative import path
    from orchestrator import DynamicTranslator


def create_llamax_prompt(query: str, src_language: str, trg_language: str) -> str:
    """
    Create the proper prompt template for LLaMAX model as specified in the official documentation.
    Based on: https://huggingface.co/LLaMAX/LLaMAX3-8B-Alpaca
    """
    instruction = f'Translate the following sentences from {src_language} to {trg_language}.'
    prompt = (
        'Below is an instruction that describes a task, paired with an input that provides further context. '
        'Write a response that appropriately completes the request.\n'
        f'### Instruction:\n{instruction}\n'
        f'### Input:\n{query}\n### Response:'
    )
    return prompt


def postprocess_llamax_output(input_prompt: str, output_llm_raw: str) -> str:
    """
    Postprocess LLaMAX model output by removing the input prompt and cleaning the response.
    
    Args:
        input_prompt (str): The original input prompt
        output_llm_raw (str): Raw output from the LLM
    
    Returns:
        str: Cleaned translation output
    """
    # Remove the input prompt from the output if it's included
    if input_prompt in output_llm_raw:
        result = output_llm_raw.split(input_prompt, 1)[1]
    else:
        result = output_llm_raw
    
    # Clean up common artifacts
    result = result.replace('"""', "").strip()
    
    # Remove any leading/trailing whitespace and newlines
    result = result.strip()
    
    return result


async def main():
    """Main function to test the HuggingFace model with fixed configuration."""
    # Create a DynamicTranslator instance
    translator = DynamicTranslator()
    output_dir = "output"
    model_id = "LLaMAX/LLaMAX3-8B-Alpaca"
    model_name = model_id.split("/")[-1]
    dataset_id = "presencesw/open-s1-small"
    dataset_name_push = f"{dataset_id}-{model_name}"
    
    # Define preprocess and postprocess functions
    preprocess = create_llamax_prompt
    postprocess = postprocess_llamax_output
    
    print("=== HUGGINGFACE MODEL TEST (GPU-ONLY) ===")
    print(f"Testing {model_id} with 4-bit quantization on GPU (no CPU offload)")
    
    # Test the HuggingFace model with GPU-only configuration
    # Uses 4-bit quantization but keeps everything on GPU (no CPU offload)
    await translator.translate(
        model_type="huggingface",
        model_id=model_id,
        dataset_type="huggingface",
        dataset_id=dataset_id,
        columns_to_process=["problem", "solution"],
        source_lang="en",
        target_lang="vi",
        push_to_hub=True,
        model_dir=os.getenv("HF_HOME"),
        data_dir=os.getenv("HF_HOME"),
        cache_dir=os.getenv(".cache"),
        device="cuda",
        # Use 4-bit quantization with GPU-only inference (no CPU offload)
        quantization="4bit",
        output_dir=output_dir,
        preprocess=preprocess,
        postprocess=postprocess
    )

    print("=== LOADING AND PUSHING RESULTS ===")
    ds = load_all_chunk(output_dir)
    ds.push_to_hub(dataset_name_push)
    print(f"✓ Dataset successfully pushed to {dataset_name_push}")


async def test_without_quantization():
    """Alternative test without quantization if the above still fails."""
    translator = DynamicTranslator()
    output_dir = "output"
    model_id = "LLaMAX/LLaMAX3-8B-Alpaca"
    model_name = model_id.split("/")[-1]
    dataset_id = "presencesw/open-s1-small"
    dataset_name_push = f"{dataset_id}-{model_name}"
    
    preprocess = create_llamax_prompt
    postprocess = postprocess_llamax_output
    
    print("=== TESTING WITHOUT QUANTIZATION (FALLBACK) ===")
    print(f"Testing {model_id} without quantization")
    
    await translator.translate(
        model_type="huggingface",
        model_id=model_id,
        dataset_type="huggingface",
        dataset_id=dataset_id,
        columns_to_process=["problem"],  # Test with just one column first
        source_lang="en",
        target_lang="vi",
        push_to_hub=True,
        model_dir=os.getenv("HF_HOME"),
        data_dir=os.getenv("HF_HOME"),
        cache_dir=".cache",
        device="cuda",
        quantization="4bit",
        output_dir=output_dir,
        preprocess=preprocess,
        postprocess=postprocess,
        # Limit processing for testing
        chunk_size=1,  # Process only 5 samples first,
        batch_size=1
    )

    print("=== LOADING AND PUSHING RESULTS (NO QUANT) ===")
    ds = load_all_chunk(output_dir)
    ds.push_to_hub(dataset_name_push)
    print(f"✓ Dataset successfully pushed to {dataset_name_push}")


if __name__ == "__main__":
    try:
        print("Running GPU-only inference test...")
        asyncio.run(main())
    except Exception as e:
        print(f"GPU-only quantization failed: {e}")
        print("Trying fallback without quantization...")
        asyncio.run(test_without_quantization()) 