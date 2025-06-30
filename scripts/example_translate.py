#!/usr/bin/env python3
"""
Example script for Local Translate using DynamicTranslator API
This script demonstrates how to translate a dataset using the LLaMAX model
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.orchestrator import DynamicTranslator

async def main():
    """Main translation function using the new API."""
    
    print("🚀 Local Translate Example")
    print("Using DynamicTranslator API")
    print("=" * 50)
    
    # Create translator instance
    translator = DynamicTranslator()
    
    # Translation configuration
    model_id = "LLaMAX/LLaMAX3-8B-Alpaca"
    dataset_id = "presencesw/open-s1-small"
    output_dir = ".cache"
    
    print(f"📊 Model: {model_id}")
    print(f"📁 Dataset: {dataset_id}")
    print(f"📂 Output: {output_dir}")
    print(f"🌐 Translation: English -> Vietnamese")
    print(f"⚡ Quantization: 4-bit (GPU-only)")
    print("=" * 50)
    
    try:
        # Run translation
        await translator.translate(
            model_type="huggingface",
            model_id=model_id,
            dataset_type="huggingface",
            dataset_id=dataset_id,
            columns_to_process=["problem"],
            source_lang="en",
            target_lang="vi",
            quantization="4bit",  # GPU-only 4-bit quantization
            device="cuda",
            model_dir=os.getenv("HF_HOME"),
            data_dir=os.getenv("HF_HOME"),
            cache_dir=".cache",
            output_dir=output_dir,
            push_to_hub=True,
            # Processing configuration
            chunk_size=1,
            batch_size=1
        )
        
        print("✅ Translation completed successfully!")
        print(f"📁 Results saved to: {output_dir}")
        print("🎉 Dataset pushed to HuggingFace Hub!")
        
    except Exception as e:
        print(f"❌ Translation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    print("🔧 Setting up CUDA device: 0")
    exit_code = asyncio.run(main())
    exit(exit_code) 