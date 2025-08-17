#!/bin/bash

# Example script for running Local Translate with the new DynamicTranslator API
# This script demonstrates how to translate a dataset using the LLaMAX model

echo "🚀 Starting Local Translate with DynamicTranslator API"
echo "Model: LLaMAX/LLaMAX3-8B-Alpaca"
echo "Dataset: knoveleng/open-s1"
echo "Translation: English -> Vietnamese"

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Create Python script for the new API
cat > temp_translate_script.py << 'EOF'
#!/usr/bin/env python3
"""
Example script for Local Translate using DynamicTranslator API
"""

import asyncio
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.orchestrator import DynamicTranslator

async def main():
    """Main translation function using the new API."""
    
    print("=== Local Translate Example ===")
    print("Using DynamicTranslator API")
    
    # Create translator instance
    translator = DynamicTranslator()
    
    # Translation configuration
    model_id = "LLaMAX/LLaMAX3-8B-Alpaca"
    dataset_id = "knoveleng/open-s1"
    output_dir = ".cache"
    
    print(f"Model: {model_id}")
    print(f"Dataset: {dataset_id}")
    print(f"Output: {output_dir}")
    
    try:
        # Run translation
        await translator.translate(
            model_type="huggingface",
            model_id=model_id,
            dataset_type="huggingface",
            dataset_id=dataset_id,
            columns_to_process=["problem", "solution"],
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
            chunk_size=20,
            batch_size=1
        )
        
        print("✅ Translation completed successfully!")
        print(f"📁 Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"❌ Translation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
EOF

# Run the translation script
echo "📝 Running translation with new API..."
python temp_translate_script.py

# Clean up temporary script
rm -f temp_translate_script.py

echo "🏁 Example script completed!"
