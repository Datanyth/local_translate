"""
Main orchestrator for the dynamic multi-model and multi-dataset system.
"""

import asyncio
import argparse
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

from .core.base import ModelConfig, DatasetConfig, ProcessingConfig
from .core.registry import ModelRegistry, DatasetRegistry, ProcessorRegistry
from .config import config, Config

# Import all models to register them
from .models.huggingface_model import HuggingFaceModel
# from .models.vllm_model import VLLMModel
# from .models.ollama_model import OllamaModel
# from .models.openrouter_model import OpenRouterModel

# Import all datasets to register them
from .data_loaders.huggingface_dataset import HuggingFaceDataset
# from .data_loaders.pandas_dataset import PandasDataset
# from .data_loaders.pytorch_dataset import PyTorchDataset

# Import all processors to register them
from .processors.translation_processor import TranslationProcessor


class DynamicTranslator:
    """Main orchestrator for dynamic multi-model and multi-dataset translation."""
    
    def __init__(self, custom_config: Optional[Config] = None):
        self.model_registry = ModelRegistry()
        self.dataset_registry = DatasetRegistry()
        self.processor_registry = ProcessorRegistry()
        
        # Use custom config if provided, otherwise use global config
        self.config_manager = custom_config if custom_config else config
    
    async def create_model(self, model_type: str, model_id: str, **kwargs) -> Any:
        """Create a model instance."""
        # Get configuration from config manager
        model_config_dict = self.config_manager.get_model_config(model_type, model_id, **kwargs)
        
        config = ModelConfig(**model_config_dict)
        return self.model_registry.create(model_type, config)
    
    async def create_dataset(self, dataset_type: str, dataset_id: str, **kwargs) -> Any:
        """Create a dataset instance."""
        # Get configuration from config manager
        dataset_config_dict = self.config_manager.get_dataset_config(dataset_type, dataset_id, **kwargs)
        
        config = DatasetConfig(**dataset_config_dict)
        return self.dataset_registry.create(dataset_type, config)
    
    async def create_processor(self, processor_type: str, model: Any, dataset: Any, **kwargs) -> Any:
        """Create a processor instance."""
        # Get configuration from config manager
        processing_config_dict = self.config_manager.get_processing_config(**kwargs)
        
        config = ProcessingConfig(**processing_config_dict)
        return self.processor_registry.create(processor_type, model, dataset, config)
    
    async def translate(self, 
                       model_type: str,
                       model_id: str,
                       dataset_type: str,
                       dataset_id: str,
                       columns_to_process: List[str],
                       source_lang: Optional[str] = None,
                       target_lang: Optional[str] = None,
                       output_dir: Optional[str] = None,
                       **kwargs) -> None:
        """Main translation workflow."""
        
        # Get translation configuration from config manager
        translation_config = self.config_manager.get_translation_config(
            source_lang=source_lang,
            target_lang=target_lang
        )
        
        source_lang = translation_config["source_lang"]
        target_lang = translation_config["target_lang"]
        output_dir = output_dir or self.config_manager.default_output_dir
        
        print(f"Starting translation with:")
        print(f"  Model: {model_type} - {model_id}")
        print(f"  Dataset: {dataset_type} - {dataset_id}")
        print(f"  Columns: {columns_to_process}")
        print(f"  Language: {source_lang} -> {target_lang}")
        print(f"  Output: {output_dir}")
        
        try:
            # Create model
            print("Creating model...")
            model = await self.create_model(model_type, model_id, **kwargs)
            
            # Create dataset
            print("Creating dataset...")
            dataset = await self.create_dataset(dataset_type, dataset_id, **kwargs)
            
            # Create processor
            print("Creating processor...")
            processor = await self.create_processor(
                "translation", 
                model, 
                dataset, 
                output_dir=output_dir,
                **kwargs
            )
            
            # Process translation
            print("Starting translation...")
            await processor.process(
                columns_to_process=columns_to_process,
                source_lang=source_lang,
                target_lang=target_lang,
                **kwargs
            )
            
            print("Translation completed successfully!")
            
        except Exception as e:
            print(f"Translation failed: {e}")
            raise
    
    async def push_to_hub(self, model_type: str, model_id: str, dataset_type: str, dataset_id: str, source_lang: Optional[str] = None, target_lang: Optional[str] = None, **kwargs) -> None:
        """Push the translated dataset to the HuggingFace Hub."""
        # Get translation configuration from config manager
        translation_config = self.config_manager.get_translation_config(
            source_lang=source_lang,
            target_lang=target_lang
        )
    
    def list_available_models(self) -> List[str]:
        """List available model types."""
        return self.model_registry.list_models()
    
    def list_available_datasets(self) -> List[str]:
        """List available dataset types."""
        return self.dataset_registry.list_datasets()
    
    def list_available_processors(self) -> List[str]:
        """List available processor types."""
        return self.processor_registry.list_processors()
    
    def show_config(self):
        """Show current configuration."""
        self.config_manager.print_config_summary()


async def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Dynamic Multi-Model Translation System")
    
    # Model arguments
    parser.add_argument("--model_type", 
                       choices=["huggingface", "vllm", "ollama", "openrouter"],
                       help="Type of model to use")
    parser.add_argument("--model_id",
                       help="Model identifier")
    parser.add_argument("--device", default=None,
                       help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--quantization", choices=["4bit", "8bit"],
                       help="Quantization to use")
    parser.add_argument("--max_length", type=int, default=None,
                       help="Maximum sequence length")
    parser.add_argument("--temperature", type=float, default=None,
                       help="Generation temperature")
    parser.add_argument("--top_p", type=float, default=None,
                       help="Top-p sampling")
    
    # Dataset arguments
    parser.add_argument("--dataset_type",
                       choices=["huggingface", "pandas", "pytorch"],
                       help="Type of dataset to use")
    parser.add_argument("--dataset_id",
                       help="Dataset identifier")
    parser.add_argument("--subset", default=None,
                       help="Dataset subset")
    parser.add_argument("--split", default=None,
                       help="Dataset split")
    parser.add_argument("--cache_dir", default=None,
                       help="Cache directory for datasets")
    
    # Processing arguments
    parser.add_argument("--columns", nargs="+",
                       help="Columns to translate")
    parser.add_argument("--source_lang", default=None,
                       help="Source language")
    parser.add_argument("--target_lang", default=None,
                       help="Target language")
    parser.add_argument("--output_dir", default=None,
                       help="Output directory")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Batch size for processing")
    parser.add_argument("--chunk_size", type=int, default=None,
                       help="Chunk size for processing")
    parser.add_argument("--resume_from", type=int, default=None,
                       help="Resume from chunk index")
    
    # API arguments
    parser.add_argument("--api_key", default=None,
                       help="API key for cloud models")
    parser.add_argument("--base_url", default=None,
                       help="Base URL for API models")
    
    # Utility arguments
    parser.add_argument("--list_models", action="store_true",
                       help="List available model types")
    parser.add_argument("--list_datasets", action="store_true",
                       help="List available dataset types")
    parser.add_argument("--list_processors", action="store_true",
                       help="List available processor types")
    parser.add_argument("--show_config", action="store_true",
                       help="Show current configuration")
    parser.add_argument("--env_file", default=None,
                       help="Path to .env file")
    
    args = parser.parse_args()
    
    # Create orchestrator with custom config if env_file is specified
    if args.env_file:
        custom_config = Config(args.env_file)
        translator = DynamicTranslator(custom_config)
    else:
        translator = DynamicTranslator()
    
    # Handle utility commands
    if args.show_config:
        translator.show_config()
        return
    
    if args.list_models:
        print("Available model types:")
        for model_type in translator.list_available_models():
            print(f"  - {model_type}")
        return
    
    if args.list_datasets:
        print("Available dataset types:")
        for dataset_type in translator.list_available_datasets():
            print(f"  - {dataset_type}")
        return
    
    if args.list_processors:
        print("Available processor types:")
        for processor_type in translator.list_available_processors():
            print(f"  - {processor_type}")
        return
    
    # Check required arguments for translation
    if not args.model_type:
        parser.error("--model_type is required for translation")
    if not args.model_id:
        parser.error("--model_id is required for translation")
    if not args.dataset_type:
        parser.error("--dataset_type is required for translation")
    if not args.dataset_id:
        parser.error("--dataset_id is required for translation")
    if not args.columns:
        parser.error("--columns is required for translation")
    
    # Prepare kwargs for translation
    translation_kwargs = {}
    
    # Add model-specific kwargs
    if args.device:
        translation_kwargs["device"] = args.device
    if args.quantization:
        translation_kwargs["quantization"] = args.quantization
    if args.max_length:
        translation_kwargs["max_length"] = args.max_length
    if args.temperature:
        translation_kwargs["temperature"] = args.temperature
    if args.top_p:
        translation_kwargs["top_p"] = args.top_p
    if args.api_key:
        translation_kwargs["api_key"] = args.api_key
    if args.base_url:
        translation_kwargs["base_url"] = args.base_url
    
    # Add dataset-specific kwargs
    if args.subset:
        translation_kwargs["subset"] = args.subset
    if args.split:
        translation_kwargs["split"] = args.split
    if args.cache_dir:
        translation_kwargs["cache_dir"] = args.cache_dir
    
    # Add processing-specific kwargs
    if args.batch_size:
        translation_kwargs["batch_size"] = args.batch_size
    if args.chunk_size:
        translation_kwargs["chunk_size"] = args.chunk_size
    if args.resume_from:
        translation_kwargs["resume_from"] = args.resume_from
    
    # Run translation
    await translator.translate(
        model_type=args.model_type,
        model_id=args.model_id,
        dataset_type=args.dataset_type,
        dataset_id=args.dataset_id,
        columns_to_process=args.columns,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        output_dir=args.output_dir,
        **translation_kwargs
    )


if __name__ == "__main__":
    asyncio.run(main()) 