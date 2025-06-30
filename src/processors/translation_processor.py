"""
Translation processor implementation.
"""

import asyncio
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
import json
from tqdm import tqdm
from ..core.base import BaseProcessor, ProcessingConfig
from ..core.registry import register_processor


@register_processor("translation")
class TranslationProcessor(BaseProcessor):
    """Translation processor implementation."""
    
    def __init__(self, model, dataset, config: ProcessingConfig):
        super().__init__(model, dataset, config)
        self.processed_data = []
        self.current_chunk = 0
        # Initialize preprocess and postprocess functions
        self.preprocess_fn = None
        self.postprocess_fn = None
    
    async def process(self, columns_to_process: List[str], source_lang: str = "English", 
                     target_lang: str = "Vietnamese", **kwargs) -> None:
        """Process the dataset with the specified columns."""
        try:
            # Extract preprocess and postprocess functions from kwargs
            self.preprocess_fn = kwargs.get('preprocess', None)
            self.postprocess_fn = kwargs.get('postprocess', None)
            
            # Load dataset if not already loaded
            if not self.dataset._loaded:
                await self.dataset.load()
            
            # Initialize model if not already initialized
            if not self.model._initialized:
                await self.model.initialize()
            
            # Get dataset size
            total_samples = self.dataset.get_size()
            print(f"Processing {total_samples} samples with {len(columns_to_process)} columns")
            
            # Process in chunks
            start_idx = self.config.resume_from * self.config.chunk_size
            end_idx = min(start_idx + self.config.chunk_size, total_samples)
            
            with tqdm(total=total_samples, initial=start_idx, desc="Processing") as pbar:
                while start_idx < total_samples:
                    # Get batch of data
                    batch_data = self.dataset.get_batch(start_idx, end_idx)
                    
                    # Process chunk
                    processed_chunk = await self.process_chunk(
                        batch_data, columns_to_process, source_lang, target_lang, **kwargs
                    )
                    
                    # Save processed chunk
                    chunk_filename = f"chunk_{self.current_chunk:04d}"
                    output_dir = self.config.output_dir or "output"
                    chunk_path = Path(output_dir) / chunk_filename
                    
                    self.dataset.save(processed_chunk, str(chunk_path))
                    
                    # Update progress
                    self.processed_data.extend(processed_chunk)
                    pbar.update(len(processed_chunk))
                    
                    # Move to next chunk
                    start_idx = end_idx
                    end_idx = min(start_idx + self.config.chunk_size, total_samples)
                    self.current_chunk += 1
            
            # Save final combined dataset
            if self.config.output_dir:
                final_output_path = Path(self.config.output_dir) / "translated_dataset"
                self.dataset.save(self.processed_data, str(final_output_path))
                print(f"Final dataset saved to: {final_output_path}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to process dataset: {e}")
    
    async def process_chunk(self, chunk_data: List[Dict[str, Any]], 
                          columns_to_process: List[str], source_lang: str, 
                          target_lang: str, **kwargs) -> List[Dict[str, Any]]:
        """Process a chunk of data."""
        processed_chunk = []
        
        for sample in chunk_data:
            processed_sample = sample.copy()
            
            # Process each specified column
            for column in columns_to_process:
                if column in sample:
                    text = sample[column]
                    if text and isinstance(text, str):
                        # Create translation prompt using appropriate method
                        if self.preprocess_fn:
                            prompt = self.preprocess_fn(text, source_lang, target_lang)
                        else:
                            prompt = self.create_prompt(text, source_lang, target_lang)
                        
                        try:
                            # Generate translation
                            translated_text = await self.model.generate(prompt, **kwargs)
                            
                            # Apply postprocessing if provided
                            if self.postprocess_fn:
                                translated_text = self.postprocess_fn(prompt, translated_text)
                            
                            # Add translated column
                            translated_column = f"{column}_translated"
                            processed_sample[translated_column] = translated_text
                            
                        except Exception as e:
                            print(f"Error translating column {column}: {e}")
                            processed_sample[f"{column}_translated"] = text  # Keep original
                    else:
                        processed_sample[f"{column}_translated"] = text  # Keep original
            
            processed_chunk.append(processed_sample)
        
        return processed_chunk
    
    def create_prompt(self, text: str, source_lang: str, target_lang: str) -> str:
        """Create a translation prompt."""
        prompt = f"""Please translate the following text from {source_lang} to {target_lang}. 
Only provide the translated text without any additional explanations or formatting.

Text to translate:
{text}

Translation:"""
        return prompt
    
    def get_processing_info(self) -> Dict[str, Any]:
        """Get processing information."""
        info = super().get_processing_info()
        info.update({
            "processed_samples": len(self.processed_data),
            "current_chunk": self.current_chunk
        })
        return info 