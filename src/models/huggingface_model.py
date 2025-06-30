"""
HuggingFace model implementation for local models.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils.quantization_config import BitsAndBytesConfig
from typing import List, Dict, Any
import asyncio
from ..core.base import BaseModel, ModelConfig
from ..core.registry import register_model


@register_model("huggingface")
class HuggingFaceModel(BaseModel):
    """HuggingFace local model implementation."""
    
    async def initialize(self) -> None:
        """Initialize the HuggingFace model."""
        try:
            # Configure quantization with proper CPU offload support
            quantization_config = None
            if self.config.quantization == "4bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            elif self.config.quantization == "8bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_id,
                cache_dir=self.config.model_dir,
                trust_remote_code=True
            )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Prepare model loading arguments
            model_kwargs = {
                "cache_dir": self.config.model_dir,
                "trust_remote_code": True,
            }
            
            # Add additional params if they exist and are a dict
            if self.config.additional_params and isinstance(self.config.additional_params, dict):
                model_kwargs.update(self.config.additional_params)
            
            # Handle quantization and device mapping (NO CPU OFFLOAD)
            if quantization_config is not None:
                model_kwargs["quantization_config"] = quantization_config
                # model_kwargs["quantization_config"].llm_int8_enable_fp32_cpu_offload = False
                # Keep model fully on GPU - no CPU offloading
                model_kwargs["device_map"] = "auto"
                # Explicitly disable CPU offload to keep model on GPU avoid pending
            else:
                # For non-quantized models, use the specified device
                model_kwargs["device_map"] = self.config.device
                model_kwargs["torch_dtype"] = torch.float16
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                **model_kwargs
            )
            
            self._initialized = True
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize HuggingFace model: {e}")
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt."""
        if not self._initialized:
            await self.initialize()
        
        # Use provided kwargs or default config values
        max_length = kwargs.get('max_length', self.config.max_length)
        temperature = kwargs.get('temperature', self.config.temperature)
        top_p = kwargs.get('top_p', self.config.top_p)
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        )
        
        # Move to device
        if self.model is not None and hasattr(self.model, 'device'):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        if self.model is None:
            raise RuntimeError("Model is not initialized")
            
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode output
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text.strip()
    
    async def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate text for multiple prompts."""
        if not self._initialized:
            await self.initialize()
        
        results = []
        for prompt in prompts:
            try:
                result = await self.generate(prompt, **kwargs)
                results.append(result)
            except Exception as e:
                print(f"Error generating for prompt: {e}")
                results.append("")
        
        return results
    
    def is_available(self) -> bool:
        """Check if the model is available and ready."""
        return self._initialized and self.model is not None
    
    async def cleanup(self) -> None:
        """Clean up model resources."""
        if self.model is not None:
            del self.model
            self.model = None
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        self._initialized = False
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache() 