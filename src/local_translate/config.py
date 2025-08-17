"""
Configuration management for the dynamic multi-model and multi-dataset system.
"""

import os
from typing import Optional, Dict, Any
from pathlib import Path
from dotenv import load_dotenv


class Config:
    """Configuration manager for environment variables and defaults."""
    
    def __init__(self, env_file: Optional[str] = None):
        """Initialize configuration manager.
        
        Args:
            env_file: Path to .env file. If None, will look for .env in current directory.
        """
        # Load environment variables
        if env_file:
            load_dotenv(env_file)
        else:
            # Try to load .env from current directory
            env_path = Path.cwd() / ".env"
            if env_path.exists():
                load_dotenv(env_path)
            else:
                # Load from project root
                project_root = Path(__file__).parent.parent
                env_path = project_root / ".env"
                if env_path.exists():
                    load_dotenv(env_path)
        
        # Set default values
        self._set_defaults()
    
    def _set_defaults(self):
        """Set default values for configuration."""
        # HuggingFace Configuration
        self.huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
        self.huggingface_cache_dir = os.getenv("HUGGINGFACE_CACHE_DIR", ".cache/huggingface")
        
        # OpenRouter Configuration
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        self.openrouter_base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        
        # Ollama Configuration
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        # VLLM Configuration
        self.vllm_gpu_memory_utilization = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.9"))
        self.vllm_max_model_len = int(os.getenv("VLLM_MAX_MODEL_LEN", "8000"))
        
        # Model Defaults
        self.default_model_temperature = float(os.getenv("DEFAULT_MODEL_TEMPERATURE", "0.7"))
        self.default_model_top_p = float(os.getenv("DEFAULT_MODEL_TOP_P", "0.9"))
        self.default_model_max_length = int(os.getenv("DEFAULT_MODEL_MAX_LENGTH", "8000"))
        
        # Processing Defaults
        self.default_batch_size = int(os.getenv("DEFAULT_BATCH_SIZE", "1"))
        self.default_chunk_size = int(os.getenv("DEFAULT_CHUNK_SIZE", "20"))
        self.default_max_workers = int(os.getenv("DEFAULT_MAX_WORKERS", "4"))
        
        # Directory Configuration
        self.default_output_dir = os.getenv("DEFAULT_OUTPUT_DIR", "output")
        self.default_cache_dir = os.getenv("DEFAULT_CACHE_DIR", ".cache")  # For processing/intermediate storage
        self.default_data_dir = os.getenv("DEFAULT_DATA_DIR", ".data")    # For raw dataset storage
        self.default_model_dir = os.getenv("DEFAULT_MODEL_DIR", ".models") # For model storage
        
        # Language Configuration
        self.default_source_language = os.getenv("DEFAULT_SOURCE_LANGUAGE", "English")
        self.default_target_language = os.getenv("DEFAULT_TARGET_LANGUAGE", "Vietnamese")
        
        # Device Configuration
        self.default_device = os.getenv("DEFAULT_DEVICE", "auto")
        
        # Logging Configuration
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.enable_progress_bar = os.getenv("ENABLE_PROGRESS_BAR", "true").lower() == "true"
    
    def get_model_config(self, model_type: str, model_id: str, **kwargs) -> Dict[str, Any]:
        """Get model configuration with defaults and environment variables.
        
        Args:
            model_type: Type of model (huggingface, vllm, ollama, openrouter)
            model_id: Model identifier
            **kwargs: Additional configuration parameters
            
        Returns:
            Dictionary with model configuration
        """
        config = {
            "model_id": model_id,
            "model_type": model_type,
            "device": kwargs.get("device", self.default_device),
            "max_length": kwargs.get("max_length", self.default_model_max_length),
            "temperature": kwargs.get("temperature", self.default_model_temperature),
            "top_p": kwargs.get("top_p", self.default_model_top_p),
            "model_dir": kwargs.get("model_dir", self.default_model_dir),
            "additional_params": kwargs.get("additional_params", {})
        }
        
        # Add model-specific configurations
        if model_type == "huggingface":
            if self.huggingface_token:
                config["additional_params"]["token"] = self.huggingface_token
            # Note: model_dir is handled separately, not in additional_params
        
        elif model_type == "openrouter":
            if self.openrouter_api_key:
                config["api_key"] = self.openrouter_api_key
            config["base_url"] = kwargs.get("base_url", self.openrouter_base_url)
        
        elif model_type == "ollama":
            config["base_url"] = kwargs.get("base_url", self.ollama_base_url)
        
        elif model_type == "vllm":
            config["additional_params"].update({
                "gpu_memory_utilization": kwargs.get("gpu_memory_utilization", self.vllm_gpu_memory_utilization),
                "max_model_len": kwargs.get("max_model_len", self.vllm_max_model_len)
            })
        
        # Handle quantization parameters
        quantization = None
        if "quantization" in kwargs:
            quantization = kwargs["quantization"]
        elif kwargs.get("use_4bit", False):
            quantization = "4bit"
        elif kwargs.get("use_8bit", False):
            quantization = "8bit"
        
        if quantization:
            config["quantization"] = quantization
        
        return config
    
    def get_dataset_config(self, dataset_type: str, dataset_id: str, **kwargs) -> Dict[str, Any]:
        """Get dataset configuration with defaults and environment variables.
        
        Args:
            dataset_type: Type of dataset (huggingface, pandas, pytorch)
            dataset_id: Dataset identifier
            **kwargs: Additional configuration parameters
            
        Returns:
            Dictionary with dataset configuration
        """
        config = {
            "dataset_id": dataset_id,
            "dataset_type": dataset_type,
            "subset": kwargs.get("subset"),
            "split": kwargs.get("split", "train"),
            "cache_dir": kwargs.get("cache_dir", self.default_cache_dir),  # For processing/intermediate storage
            "data_dir": kwargs.get("data_dir", self.default_data_dir),      # For raw dataset storage
            "additional_params": kwargs.get("additional_params", {})
        }
        
        # Add dataset-specific configurations
        if dataset_type == "huggingface":
            if self.huggingface_token:
                config["additional_params"]["token"] = self.huggingface_token
            # Use data_dir for raw HuggingFace dataset caching, cache_dir for processing
            if self.huggingface_cache_dir:
                config["data_dir"] = kwargs.get("data_dir", self.huggingface_cache_dir)
            
            # Add flexible HuggingFace-specific parameters
            config.update({
                "split_percentage": kwargs.get("split_percentage"),
                "split_seed": kwargs.get("split_seed"),
                "filter_conditions": kwargs.get("filter_conditions", {}),
                "select_columns": kwargs.get("select_columns", []),
                "exclude_columns": kwargs.get("exclude_columns", []),
                "shuffle": kwargs.get("shuffle", False),
                "shuffle_seed": kwargs.get("shuffle_seed"),
                "max_samples": kwargs.get("max_samples"),
                "offset": kwargs.get("offset")
            })
        
        return config
    
    def get_processing_config(self, **kwargs) -> Dict[str, Any]:
        """Get processing configuration with defaults and environment variables.
        
        Args:
            **kwargs: Additional configuration parameters
            
        Returns:
            Dictionary with processing configuration
        """
        config = {
            "batch_size": kwargs.get("batch_size", self.default_batch_size),
            "max_workers": kwargs.get("max_workers", self.default_max_workers),
            "chunk_size": kwargs.get("chunk_size", self.default_chunk_size),
            "resume_from": kwargs.get("resume_from", 0),
            "output_format": kwargs.get("output_format", "huggingface"),
            "output_dir": kwargs.get("output_dir", self.default_output_dir)
        }
        
        return config
    
    def get_translation_config(self, **kwargs) -> Dict[str, Any]:
        """Get translation configuration with defaults and environment variables.
        
        Args:
            **kwargs: Additional configuration parameters
            
        Returns:
            Dictionary with translation configuration
        """
        config = {
            "source_lang": kwargs.get("source_lang", self.default_source_language),
            "target_lang": kwargs.get("target_lang", self.default_target_language),
            "enable_progress_bar": kwargs.get("enable_progress_bar", self.enable_progress_bar)
        }
        
        return config
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate configuration and return validation results.
        
        Returns:
            Dictionary with validation results
        """
        validation = {
            "valid": True,
            "warnings": [],
            "errors": []
        }
        
        # Check required tokens
        if not self.huggingface_token:
            validation["warnings"].append("HUGGINGFACE_TOKEN not set - some models may not work")
        
        if not self.openrouter_api_key:
            validation["warnings"].append("OPENROUTER_API_KEY not set - OpenRouter models will not work")
        
        # Check directories
        for dir_name, dir_path in [
            ("Output", self.default_output_dir),
            ("Cache", self.default_cache_dir),
            ("Data", self.default_data_dir),
            ("Model", self.default_model_dir),
            ("HuggingFace Cache", self.huggingface_cache_dir)
        ]:
            try:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                validation["errors"].append(f"Cannot create {dir_name} directory: {e}")
                validation["valid"] = False
        
        # Check Ollama connection if needed (synchronous check)
        if self.ollama_base_url:
            try:
                import requests
                
                try:
                    response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
                    if response.status_code != 200:
                        validation["warnings"].append("Ollama server not accessible - Ollama models will not work")
                except:
                    validation["warnings"].append("Ollama server not accessible - Ollama models will not work")
            except ImportError:
                validation["warnings"].append("requests not installed - cannot check Ollama connection")
        
        return validation
    
    def print_config_summary(self):
        """Print a summary of the current configuration."""
        print("Configuration Summary:")
        print("=" * 50)
        
        print(f"HuggingFace Token: {'✓ Set' if self.huggingface_token else '✗ Not set'}")
        print(f"OpenRouter API Key: {'✓ Set' if self.openrouter_api_key else '✗ Not set'}")
        print(f"Ollama Base URL: {self.ollama_base_url}")
        print(f"Default Device: {self.default_device}")
        print(f"Default Output Dir: {self.default_output_dir}")
        print(f"Default Cache Dir: {self.default_cache_dir} (processing/intermediate)")
        print(f"Default Data Dir: {self.default_data_dir} (raw dataset storage)")
        print(f"Default Model Dir: {self.default_model_dir} (model storage)")
        print(f"Default Languages: {self.default_source_language} → {self.default_target_language}")
        print(f"Log Level: {self.log_level}")
        print(f"Progress Bar: {'✓ Enabled' if self.enable_progress_bar else '✗ Disabled'}")
        
        # Validate configuration
        validation = self.validate_config()
        if validation["warnings"]:
            print("\nWarnings:")
            for warning in validation["warnings"]:
                print(f"  ⚠️  {warning}")
        
        if validation["errors"]:
            print("\nErrors:")
            for error in validation["errors"]:
                print(f"  ❌ {error}")
        
        if validation["valid"]:
            print("\n✅ Configuration is valid!")
        else:
            print("\n❌ Configuration has errors!")


# Global configuration instance
config = Config() 