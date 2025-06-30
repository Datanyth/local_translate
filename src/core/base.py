"""
Abstract base classes for the dynamic multi-model and multi-dataset system.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import asyncio
from pydantic import BaseModel as PydanticBaseModel


@dataclass
class ModelConfig:
    """Configuration for model initialization."""
    model_id: str
    model_type: str
    device: str = "auto"
    quantization: Optional[str] = None  # "4bit", "8bit", None
    max_length: int = 8000
    temperature: float = 0.7
    top_p: float = 0.9
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model_dir: Optional[str] = None
    additional_params: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.additional_params is None:
            self.additional_params = {}


@dataclass
class DatasetConfig:
    """Configuration for dataset loading."""
    dataset_id: str
    dataset_type: str
    subset: Optional[str] = None
    split: Union[str, List[str]] = "train"  # Can be single split or list of splits
    cache_dir: Optional[str] = None         # For processing/intermediate storage
    data_dir: Optional[str] = None          # For raw dataset storage (like HF_HOME)
    additional_params: Optional[Dict[str, Any]] = None
    # New flexible parameters for HuggingFace datasets
    split_percentage: Optional[float] = None  # For percentage-based splits
    split_seed: Optional[int] = None  # Random seed for split operations
    filter_conditions: Optional[Dict[str, Any]] = None  # Filter conditions
    select_columns: Optional[List[str]] = None  # Columns to select
    exclude_columns: Optional[List[str]] = None  # Columns to exclude
    shuffle: bool = False  # Whether to shuffle the dataset
    shuffle_seed: Optional[int] = None  # Random seed for shuffling
    max_samples: Optional[int] = None  # Maximum number of samples to load
    offset: Optional[int] = None  # Offset for loading samples

    def __post_init__(self):
        if self.additional_params is None:
            self.additional_params = {}
        if self.filter_conditions is None:
            self.filter_conditions = {}
        if self.select_columns is None:
            self.select_columns = []
        if self.exclude_columns is None:
            self.exclude_columns = []


@dataclass
class ProcessingConfig:
    """Configuration for data processing."""
    batch_size: int = 1
    max_workers: int = 4
    chunk_size: int = 20
    resume_from: int = 0
    output_format: str = "huggingface"
    output_dir: Optional[str] = None


class BaseModel(ABC):
    """Abstract base class for all model implementations."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self._initialized = False
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the model."""
        pass
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt."""
        pass
    
    @abstractmethod
    async def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate text for multiple prompts."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the model is available and ready."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up model resources."""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_id": self.config.model_id,
            "model_type": self.config.model_type,
            "device": self.config.device,
            "quantization": self.config.quantization,
            "max_length": self.config.max_length
        }


class BaseDataset(ABC):
    """Abstract base class for all dataset implementations."""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.dataset = None
        self._loaded = False
    
    @abstractmethod
    async def load(self) -> None:
        """Load the dataset."""
        pass
    
    @abstractmethod
    def get_size(self) -> int:
        """Get the total number of samples in the dataset."""
        pass
    
    @abstractmethod
    def get_columns(self) -> List[str]:
        """Get the column names in the dataset."""
        pass
    
    @abstractmethod
    def get_sample(self, index: int) -> Dict[str, Any]:
        """Get a single sample by index."""
        pass
    
    @abstractmethod
    def get_batch(self, start_idx: int, end_idx: int) -> List[Dict[str, Any]]:
        """Get a batch of samples."""
        pass
    
    @abstractmethod
    def save(self, data: List[Dict[str, Any]], output_path: str) -> None:
        """Save processed data."""
        pass
    
    @abstractmethod
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get comprehensive dataset statistics."""
        pass
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get dataset information."""
        return {
            "dataset_id": self.config.dataset_id,
            "dataset_type": self.config.dataset_type,
            "subset": self.config.subset,
            "split": self.config.split,
            "size": self.get_size() if self._loaded else None,
            "columns": self.get_columns() if self._loaded else []
        }


class BaseProcessor(ABC):
    """Abstract base class for data processors."""
    
    def __init__(self, model: BaseModel, dataset: BaseDataset, config: ProcessingConfig):
        self.model = model
        self.dataset = dataset
        self.config = config
    
    @abstractmethod
    async def process(self, columns_to_process: List[str], **kwargs) -> None:
        """Process the dataset with the specified columns."""
        pass
    
    @abstractmethod
    async def process_chunk(self, chunk_data: List[Dict[str, Any]], 
                          columns_to_process: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Process a chunk of data."""
        pass
    
    @abstractmethod
    def create_prompt(self, text: str, source_lang: str, target_lang: str) -> str:
        """Create a translation prompt."""
        pass
    
    def get_processing_info(self) -> Dict[str, Any]:
        """Get processing information."""
        return {
            "model_info": self.model.get_model_info(),
            "dataset_info": self.dataset.get_dataset_info(),
            "processing_config": {
                "batch_size": self.config.batch_size,
                "chunk_size": self.config.chunk_size,
                "max_workers": self.config.max_workers,
                "resume_from": self.config.resume_from
            }
        } 