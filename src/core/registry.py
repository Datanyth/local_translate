"""
Registry system for dynamic model and dataset registration.
"""

from typing import Dict, Type, Any, Optional, Callable, List
from .base import BaseModel, BaseDataset, BaseProcessor


class ModelRegistry:
    """Registry for model implementations."""
    
    _instance = None
    _models: Dict[str, Type[BaseModel]] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def register(self, model_type: str, model_class: Type[BaseModel]) -> None:
        """Register a model implementation."""
        self._models[model_type] = model_class
    
    def get(self, model_type: str) -> Optional[Type[BaseModel]]:
        """Get a model implementation by type."""
        return self._models.get(model_type)
    
    def list_models(self) -> List[str]:
        """List all registered model types."""
        return list(self._models.keys())
    
    def create(self, model_type: str, config: Any) -> BaseModel:
        """Create a model instance."""
        model_class = self.get(model_type)
        if model_class is None:
            raise ValueError(f"Unknown model type: {model_type}")
        return model_class(config)


class DatasetRegistry:
    """Registry for dataset implementations."""
    
    _instance = None
    _datasets: Dict[str, Type[BaseDataset]] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def register(self, dataset_type: str, dataset_class: Type[BaseDataset]) -> None:
        """Register a dataset implementation."""
        self._datasets[dataset_type] = dataset_class
    
    def get(self, dataset_type: str) -> Optional[Type[BaseDataset]]:
        """Get a dataset implementation by type."""
        return self._datasets.get(dataset_type)
    
    def list_datasets(self) -> List[str]:
        """List all registered dataset types."""
        return list(self._datasets.keys())
    
    def create(self, dataset_type: str, config: Any) -> BaseDataset:
        """Create a dataset instance."""
        dataset_class = self.get(dataset_type)
        if dataset_class is None:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        return dataset_class(config)


class ProcessorRegistry:
    """Registry for processor implementations."""
    
    _instance = None
    _processors: Dict[str, Type[BaseProcessor]] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def register(self, processor_type: str, processor_class: Type[BaseProcessor]) -> None:
        """Register a processor implementation."""
        self._processors[processor_type] = processor_class
    
    def get(self, processor_type: str) -> Optional[Type[BaseProcessor]]:
        """Get a processor implementation by type."""
        return self._processors.get(processor_type)
    
    def list_processors(self) -> List[str]:
        """List all registered processor types."""
        return list(self._processors.keys())
    
    def create(self, processor_type: str, model: BaseModel, dataset: BaseDataset, config: Any) -> BaseProcessor:
        """Create a processor instance."""
        processor_class = self.get(processor_type)
        if processor_class is None:
            raise ValueError(f"Unknown processor type: {processor_type}")
        return processor_class(model, dataset, config)


# Decorators for easy registration
def register_model(model_type: str):
    """Decorator to register a model implementation."""
    def decorator(model_class: Type[BaseModel]) -> Type[BaseModel]:
        ModelRegistry().register(model_type, model_class)
        return model_class
    return decorator


def register_dataset(dataset_type: str):
    """Decorator to register a dataset implementation."""
    def decorator(dataset_class: Type[BaseDataset]) -> Type[BaseDataset]:
        DatasetRegistry().register(dataset_type, dataset_class)
        return dataset_class
    return decorator


def register_processor(processor_type: str):
    """Decorator to register a processor implementation."""
    def decorator(processor_class: Type[BaseProcessor]) -> Type[BaseProcessor]:
        ProcessorRegistry().register(processor_type, processor_class)
        return processor_class
    return decorator 