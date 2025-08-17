"""
Core module for dynamic multi-model and multi-dataset system.
"""

from .base import BaseModel, BaseProcessor
from .registry import ModelRegistry, ProcessorRegistry

__all__ = [
    "BaseModel",
    "BaseProcessor",
    "ModelRegistry",
    "ProcessorRegistry"
] 