"""
HuggingFace dataset implementation with flexible subset and split handling.
"""

import asyncio
import random
from typing import List, Dict, Any, Union, Optional, cast
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from pathlib import Path
from ..core.base import BaseDataset, DatasetConfig
from ..core.registry import register_dataset


@register_dataset("huggingface")
class HuggingFaceDataset(BaseDataset):
    """Enhanced HuggingFace dataset implementation with flexible subset and split handling."""
    
    async def load(self) -> None:
        """Load the HuggingFace dataset with flexible split and subset handling."""
        try:
            # Try to load dataset with requested subset/config
            try:
                if self.config.subset:
                    self.dataset = load_dataset(
                        self.config.dataset_id,
                        self.config.subset,
                        cache_dir=self.config.data_dir,  # Use data_dir for raw dataset storage
                        **(self.config.additional_params or {})
                    )
                else:
                    self.dataset = load_dataset(
                        self.config.dataset_id,
                        cache_dir=self.config.data_dir,  # Use data_dir for raw dataset storage
                        **(self.config.additional_params or {})
                    )
            except Exception as e:
                # If requested subset/config is not available, try 'default'
                import warnings
                warnings.warn(f"Requested subset/config '{self.config.subset}' not available for dataset '{self.config.dataset_id}'. Error: {e}\nTrying 'default' subset/config if available.")
                self.dataset = load_dataset(
                    self.config.dataset_id,
                    'default',
                    cache_dir=self.config.data_dir,  # Use data_dir for raw dataset storage
                    **(self.config.additional_params or {})
                )

            # Determine available splits/configs
            available_splits = list(cast(DatasetDict, self.dataset).keys()) if hasattr(self.dataset, 'keys') else []

            # Handle multiple splits
            if isinstance(self.config.split, list):
                # Filter out splits that are not available, warn if any missing
                missing = [s for s in self.config.split if s not in available_splits]
                if missing:
                    import warnings
                    warnings.warn(f"Requested splits {missing} not available. Available splits: {available_splits}. They will be skipped.")
                splits_to_use = [s for s in self.config.split if s in available_splits]
                if not splits_to_use and 'default' in available_splits:
                    splits_to_use = ['default']
                self.dataset = self._handle_multiple_splits(splits_to_use)
            else:
                # Handle single split with potential percentage
                split_to_use = self.config.split
                if split_to_use not in available_splits:
                    import warnings
                    warnings.warn(f"Requested split '{split_to_use}' not available. Available splits: {available_splits}. Using 'default' if available.")
                    split_to_use = 'default' if 'default' in available_splits else available_splits[0] if available_splits else None
                if not split_to_use:
                    raise RuntimeError(f"No valid split found for dataset '{self.config.dataset_id}'. Available splits: {available_splits}")
                self.dataset = self._handle_single_split(split_to_use)

            # Apply post-processing operations
            self.dataset = self._apply_post_processing()
            
            self._loaded = True
            
        except Exception as e:
            raise RuntimeError(f"Failed to load HuggingFace dataset: {e}")
    
    def _handle_multiple_splits(self, splits: List[str]) -> Dataset:
        """Handle loading and combining multiple splits."""
        datasets = []
        
        for split in splits:
            if split in self.dataset:
                split_dataset = self.dataset[split]
                datasets.append(split_dataset)
            else:
                raise ValueError(f"Split '{split}' not found in dataset. Available splits: {list(cast(DatasetDict, self.dataset).keys())}")
        
        # Concatenate all splits
        if len(datasets) > 1:
            return concatenate_datasets(datasets)
        else:
            return datasets[0]
    
    def _handle_single_split(self, split: str) -> Dataset:
        """Handle single split with potential percentage-based splitting."""
        if split not in self.dataset:
            raise ValueError(f"Split '{split}' not found in dataset. Available splits: {list(cast(DatasetDict, self.dataset).keys())}")
        
        dataset = self.dataset[split]
        
        # Handle percentage-based splits
        if self.config.split_percentage is not None:
            dataset = self._apply_percentage_split(dataset, self.config.split_percentage)
        
        return dataset
    
    def _apply_percentage_split(self, dataset: Dataset, percentage: float) -> Dataset:
        """Apply percentage-based split to the dataset."""
        if not 0 < percentage <= 1:
            raise ValueError(f"Percentage must be between 0 and 1, got {percentage}")
        
        total_size = len(dataset)
        split_size = int(total_size * percentage)
        
        # Set random seed if provided
        if self.config.split_seed is not None:
            random.seed(self.config.split_seed)
        
        # Create random indices
        indices = list(range(total_size))
        random.shuffle(indices)
        
        # Select the first percentage of samples
        selected_indices = indices[:split_size]
        
        return dataset.select(selected_indices)
    
    def _apply_post_processing(self) -> Dataset:
        """Apply post-processing operations to the dataset."""
        dataset = self.dataset
        
        # Apply filtering conditions
        if self.config.filter_conditions:
            dataset = self._apply_filtering(dataset)
        
        # Apply column selection/exclusion
        if self.config.select_columns or self.config.exclude_columns:
            dataset = self._apply_column_operations(dataset)
        
        # Apply shuffling
        if self.config.shuffle:
            dataset = self._apply_shuffling(dataset)
        
        # Apply offset and max_samples
        if self.config.offset is not None or self.config.max_samples is not None:
            dataset = self._apply_sampling(dataset)
        
        return dataset
    
    def _apply_filtering(self, dataset: Dataset) -> Dataset:
        """Apply filtering conditions to the dataset."""
        for column, condition in self.config.filter_conditions.items():
            if isinstance(condition, dict):
                # Handle range conditions
                if 'min' in condition and 'max' in condition:
                    dataset = dataset.filter(
                        lambda x: self._safe_compare(x[column], condition['min'], '>=') and 
                                 self._safe_compare(x[column], condition['max'], '<=')
                    )
                elif 'min' in condition:
                    dataset = dataset.filter(
                        lambda x: self._safe_compare(x[column], condition['min'], '>=')
                    )
                elif 'max' in condition:
                    dataset = dataset.filter(
                        lambda x: self._safe_compare(x[column], condition['max'], '<=')
                    )
                # Handle in/not_in conditions
                elif 'in' in condition:
                    dataset = dataset.filter(
                        lambda x: x[column] in condition['in']
                    )
                elif 'not_in' in condition:
                    dataset = dataset.filter(
                        lambda x: x[column] not in condition['not_in']
                    )
            else:
                # Simple equality filter
                dataset = dataset.filter(
                    lambda x: x[column] == condition
                )
        
        return dataset
    
    def _safe_compare(self, value, threshold, operator):
        """Safely compare values, handling different data types."""
        try:
            if isinstance(value, str):
                # For strings, compare length
                if operator == '>=':
                    return len(value) >= threshold
                elif operator == '<=':
                    return len(value) <= threshold
                elif operator == '>':
                    return len(value) > threshold
                elif operator == '<':
                    return len(value) < threshold
                else:
                    return len(value) == threshold
            else:
                # For numbers, compare directly
                if operator == '>=':
                    return value >= threshold
                elif operator == '<=':
                    return value <= threshold
                elif operator == '>':
                    return value > threshold
                elif operator == '<':
                    return value < threshold
                else:
                    return value == threshold
        except (TypeError, ValueError):
            # If comparison fails, return False
            return False
    
    def _apply_column_operations(self, dataset: Dataset) -> Dataset:
        """Apply column selection and exclusion operations."""
        all_columns = dataset.column_names
        
        # Start with all columns
        selected_columns = all_columns.copy()
        
        # Apply exclusion first
        if self.config.exclude_columns:
            selected_columns = [col for col in selected_columns if col not in self.config.exclude_columns]
        
        # Apply selection (overrides exclusion)
        if self.config.select_columns:
            # Validate that all selected columns exist
            missing_columns = [col for col in self.config.select_columns if col not in all_columns]
            if missing_columns:
                raise ValueError(f"Selected columns not found in dataset: {missing_columns}")
            selected_columns = self.config.select_columns
        
        return dataset.select_columns(selected_columns)
    
    def _apply_shuffling(self, dataset: Dataset) -> Dataset:
        """Apply shuffling to the dataset."""
        if self.config.shuffle_seed is not None:
            random.seed(self.config.shuffle_seed)
        
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        
        return dataset.select(indices)
    
    def _apply_sampling(self, dataset: Dataset) -> Dataset:
        """Apply offset and max_samples to the dataset."""
        start_idx = self.config.offset or 0
        end_idx = len(dataset)
        
        if self.config.max_samples is not None:
            end_idx = min(start_idx + self.config.max_samples, len(dataset))
        
        return dataset.select(range(start_idx, end_idx))
    
    def get_size(self) -> int:
        """Get the total number of samples in the dataset."""
        if not self._loaded:
            return 0
        return len(self.dataset)
    
    def get_columns(self) -> List[str]:
        """Get the column names in the dataset."""
        if not self._loaded:
            return []
        return list(self.dataset.column_names)
    
    def get_sample(self, index: int) -> Dict[str, Any]:
        """Get a single sample by index."""
        if not self._loaded:
            raise RuntimeError("Dataset not loaded")
        
        if index >= len(self.dataset):
            raise IndexError(f"Index {index} out of range")
        
        return dict(self.dataset[index])
    
    def get_batch(self, start_idx: int, end_idx: int) -> List[Dict[str, Any]]:
        """Get a batch of samples."""
        if not self._loaded:
            raise RuntimeError("Dataset not loaded")
        
        if start_idx >= len(self.dataset) or end_idx > len(self.dataset):
            raise IndexError(f"Batch indices out of range")
        
        # Get the batch data as a dataset slice
        batch_dataset = self.dataset.select(range(start_idx, end_idx))
        
        # Convert each sample to a dictionary
        batch_data = []
        for i in range(len(batch_dataset)):
            sample = batch_dataset[i]
            if isinstance(sample, dict):
                batch_data.append(sample)
            else:
                # If sample is not a dict, try to convert it
                try:
                    batch_data.append(dict(sample))
                except (TypeError, ValueError):
                    # If conversion fails, create a simple dict with the data
                    batch_data.append({"data": sample})
        
        return batch_data
    
    def get_available_splits(self) -> List[str]:
        """Get available splits in the dataset."""
        if not hasattr(self, 'dataset') or self.dataset is None:
            return []
        
        if hasattr(self.dataset, 'keys'):
            return list(self.dataset.keys())
        elif hasattr(self.dataset, 'column_names'):
            return list(self.dataset.column_names)
        else:
            return []
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get comprehensive dataset statistics."""
        if not self._loaded:
            return {}
        
        stats = {
            "total_samples": len(self.dataset),
            "columns": self.get_columns(),
            "column_types": {},
            "memory_usage": None
        }
        
        # Get column types and sample values
        if hasattr(self.dataset, '__len__') and len(self.dataset) > 0:
            sample = self.dataset[0]
            for column in self.get_columns():
                value = sample[column]
                stats["column_types"][column] = type(value).__name__
        
        # Try to get memory usage
        try:
            stats["memory_usage"] = self.dataset.data.nbytes if hasattr(self.dataset, 'data') else None
        except:
            pass
        
        return stats
    
    def save(self, data: List[Dict[str, Any]], output_path: str) -> None:
        """Save processed data."""
        try:
            # Create output directory if it doesn't exist
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Convert to HuggingFace dataset
            dataset = Dataset.from_list(data)
            
            # Save dataset
            dataset.save_to_disk(output_path)
            
        except Exception as e:
            raise RuntimeError(f"Failed to save dataset: {e}") 