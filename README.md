<p align="center">
    <img src="https://huggingface.co/datasets/TransTa/images_icons/resolve/main/llama_translate.jpg" align="center" width="30%">
</p>
<p align="center"><h1 align="center">Local Translate</h1></p>
<p align="center">
   A powerful tool for translating datasets using local Large Language Models (LLMs) with efficient chunked processing and HuggingFace integration.
</p>
<p align="center">
	<img src="https://img.shields.io/github/license/Datanyth/translate_data_huggingface_with_llm?style=default&logo=opensourceinitiative&logoColor=white&color=0080ff" alt="license">
	<img src="https://img.shields.io/github/last-commit/Datanyth/translate_data_huggingface_with_llm?style=default&logo=git&logoColor=white&color=0080ff" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/Datanyth/translate_data_huggingface_with_llm?style=default&color=0080ff" alt="repo-top-language">
	<img src="https://img.shields.io/github/languages/count/Datanyth/translate_data_huggingface_with_llm?style=default&color=0080ff" alt="repo-language-count">
</p>
<br>

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#-prerequisites)
  - [Installation](#-installation)
  - [Usage](#-usage)
  - [Example](#-example)
- [Configuration](#-configuration)
- [Model Zoo](#-model-zoo)
- [Troubleshooting](#-troubleshooting)
- [Project Roadmap](#-project-roadmap)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 Overview

**Local Translate** is a specialized tool designed for translating large datasets using local Large Language Models (LLMs). It leverages the power of models like LLaMAX to efficiently translate text data from one language to another, with robust chunked processing to handle large datasets and seamless integration with the HuggingFace ecosystem.

### Key Capabilities:
- **Local LLM Translation**: Uses local LLMs for privacy and offline translation
- **Dynamic Multi-Model Architecture**: Flexible system supporting multiple model types and datasets
- **Chunked Processing**: Handles large datasets efficiently through chunked processing
- **HuggingFace Integration**: Direct integration with HuggingFace datasets and model hub
- **Resumable Processing**: Can resume translation from any chunk index
- **GPU-Only Inference**: Optimized for GPU-only execution without CPU offloading
- **Quantization Support**: Supports 4-bit and 8-bit quantization for memory efficiency
- **Multi-Column Translation**: Translate multiple columns in a single pass
- **Automatic Push to Hub**: Automatically push translated datasets to HuggingFace Hub

---

## ✨ Features

- 🔒 **Privacy-First**: All translation happens locally using your own LLM models
- 🏗️ **Dynamic Architecture**: Modular system supporting multiple model types (HuggingFace, VLLM, Ollama, OpenRouter)
- 📊 **Dataset Support**: Works with any HuggingFace dataset format
- 🚀 **Efficient Processing**: Chunked processing with configurable batch sizes
- 💾 **Memory Optimization**: Support for 4-bit and 8-bit quantization with GPU-only inference
- 🔄 **Resumable**: Continue translation from where you left off
- 🌐 **Multi-Language**: Support for any language pair your model supports
- 📈 **Progress Tracking**: Real-time progress monitoring with tqdm
- 🎯 **Flexible Columns**: Translate specific columns while preserving others
- 🔧 **Easy Configuration**: Simple command-line interface with comprehensive options
- ⚡ **GPU-Optimized**: No CPU offloading - everything runs on GPU for maximum performance

---

## 📁 Project Structure

```
local_translate/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── .pre-commit-config.yaml   # Pre-commit hooks configuration
├── scripts/
│   └── example_run.sh        # Example usage script
├── src/
│   ├── orchestrator.py       # Main orchestrator for dynamic translation
│   ├── config.py             # Configuration management
│   ├── core/
│   │   ├── base.py           # Base classes for models, datasets, processors
│   │   └── registry.py       # Registry system for components
│   ├── models/
│   │   └── huggingface_model.py  # HuggingFace model implementation
│   ├── data_loaders/
│   │   └── huggingface_dataset.py # HuggingFace dataset loader
│   └── processors/
│       └── translation_processor.py # Translation processing logic
├── test/
│   └── test_hf_models.py     # Test suite for HuggingFace models
├── utils/
│   └── utils.py              # Utility functions
└── output/                   # Output directory for translated datasets
```

### Core Components:

- **`DynamicTranslator`**: Main orchestrator handling the translation workflow
- **`HuggingFaceModel`**: Optimized HuggingFace model wrapper with GPU-only inference
- **`TranslationProcessor`**: Handles dataset translation with chunked processing
- **Registry System**: Dynamic registration and management of models, datasets, and processors

---

## 🚀 Getting Started

### Prerequisites

Before using Local Translate, ensure your system meets these requirements:

- **Python**: 3.8 or higher
- **CUDA**: Compatible GPU with CUDA support (recommended)
- **Memory**: Sufficient RAM/VRAM for your chosen model
- **Storage**: Adequate disk space for datasets and translated outputs

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Datanyth/translate_data_huggingface_with_llm
cd translate_data_huggingface_with_llm
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Verify installation:**
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

### Usage

#### New Dynamic Translator API (Recommended)

```python
import asyncio
from src.orchestrator import DynamicTranslator

async def translate_dataset():
    translator = DynamicTranslator()
    
    await translator.translate(
        model_type="huggingface",
        model_id="LLaMAX/LLaMAX3-8B-Alpaca",
        dataset_type="huggingface",
        dataset_id="presencesw/open-s1-small",
        columns_to_process=["problem", "solution"],
        source_lang="en",
        target_lang="vi",
        quantization="4bit",  # GPU-only 4-bit quantization
        device="cuda",
        output_dir="output"
    )

# Run the translation
asyncio.run(translate_dataset())
```

#### Test Script

Run the included test script to verify everything works:

```bash
python test/test_hf_models.py
```

### Example

Here's a complete example translating an English dataset to Vietnamese:

```python
#!/usr/bin/env python3
import asyncio
import os
from src.orchestrator import DynamicTranslator

async def main():
    translator = DynamicTranslator()
    
    await translator.translate(
        model_type="huggingface",
        model_id="LLaMAX/LLaMAX3-8B-Alpaca",
        dataset_type="huggingface",
        dataset_id="presencesw/open-s1-small",
        columns_to_process=["problem", "solution"],
        source_lang="en",
        target_lang="vi",
        quantization="4bit",  # GPU-only quantization
        device="cuda",
        model_dir=os.getenv("HF_HOME"),
        data_dir=os.getenv("HF_HOME"),
        cache_dir=".cache",
        output_dir="output",
        push_to_hub=True
    )

if __name__ == "__main__":
    asyncio.run(main())
```

---

## ⚙️ Configuration

### Model Configuration

| Parameter | Description | Example |
|-----------|-------------|---------|
| `model_type` | Type of model to use | `"huggingface"` |
| `model_id` | HuggingFace model identifier | `"LLaMAX/LLaMAX3-8B-Alpaca"` |
| `quantization` | Quantization level | `"4bit"` or `"8bit"` |
| `device` | Device to use | `"cuda"` |

### Dataset Configuration

| Parameter | Description | Example |
|-----------|-------------|---------|
| `dataset_type` | Type of dataset | `"huggingface"` |
| `dataset_id` | HuggingFace dataset identifier | `"presencesw/open-s1-small"` |
| `columns_to_process` | Columns to translate | `["problem", "solution"]` |

### Processing Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `source_lang` | Source language | `"en"` |
| `target_lang` | Target language | `"vi"` |
| `output_dir` | Output directory | `"output"` |
| `chunk_size` | Processing chunk size | `20` |
| `batch_size` | Processing batch size | `1` |

### Performance Tips

- **Use 4-bit quantization** for memory efficiency (recommended)
- **GPU-only inference** - no CPU offloading for maximum performance
- **Adjust chunk_size** based on your available memory
- **Use environment variables** for cache directories (`HF_HOME`, etc.)

---

## 🤖 Model Zoo

Local Translate is optimized for the **LLaMAX3-8B-Alpaca** model, which provides an excellent balance of translation quality and computational efficiency.

### 🏆 Recommended Model

| Model | Size | HuggingFace ID | Memory (FP16) | Memory (4-bit) | Memory (8-bit) | Languages | Notes |
|-------|------|----------------|---------------|----------------|----------------|-----------|-------|
| **LLaMAX3-8B-Alpaca** | 8B | `LLaMAX/LLaMAX3-8B-Alpaca` | ~16GB | ~4GB | ~8GB | Multi | ⭐ **Recommended** - Perfect balance of quality and efficiency |

### 💡 Model Selection Guide

#### **For Production Use**
- **Recommended**: `LLaMAX/LLaMAX3-8B-Alpaca` (4-bit quantization)
- **Memory requirement**: ~4GB VRAM
- **Best for**: Most translation tasks, excellent quality/speed balance

### 🔧 Model Usage Examples

#### GPU-Only 4-bit Quantization (Recommended):
```python
await translator.translate(
    model_type="huggingface",
    model_id="LLaMAX/LLaMAX3-8B-Alpaca",
    quantization="4bit",  # GPU-only, no CPU offload
    device="cuda",
    # ... other parameters
)
```

#### Full Precision Usage (if you have sufficient VRAM):
```python
await translator.translate(
    model_type="huggingface",
    model_id="LLaMAX/LLaMAX3-8B-Alpaca",
    device="cuda",
    # ... other parameters
)
```

### 📊 Memory Requirements

| Quantization Level | Memory Requirement | Recommended For |
|-------------------|-------------------|-----------------|
| FP16 (Full Precision) | ~16GB | High-end GPUs with 24GB+ VRAM |
| 8-bit Quantization | ~8GB | Mid-range GPUs with 12GB+ VRAM |
| 4-bit Quantization | ~4GB | Most GPUs with 6GB+ VRAM |

### ⚠️ Important Notes

- **GPU-Only Inference**: No CPU offloading - everything runs on GPU for maximum performance
- **Quantization**: Always use `quantization="4bit"` for memory efficiency unless you have high-end hardware
- **Model Compatibility**: Uses LLaMA architecture (`LlamaForCausalLM`)
- **Download Time**: ~15GB download on first use
- **Performance**: Excellent translation quality with reasonable speed
- **Memory**: Ensure your GPU has at least 4GB VRAM for 4-bit quantization

---

## 🔧 Troubleshooting

### Common Issues

#### Model Loading Hangs
- **Issue**: Model loading gets stuck during initialization
- **Solution**: Use GPU-only configuration with `quantization="4bit"` and ensure `llm_int8_enable_fp32_cpu_offload=False`

#### Memory Issues
- **Issue**: CUDA out of memory errors
- **Solution**: Use 4-bit quantization and reduce chunk_size

#### Slow Performance
- **Issue**: Translation is too slow
- **Solution**: Ensure GPU-only inference (no CPU offloading) and use appropriate quantization

### Performance Optimization

1. **Use 4-bit quantization** for best memory efficiency
2. **Keep everything on GPU** - avoid CPU offloading
3. **Adjust chunk_size** based on your GPU memory
4. **Use environment variables** for cache directories

### Debug Mode

Run the test script to verify your setup:

```bash
python test/test_hf_models.py
```

This will test the complete pipeline and identify any issues.

---

## 🎯 Project Roadmap

### Planned Features

- [ ] **Enhanced Model Support**: Add support for additional model architectures
- [ ] **Multi-GPU Support**: Parallel processing across multiple GPUs
- [ ] **Advanced Quantization**: Support for more quantization methods
- [ ] **Benchmark Suite**: Comprehensive benchmarking and metrics
- [ ] **Web Interface**: User-friendly web UI for configuration and monitoring
- [ ] **Batch Inference**: Optimized batch processing for faster translation
- [ ] **Language Detection**: Automatic language detection for source text
- [ ] **Quality Metrics**: Translation quality assessment tools
- [ ] **API Support**: REST API for integration with other tools
- [ ] **Docker Support**: Containerized deployment options

### Recent Updates

- ✅ **Dynamic Architecture**: Modular system supporting multiple model types
- ✅ **GPU-Only Inference**: Optimized for GPU-only execution without CPU offloading
- ✅ **Fixed Hanging Issues**: Resolved model loading and inference hanging problems
- ✅ **4-bit Quantization**: Memory-efficient quantization with proper configuration
- ✅ **Chunked Processing**: Efficient handling of large datasets
- ✅ **Resumable Processing**: Continue from any chunk index
- ✅ **HuggingFace Integration**: Seamless dataset and model management

---

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### Ways to Contribute

- **💬 Discussions**: Join our [GitHub Discussions](https://github.com/Datanyth/translate_data_huggingface_with_llm/discussions) to share ideas and ask questions
- **🐛 Bug Reports**: Report issues and bugs through [GitHub Issues](https://github.com/Datanyth/translate_data_huggingface_with_llm/issues)
- **💡 Feature Requests**: Suggest new features and improvements
- **📝 Code Contributions**: Submit pull requests with code improvements

### Development Setup

1. **Fork the repository**
2. **Clone your fork:**
   ```bash
   git clone https://github.com/your-username/translate_data_huggingface_with_llm
   cd translate_data_huggingface_with_llm
   ```
3. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Install development dependencies:**
   ```bash
   pip install -r requirements.txt
   pre-commit install
   ```
5. **Make your changes and test them**
6. **Commit with a clear message:**
   ```bash
   git commit -m "Add feature: description of changes"
   ```
7. **Push and create a pull request**

### Code Style

This project uses:
- **Black** for code formatting
- **Ruff** for linting and import sorting
- **Pre-commit hooks** for automated code quality checks

<details>
<summary>Contributor Graph</summary>
<br>
<p align="left">
   <a href="https://github.com/Datanyth/translate_data_huggingface_with_llm/graphs/contributors">
      <img src="https://contrib.rocks/image?repo=Datanyth/translate_data_huggingface_with_llm">
   </a>
</p>
</details>

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **HuggingFace**: For the excellent transformers library and dataset ecosystem
- **LLaMAX Team**: For providing the base models used in this project
- **Open Source Community**: For the various tools and libraries that make this project possible
- **Contributors**: Everyone who has contributed to improving this tool

---

## 📞 Support

If you need help or have questions:

- 📖 **Documentation**: Check this README and the code comments
- 💬 **Discussions**: Join our [GitHub Discussions](https://github.com/Datanyth/translate_data_huggingface_with_llm/discussions)
- 🐛 **Issues**: Report bugs via [GitHub Issues](https://github.com/Datanyth/translate_data_huggingface_with_llm/issues)
- ⭐ **Star the repo**: If you find this project useful!

---

*Made with ❤️ for the open-source community*
