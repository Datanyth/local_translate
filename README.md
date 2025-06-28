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
- [Project Roadmap](#-project-roadmap)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 Overview

**Local Translate** is a specialized tool designed for translating large datasets using local Large Language Models (LLMs). It leverages the power of models like LLaMAX to efficiently translate text data from one language to another, with robust chunked processing to handle large datasets and seamless integration with the HuggingFace ecosystem.

### Key Capabilities:
- **Local LLM Translation**: Uses local LLMs for privacy and offline translation
- **Chunked Processing**: Handles large datasets efficiently through chunked processing
- **HuggingFace Integration**: Direct integration with HuggingFace datasets and model hub
- **Resumable Processing**: Can resume translation from any chunk index
- **Quantization Support**: Supports 4-bit and 8-bit quantization for memory efficiency
- **Multi-Column Translation**: Translate multiple columns in a single pass
- **Automatic Push to Hub**: Automatically push translated datasets to HuggingFace Hub

---

## ✨ Features

- 🔒 **Privacy-First**: All translation happens locally using your own LLM models
- 📊 **Dataset Support**: Works with any HuggingFace dataset format
- 🚀 **Efficient Processing**: Chunked processing with configurable batch sizes
- 💾 **Memory Optimization**: Support for 4-bit and 8-bit quantization
- 🔄 **Resumable**: Continue translation from where you left off
- 🌐 **Multi-Language**: Support for any language pair your model supports
- 📈 **Progress Tracking**: Real-time progress monitoring with tqdm
- 🎯 **Flexible Columns**: Translate specific columns while preserving others
- 🔧 **Easy Configuration**: Simple command-line interface with comprehensive options

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
│   ├── main.py              # Main translation processor
│   └── model.py             # LLM model wrapper
└── utils/
    └── utils.py             # Utility functions
```

### Core Components:

- **`TranslateProcessor`**: Main class handling dataset translation workflow
- **`TranslateModel`**: Wrapper for LLM models with quantization support
- **Utility Functions**: Dataset management, chunk loading, and HuggingFace integration

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

The main translation script can be run with various command-line arguments:

```bash
python -m src.main \
  --model_id <MODEL_ID> \
  --repo_id <REPO_ID> \
  --src_language <SOURCE_LANGUAGE> \
  --trg_language <TARGET_LANGUAGE> \
  --dataset_name <DATASET_NAME> \
  --column_name <COLUMN1> <COLUMN2> \
  --translated_dataset_dir <OUTPUT_DIR>
```

### Example

Here's a complete example translating an English dataset to Vietnamese:

```bash
export CUDA_VISIBLE_DEVICES=0
python -m src.main \
  --model_id LLaMAX/LLaMAX3-8B-Alpaca \
  --repo_id your-username/translated-dataset \
  --src_language English \
  --trg_language Vietnamese \
  --max_length_token 12800 \
  --dataset_name knoveleng/open-s1 \
  --column_name problem solution \
  --translated_dataset_dir ".cache" \
  --download_dataset_dir ".cache" \
  --start_inter 0 \
  --writer_batch_size 20 \
  --use_4bit
```

Or use the provided script:
```bash
chmod +x scripts/example_run.sh
./scripts/example_run.sh
```

---

## ⚙️ Configuration

### Required Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--model_id` | HuggingFace model identifier | `LLaMAX/LLaMAX3-8B-Alpaca` |
| `--repo_id` | Target HuggingFace repo for translated dataset | `username/dataset-name` |
| `--src_language` | Source language name | `English` |
| `--trg_language` | Target language name | `Vietnamese` |
| `--dataset_name` | HuggingFace dataset name | `knoveleng/open-s1` |
| `--column_name` | Columns to translate (space-separated) | `problem solution` |
| `--translated_dataset_dir` | Output directory for translated data | `.cache` |

### Optional Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--max_length_token` | Maximum tokens for model input | `8000` |
| `--subset` | Dataset subset to use | `train` |
| `--start_inter` | Chunk index to start from | `0` |
| `--batch_size` | Processing batch size | `1` |
| `--writer_batch_size` | Records per chunk | `20` |
| `--download_dataset_dir` | Local dataset cache directory | `None` |
| `--use_4bit` | Enable 4-bit quantization | `False` |
| `--use_8bit` | Enable 8-bit quantization | `False` |
| `--push` | Push to HuggingFace Hub | `True` |
| `--warning_skip` | Suppress warnings | `True` |

### Performance Tips

- **Use quantization** (`--use_4bit` or `--use_8bit`) for memory efficiency
- **Adjust `writer_batch_size`** based on your available memory
- **Set `start_inter`** to resume interrupted translations
- **Use `download_dataset_dir`** to cache datasets locally for faster re-runs

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

#### Basic Usage (Recommended with 4-bit quantization):
```bash
python -m src.main \
  --model_id LLaMAX/LLaMAX3-8B-Alpaca \
  --use_4bit \
  --max_length_token 12800 \
  # ... other arguments
```

#### Full Precision Usage (if you have sufficient VRAM):
```bash
python -m src.main \
  --model_id LLaMAX/LLaMAX3-8B-Alpaca \
  --max_length_token 12800 \
  # ... other arguments
```

### 📊 Memory Requirements

| Quantization Level | Memory Requirement | Recommended For |
|-------------------|-------------------|-----------------|
| FP16 (Full Precision) | ~16GB | High-end GPUs with 24GB+ VRAM |
| 8-bit Quantization | ~8GB | Mid-range GPUs with 12GB+ VRAM |
| 4-bit Quantization | ~4GB | Most GPUs with 6GB+ VRAM |

### ⚠️ Important Notes

- **Quantization**: Always use `--use_4bit` for memory efficiency unless you have high-end hardware
- **Model Compatibility**: Uses LLaMA architecture (`LlamaForCausalLM`)
- **Download Time**: ~15GB download on first use
- **Performance**: Excellent translation quality with reasonable speed
- **Memory**: Ensure your GPU has at least 4GB VRAM for 4-bit quantization

### 🔄 Adding Custom Models

To use a custom model, ensure it:
1. **Is LLaMA-compatible** (uses `LlamaForCausalLM` architecture)
2. **Has a HuggingFace model ID** or local path
3. **Supports the required tokenizer**

Example with custom model:
```bash
python -m src.main \
  --model_id your-username/your-custom-llama-model \
  --use_4bit \
  # ... other arguments
```

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

- ✅ **Chunked Processing**: Efficient handling of large datasets
- ✅ **Quantization Support**: 4-bit and 8-bit model quantization
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
