# Llama 3.3 70B Conversational Agent Fine-tuning

A comprehensive framework for fine-tuning Llama 3.3 70B Instruct for custom conversational agents using QLoRA on H100 hardware. This repository provides production-ready tools for training, inference, and deployment of specialized instruction-following models.

[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.0-red)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.45.0+-yellow)](https://huggingface.co/transformers/)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA-blue)](https://github.com/huggingface/peft)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 📦 Installation

### Prerequisites
First, install `uv` (recommended) for fast dependency management:
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
# or with pip
pip install uv
```

### Full Environment (Recommended)
```bash
# Clone the repository
git clone https://github.com/juanmvsa/llama3-3-70b-finetuning.git
cd llama3-3-70b-finetuning

# Install all dependencies with uv
uv sync

# Activate the virtual environment
source .venv/bin/activate  # Linux/Mac
# or .venv\Scripts\activate  # Windows
```

### For Inference Only
```bash
# Install minimal dependencies for inference
uv add transformers torch peft bitsandbytes accelerate

# Alternative: traditional pip (slower)
pip install transformers torch peft bitsandbytes accelerate
```

### Alternative: Conda Environment
```bash
# If you prefer conda over uv
conda env create -f environment.yml
conda activate llama-finetuning
```

## 🚀 Quick Start

### Prepare Your Data
Create a JSON file with instruction-response pairs:
```json
[
  {
    "instruction": "Your instruction here",
    "response": "The expected response here"
  },
  {
    "instruction": "Another instruction",
    "response": "Another response"
  }
]
```

### Training
```bash
# Install dependencies
uv sync

# Train your custom model
uv run python finetune_llama33_70b.py --data_path your_data.json --output_dir ./results
```

### Inference
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model and your trained adapter
base_model_id = "meta-llama/Llama-3.3-70B-Instruct"
adapter_path = "./results"  # Path to your trained adapter

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    load_in_4bit=True  # Enable 4-bit quantization
)

# Load your custom adapter
model = PeftModel.from_pretrained(base_model, adapter_path)

# Example usage
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Your question here"},
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

outputs = model.generate(
    input_ids,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
)
response = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))
```

## 📑 Table of Contents

- [📦 Installation](#-installation)
- [🚀 Quick Start](#-quick-start)
- [⚡ Core Operations](#-core-operations)
- [🤖 Features](#-features)
- [⚙️ Configuration](#️-configuration)
- [📁 Project Structure](#-project-structure)
- [📚 Example Use Cases](#-example-use-cases)
- [🔧 Troubleshooting](#-troubleshooting)
- [📞 Support](#-support)


## 🚀 Core Operations

### Training Your Custom Model
```bash
# Ensure dependencies are installed
uv sync

# Basic training with your dataset
uv run python finetune_llama33_70b.py --data_path your_data.json --output_dir ./results

# Advanced training with custom parameters
uv run python finetune_llama33_70b.py \
  --data_path your_data.json \
  --output_dir ./results \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 32 \
  --learning_rate 1e-4 \
  --lora_r 128 \
  --lora_alpha 32
```

### Local Inference
```bash
# Interactive chat with your trained model (requires inference.py - not included in this repository)
# You'll need to implement your own inference script or use the training script's inference mode

# Single prompt inference using the training script
uv run python finetune_llama33_70b.py --inference --adapter_path ./results --prompt "Your question here" --token hf_your_token_here

# Compare with base model
uv run python finetune_llama33_70b.py --inference --compare --adapter_path ./results --token hf_your_token_here
```

### Deployment
```bash
# Deploy to HuggingFace Hub
uv run python upload_to_hf.py --model_dir ./results --repo_name "username/your-model-name"
```

## 📁 Project Structure

```
llama3-3-70b-finetuning/
├── 🐍 Core Scripts
│   ├── finetune_llama33_70b.py           # H100-optimized training pipeline
│   └── upload_to_hf.py                   # HuggingFace Hub deployment utility
├── ⚙️ Configuration
│   ├── pyproject.toml                    # Complete dependency management and project config
│   ├── adapter_config.json              # LoRA adapter configuration (r=128, α=32)
│   ├── config.json                       # Model architecture configuration
│   ├── generation_config.json           # Inference generation parameters
│   ├── tokenizer_config.json            # Tokenizer configuration
│   └── special_tokens_map.json          # Special tokens mapping
└── 📚 Documentation
    └── README.md                         # Project overview and usage guide

Note: Some files (training data, inference scripts, and documentation) are in .gitignore
and not included in the repository. You'll need to create your own training data
and implement inference scripts as needed.
```

## 🤖 Features

### Core Capabilities
- **QLoRA Fine-tuning**: Efficient 4-bit quantization with LoRA adapters
- **H100 Optimization**: Optimized for NVIDIA H100 hardware with dynamic memory management
- **Conversation Quality Metrics**: Multi-dimensional evaluation framework
- **Interactive Inference**: Built-in chat interface with model comparison
- **Production Ready**: Complete pipeline from training to deployment

### Advanced Features
- **Dynamic Batch Size**: Automatic optimization for available GPU memory
- **Conversation Validation**: Multi-layered data quality assessment
- **Memory Optimization**: Automatic CUDA cache management and OOM recovery
- **Multi-turn Dialogue**: Support for complex conversational contexts
- **Flexible Data Format**: Simple JSON instruction-response format

## ⚙️ Configuration

### Default Training Parameters
- **Base Model**: Llama 3.3 70B Instruct
- **Method**: QLoRA (4-bit quantization + LoRA)
- **LoRA Settings**: r=128, alpha=32, dropout=0.05
- **Learning Rate**: 1e-4 with cosine scheduling
- **Per-Device Batch Size**: 1 (H100 optimized)
- **Gradient Accumulation**: 32 steps (effective batch size: 32)
- **Epochs**: 3

### Customization Options
All parameters can be adjusted via command-line arguments:
```bash
uv run python finetune_llama33_70b.py \
  --data_path your_data.json \
  --output_dir ./results \
  --num_train_epochs 5 \
  --learning_rate 2e-4 \
  --lora_r 256 \
  --lora_alpha 64 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16
```

### System Requirements
| Component | Training | Inference (Quantized) | Inference (Full) |
|-----------|----------|----------------------|------------------|
| **GPU** | H100 PCIe 80GB | 8GB+ VRAM | 40GB+ VRAM |
| **RAM** | 32GB+ | 16GB+ | 64GB+ |
| **Storage** | 100GB+ | 20GB+ | 150GB+ |
| **CUDA** | 11.8+ | 11.8+ | 11.8+ |

## 📚 Example Use Cases

This framework can be used to create specialized conversational agents for various domains:

### 🏥 Healthcare Assistant
```json
[
  {
    "instruction": "What are the symptoms of common cold?",
    "response": "Common cold symptoms include runny nose, sneezing, cough, sore throat, and mild fatigue. Symptoms typically last 7-10 days."
  }
]
```

### 💼 Customer Support Agent
```json
[
  {
    "instruction": "How do I return a product?",
    "response": "To return a product, please visit our returns page, print a return label, and ship the item back within 30 days of purchase."
  }
]
```

### 🎓 Educational Tutor
```json
[
  {
    "instruction": "Explain photosynthesis in simple terms",
    "response": "Photosynthesis is how plants make food using sunlight, water, and carbon dioxide. They combine these to create sugar and release oxygen as a byproduct."
  }
]
```

### 🔧 Technical Support
```json
[
  {
    "instruction": "My computer won't start. What should I check?",
    "response": "First, ensure the power cable is securely connected. Check if the power button lights up. If not, try a different power outlet or cable."
  }
]
```

### Example Implementation
This framework was successfully used to create specialized conversational agents for:
- **Spanish workplace violence prevention** - Achieved 97% memory efficiency on H100
- **Multi-dimensional conversation quality evaluation** - 7-dimensional assessment framework
- **Professional empathy balance** - 0.73 weighted composite score

*Note: Specific implementation files and model cards are not included in this repository
but can serve as a reference for your own projects.*


## 🔧 Troubleshooting

### Common Issues

**"CUDA out of memory" during training:**
```bash
# Use 4-bit quantization and reduce batch size
export CUDA_VISIBLE_DEVICES=0
uv run python finetune_llama33_70b.py --per_device_train_batch_size 1 --gradient_accumulation_steps 64 --data_path your_data.json

# For inference, implement your own inference script with 4-bit quantization enabled
```

**"Model not found" error:**
```bash
# Ensure you have HuggingFace access
uv run huggingface-cli login
# Or set token in environment
export HUGGINGFACE_HUB_TOKEN=your_token_here
```

**"Connection timeout" during model download:**
```bash
# Set longer timeout and resume downloads
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_HUB_ENABLE_HF_TRANSFER=1
```

**Dependencies not found:**
```bash
# Reinstall dependencies with uv
uv sync --reinstall

# Or check virtual environment activation
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows
```

**Slow inference on CPU:**
- Use GPU with quantization: `--load_in_4bit`
- Reduce max_new_tokens: `--max_new_tokens 256`
- Consider using smaller base models for CPU inference

### Hardware-Specific Tips

**For RTX 4090 / RTX 3090:**
```python
# Enable 4-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    load_in_4bit=True,
    torch_dtype=torch.float16
)
```

**For Mac with Apple Silicon:**
```python
# Use MPS backend
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

## 📞 Support

### Technical Issues
- Check [troubleshooting section](#-troubleshooting) first
- Verify hardware requirements match your system
- Ensure proper environment setup with `uv sync`
- Create your own training data in the expected JSON format
- Implement inference scripts as needed for your use case

### Getting Started
- Start with the [quick start](#-quick-start) section
- Review [example use cases](#-example-use-cases) for inspiration
- Check system requirements before training
- Ensure you have proper HuggingFace authentication

---

**📝 License**: This project is licensed under the MIT License - see the LICENSE file for details.

**🤝 Contributing**: Contributions are welcome! Please read the contributing guidelines before submitting pull requests.
