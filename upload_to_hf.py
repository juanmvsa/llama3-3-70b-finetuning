#!/usr/bin/env python3
"""
upload trained llama 3.3 70b model to huggingface hub.
supports both adapter-only and merged model uploads with proper metadata.
"""
import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, LoraConfig, get_peft_model
from huggingface_hub import HfApi, login, create_repo

# configure logging.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HuggingFaceUploader:
    """handles uploading trained models to huggingface hub."""

    def __init__(self, model_dir: str, repo_name: str, token: Optional[str] = None):
        self.model_dir = Path(model_dir)
        self.repo_name = repo_name
        self.token = token
        self.api = HfApi()

        # authenticate with huggingface.
        if token:
            login(token=token)
        else:
            login()  # use cached token or prompt for login.

        logger.info(f"initialized uploader for {repo_name}")

    def load_metadata(self) -> Dict[str, Any]:
        """load training metadata from model directory."""
        metadata_path = self.model_dir / "hf_upload_metadata.json"

        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            logger.info("loaded training metadata")
            return metadata
        else:
            logger.warning("no metadata found, using defaults")
            return {
                "base_model": "meta-llama/Llama-3.3-70B-Instruct",
                "training_data_size": "unknown",
                "training_time": "unknown",
                "training_loss": "unknown"
            }

    def load_conversation_config(self) -> Dict[str, Any]:
        """load conversation configuration."""
        config_path = self.model_dir / "conversation_config.json"

        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            logger.info("loaded conversation configuration")
            return config
        else:
            logger.warning("no conversation config found")
            return {}

    def create_model_card(self, metadata: Dict[str, Any], conversation_config: Dict[str, Any]) -> str:
        """create model card content."""

        # extract system message for display.
        system_message = conversation_config.get("system_message", "conversational ai assistant")

        model_card = f"""---
language:
- en
- es
library_name: transformers
pipeline_tag: text-generation
tags:
- llama
- llama-3.3
- conversational-ai
- finetuned
- qlora
base_model: {metadata.get('base_model', 'meta-llama/Llama-3.3-70B-Instruct')}
license: llama3.3
---

# Llama 3.3 70B Finetuned Model

This is a finetuned version of {metadata.get('base_model', 'meta-llama/Llama-3.3-70B-Instruct')} trained using qlora (quantized lora) on h100 gpus.

## model details

- **base model**: {metadata.get('base_model', 'meta-llama/Llama-3.3-70B-Instruct')}
- **training method**: qlora (4-bit quantization + lora)
- **training data size**: {metadata.get('training_data_size', 'unknown')} samples
- **training time**: {metadata.get('training_time', 'unknown')}
- **final training loss**: {metadata.get('training_loss', 'unknown')}

## system message

```
{system_message}
```

## usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# load base model and tokenizer.
base_model_name = "{metadata.get('base_model', 'meta-llama/Llama-3.3-70B-Instruct')}"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# load the finetuned adapter.
model = PeftModel.from_pretrained(base_model, "{self.repo_name}")

# generate response.
messages = [
    {{"role": "system", "content": "{system_message}"}},
    {{"role": "user", "content": "your question here"}}
]

input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.1
    )

response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print(response)
```

## training configuration

- **lora rank**: 128
- **lora alpha**: 32
- **lora dropout**: 0.05
- **quantization**: 4-bit nf4 with bfloat16
- **batch size**: 2 (h100 optimized)
- **learning rate**: 1e-4 with cosine scheduling
- **validation dataset**: spanish alpaca

## hardware requirements

- **inference**: 8gb+ gpu memory (with quantization), 20gb+ (without)
- **training**: h100 80gb (optimized)

## license

this model follows the llama 3.3 license terms.
"""
        return model_card

    def upload_adapter(self, private: bool = False) -> str:
        """upload adapter-only model to huggingface hub."""

        # create repository.
        try:
            repo_url = create_repo(
                repo_id=self.repo_name,
                private=private,
                exist_ok=True
            )
            logger.info(f"repository created/verified: {repo_url}")
        except Exception as e:
            logger.error(f"failed to create repository: {e}")
            raise

        # load metadata and config.
        metadata = self.load_metadata()
        conversation_config = self.load_conversation_config()

        # create model card.
        model_card_content = self.create_model_card(metadata, conversation_config)
        model_card_path = self.model_dir / "README.md"

        with open(model_card_path, "w", encoding="utf-8") as f:
            f.write(model_card_content)

        logger.info("created model card")

        # upload all files in model directory.
        try:
            self.api.upload_folder(
                folder_path=str(self.model_dir),
                repo_id=self.repo_name,
                commit_message="upload finetuned llama 3.3 70b adapter"
            )
            logger.info(f"adapter uploaded successfully to {self.repo_name}")
            return f"https://huggingface.co/{self.repo_name}"

        except Exception as e:
            logger.error(f"upload failed: {e}")
            raise

    def merge_and_upload(self, private: bool = False) -> str:
        """merge adapter with base model and upload full model."""

        logger.info("merging adapter with base model...")

        # load metadata to get base model name.
        metadata = self.load_metadata()
        base_model_name = metadata.get("base_model", "meta-llama/Llama-3.3-70B-Instruct")

        try:
            # load base model.
            logger.info(f"loading base model: {base_model_name}")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )

            # load tokenizer.
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)

            # load adapter.
            logger.info(f"loading adapter from {self.model_dir}")
            model = PeftModel.from_pretrained(base_model, str(self.model_dir))

            # merge adapter into base model.
            logger.info("merging adapter...")
            merged_model = model.merge_and_unload()

            # create merged model directory.
            merged_dir = self.model_dir.parent / f"{self.model_dir.name}_merged"
            merged_dir.mkdir(exist_ok=True)

            # save merged model.
            logger.info(f"saving merged model to {merged_dir}")
            merged_model.save_pretrained(str(merged_dir))
            tokenizer.save_pretrained(str(merged_dir))

            # copy metadata files.
            for file_name in ["hf_upload_metadata.json", "conversation_config.json"]:
                src_path = self.model_dir / file_name
                if src_path.exists():
                    dst_path = merged_dir / file_name
                    with open(src_path, "r") as src, open(dst_path, "w") as dst:
                        dst.write(src.read())

            # create model card for merged model.
            conversation_config = self.load_conversation_config()
            model_card_content = self.create_model_card(metadata, conversation_config)

            with open(merged_dir / "README.md", "w", encoding="utf-8") as f:
                f.write(model_card_content)

            # create merged model repository.
            merged_repo_name = f"{self.repo_name}-merged"

            try:
                repo_url = create_repo(
                    repo_id=merged_repo_name,
                    private=private,
                    exist_ok=True
                )
                logger.info(f"merged repository created: {repo_url}")
            except Exception as e:
                logger.error(f"failed to create merged repository: {e}")
                raise

            # upload merged model.
            self.api.upload_folder(
                folder_path=str(merged_dir),
                repo_id=merged_repo_name,
                commit_message="upload merged llama 3.3 70b model"
            )

            logger.info(f"merged model uploaded successfully to {merged_repo_name}")
            return f"https://huggingface.co/{merged_repo_name}"

        except Exception as e:
            logger.error(f"merge and upload failed: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description="upload finetuned llama model to huggingface hub")
    parser.add_argument("--model_dir", type=str, default="./results",
                       help="directory containing the finetuned model")
    parser.add_argument("--repo_name", type=str, required=True,
                       help="huggingface repository name (username/model-name)")
    parser.add_argument("--token", type=str, default=None,
                       help="huggingface access token")
    parser.add_argument("--adapter_only", action="store_true",
                       help="upload adapter only (default: upload both adapter and merged)")
    parser.add_argument("--merged_only", action="store_true",
                       help="upload merged model only")
    parser.add_argument("--private", action="store_true",
                       help="create private repository")

    args = parser.parse_args()

    # validate model directory.
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        logger.error(f"model directory {model_dir} does not exist")
        return

    # check for adapter files.
    adapter_files = ["adapter_config.json", "adapter_model.safetensors"]
    has_adapter = all((model_dir / f).exists() for f in adapter_files)

    if not has_adapter:
        logger.error("adapter files not found in model directory")
        logger.error(f"expected files: {adapter_files}")
        return

    # initialize uploader.
    uploader = HuggingFaceUploader(
        model_dir=str(model_dir),
        repo_name=args.repo_name,
        token=args.token
    )

    try:
        if args.merged_only:
            # upload merged model only.
            url = uploader.merge_and_upload(private=args.private)
            logger.info(f"merged model available at: {url}")

        elif args.adapter_only:
            # upload adapter only.
            url = uploader.upload_adapter(private=args.private)
            logger.info(f"adapter available at: {url}")

        else:
            # upload both adapter and merged model.
            adapter_url = uploader.upload_adapter(private=args.private)
            logger.info(f"adapter available at: {adapter_url}")

            merged_url = uploader.merge_and_upload(private=args.private)
            logger.info(f"merged model available at: {merged_url}")

    except Exception as e:
        logger.error(f"upload process failed: {e}")
        return


if __name__ == "__main__":
    main()