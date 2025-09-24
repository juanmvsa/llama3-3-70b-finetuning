#!/usr/bin/env python3
"""
llama 3.3 70b instruct finetuning with qlora on h100 gpu.
optimized for redhat server with cuda 12.1 and 80gb h100.
"""
import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional
import argparse
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

# configure logging first.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# check torch version for compatibility.
torch_version = torch.__version__
logger.info(f"torch version: {torch_version}")

# check for known incompatible combinations.
torch_major, torch_minor = torch_version.split('.')[:2]
torch_version_tuple = (int(torch_major), int(torch_minor))

if torch_version_tuple < (2, 0):
    logger.warning(f"torch {torch_version} may have compatibility issues")
    logger.warning("recommended: torch >= 2.0.0 for best compatibility")
else:
    logger.info("torch version compatible")

# handle version-specific imports with fallbacks.
try:
    import transformers
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        TrainingArguments,
        DataCollatorForLanguageModeling
    )
    logger.info(f"transformers version: {transformers.__version__}")
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    logger.error(f"transformers import failed: {e}")
    raise

# handle trainer import separately due to peft dependency issues.
try:
    from transformers import Trainer
    TRAINER_AVAILABLE = True
    logger.info("transformers.Trainer available")
except (ImportError, RuntimeError, AttributeError) as e:
    logger.warning(f"transformers.Trainer import failed: {e}")
    logger.warning("will implement basic training loop")
    Trainer = None
    TRAINER_AVAILABLE = False

# handle bitsandbytes import with torch compatibility.
try:
    from transformers import BitsAndBytesConfig
    QUANTIZATION_AVAILABLE = True
    logger.info("quantization available")
except ImportError as e:
    logger.warning(f"quantization not available: {e}")
    BitsAndBytesConfig = None
    QUANTIZATION_AVAILABLE = False

# handle peft import with version compatibility.
try:
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    PEFT_AVAILABLE = True
    logger.info("peft available")
except ImportError as e:
    logger.warning(f"peft not available: {e}")
    LoraConfig = None
    get_peft_model = None
    TaskType = None
    prepare_model_for_kbit_training = None
    PEFT_AVAILABLE = False

try:
    from datasets import Dataset as HFDataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not available - training will run without logging")
# handle trl import with version compatibility.
try:
    from trl import SFTTrainer
    TRL_AVAILABLE = True
    logger.info("trl.SFTTrainer available")
except ImportError as e:
    logger.warning(f"trl import failed: {e}")
    if TRAINER_AVAILABLE:
        logger.info("fallback: using transformers.Trainer")
        SFTTrainer = Trainer
    else:
        logger.error("neither trl nor transformers.Trainer available")
        SFTTrainer = None
    TRL_AVAILABLE = False

# handle data collator import.
try:
    from trl import DataCollatorForCompletionOnlyLM
    DataCollatorForCompletionOnlyLM_AVAILABLE = True
except ImportError:
    from transformers import DataCollatorForLanguageModeling
    DataCollatorForCompletionOnlyLM = None
    DataCollatorForCompletionOnlyLM_AVAILABLE = False

# logging already configured above.

class WorkplaceViolenceDataset:
    """dataset handler for workplace violence prevention training data with conversation validation."""

    def __init__(self, data_path: str, tokenizer: AutoTokenizer, validation_dataset: str = "bertin-project/alpaca-spanish"):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.validation_dataset = validation_dataset
        self.data = self._load_data()
        self.validation_data = self._load_validation_data()

    def _load_data(self) -> list:
        """load and validate training data."""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # validate conversational quality.
        validated_data = self._validate_conversations(data)
        logger.info(f"loaded {len(validated_data)} validated training samples from {self.data_path}")
        return validated_data

    def _load_validation_data(self) -> list:
        """load spanish conversational validation dataset."""
        try:
            from datasets import load_dataset
            dataset = load_dataset(self.validation_dataset, split="train")
            validation_data = []
            for item in dataset:
                if len(item.get("instruction", "")) > 10 and len(item.get("output", "")) > 10:
                    validation_data.append({
                        "instruction": item["instruction"],
                        "response": item["output"]
                    })
            logger.info(f"loaded {len(validation_data)} validation samples from {self.validation_dataset}")
            return validation_data[:1000]  # limit for efficiency.
        except Exception as e:
            logger.warning(f"failed to load validation dataset: {e}")
            return []

    def _validate_conversations(self, data: list) -> list:
        """validate conversations for agent suitability."""
        validated = []
        for item in data:
            if self._is_valid_conversation(item):
                validated.append(item)
            else:
                logger.debug(f"filtered out invalid conversation: {item.get('instruction', '')[:50]}...")
        return validated

    def _is_valid_conversation(self, item: dict) -> bool:
        """check if conversation item is suitable for agent training."""
        instruction = item.get("instruction", "")
        response = item.get("response", "")

        # basic length checks.
        if len(instruction.strip()) < 5 or len(response.strip()) < 10:
            return False

        # enhance response quality during validation.
        item["response"] = self._enhance_response_quality(response)
        response = item["response"]

        # check for proper conversational structure.
        if not instruction.strip().endswith(('.', '?', '!')):
            # add proper punctuation for questions.
            if any(word in instruction.lower() for word in ['qué', 'cómo', 'cuándo', 'dónde', 'por qué', 'quién']):
                item["instruction"] = instruction.strip() + "?"
            else:
                item["instruction"] = instruction.strip() + "."

        # validate workplace violence domain relevance.
        if not self._validate_workplace_violence_content(instruction + " " + response):
            return False

        # ensure response sounds conversational.
        response_lower = response.lower()
        conversational_markers = ['puedo', 'te', 'le', 'nos', 'les', 'ayuda', 'apoyo', 'importante', 'entiendo']
        has_conversational_tone = any(marker in response_lower for marker in conversational_markers)

        # validate empathy and professional tone.
        has_empathy = self._validate_empathy_tone(response)
        has_professional_tone = self._validate_professional_language(response)

        return has_conversational_tone and has_empathy and has_professional_tone

    def _enhance_response_quality(self, response: str) -> str:
        """enhance response quality with proper formatting and tone."""
        # basic cleaning.
        response = response.strip()

        # ensure proper capitalization at sentence start.
        sentences = response.split('. ')
        enhanced_sentences = []

        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                # capitalize first letter.
                sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
                # ensure proper ending punctuation.
                if not sentence.endswith(('.', '!', '?')):
                    sentence += '.'
                enhanced_sentences.append(sentence)

        enhanced_response = '. '.join(enhanced_sentences)

        # ensure response starts with empathetic language when appropriate.
        empathy_starters = ['entiendo', 'comprendo', 'es importante', 'tienes razón']
        response_lower = enhanced_response.lower()

        if not any(starter in response_lower[:50] for starter in empathy_starters):
            # add empathetic opening for sensitive topics.
            sensitive_keywords = ['acoso', 'violencia', 'discriminación', 'abuso', 'maltrato']
            if any(keyword in response_lower for keyword in sensitive_keywords):
                enhanced_response = "Entiendo tu preocupación. " + enhanced_response

        return enhanced_response

    def _validate_workplace_violence_content(self, text: str) -> bool:
        """validate content is relevant to workplace violence prevention."""
        text_lower = text.lower()

        # core workplace violence prevention keywords.
        workplace_keywords = [
            'trabajo', 'laboral', 'empresa', 'oficina', 'empleado', 'empleada',
            'compañero', 'compañera', 'jefe', 'jefa', 'supervisor', 'supervisora'
        ]

        violence_prevention_keywords = [
            'acoso', 'violencia', 'discriminación', 'maltrato', 'abuso',
            'prevención', 'protocolo', 'denuncia', 'queja', 'reporte',
            'seguridad', 'protección', 'derechos', 'apoyo', 'ayuda'
        ]

        has_workplace_context = any(keyword in text_lower for keyword in workplace_keywords)
        has_prevention_content = any(keyword in text_lower for keyword in violence_prevention_keywords)

        return has_workplace_context or has_prevention_content

    def _validate_empathy_tone(self, response: str) -> bool:
        """validate response contains empathetic language."""
        response_lower = response.lower()

        empathy_markers = [
            'entiendo', 'comprendo', 'lamento', 'siento', 'apoyo',
            'importante', 'válido', 'valida', 'normal', 'natural',
            'puedo ayudar', 'estoy aquí', 'no estás solo', 'no estás sola',
            'mereces', 'tienes derecho', 'es comprensible'
        ]

        supportive_language = [
            'siempre', 'nunca dudes', 'recuerda', 'es fundamental',
            'no tengas miedo', 'puedes contar', 'estamos contigo'
        ]

        has_empathy = any(marker in response_lower for marker in empathy_markers)
        has_support = any(phrase in response_lower for phrase in supportive_language)

        return has_empathy or has_support

    def _validate_professional_language(self, response: str) -> bool:
        """validate response maintains professional tone while being warm."""
        response_lower = response.lower()

        # check for inappropriate informal language.
        overly_informal = ['wey', 'neta', 'chido', 'gacho', 'órale']
        has_inappropriate = any(word in response_lower for word in overly_informal)

        if has_inappropriate:
            return False

        # ensure professional resource mentions.
        professional_indicators = [
            'protocolo', 'procedimiento', 'recursos', 'departamento',
            'profesional', 'especialista', 'asesoría', 'orientación',
            'legal', 'jurídico', 'rrhh', 'recursos humanos'
        ]

        # check for action-oriented language.
        action_oriented = [
            'puedes', 'debes', 'es recomendable', 'te sugiero',
            'considera', 'evalúa', 'documenta', 'reporta'
        ]

        has_professional_tone = any(indicator in response_lower for indicator in professional_indicators)
        has_action_guidance = any(action in response_lower for action in action_oriented)

        # at least one professional indicator OR action guidance required.
        return has_professional_tone or has_action_guidance

    def format_prompt(self, instruction: str, response: str) -> str:
        """format instruction-response pair for llama 3.3 instruct with chat template."""
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

eres un asistente especializado en prevención de violencia laboral y acoso sexual en el entorno de trabajo. tu objetivo es proporcionar apoyo empático, información precisa y recursos específicos a personas que puedan estar experimentando situaciones difíciles en su lugar de trabajo.

IMPORTANTE: siempre mantén un tono profesional pero cálido, valida las emociones del usuario, y proporciona información práctica basada en protocolos establecidos. nunca minimices las experiencias reportadas y siempre ofrece recursos concretos de ayuda.

responde de forma conversacional y empática. usa un tono profesional pero cálido. siempre ofrece apoyo y recursos específicos.<|eot_id|><|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{response}<|eot_id|>"""

    def to_hf_dataset(self) -> tuple:
        """convert to hugging face dataset format with validation split."""
        formatted_train_data = []
        formatted_val_data = []

        # format training data.
        for item in self.data:
            formatted_prompt = self.format_prompt(
                item["instruction"],
                item["response"]
            )
            formatted_train_data.append({"text": formatted_prompt})

        # format validation data for conversation evaluation.
        for item in self.validation_data:
            formatted_prompt = self.format_prompt(
                item["instruction"],
                item["response"]
            )
            formatted_val_data.append({"text": formatted_prompt})

        train_dataset = HFDataset.from_list(formatted_train_data)
        val_dataset = HFDataset.from_list(formatted_val_data)

        return train_dataset, val_dataset

class LlamaFinetunConfig:
    """configuration for llama 3.3 70b finetuning with qlora."""
    
    def __init__(self, output_dir: str = "./results"):
        # model configuration with fallback.
        self.model_name = "meta-llama/Llama-3.3-70B-Instruct"
        self.fallback_model = "meta-llama/Llama-3.1-70B-Instruct"  # fallback if 3.3 fails.
        self.cache_dir = "./model_cache"
        
        # quantization configuration optimized for h100 80gb.
        if QUANTIZATION_AVAILABLE:
            self.bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                # h100-specific optimizations.
                bnb_4bit_quant_storage=torch.bfloat16,  # native h100 format.
            )
            logger.info("quantization enabled for memory efficiency")
        else:
            self.bnb_config = None
            logger.warning("quantization disabled - higher memory usage expected")
        
        # lora configuration optimized for conversational agents.
        if PEFT_AVAILABLE:
            self.lora_config = LoraConfig(
                r=128,  # higher rank for better conversation modeling.
                lora_alpha=32,  # increased alpha for stronger adaptation.
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                    "embed_tokens", "lm_head"  # include embeddings for better conversation handling.
                ],
                lora_dropout=0.05,  # reduced dropout for conversation stability.
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False
            )
            logger.info("lora configuration enabled for efficient training")
        else:
            self.lora_config = None
            logger.warning("peft not available - full model training will be used")
        
        # training arguments optimized for h100 80gb and conversational performance.
        self.training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=2,  # will be optimized dynamically.
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=16,  # reduced due to larger batch size (effective batch = 32).

            # conversational-specific improvements.
            save_strategy="steps",
            eval_strategy="steps",
            eval_steps=25,  # more frequent evaluation for conversation quality.
            save_steps=25,  # more frequent checkpoints.

            # learning rate scheduling for conversational stability.
            learning_rate=1e-4,
            lr_scheduler_type="cosine",  # more stable than cosine_with_restarts.
            warmup_steps=100,  # more granular than warmup_ratio for better convergence.

            # precision and optimization.
            fp16=False,
            bf16=True,  # native h100 precision.
            tf32=True,  # enable tensor cores on h100.
            optim="adamw_torch_fused",  # h100-optimized fused adamw.

            # regularization for better generalization.
            weight_decay=0.01,  # helps prevent overfitting on small conversational dataset.
            max_grad_norm=0.5,  # more conservative gradient clipping for stability.

            # logging and monitoring.
            logging_steps=5,
            report_to="wandb",
            run_name="llama-33-70b-conversational-agent-h100",

            # model saving and evaluation.
            save_total_limit=10,  # keep more checkpoints for conversation quality analysis.
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,

            # data handling optimizations.
            remove_unused_columns=False,
            dataloader_pin_memory=True,
            dataloader_num_workers=8,  # increased for h100 nvme speed.
            group_by_length=True,  # helps with conversational data efficiency.
            dataloader_drop_last=False,

            # memory and compute optimizations.
            gradient_checkpointing=True,
            auto_find_batch_size=False,  # will be handled by dynamic optimization.

            # h100-specific optimizations.
            ddp_timeout=7200,  # handle h100 initialization time.
            max_steps=-1,
            prediction_loss_only=False,
            seed=42,
            data_seed=42,
            save_safetensors=True,

            # conversational training specific.
            label_smoothing_factor=0.0,  # keep sharp for conversational responses.
            include_inputs_for_metrics=False,  # focus on response quality metrics.
        )

class LlamaFinetuner:
    """main finetuning orchestrator."""
    
    def __init__(self, config: LlamaFinetunConfig, data_path: str):
        self.config = config
        self.data_path = data_path
        self.model = None
        self.tokenizer = None
        self.dataset = None
        
    def find_optimal_batch_size(self):
        """dynamically find optimal batch size for h100."""
        logger.info("finding optimal batch size for h100 gpu...")

        # test batch sizes in descending order.
        test_batch_sizes = [4, 3, 2, 1]

        for batch_size in test_batch_sizes:
            try:
                logger.info(f"testing batch size: {batch_size}")

                # create a small test batch.
                test_input = {
                    'input_ids': torch.randint(0, 1000, (batch_size, 512)).cuda(),
                    'attention_mask': torch.ones(batch_size, 512).cuda(),
                    'labels': torch.randint(0, 1000, (batch_size, 512)).cuda()
                }

                # test forward pass.
                with torch.no_grad():
                    outputs = self.model(**test_input)
                    loss = outputs.loss

                # test backward pass for memory requirements.
                loss.backward()

                # clear cache.
                torch.cuda.empty_cache()

                # if we reach here, this batch size works.
                logger.info(f"✅ optimal batch size found: {batch_size}")

                # update training arguments with optimal batch size.
                self.config.training_args.per_device_train_batch_size = batch_size
                self.config.training_args.per_device_eval_batch_size = batch_size

                # adjust gradient accumulation to maintain effective batch size.
                target_effective_batch = 32
                new_grad_accum = max(1, target_effective_batch // batch_size)
                self.config.training_args.gradient_accumulation_steps = new_grad_accum

                logger.info(f"updated gradient accumulation steps: {new_grad_accum}")
                logger.info(f"effective batch size: {batch_size * new_grad_accum}")

                return batch_size

            except torch.cuda.OutOfMemoryError as e:
                logger.warning(f"batch size {batch_size} caused oom: {e}")
                torch.cuda.empty_cache()
                continue
            except Exception as e:
                logger.warning(f"batch size {batch_size} failed: {e}")
                torch.cuda.empty_cache()
                continue

        # fallback to batch size 1 if all fail.
        logger.warning("all batch sizes failed, using batch size 1")
        self.config.training_args.per_device_train_batch_size = 1
        self.config.training_args.per_device_eval_batch_size = 1
        self.config.training_args.gradient_accumulation_steps = 32
        return 1

    def _optimize_h100_memory(self):
        """apply h100-specific memory optimizations."""
        logger.info("applying h100 memory optimizations...")

        # check current gpu memory.
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            total_memory = torch.cuda.get_device_properties(device).total_memory / 1e9
            allocated_memory = torch.cuda.memory_allocated(device) / 1e9
            cached_memory = torch.cuda.memory_reserved(device) / 1e9

            logger.info(f"gpu memory status:")
            logger.info(f"  total: {total_memory:.1f}gb")
            logger.info(f"  allocated: {allocated_memory:.1f}gb")
            logger.info(f"  cached: {cached_memory:.1f}gb")
            logger.info(f"  free: {total_memory - allocated_memory:.1f}gb")

            # apply h100-specific optimizations based on memory usage.
            if allocated_memory > 70:  # if using more than 70gb.
                logger.warning("high memory usage detected, applying aggressive optimizations")

                # reduce cache retention.
                torch.cuda.empty_cache()

                # enable memory fraction limit.
                torch.cuda.set_per_process_memory_fraction(0.95)

                # set smaller max memory allocation.
                self.config.training_args.dataloader_num_workers = 4

            # log final memory status.
            final_allocated = torch.cuda.memory_allocated(device) / 1e9
            logger.info(f"memory optimization complete. allocated: {final_allocated:.1f}gb")

    def setup_model_and_tokenizer(self):
        """initialize model and tokenizer with conversation-specific optimizations."""
        logger.info(f"loading model for conversational agent: {self.config.model_name}")

        # load tokenizer with conversation-specific settings and fallback.
        try:
            logger.info(f"loading tokenizer: {self.config.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                cache_dir=self.config.cache_dir,
                trust_remote_code=True
            )
            logger.info("✅ llama 3.3 tokenizer loaded successfully")
        except Exception as e:
            logger.warning(f"llama 3.3 tokenizer failed: {e}")
            logger.info(f"falling back to: {self.config.fallback_model}")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config.fallback_model,
                    cache_dir=self.config.cache_dir,
                    trust_remote_code=True
                )
                # update model name for consistency.
                self.config.model_name = self.config.fallback_model
                logger.info("✅ llama 3.1 tokenizer loaded successfully")
            except Exception as fallback_error:
                logger.error(f"fallback tokenizer also failed: {fallback_error}")
                raise fallback_error

        # conversation-specific tokenizer setup.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        # set chat template for conversation handling.
        if not hasattr(self.tokenizer, 'chat_template') or self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = "{% for message in messages %}<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>{% endfor %}<|start_header_id|>assistant<|end_header_id|>\n\n"

        # load model with compatibility handling.
        model_kwargs = {
            "device_map": "auto",
            "cache_dir": self.config.cache_dir,
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,  # correct parameter name.
            "use_cache": False,  # required for training mode.
            "low_cpu_mem_usage": True,
            "max_memory": {0: "78GB"},  # reserve 2gb for overhead on h100 80gb.
        }

        # add quantization config if available.
        if self.config.bnb_config is not None:
            model_kwargs["quantization_config"] = self.config.bnb_config

        # use sdpa (scaled dot product attention) as primary choice - pytorch native optimized attention.
        # sdpa is pytorch's native attention implementation that works well with h100 tensor cores.
        logger.info("using pytorch sdpa attention for h100 optimization")
        attention_attempts = [("sdpa", "SDPA"), (None, "Standard")]

        model_loaded = False
        for attempt, (attn_impl, desc) in enumerate(attention_attempts, 1):
            try:
                if attn_impl:
                    model_kwargs["attn_implementation"] = attn_impl
                else:
                    # remove attention implementation to use default
                    model_kwargs.pop("attn_implementation", None)

                logger.info(f"attempt {attempt}: loading with {desc}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name, **model_kwargs
                )
                logger.info(f"✅ {desc} model loaded successfully")
                model_loaded = True
                break
            except Exception as e:
                if "data did not match" in str(e):
                    logger.warning(f"{desc} failed: {e}")
                    if "data did not match" in str(e) and self.config.model_name != self.config.fallback_model:
                        logger.warning("llama 3.3 model loading failed, trying 3.1 fallback")
                        self.config.model_name = self.config.fallback_model
                        logger.info(f"switching to fallback model: {self.config.model_name}")
                        # restart attempts with fallback model
                        break
                    continue
                else:
                    raise e

        if not model_loaded:
            logger.error("all model loading attempts failed")
            raise RuntimeError("failed to load any compatible model")

        # prepare for training with available components.
        if PEFT_AVAILABLE and self.config.lora_config is not None:
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=True
            )
            self.model = get_peft_model(self.model, self.config.lora_config)
            logger.info("peft model prepared for efficient training")
            self.model.print_trainable_parameters()
        else:
            logger.warning("training full model without peft - requires significant memory")

        # enable training mode for conversations.
        self.model.train()

        # apply h100 memory optimizations.
        self._optimize_h100_memory()

        # find optimal batch size for h100.
        optimal_batch_size = self.find_optimal_batch_size()

        logger.info("conversational model and tokenizer setup complete")
        logger.info(f"using optimized batch size: {optimal_batch_size}")
    
    def prepare_dataset(self):
        """prepare conversation-optimized training dataset."""
        logger.info("preparing conversational dataset with validation")

        # load datasets with conversation validation.
        dataset_handler = WorkplaceViolenceDataset(self.data_path, self.tokenizer)
        train_dataset, val_dataset = dataset_handler.to_hf_dataset()

        # use custom validation dataset for conversation quality.
        if len(val_dataset) > 0:
            self.train_dataset = train_dataset
            self.eval_dataset = val_dataset
            logger.info("using spanish alpaca dataset for conversation validation")
        else:
            # fallback to train/test split.
            split_dataset = train_dataset.train_test_split(test_size=0.2, seed=42)
            self.train_dataset = split_dataset["train"]
            self.eval_dataset = split_dataset["test"]
            logger.info("using internal train/test split")

        logger.info(f"train samples: {len(self.train_dataset)}")
        logger.info(f"eval samples: {len(self.eval_dataset)}")

        # log sample conversation for validation.
        if len(self.train_dataset) > 0:
            sample = self.train_dataset[0]["text"]
            logger.info(f"sample conversation format:\n{sample[:500]}...")
    
    def train(self):
        """execute conversational agent finetuning."""
        logger.info("starting conversational agent finetuning")

        # initialize wandb with conversation-specific metrics.
        if WANDB_AVAILABLE:
            wandb_config = {
                "model_name": self.config.model_name,
                "task": "conversational_agent",
                "validation_dataset": "bertin-project/alpaca-spanish",
                "learning_rate": self.config.training_args.learning_rate,
                "batch_size": self.config.training_args.per_device_train_batch_size,
                "gradient_accumulation_steps": self.config.training_args.gradient_accumulation_steps,
            }
            if self.config.lora_config is not None:
                wandb_config.update({
                    "lora_r": self.config.lora_config.r,
                    "lora_alpha": self.config.lora_config.lora_alpha,
                    "target_modules": self.config.lora_config.target_modules
                })
            wandb.init(
                project="llama-33-70b-conversational-agent",
                name="conversational-qlora-finetuning",
                config=wandb_config
            )
            logger.info("wandb logging initialized")
        else:
            logger.warning("training without wandb logging")

        # setup data collator for training.
        if DataCollatorForCompletionOnlyLM_AVAILABLE:
            response_template = "<|start_header_id|>assistant<|end_header_id|>"
            data_collator = DataCollatorForCompletionOnlyLM(
                response_template=response_template,
                tokenizer=self.tokenizer
            )
            logger.info("using DataCollatorForCompletionOnlyLM for response-only training")
        else:
            # fallback to standard language modeling collator.
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False  # causal language modeling.
            )
            logger.warning("using standard DataCollatorForLanguageModeling")

        # setup trainer with fallback handling.
        if TRL_AVAILABLE:
            # use sfttrainer with trl-specific features.
            # note: newer trl versions handle tokenizer automatically from model.
            # sfttrainer handles data collation internally
            trainer = SFTTrainer(
                model=self.model,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                args=self.config.training_args,
            )
            logger.info("using trl.SFTTrainer for conversation training")
        elif TRAINER_AVAILABLE and SFTTrainer is not None:
            # fallback to standard transformers trainer.
            trainer = SFTTrainer(
                model=self.model,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                args=self.config.training_args,
                data_collator=data_collator,
            )
            logger.warning("using standard transformers.Trainer (trl not available)")
        else:
            # no trainer available - cannot proceed.
            logger.error("neither trl.SFTTrainer nor transformers.Trainer is available")
            logger.error("cannot proceed with training due to compatibility issues")
            logger.error("please fix package compatibility or use a different environment")
            raise RuntimeError("No compatible trainer implementation available")

        # add conversation quality metrics computation.
        trainer.compute_metrics = self._compute_conversation_metrics

        # start training with conversation monitoring.
        logger.info("beginning conversation training...")
        trainer.train()

        # save final model with conversation capabilities.
        logger.info("saving conversational model...")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config.training_args.output_dir)

        # save conversation template.
        self._save_conversation_config()

        logger.info("conversational agent finetuning completed")
        if WANDB_AVAILABLE:
            wandb.finish()

        # save training metadata for hugging face upload.
        self._save_training_metadata()

        # copy all necessary files for transformers inference.
        self._copy_inference_files()

    def _compute_conversation_metrics(self, eval_preds):
        """compute domain-specific conversation quality metrics."""
        import numpy as np
        from collections import Counter
        import re

        predictions, labels = eval_preds

        # decode predictions and labels to text.
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        metrics = {}

        # 1. response length analysis.
        pred_lengths = [len(pred.split()) for pred in decoded_preds]
        metrics["avg_response_length"] = np.mean(pred_lengths)
        metrics["response_length_std"] = np.std(pred_lengths)

        # 2. empathy score based on empathetic language markers.
        empathy_markers = [
            'entiendo', 'comprendo', 'lamento', 'siento', 'apoyo',
            'importante', 'válido', 'valida', 'normal', 'natural',
            'puedo ayudar', 'estoy aquí', 'no estás solo', 'no estás sola',
            'mereces', 'tienes derecho', 'es comprensible'
        ]

        empathy_scores = []
        for pred in decoded_preds:
            pred_lower = pred.lower()
            empathy_count = sum(1 for marker in empathy_markers if marker in pred_lower)
            # normalize by response length.
            empathy_score = empathy_count / max(len(pred.split()), 1)
            empathy_scores.append(empathy_score)

        metrics["avg_empathy_score"] = np.mean(empathy_scores)

        # 3. domain relevance score.
        workplace_keywords = [
            'trabajo', 'laboral', 'empresa', 'oficina', 'empleado', 'empleada',
            'acoso', 'violencia', 'discriminación', 'maltrato', 'abuso',
            'protocolo', 'denuncia', 'reporte', 'recursos', 'apoyo'
        ]

        domain_scores = []
        for pred in decoded_preds:
            pred_lower = pred.lower()
            domain_count = sum(1 for keyword in workplace_keywords if keyword in pred_lower)
            domain_score = min(domain_count / 3.0, 1.0)  # normalize to max 1.0.
            domain_scores.append(domain_score)

        metrics["avg_domain_relevance"] = np.mean(domain_scores)

        # 4. professional tone score.
        professional_indicators = [
            'protocolo', 'procedimiento', 'recursos', 'departamento',
            'profesional', 'especialista', 'asesoría', 'orientación',
            'legal', 'jurídico', 'rrhh', 'recursos humanos',
            'puedes', 'debes', 'es recomendable', 'te sugiero'
        ]

        professional_scores = []
        for pred in decoded_preds:
            pred_lower = pred.lower()
            professional_count = sum(1 for indicator in professional_indicators if indicator in pred_lower)
            professional_score = min(professional_count / 2.0, 1.0)  # normalize to max 1.0.
            professional_scores.append(professional_score)

        metrics["avg_professional_tone"] = np.mean(professional_scores)

        # 5. conversation structure quality.
        structure_scores = []
        for pred in decoded_preds:
            # check for proper sentence structure.
            sentences = pred.split('.')
            complete_sentences = sum(1 for s in sentences if len(s.strip()) > 5)
            structure_score = min(complete_sentences / max(len(sentences), 1), 1.0)
            structure_scores.append(structure_score)

        metrics["avg_structure_quality"] = np.mean(structure_scores)

        # 6. repetition analysis.
        repetition_scores = []
        for pred in decoded_preds:
            words = pred.lower().split()
            if len(words) > 0:
                word_counts = Counter(words)
                # calculate repetition ratio.
                max_repetition = max(word_counts.values()) if word_counts else 1
                repetition_ratio = max_repetition / len(words)
                # lower repetition is better, so invert.
                repetition_score = max(0, 1 - repetition_ratio)
                repetition_scores.append(repetition_score)

        metrics["avg_repetition_quality"] = np.mean(repetition_scores) if repetition_scores else 0.0

        # 7. overall conversation quality composite score.
        conversation_quality = (
            metrics["avg_empathy_score"] * 0.3 +
            metrics["avg_domain_relevance"] * 0.25 +
            metrics["avg_professional_tone"] * 0.25 +
            metrics["avg_structure_quality"] * 0.1 +
            metrics["avg_repetition_quality"] * 0.1
        )
        metrics["conversation_quality_score"] = conversation_quality

        # log sample predictions for qualitative analysis.
        if len(decoded_preds) > 0:
            logger.info(f"sample prediction: {decoded_preds[0][:200]}...")

        return metrics

    def _save_conversation_config(self):
        """save conversation-specific configuration."""
        conversation_config = {
            "chat_template": self.tokenizer.chat_template,
            "response_template": "<|start_header_id|>assistant<|end_header_id|>",
            "system_message": "eres un asistente especializado en prevención de violencia laboral y acoso sexual en el entorno de trabajo. proporciona información precisa, empática y orientada a la acción basada en protocolos de prevención.\n\nresponde de forma conversacional y empática. usa un tono profesional pero cálido. siempre ofrece apoyo y recursos específicos.",
            "conversation_format": "llama3_instruct",
            "max_context_length": 2048,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True
        }

        config_path = Path(self.config.training_args.output_dir) / "conversation_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(conversation_config, f, indent=2, ensure_ascii=False)

        logger.info(f"conversation configuration saved to {config_path}")

    def _save_training_metadata(self):
        """save training metadata for hugging face upload."""
        
        metadata = {
            "base_model": self.config.model_name,
            "technique": "qlora",
            "task_type": "conversational_agent",
            "validation_dataset": "bertin-project/alpaca-spanish",
            "precision": "bfloat16",
            "quantization": "4bit",
            "epochs": self.config.training_args.num_train_epochs,
            "learning_rate": self.config.training_args.learning_rate,
            "batch_size": (
                self.config.training_args.per_device_train_batch_size *
                self.config.training_args.gradient_accumulation_steps
            ),
            "lora_r": self.config.lora_config.r,
            "lora_alpha": self.config.lora_config.lora_alpha,
            "lora_dropout": self.config.lora_config.lora_dropout,
            "target_modules": list(self.config.lora_config.target_modules) if isinstance(self.config.lora_config.target_modules, set) else self.config.lora_config.target_modules,
            "warmup_ratio": self.config.training_args.warmup_ratio,
            "lr_scheduler": self.config.training_args.lr_scheduler_type,
            "training_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_samples": len(self.train_dataset) + len(self.eval_dataset) if hasattr(self, 'train_dataset') and hasattr(self, 'eval_dataset') else "unknown",
            "conversation_features": {
                "chat_template": True,
                "response_only_training": True,
                "conversation_validation": True,
                "max_seq_length": 2048,
                "packing": False
            },
            "intended_use": "workplace_violence_prevention_chatbot",
            "language": "spanish",
            "conversational_capabilities": [
                "empathetic_responses",
                "workplace_safety_guidance",
                "resource_recommendations",
                "multi_turn_conversations"
            ]
        }
        
        # save metadata.
        metadata_path = Path(self.config.training_args.output_dir) / "hf_upload_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"training metadata saved to {metadata_path}")

    def _copy_inference_files(self):
        """copy all necessary files for transformers inference to output directory."""
        import shutil

        output_dir = Path(self.config.training_args.output_dir)

        # files to copy for complete transformers inference support.
        inference_files = [
            "config.json",
            "generation_config.json",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "HUGGINGFACE_MODEL_CARD.md",
            "README.md"
        ]

        logger.info("copying inference files to output directory...")

        for file_name in inference_files:
            src_path = Path(".") / file_name
            dst_path = output_dir / file_name

            if src_path.exists():
                shutil.copy2(src_path, dst_path)
                logger.info(f"copied {file_name} to {dst_path}")
            else:
                logger.warning(f"file {file_name} not found, skipping...")

        logger.info("all inference files copied successfully")


def main():
    """main entry point."""
    parser = argparse.ArgumentParser(description="Finetune Llama 3.3 70B model.")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save the finetuned model.")
    args = parser.parse_args()

    # h100-optimized environment setup.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # use first gpu only.
    # h100-specific memory configuration.
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,roundup_power2_divisions:16"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # async for h100 performance.
    # h100 tensor core optimizations.
    os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # pytorch sdpa attention for h100.
    os.environ["PYTORCH_ENABLE_SDPA"] = "1"  # enable sdpa attention.
    # h100 nvlink optimizations.
    os.environ["NCCL_P2P_DISABLE"] = "0"
    os.environ["NCCL_IB_DISABLE"] = "1"
    
    # check gpu availability with proper error handling.
    if not torch.cuda.is_available():
        raise RuntimeError("cuda not available - check drivers and cuda installation")
    
    device_count = torch.cuda.device_count()
    logger.info(f"detected {device_count} cuda device(s)")
    
    if device_count == 0:
        raise RuntimeError("no cuda devices detected")
    
    # use device 0 safely.
    current_device = torch.cuda.current_device()
    logger.info(f"using cuda device: {current_device}")
    
    try:
        gpu_name = torch.cuda.get_device_name(current_device)
        gpu_props = torch.cuda.get_device_properties(current_device)
        logger.info(f"gpu: {gpu_name}")
        logger.info(f"gpu memory: {gpu_props.total_memory / 1e9:.1f}gb")

        # h100 specific optimizations.
        if "H100" in gpu_name.upper():
            logger.info("h100 detected - applying h100-specific optimizations")
            os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
            os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
            # enable h100 tensor cores for maximum performance.
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        else:
            logger.warning(f"gpu {gpu_name} is not h100 - performance may be suboptimal")

    except Exception as e:
        logger.warning(f"could not get gpu properties: {e}")
        logger.info("continuing with training anyway")
    
    # initialize configuration.
    config = LlamaFinetunConfig(output_dir=args.output_dir)
    
    # check data file exists.
    data_path = "ft_data.json"
    if not Path(data_path).exists():
        raise FileNotFoundError(f"training data not found: {data_path}")
    
    # initialize finetuner.
    finetuner = LlamaFinetuner(config, data_path)
    
    try:
        # setup model and data.
        finetuner.setup_model_and_tokenizer()
        finetuner.prepare_dataset()
        
        # start training.
        finetuner.train()
        
    except Exception as e:
        logger.error(f"training failed: {e}")
        raise
    
    logger.info("training pipeline completed successfully")

if __name__ == "__main__":
    main()