"""
Simple script to test OlmOCR dataset loading with YAML configuration.
"""

import argparse
import logging
import math
import os
import shutil
from typing import Any, Dict, Optional

# Set environment variables before importing other modules
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import wandb
from accelerate import Accelerator
from torch.amp import autocast
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader, Subset
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    get_scheduler,
)

from olmocr.train.config import Config
from olmocr.train.dataloader import BaseMarkdownPDFDataset
from olmocr.train.muon import SingleDeviceMuonWithAuxAdam

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def prepare_lora_model(model: torch.nn.Module, model_cfg) -> torch.nn.Module:
    """Wrap the model with a LoRA adapter according to the configuration."""
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise ImportError("LoRA training requires the `peft` package. Install it with `pip install peft`.") from exc

    lora_kwargs = dict(
        r=model_cfg.lora_rank,
        lora_alpha=model_cfg.lora_alpha,
        lora_dropout=model_cfg.lora_dropout,
        target_modules=model_cfg.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    if model_cfg.lora_modules_to_save:
        lora_kwargs["modules_to_save"] = model_cfg.lora_modules_to_save

    lora_config = LoraConfig(**lora_kwargs)
    model = get_peft_model(model, lora_config)

    if hasattr(model, "config"):
        model.config.base_model_name_or_path = model_cfg.name
    base_model = getattr(model, "base_model", None)
    if base_model is not None:
        inner_model = getattr(base_model, "model", None)
        if inner_model is not None and hasattr(inner_model, "config"):
            inner_model.config._name_or_path = model_cfg.name

    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    return model


def is_lora_checkpoint(checkpoint_dir: str) -> bool:
    """Detect whether a checkpoint directory contains LoRA adapter weights."""
    return os.path.exists(os.path.join(checkpoint_dir, "adapter_config.json"))


def ensure_bbox_special_tokens(processor) -> bool:
    """Make sure the tokenizer knows about bbox start/end markers."""

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        return False

    required_tokens = ["[BBOX_BLK_END]"]
    vocab = tokenizer.get_vocab()
    missing_tokens = [token for token in required_tokens if token not in vocab]
    if not missing_tokens:
        return False

    tokenizer.add_special_tokens({"additional_special_tokens": missing_tokens})
    logger.info("Registered bbox special tokens: %s", missing_tokens)
    return True


class QwenDataCollator:
    """Data collator for vision-language models that handles numpy arrays."""

    def __init__(self, max_token_len: Optional[int] = None, device: Optional[torch.device] = None):
        self.max_token_len = max_token_len
        self.device = device

    def __call__(self, examples):
        # Filter out None values and extract the fields we need
        batch = {"input_ids": [], "attention_mask": [], "labels": [], "pixel_values": [], "image_grid_thw": []}

        for example in examples:
            if example is not None:
                # Convert numpy arrays to tensors
                input_ids = torch.from_numpy(example["input_ids"]) if isinstance(example["input_ids"], np.ndarray) else example["input_ids"]
                attention_mask = torch.from_numpy(example["attention_mask"]) if isinstance(example["attention_mask"], np.ndarray) else example["attention_mask"]
                labels = torch.from_numpy(example["labels"]) if isinstance(example["labels"], np.ndarray) else example["labels"]

                # Trim to max_token_len if specified
                if self.max_token_len is not None:
                    input_ids = input_ids[: self.max_token_len]
                    attention_mask = attention_mask[: self.max_token_len]
                    labels = labels[: self.max_token_len]

                batch["input_ids"].append(input_ids)
                batch["attention_mask"].append(attention_mask)
                batch["labels"].append(labels)

                # Handle pixel_values which might be numpy array or already a tensor
                pixel_values = example["pixel_values"]
                if isinstance(pixel_values, np.ndarray):
                    pixel_values = torch.from_numpy(pixel_values)
                batch["pixel_values"].append(pixel_values)

                # Handle image_grid_thw
                image_grid_thw = example["image_grid_thw"]
                if isinstance(image_grid_thw, np.ndarray):
                    image_grid_thw = torch.from_numpy(image_grid_thw)
                batch["image_grid_thw"].append(image_grid_thw)

        # Check if we have any valid samples
        if not batch["input_ids"]:
            return None

        # Convert lists to tensors with proper padding
        # Note: For Qwen2-VL, we typically handle variable length sequences
        # The model's processor should handle the padding internally
        try:
            result = {
                "input_ids": torch.stack(batch["input_ids"]),
                "attention_mask": torch.stack(batch["attention_mask"]),
                "labels": torch.stack(batch["labels"]),
                "pixel_values": torch.stack(batch["pixel_values"]),  # Stack into tensor
                "image_grid_thw": torch.stack(batch["image_grid_thw"]),
            }
        except Exception as e:
            # Log detailed error info for debugging
            logger.error(f"Error stacking tensors in data collator: {e}")
            logger.error(f"Batch lengths: input_ids={len(batch['input_ids'])}, "
                        f"attention_mask={len(batch['attention_mask'])}, "
                        f"labels={len(batch['labels'])}, "
                        f"pixel_values={len(batch['pixel_values'])}, "
                        f"image_grid_thw={len(batch['image_grid_thw'])}")
            raise e
        
        # Ensure tensors are on the correct device if specified
        if self.device is not None:
            try:
                result = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in result.items()}
            except Exception as e:
                logger.error(f"Error moving tensors to device {self.device}: {e}")
                # Force tensors to CUDA if available
                if torch.cuda.is_available():
                    cuda_device = torch.device("cuda")
                    result = {k: v.to(cuda_device) if isinstance(v, torch.Tensor) else v for k, v in result.items()}
        
        return result


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any,
    epoch: float,
    global_step: int,
    samples_seen: int,
    best_metric: float,
    output_dir: str,
    save_total_limit: Optional[int] = None,
    processor: Optional[Any] = None,
):
    """Save model, optimizer, scheduler, and training state."""
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save model
    model.save_pretrained(checkpoint_dir)

    # Save tokenizer/processor assets first so they share the same output dir
    if processor is not None:
        try:
            processor.save_pretrained(checkpoint_dir)
        except Exception as exc:
            logger.warning("Failed to save processor assets to %s: %s", checkpoint_dir, exc)

    # Save optimizer and scheduler
    torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
    torch.save(lr_scheduler.state_dict(), os.path.join(checkpoint_dir, "scheduler.pt"))

    # Save training state
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "samples_seen": samples_seen,
        "best_metric": best_metric,
    }
    torch.save(state, os.path.join(checkpoint_dir, "training_state.pt"))

    logger.info(f"Saved checkpoint to {checkpoint_dir}")

    # Enforce save_total_limit by removing oldest checkpoints
    if save_total_limit is not None and save_total_limit > 0:
        checkpoints = sorted([d for d in os.listdir(output_dir) if d.startswith("checkpoint-")], key=lambda x: int(x.split("-")[1]))
        while len(checkpoints) > save_total_limit:
            oldest = checkpoints.pop(0)
            shutil.rmtree(os.path.join(output_dir, oldest))
            logger.info(f"Deleted old checkpoint: {oldest}")


def load_checkpoint(
    model_class: type,
    init_kwargs: Dict[str, Any],
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any,
    checkpoint_dir: str,
    device: torch.device,
    *,
    base_model_path: Optional[str] = None,
    use_lora: bool = False,
) -> tuple[torch.nn.Module, Dict[str, Any]]:
    """Load model, optimizer, scheduler, and training state from checkpoint."""
    checkpoint_has_lora = is_lora_checkpoint(checkpoint_dir)

    if checkpoint_has_lora or use_lora:
        if base_model_path is None:
            raise ValueError("base_model_path must be provided when loading LoRA checkpoints.")

        try:
            from peft import PeftModel
        except ImportError as exc:  # pragma: no cover - optional dependency guard
            raise ImportError("Loading a LoRA checkpoint requires the `peft` package. Install it with `pip install peft`.") from exc

        base_model = model_class.from_pretrained(base_model_path, **init_kwargs)
        model = PeftModel.from_pretrained(base_model, checkpoint_dir, is_trainable=True)
        if hasattr(model, "config"):
            model.config.base_model_name_or_path = base_model_path
    else:
        model = model_class.from_pretrained(checkpoint_dir, **init_kwargs)

    model.to(device)

    # For DeepSpeed, we'll skip optimizer/scheduler loading here
    # They need to be loaded after accelerator.prepare()
    
    # Load training state - this is always safe to load
    training_state_path = os.path.join(checkpoint_dir, "training_state.pt")
    if os.path.exists(training_state_path):
        try:
            state = torch.load(training_state_path, map_location=device, weights_only=False)
            logger.info(f"Resumed from checkpoint: {checkpoint_dir} at epoch {state.get('epoch', 0):.2f}, step {state.get('global_step', 0)}, samples seen {state.get('samples_seen', 0)}")
        except Exception as e:
            logger.error(f"Failed to load training state from {training_state_path}: {e}")
            # Create minimal state to allow training to continue
            state = {"global_step": 0, "samples_seen": 0, "best_metric": float("inf"), "epoch": 0.0}
            logger.warning("Created minimal training state for fresh start")
    else:
        logger.warning(f"Training state not found at {training_state_path}, creating fresh state")
        state = {"global_step": 0, "samples_seen": 0, "best_metric": float("inf"), "epoch": 0.0}
    
    return model, state


def load_deepspeed_checkpoint_state(
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any,
    checkpoint_dir: str,
    device: torch.device,
) -> None:
    """Load optimizer and scheduler state for DeepSpeed after accelerator.prepare()."""
    logger = logging.getLogger(__name__)
    
    # Load optimizer state with error handling
    optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
    if os.path.exists(optimizer_path):
        try:
            optimizer_checkpoint = torch.load(optimizer_path, map_location=device, weights_only=False)
            # Check if this is a DeepSpeed checkpoint
            if "base_optimizer_state" in optimizer_checkpoint:
                logger.info("Loading DeepSpeed optimizer checkpoint")
                optimizer_state = optimizer_checkpoint["base_optimizer_state"]
            else:
                optimizer_state = optimizer_checkpoint
                
            if "param_groups" in optimizer_state:
                optimizer.load_state_dict(optimizer_state)
                logger.info(f"Successfully loaded optimizer state from {optimizer_path}")
            else:
                logger.warning(f"Optimizer checkpoint {optimizer_path} is missing 'param_groups' key, skipping optimizer loading")
        except Exception as e:
            logger.warning(f"Failed to load optimizer state from {optimizer_path}: {e}")
    else:
        logger.warning(f"Optimizer checkpoint not found at {optimizer_path}, starting with fresh optimizer")

    # Load scheduler state with error handling  
    scheduler_path = os.path.join(checkpoint_dir, "scheduler.pt")
    if os.path.exists(scheduler_path):
        try:
            lr_scheduler.load_state_dict(torch.load(scheduler_path, map_location=device, weights_only=False))
            logger.info(f"Successfully loaded scheduler state from {scheduler_path}")
        except Exception as e:
            logger.warning(f"Failed to load scheduler state from {scheduler_path}: {e}")
    else:
        logger.warning(f"Scheduler checkpoint not found at {scheduler_path}, starting with fresh scheduler")


def evaluate_model(
    model: torch.nn.Module,
    eval_dataloaders: Dict[str, DataLoader],
    device: torch.device,
    accelerator = None,
) -> Dict[str, float]:
    """Evaluate on all eval datasets and return average loss per dataset."""
    model.eval()
    eval_metrics = {}

    for dataset_name, dataloader in eval_dataloaders.items():
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                # Skip if batch is None (all samples were filtered out)
                if batch is None:
                    continue
                # Remove manual device movement if accelerator is used
                if accelerator is None:
                    batch = {k: v.to(device) for k, v in batch.items()}
                with autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                    outputs = model(**batch)
                total_loss += outputs.loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        eval_metrics[f"eval_{dataset_name}_loss"] = avg_loss
        if accelerator and accelerator.is_main_process:
            logger.info(f"Eval {dataset_name} loss: {avg_loss:.4f}")
        elif accelerator is None:  # Backward compatibility
            logger.info(f"Eval {dataset_name} loss: {avg_loss:.4f}")

    # Compute overall eval loss as average across datasets (or customize as needed)
    if eval_metrics:
        overall_loss = sum(eval_metrics.values()) / len(eval_metrics)
        eval_metrics["eval_loss"] = overall_loss

    return eval_metrics


def create_train_dataloader(
    train_dataset,
    config,
    data_collator,
    seed_worker,
    epoch_num: int = 0,
) -> DataLoader:
    """Create a training dataloader with epoch-specific shuffling.

    Args:
        train_dataset: The training dataset
        config: Training configuration
        data_collator: Data collator for batching
        seed_worker: Worker initialization function
        epoch_num: Current epoch number for seed generation

    Returns:
        DataLoader with epoch-specific shuffling
    """
    # Create generator with epoch-specific seed for different shuffling each epoch
    epoch_generator = torch.Generator()
    if config.training.data_seed is not None:
        # Use epoch number to ensure different shuffling each epoch while maintaining reproducibility
        epoch_generator.manual_seed(config.training.data_seed + epoch_num)
    else:
        # Use a random seed if no data_seed specified
        epoch_generator.manual_seed(int(torch.randint(0, 2**32 - 1, (1,)).item()))

    dataloader_kwargs: Dict[str, Any] = dict(
        batch_size=config.training.per_device_train_batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=config.training.dataloader_num_workers,
        drop_last=config.training.dataloader_drop_last,
        worker_init_fn=seed_worker,
        generator=epoch_generator,
    )

    prefetch_factor = getattr(config.training, "dataloader_prefetch_factor", None)
    if (
        prefetch_factor is not None
        and config.training.dataloader_num_workers > 0
        and prefetch_factor > 0
    ):
        dataloader_kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(train_dataset, **dataloader_kwargs)


def main():
    # Set tokenizer parallelism environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["RAYON_NUM_THREADS"] = "1"
    
    # Increase NCCL timeout to prevent collective timeouts
    os.environ["NCCL_TIMEOUT"] = "1800"  # 30 minutes
    os.environ["NCCL_BLOCKING_WAIT"] = "1"
    
    # Initialize Accelerator for distributed training
    accelerator = Accelerator()
    
    parser = argparse.ArgumentParser(description="Train OlmOCR model")
    parser.add_argument("--config", type=str, default="olmocr/train/configs/example_config.yaml", help="Path to YAML configuration file")

    args = parser.parse_args()

    # Load configuration
    if accelerator.is_main_process:
        logger.info(f"Loading configuration from: {args.config}")
    config = Config.from_yaml(args.config)

    # Validate configuration
    try:
        config.validate()
    except ValueError as e:
        if accelerator.is_main_process:
            logger.error(f"Configuration validation failed: {e}")
        return

    # Set wandb project from config
    if config.project_name:
        os.environ["WANDB_PROJECT"] = config.project_name
        if accelerator.is_main_process:
            logger.info(f"Setting WANDB_PROJECT to: {config.project_name}")

    # Initialize wandb if reporting to it (only on main process)
    if "wandb" in config.training.report_to and accelerator.is_main_process:
        wandb.init(project=config.project_name, name=config.run_name, config=config.to_dict())

    # Load processor for tokenization
    if accelerator.is_main_process:
        logger.info(f"Loading processor: {config.model.name}")
    processor = AutoProcessor.from_pretrained(
        config.model.name,
    )
    ensure_bbox_special_tokens(processor)

    # Model init kwargs to reuse for loading checkpoints
    model_init_kwargs = {
        "torch_dtype": getattr(torch, config.model.torch_dtype) if config.model.torch_dtype != "auto" else "auto",
        "device_map": config.model.device_map,
        "trust_remote_code": config.model.trust_remote_code,
        "attn_implementation": config.model.attn_implementation if config.model.use_flash_attention else None,
    }

    # Load model
    if accelerator.is_main_process:
        logger.info(f"Loading model: {config.model.name}")
    if "qwen2.5-vl" in config.model.name.lower() or "olmocr-2-7b-1025" in config.model.name.lower():
        model_class = Qwen2_5_VLForConditionalGeneration
        model = model_class.from_pretrained(config.model.name, **model_init_kwargs)
    elif "qwen2-vl" in config.model.name.lower():
        model_class = Qwen2VLForConditionalGeneration
        model = model_class.from_pretrained(config.model.name, **model_init_kwargs)
    else:
        raise NotImplementedError()

    tokenizer_vocab_size = len(processor.tokenizer)
    input_embeddings = model.get_input_embeddings()
    if input_embeddings is not None:
        current_vocab_size = input_embeddings.weight.size(0)
        if current_vocab_size != tokenizer_vocab_size:
            model.resize_token_embeddings(tokenizer_vocab_size)
            if accelerator.is_main_process:
                logger.info("Resized token embeddings to match tokenizer vocab size: %d", tokenizer_vocab_size)

    if hasattr(model.config, "vocab_size"):
        model.config.vocab_size = tokenizer_vocab_size

    if config.model.use_lora:
        if accelerator.is_main_process:
            logger.info("Applying LoRA adapters as specified in the config.")
        model = prepare_lora_model(model, config.model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_ratio = (trainable_params / total_params * 100) if total_params else 0.0
    if accelerator.is_main_process:
        logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_ratio:.2f}%)")

    # Enable gradient checkpointing if configured
    if config.training.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=config.training.gradient_checkpointing_kwargs)

    # Create training datasets
    if accelerator.is_main_process:
        logger.info("Creating training datasets...")
    train_datasets = []
    for i, dataset_cfg in enumerate(config.dataset.train):
        root_dir = dataset_cfg["root_dir"]
        pipeline_steps = config.get_pipeline_steps(dataset_cfg["pipeline"], processor)

        if accelerator.is_main_process:
            logger.info(f"Creating training dataset {i+1} from: {root_dir}")
        dataset = BaseMarkdownPDFDataset(root_dir, pipeline_steps)
        if accelerator.is_main_process:
            logger.info(f"Found {len(dataset)} samples")

        if len(dataset) > 0:
            train_datasets.append(dataset)

    # Combine all training datasets
    train_dataset = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
    total_train_samples = len(train_dataset)
    if accelerator.is_main_process:
        logger.info(f"Total training samples: {total_train_samples}")

    if config.training.max_train_samples is not None and total_train_samples > config.training.max_train_samples:
        subset_size = max(0, config.training.max_train_samples)
        if subset_size == 0:
            raise ValueError("config.training.max_train_samples must be greater than 0 when specified.")
        if accelerator.is_main_process:
            logger.info(
                f"Limiting training dataset to first {subset_size} samples (from {total_train_samples})."
            )
        train_dataset = Subset(train_dataset, range(subset_size))
        total_train_samples = subset_size

    # Create evaluation datasets
    if accelerator.is_main_process:
        logger.info("Creating evaluation datasets...")
    eval_datasets = {}
    for i, dataset_cfg in enumerate(config.dataset.eval):
        root_dir = dataset_cfg["root_dir"]
        pipeline_steps = config.get_pipeline_steps(dataset_cfg["pipeline"], processor)

        # Use dataset name if provided, otherwise use root_dir as name
        dataset_name = dataset_cfg.get("name", f"eval_dataset_{i+1}")

        if accelerator.is_main_process:
            logger.info(f"Creating evaluation dataset '{dataset_name}' from: {root_dir}")
        dataset = BaseMarkdownPDFDataset(root_dir, pipeline_steps)
        if accelerator.is_main_process:
            logger.info(f"Found {len(dataset)} samples")

        if len(dataset) > 0:
            if (
                config.training.max_eval_samples is not None
                and len(dataset) > config.training.max_eval_samples
            ):
                subset_size = max(0, config.training.max_eval_samples)
                if subset_size == 0:
                    raise ValueError("config.training.max_eval_samples must be greater than 0 when specified.")
                if accelerator.is_main_process:
                    logger.info(
                        f"Limiting evaluation dataset '{dataset_name}' to first {subset_size} samples (from {len(dataset)})."
                    )
                dataset = Subset(dataset, range(subset_size))
            eval_datasets[dataset_name] = dataset

    # Log total evaluation samples across all datasets
    total_eval_samples = sum(len(dataset) for dataset in eval_datasets.values())
    if accelerator.is_main_process:
        logger.info(f"Total evaluation samples across {len(eval_datasets)} datasets: {total_eval_samples}")

    # Construct full output directory by appending run_name to base output_dir
    full_output_dir = os.path.join(config.training.output_dir, config.run_name)
    if accelerator.is_main_process:
        logger.info(f"Setting output directory to: {full_output_dir}")
    os.makedirs(full_output_dir, exist_ok=True)

    # Check for existing checkpoints if any
    found_resumable_checkpoint = None
    if os.path.exists(full_output_dir):
        # Look for checkpoint directories
        checkpoint_dirs = [d for d in os.listdir(full_output_dir) if d.startswith("checkpoint-") and os.path.isdir(os.path.join(full_output_dir, d))]
        if checkpoint_dirs:
            # Sort by checkpoint number and get the latest
            checkpoint_dirs.sort(key=lambda x: int(x.split("-")[1]))
            latest_checkpoint = os.path.join(full_output_dir, checkpoint_dirs[-1])
            if accelerator.is_main_process:
                logger.info(f"Found existing checkpoint: {latest_checkpoint}")
            found_resumable_checkpoint = latest_checkpoint
        else:
            if accelerator.is_main_process:
                logger.info("No existing checkpoints found in output directory")

    # Set seeds
    torch.manual_seed(config.training.seed)

    # Set up data loader seed worker function
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        import random

        random.seed(worker_seed)

    # Device setup - use accelerator device
    device = accelerator.device

    # Apply torch compile if enabled
    if config.training.torch_compile:
        if accelerator.is_main_process:
            logger.info(f"Compiling model with torch.compile (backend={config.training.torch_compile_backend}, mode={config.training.torch_compile_mode})")
        model = torch.compile(
            model,
            backend=config.training.torch_compile_backend,
            mode=config.training.torch_compile_mode,
            fullgraph=config.training.torch_compile_fullgraph,
            dynamic=config.training.torch_compile_dynamic,
        )
        if accelerator.is_main_process:
            logger.info("Model compilation complete")

    # Set up optimizer
    trainable_named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not trainable_named_params:
        raise ValueError("No trainable parameters found. Check model fine-tuning configuration.")

    if config.training.optim == "adamw_torch":
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in trainable_named_params if not any(nd in n for nd in no_decay)],
                "weight_decay": config.training.weight_decay,
            },
            {
                "params": [p for n, p in trainable_named_params if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=float(config.training.learning_rate),
            betas=(config.training.adam_beta1, config.training.adam_beta2),
            eps=float(config.training.adam_epsilon),
        )
    elif config.training.optim == "muon":
        if config.model.use_lora:
            raise NotImplementedError("LoRA training is not currently supported with the Muon optimizer in this loop.")

        # Separate parameters for Muon (hidden matrices) and Adam (embeddings, scalars, head)
        hidden_matrix_params = [p for n, p in trainable_named_params if p.ndim >= 2 and "embed" not in n and "lm_head" not in n]
        embed_params = [p for n, p in trainable_named_params if "embed" in n]
        scalar_params = [p for n, p in trainable_named_params if p.ndim < 2]
        head_params = [p for n, p in trainable_named_params if "lm_head" in n]

        # Create Adam groups with different learning rates
        adam_groups = [
            dict(params=head_params, lr=float(config.training.learning_rate) * config.training.muon_lr_multiplier_head, use_muon=False),
            dict(params=embed_params, lr=float(config.training.learning_rate) * config.training.muon_lr_multiplier_embed, use_muon=False),
            dict(params=scalar_params, lr=float(config.training.learning_rate) * config.training.muon_lr_multiplier_scalar, use_muon=False),
        ]

        # Add Adam hyperparameters to groups
        for g in adam_groups:
            g["betas"] = (config.training.adam_beta1, config.training.adam_beta2)
            g["eps"] = float(config.training.adam_epsilon)
            g["weight_decay"] = config.training.weight_decay

        # Create Muon group
        muon_group = dict(
            params=hidden_matrix_params,
            lr=float(config.training.learning_rate),
            momentum=config.training.muon_momentum,
            weight_decay=config.training.weight_decay,
            use_muon=True,
        )

        # Combine all groups
        param_groups = [*adam_groups, muon_group]
        optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
    else:
        raise NotImplementedError(f"Optimizer {config.training.optim} not supported in custom loop")

    # Total training steps calculation
    samples_per_step = config.training.per_device_train_batch_size * config.training.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(total_train_samples / samples_per_step)
    max_train_steps = int(math.ceil(config.training.num_train_epochs * num_update_steps_per_epoch))
    max_train_samples = int(math.ceil(config.training.num_train_epochs * total_train_samples))

    # Set up scheduler
    lr_scheduler = get_scheduler(
        name=config.training.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=int(max_train_steps * config.training.warmup_ratio),
        num_training_steps=max_train_steps,
        scheduler_specific_kwargs=config.training.lr_scheduler_kwargs,
    )

    # Data collator
    data_collator = QwenDataCollator(
        max_token_len=config.training.collator_max_token_len,
        device=accelerator.device
    )

    # Resume from checkpoint if available
    global_step = 0
    samples_seen = 0
    best_metric = float("inf") if not config.training.greater_is_better else -float("inf")

    # Store checkpoint info but don't load yet (need to load after accelerator.prepare for DeepSpeed)
    checkpoint_state = None
    if found_resumable_checkpoint:
        if accelerator.is_main_process:
            logger.info(f"Found checkpoint to resume from: {found_resumable_checkpoint}")

    # Create dataloaders - use epoch 0 initially (will be recreated with proper epoch if resuming)
    current_epoch_num = 0  # Will be updated after loading checkpoint
    train_dataloader = create_train_dataloader(
        train_dataset,
        config,
        data_collator,
        seed_worker,
        epoch_num=current_epoch_num,
    )

    eval_dataloaders = {
        name: DataLoader(
            dataset,
            batch_size=config.training.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=data_collator,
            num_workers=0,  # Disable multiprocessing for tokenizer compatibility
            drop_last=False,
        )
        for name, dataset in eval_datasets.items()
    }

    # Prepare model, optimizer, and dataloaders for distributed training
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
    # Prepare eval dataloaders separately
    eval_dataloaders = {
        name: accelerator.prepare(dataloader) 
        for name, dataloader in eval_dataloaders.items()
    }

    # NOW load checkpoint after accelerator.prepare() for DeepSpeed compatibility
    if found_resumable_checkpoint:
        model, checkpoint_state = load_checkpoint(
            model_class,
            model_init_kwargs,
            optimizer,
            lr_scheduler,
            found_resumable_checkpoint,
            device,
            base_model_path=config.model.name,
            use_lora=config.model.use_lora,
        )
        global_step = checkpoint_state["global_step"]
        best_metric = checkpoint_state["best_metric"]
        samples_seen = checkpoint_state["samples_seen"]
        
        # Load optimizer and scheduler state after DeepSpeed preparation
        load_deepspeed_checkpoint_state(
            optimizer,
            lr_scheduler,
            found_resumable_checkpoint,
            device
        )
        
        # Recreate dataloaders with correct epoch if needed
        current_epoch_num = int(samples_seen / total_train_samples) if samples_seen > 0 else 0
        if current_epoch_num > 0:
            train_dataloader = create_train_dataloader(
                train_dataset,
                config,
                data_collator,
                seed_worker,
                epoch_num=current_epoch_num,
            )
            # Re-prepare the new dataloader
            train_dataloader = accelerator.prepare(train_dataloader)

    # Always evaluate on start
    metrics = evaluate_model(model, eval_dataloaders, device, accelerator)
    if accelerator.is_main_process:
        logger.info(f"Initial evaluation: {metrics}")
    if "wandb" in config.training.report_to and accelerator.is_main_process:
        wandb.log(metrics, step=global_step)

    # Main training loop
    current_epoch = samples_seen / total_train_samples
    if accelerator.is_main_process:
        logger.info(f"Starting training from epoch {current_epoch:.2f} (step {global_step}, samples {samples_seen}) to {config.training.num_train_epochs} epochs")
        logger.info(f"Total training steps: {max_train_steps}, Total samples to process: {max_train_samples}")

    if samples_seen >= max_train_samples:
        if accelerator.is_main_process:
            logger.info("Training already completed based on samples seen!")
            logger.info("Skipping to final model save.")
    else:
        model.train()
        accumulated_loss = 0.0
        num_losses_accumulated = 0

        # Create epoch iterator and skip samples if resuming
        epoch_iterator = iter(train_dataloader)
        if samples_seen > 0:
            samples_to_skip = samples_seen % total_train_samples
            batches_to_skip = samples_to_skip // config.training.per_device_train_batch_size
            if accelerator.is_main_process:
                logger.info(f"Resuming training: skipping {batches_to_skip} batches ({samples_to_skip} samples) to reach position {samples_seen}")

            # Skip batches to resume from the correct position within the epoch
            for _ in range(batches_to_skip):
                try:
                    next(epoch_iterator)
                except StopIteration:
                    # We've reached the end of the epoch while skipping
                    # This shouldn't normally happen, but handle it gracefully
                    if accelerator.is_main_process:
                        logger.warning(f"Reached end of epoch while skipping batches. Creating new epoch.")
                    current_epoch_num += 1
                    train_dataloader = create_train_dataloader(
                        train_dataset,
                        config,
                        data_collator,
                        seed_worker,
                        epoch_num=current_epoch_num,
                    )
                    epoch_iterator = iter(train_dataloader)
                    break

        # Create progress bar (only on main process)
        pbar = tqdm(total=max_train_samples - samples_seen, desc=f"Training from step {global_step}", unit="samples", disable=not accelerator.is_main_process)

        while samples_seen < max_train_samples and global_step < max_train_steps:
            try:
                batch = next(epoch_iterator)
            except StopIteration:
                # End of epoch, create new dataloader with fresh shuffle
                current_epoch = samples_seen / total_train_samples
                if accelerator.is_main_process:
                    logger.info(f"Completed epoch {current_epoch:.2f}")

                # Increment epoch number for new shuffle seed
                current_epoch_num += 1

                # Recreate dataloader with new generator for fresh shuffle
                train_dataloader = create_train_dataloader(
                    train_dataset,
                    config,
                    data_collator,
                    seed_worker,
                    epoch_num=current_epoch_num,
                )
                epoch_iterator = iter(train_dataloader)
                batch = next(epoch_iterator)

            # Skip if batch is None (all samples were filtered out)
            if batch is None:
                continue

            # Periodic device and memory monitoring
            if global_step % 100 == 0 and accelerator.is_main_process:
                logger.info(f"Step {global_step} - GPU memory: {torch.cuda.memory_allocated(accelerator.device) / 1e9:.2f}GB / {torch.cuda.memory_reserved(accelerator.device) / 1e9:.2f}GB")
                
            # Ensure all batch tensors are on the correct device
            try:
                # Check batch device consistency before processing
                batch_devices = {k: v.device if isinstance(v, torch.Tensor) else None for k, v in batch.items()}
                expected_device = accelerator.device
                
                # Double-check device placement
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor) and value.device != expected_device:
                        logger.warning(f"Step {global_step}: Moving {key} from {value.device} to {expected_device}")
                        batch[key] = value.to(expected_device)
                
                # Log batch info for debugging
                if global_step % 500 == 0 and accelerator.is_main_process:
                    logger.info(f"Step {global_step} batch info:")
                    for key, value in batch.items():
                        if isinstance(value, torch.Tensor):
                            logger.info(f"  {key}: shape={value.shape}, device={value.device}, dtype={value.dtype}")
                
                # Remove manual device movement - accelerator handles this
                with autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                    outputs = model(**batch)
            except RuntimeError as e:
                if "Expected all tensors to be on the same device" in str(e):
                    logger.error(f"Device mismatch error at step {global_step}: {e}")
                    logger.error(f"Batch device info:")
                    for key, value in batch.items():
                        if isinstance(value, torch.Tensor):
                            logger.error(f"  {key}: device={value.device}, shape={value.shape}")
                    logger.error(f"Model device: {next(model.parameters()).device}")
                    logger.error(f"Accelerator device: {accelerator.device}")
                    logger.error(f"CUDA devices available: {torch.cuda.device_count()}")
                    logger.error(f"Current CUDA device: {torch.cuda.current_device()}")
                    raise e
                else:
                    raise e
            loss = outputs.loss / config.training.gradient_accumulation_steps
            accelerator.backward(loss)

            accumulated_loss += outputs.loss.item()  # Use undivided loss for logging
            num_losses_accumulated += 1
            samples_seen += config.training.per_device_train_batch_size

            # Update progress bar (only on main process)
            if accelerator.is_main_process:
                pbar.update(config.training.per_device_train_batch_size)

            # Check if we should do a gradient update
            if samples_seen % samples_per_step == 0 or samples_seen >= max_train_samples:
                # Clip gradients
                accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

                # Step optimizer and scheduler
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                current_epoch = samples_seen / total_train_samples

                # Update progress bar with current stats (only on main process)
                if accelerator.is_main_process:
                    current_lr = lr_scheduler.get_last_lr()[0]
                    avg_loss = accumulated_loss / num_losses_accumulated if num_losses_accumulated > 0 else 0
                    pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{current_lr:.2e}", "epoch": f"{current_epoch:.2f}", "step": global_step})

                # Logging (only on main process)
                if config.training.logging_steps > 0 and global_step % config.training.logging_steps == 0 and accelerator.is_main_process:
                    avg_train_loss = accumulated_loss / num_losses_accumulated if num_losses_accumulated > 0 else 0
                    logs = {
                        "train_loss": avg_train_loss,
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                        "epoch": current_epoch,
                        "samples_seen": samples_seen,
                    }
                    if accelerator.is_main_process:
                        logger.info(f"Step {global_step}: epoch={current_epoch:.3f}, loss={avg_train_loss:.4f}, lr={lr_scheduler.get_last_lr()[0]:.2e}")
                    if "wandb" in config.training.report_to and accelerator.is_main_process:
                        wandb.log(logs, step=global_step)

                    accumulated_loss = 0.0
                    num_losses_accumulated = 0

                # Evaluation
                if config.training.eval_steps > 0 and global_step % config.training.eval_steps == 0 and global_step > 0:
                    metrics = evaluate_model(model, eval_dataloaders, device, accelerator)
                    if accelerator.is_main_process:
                        logger.info(f"Evaluation at step {global_step}: {metrics}")
                    if "wandb" in config.training.report_to and accelerator.is_main_process:
                        wandb.log(metrics, step=global_step)

                    # Update best metric
                    current_metric = metrics.get(config.training.metric_for_best_model, None)
                    if current_metric is not None:
                        if (config.training.greater_is_better and current_metric > best_metric) or (
                            not config.training.greater_is_better and current_metric < best_metric
                        ):
                            best_metric = current_metric

                    # Return to training mode
                    model.train()

                # Saving (only on main process, but wait for everyone)
                if config.training.save_steps > 0 and global_step % config.training.save_steps == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        # Unwrap model for saving
                        unwrapped_model = accelerator.unwrap_model(model)
                        save_checkpoint(
                            unwrapped_model,
                            optimizer,
                            lr_scheduler,
                            current_epoch,
                            global_step,
                            samples_seen,
                            best_metric,
                            full_output_dir,
                            config.training.save_total_limit,
                            processor=processor,
                        )

            # Check if we've reached our training limit
            if samples_seen >= max_train_samples or global_step >= max_train_steps:
                break

        # Close progress bar (only on main process)
        if accelerator.is_main_process:
            pbar.close()

    # Save the final checkpoint with step number
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info(f"Saving final checkpoint at step {global_step}...")
        unwrapped_model = accelerator.unwrap_model(model)
        save_checkpoint(
            unwrapped_model,
            optimizer,
            lr_scheduler,
            current_epoch,
            global_step,
            samples_seen,
            best_metric,
            full_output_dir,
            config.training.save_total_limit,
            processor=processor,
        )

    # Log final training state
    final_epoch = samples_seen / total_train_samples
    if accelerator.is_main_process:
        logger.info(f"Training completed at epoch {final_epoch:.3f}, step {global_step}, samples {samples_seen}")

    # Final evaluation
    final_metrics = evaluate_model(model, eval_dataloaders, device, accelerator)
    if accelerator.is_main_process:
        logger.info(f"Final evaluation metrics: {final_metrics}")
    if "wandb" in config.training.report_to and accelerator.is_main_process:
        wandb.log(final_metrics, step=global_step)
        wandb.finish()


if __name__ == "__main__":
    main()
