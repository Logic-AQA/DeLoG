"""
SFT Best ASQA Reproduction Script
==================================
Reproduces the SFT baseline that achieved Trust-Score 68.56 on ASQA.

Configuration (frozen):
  - Base model: provided via --model_path
  - Data: data_sft.json (4,331 samples)
  - Epochs: 2, LR: 5e-6, ZeRO-2, bf16
  - Max len: 3584, Batch/GPU: 1, Grad accum: 8
  - Token weights: answer=2.5, think critical=1.2, struct=1.0, boiler=0.3, other=0.8

Launch:
    MODEL_PATH=<local-base-model> CUDA_VISIBLE_DEVICES=0,1 deepspeed --num_gpus 2 train_sft.py
Or single-GPU debug:
    python3 train_sft.py --local_rank -1
"""

import os
import sys
import math
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from dataclasses import dataclass
from typing import List, Any
import deepspeed

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
from model_adapter import build_chat_prompt, find_think_boundaries

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_MODEL_PATH = os.environ.get("MODEL_PATH", "")
DATA_PATH       = "data_sft.json"
SAVE_PATH       = os.path.join("saves", "sft-best-asqa-repro")

# ── Hyper-parameters ─────────────────────────────────────────────────────────
MAX_LEN         = 3584
BATCH_PER_GPU   = 1
GRAD_ACCUM      = 8
EPOCHS          = 2
LR              = 5e-6
WEIGHT_DECAY    = 0.01
WARMUP_RATIO    = 0.05
LOG_EVERY       = 10

FULL_SYSTEM_PROMPT = (
    "Your task is to generate a Datalog-style logic programing process between "
    "<think></think>, a step-by-step citation plan between <cite_plan></cite_plan>, "
    "and a long-form answer between <answer></answer> with citation [*] "
    "for a given question and documents"
)

THINK_ANSWER_SYSTEM_PROMPT = (
    "Your task is to generate a Datalog-style logic programing process between "
    "<think></think> and a long-form answer between <answer></answer> with citation [*] "
    "for a given question and documents"
)

ANSWER_ONLY_SYSTEM_PROMPT = (
    "Your task is to generate a long-form answer between <answer></answer> "
    "with citation [*] for a given question and documents"
)


def get_system_prompt(output_mode: str) -> str:
    if output_mode == "think_cite_plan_answer":
        return FULL_SYSTEM_PROMPT
    if output_mode == "think_answer":
        return THINK_ANSWER_SYSTEM_PROMPT
    if output_mode == "answer_only":
        return ANSWER_ONLY_SYSTEM_PROMPT
    raise ValueError(f"unsupported output_mode: {output_mode}")


def build_target(item: dict, output_mode: str) -> str:
    answer = item["long_answer_with_citation"]
    if output_mode == "think_cite_plan_answer":
        cite_plan_text = item.get("cite_plan", "")
        return (
            f"<think>\n{item['logic_program']}\n</think>\n"
            f"<cite_plan>\n{cite_plan_text}\n</cite_plan>\n"
            f"<answer>{answer}</answer><|im_end|>"
        )
    if output_mode == "think_answer":
        return (
            f"<think>\n{item['logic_program']}\n</think>\n"
            f"<answer>{answer}</answer><|im_end|>"
        )
    if output_mode == "answer_only":
        return f"<answer>{answer}</answer><|im_end|>"
    raise ValueError(f"unsupported output_mode: {output_mode}")


# ── Dataset ──────────────────────────────────────────────────────────────────

class SFTDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_len: int = 3072,
                 output_mode: str = "think_cite_plan_answer"):
        with open(data_path, "r") as f:
            self.data = json.load(f)
        self.tok = tokenizer
        self.max_len = max_len
        self.output_mode = output_mode
        self.system_prompt = get_system_prompt(output_mode)

    def __len__(self) -> int:
        return len(self.data)

    def _assign_weights(self, input_ids: List[int], prompt_len: int) -> List[float]:
        """Per-token loss weights for improve-correctness setting."""
        CRITICAL_KWS = [
            'valid_result', 'action_atomic', 'action_convergence',
            'action_interleaved', 'action_composite',
            '"d1"', '"d2"', '"d3"', '"d4"', '"d5"',
        ]
        STRUCT_KWS = ['feature', 'keep', 'discard', 'cluster', 'query_score']
        BOILER_KWS = ['lib']

        weights = [1.0] * len(input_ids)

        think_start, think_end = find_think_boundaries(input_ids, self.tok, prompt_len)

        if think_start == -1:
            for i in range(prompt_len, len(input_ids)):
                weights[i] = 2.5
            return weights

        # Find <cite_plan> and </cite_plan> boundaries
        cp_start_ids = self.tok.encode("<cite_plan>", add_special_tokens=False)
        cp_end_ids   = self.tok.encode("</cite_plan>", add_special_tokens=False)
        ans_start_ids = self.tok.encode("<answer>", add_special_tokens=False)

        cite_plan_start_tok = -1
        cite_plan_end_tok   = -1
        answer_start_tok    = -1

        for i in range(think_end, len(input_ids) - len(cp_start_ids) + 1):
            if input_ids[i:i+len(cp_start_ids)] == cp_start_ids:
                cite_plan_start_tok = i + len(cp_start_ids)
                break

        for i in range(think_end, len(input_ids) - len(cp_end_ids) + 1):
            if input_ids[i:i+len(cp_end_ids)] == cp_end_ids:
                cite_plan_end_tok = i
                break

        for i in range(think_end, len(input_ids) - len(ans_start_ids) + 1):
            if input_ids[i:i+len(ans_start_ids)] == ans_start_ids:
                answer_start_tok = i + len(ans_start_ids)
                break

        for i in range(len(input_ids)):
            if i < prompt_len or i <= think_start:
                weights[i] = 1.0
            elif i < think_end:
                decoded = self.tok.decode([input_ids[i]])
                if any(k in decoded for k in CRITICAL_KWS):
                    weights[i] = 1.2
                elif any(k in decoded for k in STRUCT_KWS):
                    weights[i] = 1.0
                elif any(k in decoded for k in BOILER_KWS):
                    weights[i] = 0.3
                else:
                    weights[i] = 0.8
            elif cite_plan_start_tok != -1 and cite_plan_end_tok != -1 \
                    and cite_plan_start_tok <= i < cite_plan_end_tok:
                weights[i] = 2.5
            elif answer_start_tok != -1 and i >= answer_start_tok:
                weights[i] = 2.5
            else:
                weights[i] = 1.5

        return weights

    def __getitem__(self, idx: int):
        item = self.data[idx]

        docs = item.get('docs', [])
        processed_docs = []
        for d in docs:
            if isinstance(d, str):
                processed_docs.append({"title": "", "text": d})
            else:
                processed_docs.append(d)
        docs_text = "\n".join([
            f"Title: {d.get('title', '')} Content: {d.get('text', '')}"
            for d in processed_docs
        ])

        user_content = f"Question: {item['question']} \n Documents: {docs_text}"
        prompt = build_chat_prompt(self.tok, self.system_prompt, user_content)

        target = build_target(item, self.output_mode)
        full_text = prompt + target

        encoded = self.tok(
            full_text,
            truncation=True,
            max_length=self.max_len,
            add_special_tokens=False,
        )
        input_ids: List[int] = encoded["input_ids"]

        prompt_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        prompt_len = len(prompt_ids)
        labels = [-100] * prompt_len + input_ids[prompt_len:]
        labels = labels[:self.max_len]

        token_weights = self._assign_weights(input_ids, prompt_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "token_weights": torch.tensor(token_weights, dtype=torch.float32),
        }


@dataclass
class SFTDataCollator:
    tokenizer: Any

    def __call__(self, features: List[dict]) -> dict:
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        input_ids = torch.nn.utils.rnn.pad_sequence(
            [f["input_ids"] for f in features], batch_first=True, padding_value=pad_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [f["labels"] for f in features], batch_first=True, padding_value=-100
        )
        token_weights = torch.nn.utils.rnn.pad_sequence(
            [f["token_weights"] for f in features], batch_first=True, padding_value=1.0
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [f["attention_mask"] for f in features], batch_first=True, padding_value=0
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "token_weights": token_weights,
        }


# ── Training ─────────────────────────────────────────────────────────────────

def compute_loss(model, batch):
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )
    logits        = outputs.logits
    shift_logits  = logits[..., :-1, :].contiguous()
    shift_labels  = batch["labels"][..., 1:].contiguous()
    shift_weights = batch["token_weights"][..., 1:].contiguous()

    ce_fct  = nn.CrossEntropyLoss(reduction='none')
    ce_flat = ce_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )
    ce_mat  = ce_flat.view(shift_labels.size())
    lm_mask = (shift_labels != -100).float()
    loss = (
        (ce_mat * shift_weights * lm_mask).sum()
        / (lm_mask.sum() + 1e-8)
    )
    return loss


def get_ds_config(
    world_size: int = 1,
    reduce_bucket_size: int | None = None,
    offload_optimizer: bool = False,
) -> dict:
    zero_config = {
        "stage":                2,
        "allgather_partitions": True,
        "reduce_scatter":       True,
        "overlap_comm":         True,
    }
    if reduce_bucket_size is not None:
        zero_config["reduce_bucket_size"] = reduce_bucket_size
        zero_config["allgather_bucket_size"] = reduce_bucket_size
    if offload_optimizer:
        zero_config["offload_optimizer"] = {
            "device": "cpu",
            "pin_memory": True,
        }

    cfg = {
        "train_micro_batch_size_per_gpu": BATCH_PER_GPU,
        "gradient_accumulation_steps":    GRAD_ACCUM,
        "fp16": {"enabled": False},
        "bf16": {"enabled": True},
        "gradient_clipping": 1.0,
        "zero_optimization": zero_config,
    }
    if offload_optimizer:
        cfg["zero_force_ds_cpu_optimizer"] = False
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--model_path", type=str, default=BASE_MODEL_PATH,
                        help="Path to local base model checkpoint")
    parser.add_argument("--save_path", type=str, default=SAVE_PATH,
                        help="Path to save trained model")
    parser.add_argument(
        "--data_path",
        type=str,
        default=DATA_PATH,
        help="Path to SFT JSON data",
    )
    parser.add_argument(
        "--output_mode",
        type=str,
        default="think_cite_plan_answer",
        choices=["think_cite_plan_answer", "think_answer", "answer_only"],
        help="Ablation target format",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to reduce activation memory",
    )
    parser.add_argument(
        "--reduce_bucket_size",
        type=int,
        default=None,
        help="Optional DeepSpeed ZeRO reduce/allgather bucket size",
    )
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate")
    parser.add_argument(
        "--offload_optimizer",
        action="store_true",
        help="Offload ZeRO optimizer state to CPU",
    )
    parser.add_argument(
        "--max_train_batches",
        type=int,
        default=None,
        help="Optional micro-batch limit for stability smoke tests",
    )
    parser.add_argument(
        "--skip_save",
        action="store_true",
        help="Skip final model save for stability smoke tests",
    )
    args = parser.parse_args()

    if not args.model_path:
        raise ValueError("Provide --model_path or set MODEL_PATH to a local base model checkpoint")

    is_main = args.local_rank in (-1, 0)

    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        deepspeed.init_distributed()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset  = SFTDataset(args.data_path, tokenizer, max_len=MAX_LEN,
                          output_mode=args.output_mode)
    collator = SFTDataCollator(tokenizer)

    sampler    = DistributedSampler(dataset) if args.local_rank != -1 else None
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_PER_GPU,
        collate_fn=collator,
        sampler=sampler,
        shuffle=(sampler is None),
        pin_memory=True,
        num_workers=2,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)

    world_size = torch.distributed.get_world_size() if args.local_rank != -1 else 1
    steps_per_epoch = math.ceil(len(dataset) / (BATCH_PER_GPU * GRAD_ACCUM * world_size))
    total_steps  = steps_per_epoch * EPOCHS
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))

    if is_main:
        print(f"Model: {args.model_path}")
        print(f"Dataset: {args.data_path} ({len(dataset)} samples)")
        print(f"Output mode: {args.output_mode}")
        print(f"LR: {args.lr}")
        print(f"Total optimiser steps: {total_steps}, warmup: {warmup_steps}")

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        args=args,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=get_ds_config(
            world_size,
            reduce_bucket_size=args.reduce_bucket_size,
            offload_optimizer=args.offload_optimizer,
        ),
    )

    global_step = 0
    stop_training = False
    for epoch in range(EPOCHS):
        if sampler is not None:
            sampler.set_epoch(epoch)

        model_engine.train()
        for step, batch in enumerate(dataloader):
            batch = {k: v.to(model_engine.device) for k, v in batch.items()}

            loss = compute_loss(model_engine.module, batch)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss at epoch={epoch+1}, micro_step={global_step+1}: "
                    f"{loss.item()}"
                )

            model_engine.backward(loss)
            model_engine.step()
            global_step += 1

            if is_main and global_step % LOG_EVERY == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch+1}/{EPOCHS}  Step {global_step}  "
                    f"Loss: {loss.item():.4f}  LR: {current_lr:.2e}"
                )
            if args.max_train_batches is not None and global_step >= args.max_train_batches:
                stop_training = True
                break
        if stop_training:
            break

    if is_main and not args.skip_save:
        os.makedirs(args.save_path, exist_ok=True)
        model_engine.module.save_pretrained(args.save_path)
        tokenizer.save_pretrained(args.save_path)
        print(f"Model saved to {args.save_path}")
    elif is_main:
        print("Skipped model save")


if __name__ == "__main__":
    main()
