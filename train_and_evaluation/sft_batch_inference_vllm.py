"""
SFT Batch Inference — vLLM 加速版
=================================
使用 vLLM 引擎进行高速批量推理，支持多卡并行。
提取 <answer>...</answer> 内容，替换 [FALSE] 为评估器识别的拒绝字符串。

用法:
    python3 train_and_evaluation/sft_batch_inference_vllm.py \
        --model_path saves/sft-model \
        --input_path data/eval_dataset.json \
        --output_path results/inference.json \
        --batch_size 256 \
        --num_docs 5
"""

import argparse
import json
import os
import re
import sys
from typing import List, Dict
from tqdm import tqdm

from qampari_sft_shared import QAMPARI_INSTRUCTION

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(THIS_DIR, ".."))
sys.path.insert(0, REPO_ROOT)
from model_adapter import build_chat_prompt
from inference_logic_guid_aqa import (
    SYMBOLIC_REFUSAL_ANSWER,
    compute_cite_pattern,
    generate_long_answer_for_item,
)

SYSTEM_PROMPT = (
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

# Keep Qampari inference aligned with the origin SFT training prompt.
QAMPARI_SYSTEM_PROMPT = QAMPARI_INSTRUCTION

# Refusal flag recognized by Trust-Eval evaluator (from eval_config.yaml)
REFUSAL_FLAG = "I apologize, but I couldn't find an answer"


def get_system_prompt(output_mode: str) -> str:
    if output_mode == "think_cite_plan_answer":
        return SYSTEM_PROMPT
    if output_mode == "think_answer":
        return THINK_ANSWER_SYSTEM_PROMPT
    if output_mode == "answer_only":
        return ANSWER_ONLY_SYSTEM_PROMPT
    raise ValueError(f"unsupported output_mode: {output_mode}")


def build_prompt(
    item: dict,
    tokenizer,
    num_docs: int = 5,
    dataset_type: str = "asqa",
    output_mode: str = "think_cite_plan_answer",
) -> str:
    """构建单个样本的 prompt，与训练时 sft_dataset.py 严格对齐"""
    docs = item['docs'][:num_docs]
    docs_text = "\n".join([
        f"Document: {d['text']}" if isinstance(d, str)
        else f"Title: {d.get('title', '')} Content: {d.get('text', '')}"
        for d in docs
    ])
    system_prompt = QAMPARI_SYSTEM_PROMPT if dataset_type == "qampari" else get_system_prompt(output_mode)
    user_content = f"Question: {item['question']} \n Documents: {docs_text}"
    prompt = build_chat_prompt(tokenizer, system_prompt, user_content)
    return prompt


def extract_answer(text: str, dataset_type: str = "asqa") -> str:
    """Extract answer content and normalize it for the target dataset."""
    m = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if not m:
        m = re.search(r'<short_answer>(.*?)</short_answer>', text, re.DOTALL)
    if m:
        raw = m.group(1).strip()
    else:
        if ("<think>" in text and "</think>" not in text) or (
            "<cite_plan>" in text and "</cite_plan>" not in text
        ):
            return REFUSAL_FLAG
        m2 = re.search(r'</think>(.*)', text, re.DOTALL)
        raw = m2.group(1).strip() if m2 else text.strip()
        raw = re.sub(r'<cite_plan>.*?</cite_plan>', '', raw, flags=re.DOTALL).strip()
        raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    return postprocess_answer(raw, dataset_type=dataset_type)


def extract_think_process(text: str) -> str:
    """Extract the model's think process for downstream symbolic reasoning."""
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"</think>", text, re.DOTALL)
    if m:
        return text[:m.start()].strip()
    return text.strip()


def build_symbolic_long_answer(
    item: dict,
    full_output: str,
    num_docs: int = 5,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    llm_base_url: str | None = None,
    llm_api_key: str | None = None,
    llm_timeout: int | None = None,
) -> tuple[str, list, str]:
    """Convert model output into a symbolically grounded long answer.

    Returns:
        long_answer_with_nemo_guidance, cite_pattern, think_process
    """
    docs = item["docs"][:num_docs]
    documents = [
        f"[D{i+1}]: {d['title']}\t\t{d['text']}" if isinstance(d, dict)
        else f"[D{i+1}]: {d}"
        for i, d in enumerate(docs)
    ]
    think_process = extract_think_process(full_output)
    cite_pattern = compute_cite_pattern(think_process) if think_process else []
    if not cite_pattern:
        return SYMBOLIC_REFUSAL_ANSWER, [], think_process
    long_answer = generate_long_answer_for_item(
        cite_pattern,
        item["question"],
        documents,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
        llm_timeout=llm_timeout,
    )
    return long_answer, cite_pattern, think_process


def _clean_qampari_fragment(fragment: str) -> str:
    fragment = fragment.strip()
    fragment = re.sub(r'^[\-\*\u2022]+\s*', '', fragment)
    fragment = re.sub(
        r'^(?:answer|the answer|final answer|short answer|entities|the entities|it is|they are|this is|there is|there are)\s*(?:is|are)?\s*[:\-]?\s*',
        '',
        fragment,
        flags=re.IGNORECASE,
    )
    fragment = fragment.strip().strip('"').strip("'").strip()
    fragment = re.sub(r'^[,;:\-\s]+', '', fragment)
    fragment = re.sub(r'[,;:\-\s]+$', '', fragment)
    fragment = fragment.rstrip(".!?")
    return fragment


def _extract_qampari_final_answer(raw: str) -> str:
    match = re.search(r'final answer\s*:\s*(.*)', raw, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return raw
    final = match.group(1).strip()
    final = re.sub(r'</?answer>', '', final, flags=re.IGNORECASE).strip()
    final = re.sub(r'</?short_answer>', '', final, flags=re.IGNORECASE).strip()
    lines = final.splitlines()
    if not lines:
        return ""
    final = lines[0].strip()
    return final


def _normalize_qampari_answer(answer: str) -> str:
    raw = answer.strip()
    if not raw:
        return raw
    if "[FALSE]" in raw:
        return REFUSAL_FLAG

    extracted = _extract_qampari_final_answer(raw)
    extracted_final_answer = extracted != raw
    raw = extracted
    raw = raw.replace("\n", " ").strip()
    raw = re.sub(
        r'^\s*(?:answer(?: is)?|the answer(?: is)?|final answer(?: is)?|short answer(?: is)?)\s*[:\-]?\s*',
        '',
        raw,
        flags=re.IGNORECASE,
    )

    fragments = re.split(r'(?<=[.!?])\s+', raw)
    cleaned = []
    seen = set()
    for fragment in fragments:
        fragment = _clean_qampari_fragment(fragment)
        if not fragment:
            continue
        norm = re.sub(r'\[\d+\]', '', fragment).strip().lower()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        cleaned.append(fragment)

    if not cleaned:
        return raw
    if len(cleaned) == 1:
        if extracted_final_answer and raw.endswith((".", "!", "?")):
            return cleaned[0] + raw[-1]
        return cleaned[0]
    return ', '.join(cleaned).rstrip('.') + '.'


def postprocess_answer(answer: str, dataset_type: str = "asqa") -> str:
    """
    1. Replace [FALSE] with evaluator-recognized refusal flag.
    2. Deduplicate repeated sentences (keep first occurrence).
    """
    if "[FALSE]" in answer:
        return REFUSAL_FLAG

    if dataset_type == "qampari":
        return _normalize_qampari_answer(answer)

    sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
    seen_norms, deduped = [], []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        norm = re.sub(r'\[\d+\]', '', s).strip().lower()
        if norm and norm not in seen_norms:
            seen_norms.append(norm)
            deduped.append(s)
    return ' '.join(deduped)


def build_engine_kwargs(args):
    engine_kwargs = dict(
        model=args.model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        disable_log_stats=False,
        enforce_eager=args.enforce_eager,
    )
    if args.max_num_seqs is not None:
        engine_kwargs["max_num_seqs"] = args.max_num_seqs
    if args.adapter_path:
        engine_kwargs["enable_lora"] = True
        engine_kwargs["max_lora_rank"] = 64
    return engine_kwargs


def main():
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=os.path.join("saves", "sft-model"))
    parser.add_argument("--adapter_path", default=None,
                        help="Path to LoRA adapter (optional)")
    parser.add_argument("--input_path",
                        default=os.path.join("data", "eval_dataset.json"))
    parser.add_argument("--output_path",
                        default=os.path.join("results", "inference.json"))
    parser.add_argument("--batch_size", type=int, default=256,
                        help="vLLM batch size (can be much larger than transformers)")
    parser.add_argument("--num_docs", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--max_model_len", type=int, default=5120,
                        help="Max context length for vLLM (input + output)")
    parser.add_argument("--tensor_parallel_size", type=int, default=2,
                        help="Number of GPUs for tensor parallelism")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                        help="GPU memory utilization (0.0-1.0)")
    parser.add_argument("--max_num_seqs", type=int, default=None,
                        help="Optional vLLM max_num_seqs to reduce warmup/concurrency memory")
    parser.add_argument("--enforce_eager", action="store_true",
                        help="Disable CUDA graph capture in vLLM; useful for long-context LoRA inference")
    parser.add_argument("--dataset_type", type=str, default="asqa",
                        choices=["asqa", "qampari", "eli5", "expertqa"],
                        help="Dataset type: asqa (long-form) or qampari (comma-separated short answers)")
    parser.add_argument("--output_mode", type=str, default="think_cite_plan_answer",
                        choices=["think_cite_plan_answer", "think_answer", "answer_only"],
                        help="Prompt format used by the SFT checkpoint.")
    parser.add_argument(
        "--symbolic_eval",
        action="store_true",
        help="Extract <think>, run cite-pattern reasoning, and generate the final long answer symbolically.",
    )
    parser.add_argument(
        "--symbolic_llm_provider",
        type=str,
        default=os.environ.get("LONG_ANSWER_PROVIDER", "qwen"),
        choices=["qwen", "gpt", "claude", "openai"],
        help="Provider used to synthesize the long answer from cite_pattern.",
    )
    parser.add_argument("--symbolic_llm_model", type=str, default=None)
    parser.add_argument("--symbolic_llm_base_url", type=str, default=None)
    parser.add_argument("--symbolic_llm_api_key", type=str, default=None)
    parser.add_argument("--symbolic_llm_timeout", type=int, default=None)
    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────────────────────
    with open(args.input_path) as f:
        raw_data = json.load(f)
    print(f"Loaded {len(raw_data)} eval samples.")

    # Resume support
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    results = {}
    done_qs = set()

    if os.path.exists(args.output_path):
        try:
            with open(args.output_path) as f:
                existing = json.load(f)
            done_qs = {item["question"] for item in existing}
            results = {i: item for i, item in enumerate(existing)}
            print(f"Resuming: {len(done_qs)} already done.")
        except Exception:
            pass

    todo_data = [(i, item) for i, item in enumerate(raw_data)
                 if item["question"] not in done_qs]
    print(f"Todo: {len(todo_data)} samples.")

    if not todo_data:
        print("All done.")
        return

    # ── Initialize vLLM ───────────────────────────────────────────────────────
    print(f"\nInitializing vLLM with {args.tensor_parallel_size} GPUs...")
    engine_kwargs = build_engine_kwargs(args)
    llm = LLM(**engine_kwargs)

    # Load tokenizer for prompt construction (must match training format)
    print(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)

    lora_request = None
    if args.adapter_path:
        lora_request = LoRARequest("grpo_adapter", 1, args.adapter_path)

    sampling_params = SamplingParams(
        temperature=0.0,  # greedy decoding for consistency
        max_tokens=args.max_new_tokens,
        skip_special_tokens=True,
    )

    # ── Batch inference ───────────────────────────────────────────────────────
    todo_indices, todo_items = zip(*todo_data)
    prompts = [
        build_prompt(item, tokenizer, args.num_docs, args.dataset_type, args.output_mode)
        for item in todo_items
    ]

    print(f"\nRunning vLLM inference (batch_size={args.batch_size})...")
    all_outputs = []

    for i in tqdm(range(0, len(prompts), args.batch_size), desc="vLLM inference"):
        batch_prompts = prompts[i:i + args.batch_size]
        batch_outputs = llm.generate(batch_prompts, sampling_params, lora_request=lora_request)
        all_outputs.extend(batch_outputs)

    # ── Process results ───────────────────────────────────────────────────────
    print("\nProcessing outputs...")
    for idx, (global_idx, item, output) in enumerate(zip(todo_indices, todo_items, all_outputs)):
        full_output = output.outputs[0].text

        result_item = dict(item)
        result_item["original_response"] = full_output
        if args.symbolic_eval:
            answer, cite_pattern, think_process = build_symbolic_long_answer(
                item,
                full_output,
                num_docs=args.num_docs,
                llm_provider=args.symbolic_llm_provider,
                llm_model=args.symbolic_llm_model,
                llm_base_url=args.symbolic_llm_base_url,
                llm_api_key=args.symbolic_llm_api_key,
                llm_timeout=args.symbolic_llm_timeout,
            )
            result_item["think_process"] = think_process
            result_item["cite_pattern"] = cite_pattern
            result_item["symbolic_eval"] = True
            result_item["symbolic_llm_provider"] = args.symbolic_llm_provider
            result_item["symbolic_llm_model"] = args.symbolic_llm_model
        else:
            answer = extract_answer(full_output, dataset_type=args.dataset_type)
        result_item["long_answer_with_nemo_guidance"] = answer
        results[global_idx] = result_item

        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(todo_data)} | Last: {answer[:80]}...")

    # ── Save results ──────────────────────────────────────────────────────────
    sorted_results = [results[i] for i in sorted(results.keys())]
    with open(args.output_path, "w") as f:
        json.dump(sorted_results, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Saved to {args.output_path} ({len(sorted_results)} samples)")


if __name__ == "__main__":
    main()
