"""
GRPO Training Script (Unsloth Version)
=======================================
Uses unsloth + TRL GRPOTrainer for memory-efficient GRPO training.

Advantages over the hand-written loop:
  - Chunked log softmax (avoids materializing full [B, L, V] logits)
  - Unsloth gradient checkpointing (1.0x vs HF 1.5x)
  - Padding-free training
  - Built-in reference model management (no ref_server needed)
  - Built-in vLLM generation (no separate vLLM process needed)

Single-GPU:
    python train_grpo.py

Multi-GPU (DDP):
    torchrun --nproc_per_node=4 train_grpo.py

Changelog:
  2026-05-08  (1) Citation reward enabled by default (cite_rec/cite_prec weights 0.15).
              (2) Answer extracted directly from model output (<answer>...</answer>)
                  in reward.py partial_reward(), with length penalty to discourage
                  overly long answers (linear decay 1.0 -> 0.5 for 100-200 words;
                  thresholds calibrated against original run answered_length ~83).
              (3) MAX_STEPS reduced from 1400 to 1200 based on eval results.
  2026-05-24  Rebalanced reward toward QA correctness: QA=0.55, citation=0.25,
              logic+feature=0.20, anneal starts at step 400, annealed refusal
              signal, and citation-missing penalty instead of capping QA reward.
"""

import os
import sys
import re
import json
import argparse
import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = THIS_DIR
TRAIN_EVAL_DIR = os.path.join(REPO_ROOT, "train_and_evaluation")
sys.path.insert(0, TRAIN_EVAL_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, "unsloth-main"))

from grpo_reward_components import (
    answerability_reward,
    classify_pareto_route,
    group_minmax,
    group_minmax_masked,
    is_refusal_answer,
    length_reward,
    pareto_gate_reward,
    weighted_sum,
)

BASE_MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join("saves", "sft-best-asqa-repro"),
)
DATA_PATH = "data_grpo.json"
SAVE_PATH = os.path.join("saves", "grpo-improve-correctness-refusal")

# ── Hyper-parameters ─────────────────────────────────────────────────────────
LORA_R = 64
LORA_ALPHA = 64
MAX_SEQ_LENGTH = 3200
MAX_PROMPT_LENGTH = 1024
MAX_COMPLETION_LENGTH = 2048
NUM_GENERATIONS = 16
MAX_STEPS = 1500
SAVE_STEPS = 300

# SYSTEM_PROMPT = (
#     "Your task is to generate a Datalog-style logic programing process between "
#     "<think></think> and a long-form answer between <answer></answer> with citation [*] "
#     "for a given question and documents"
# )

SYSTEM_PROMPT = (
    "Your task is to generate a Datalog-style logic programing process between "
    "<think></think> and a long-form answer between <answer></answer> with citation [*] "
    "for a given question and documents"
)

# ── Reward defaults ──────────────────────────────────────────────────────────
REWARD_WEIGHTS = {
    "cite_rec":  0.125,
    "cite_prec": 0.125,
    "qa":        0.55,
    "logic":     0.12,
    "feature":   0.08,
    "illegal":   0.00,
}
REWARD_ENABLED = {
    "cite_rec":  True,
    "cite_prec": True,
    "qa":        True,
    "logic":     True,
    "feature":   True,
    "illegal":   False,
}

# Global reward computation config (populated from CLI in main)
REWARD_CONFIG = {
    "normalize_per_component": False,  # Let GRPOTrainer handle group normalization
    "group_size": NUM_GENERATIONS,
    "reward_mode": "current",
}
REWARD_DEBUG_CONFIG = {
    "enabled": False,
    "max_batches": 0,
    "max_samples": 0,
    "calls": 0,
}
BOUNDARY_GDPO_COMPONENT_CACHE = {
    "key": None,
    "components": None,
}
ANNEAL_START_STEP = 400
ANNEAL_DECAY_STEPS = 400
DOC_REFUSAL_BASE_WEIGHT = 0.10
DOC_REFUSAL_MIN_MULTIPLIER = 0.40
DOC_REFUSAL_REWARD = 0.40
DOC_REFUSAL_PENALTY = -0.10
CITATION_MISSING_PENALTY = -0.05


# ── Reward functions ─────────────────────────────────────────────────────────

def _extract_logic(text: str) -> str:
    """Extract Datalog logic from text.

    Backward-compatible with two formats:
      - Old: model outputs `<think>...logic...</think>`
      - V3.1+: prompt already contains `<think>`, so model outputs `...logic...</think>`
    """
    m = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r'</think>', text, re.DOTALL)
    if m:
        return text[:m.start()].strip()
    return text.strip()


def _doc_refusal_weight(anneal: float) -> float:
    return DOC_REFUSAL_BASE_WEIGHT * max(DOC_REFUSAL_MIN_MULTIPLIER, anneal)


def _reward_anneal(step: int) -> float:
    if REWARD_CONFIG.get("disable_reward_anneal", False):
        return 1.0
    anneal_start = int(REWARD_CONFIG.get("anneal_start_step", ANNEAL_START_STEP))
    anneal_decay = max(1, int(REWARD_CONFIG.get("anneal_decay_steps", ANNEAL_DECAY_STEPS)))
    return math.exp(-max(0, step - anneal_start) / anneal_decay)


def _compute_single_reward(completion_text: str, sample: dict, **kwargs) -> tuple:
    """Compute reward for one completion against one sample.
    Returns (total, qa_raw, logic_norm, feature, cite_rec, cite_prec, illegal, skip_cite,
             anneal, logic_norm_annealed, feature_annealed, cite_rec_annealed, cite_prec_annealed,
             doc_refusal_r, citation_missing_penalty_r, gt_refusal_correct, gt_refusal_wrong).
    Skips expensive computations (AutoAIS, Qwen API) for disabled reward components.
    """
    from reward import partial_reward, citation_recall_precision, is_refusal_ground_truth, docs_support_answer

    reward_mode = REWARD_CONFIG.get("reward_mode", "current")
    qa_citation_mode = reward_mode == "qa_citation"

    # Progressive annealing: process rewards (logic/feature/citation) decay after
    # step 400, shifting optimization pressure toward QA once citation and format
    # behavior are established.
    # FIX: use trainer_state.global_step instead of wandb.run.step, because wandb.run.step
    # auto-increments on every wandb.log() call and quickly diverges from the real step.
    trainer_state = kwargs.get("trainer_state")
    step = trainer_state.global_step if trainer_state is not None else 0
    anneal = _reward_anneal(step)

    logic_process = _extract_logic(completion_text)
    # Pass raw completion text so QA/citation rewards are computed on the model-generated answer
    qa_r, logic_r, feat_r, illegal_r, long_ans = partial_reward(
        logic_process, sample, completion_text=completion_text
    )

    cite_rec = 0.0
    cite_prec = 0.0
    skip_cite = False
    citation_computed = False

    # ── Refusal signals ──────────────────────────────────────────────────────
    # Ground-truth refusal and document-unsupported normal QA are different:
    #   - ground-truth refusal keeps the main QA reward/penalty from reward.py;
    #   - normal QA with unsupported retrieved docs gets only a small document-
    #     refusal shaping signal, so retrieval misses do not override labels.
    doc_refusal_r = 0.0
    citation_missing_penalty_r = 0.0
    gt_refusal_correct = 0.0
    gt_refusal_wrong = 0.0
    is_gt_refusal = False if qa_citation_mode else is_refusal_ground_truth(sample)
    has_false_answer = long_ans is not None and "[FALSE]" in long_ans

    if is_gt_refusal:
        skip_cite = True
        if has_false_answer:
            gt_refusal_correct = 1.0
        else:
            gt_refusal_wrong = 1.0
            # Wrong answers on true refusal samples should not be rescued by
            # format/process rewards.
            logic_r = 0.0
            feat_r = 0.0
            illegal_r = 0.0
    elif not qa_citation_mode and not docs_support_answer(sample):
        # Neutralise QA / citation for retrieval-unsupported normal samples.
        qa_r = 0.0
        skip_cite = True
        # Refusal signal: reward correct refusal, lightly penalise blind answers.
        # The support detector can be conservative, so the negative side is
        # intentionally weaker than the positive side.
        doc_refusal_r = DOC_REFUSAL_REWARD if has_false_answer else DOC_REFUSAL_PENALTY
        # Logic / feature / illegal still computed normally (they reflect format quality)
        # but re-assemble total below will use the neutralised values above.
    # ─────────────────────────────────────────────────────────────────────────

    # Only short-circuit if *everything* is zero and we have no long answer for citation
    if qa_r == 0 and logic_r == 0 and feat_r == 0 and illegal_r == 0 and long_ans is None and doc_refusal_r == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # Citation reward (expensive AutoAIS calls) – skip entirely for refusal /
    # unsupported samples, and when both citation components are disabled.
    if (
        not skip_cite
        and (REWARD_ENABLED["cite_rec"] or REWARD_ENABLED["cite_prec"])
        and logic_r > 0
        and long_ans is not None
    ):
        cite_rec, cite_prec = citation_recall_precision(long_ans, sample)
        citation_computed = True
        # For correct refusals, mark citation as "missing" so group normalization
        # treats it as neutral (0) instead of pulling the sample to the minimum.
        if cite_rec is None and cite_prec is None:
            skip_cite = True
        if not REWARD_ENABLED["cite_rec"]:
            cite_rec = 0.0
        if not REWARD_ENABLED["cite_prec"]:
            cite_prec = 0.0

    # Coupling: normal sample with zero citation recall receives a small penalty.
    # Keep the QA reward intact so correct answers are not flattened into the same
    # bucket as partially correct answers.
    if (
        not qa_citation_mode
        and not is_gt_refusal
        and citation_computed
        and not skip_cite
        and qa_r > 0.5
        and cite_rec == 0.0
    ):
        citation_missing_penalty_r = CITATION_MISSING_PENALTY

    # Normalize logic reward
    logic_r_norm = (logic_r / 2.0) if REWARD_ENABLED["logic"] else 0.0

    # Apply progressive annealing to process rewards (QA is untouched)
    logic_r_norm_annealed = logic_r_norm * anneal
    feat_r_annealed = feat_r * anneal
    # citation may be None for correct refusals — guard before multiplying
    cite_rec_annealed = cite_rec * anneal if cite_rec is not None else None
    cite_prec_annealed = cite_prec * anneal if cite_prec is not None else None

    # Assemble total reward only for enabled components
    total = 0.0
    if REWARD_ENABLED["cite_rec"] and not skip_cite:
        total += REWARD_WEIGHTS["cite_rec"] * cite_rec_annealed
    if REWARD_ENABLED["cite_prec"] and not skip_cite:
        total += REWARD_WEIGHTS["cite_prec"] * cite_prec_annealed
    if REWARD_ENABLED["qa"]:
        total += REWARD_WEIGHTS["qa"] * qa_r
    if REWARD_ENABLED["logic"]:
        total += REWARD_WEIGHTS["logic"] * logic_r_norm_annealed
    if REWARD_ENABLED["feature"]:
        total += REWARD_WEIGHTS["feature"] * feat_r_annealed
    if REWARD_ENABLED["illegal"]:
        total += REWARD_WEIGHTS["illegal"] * illegal_r
    # Document-driven refusal signal is annealed with a floor. It remains a
    # safety boundary without becoming the dominant late-training signal.
    total += _doc_refusal_weight(anneal) * doc_refusal_r
    total += citation_missing_penalty_r

    return (
        float(total),
        float(qa_r) if REWARD_ENABLED["qa"] else 0.0,
        float(logic_r_norm),
        float(feat_r) if REWARD_ENABLED["feature"] else 0.0,
        cite_rec if skip_cite else float(cite_rec),
        cite_prec if skip_cite else float(cite_prec),
        float(illegal_r) if REWARD_ENABLED["illegal"] else 0.0,
        skip_cite,
        float(anneal),
        float(logic_r_norm_annealed),
        float(feat_r_annealed),
        float(cite_rec_annealed) if (cite_rec_annealed is not None and not skip_cite) else 0.0,
        float(cite_prec_annealed) if (cite_prec_annealed is not None and not skip_cite) else 0.0,
        float(doc_refusal_r),
        float(citation_missing_penalty_r),
        float(gt_refusal_correct),
        float(gt_refusal_wrong),
    )


def _validate_boundary_answerability_labels(labels, expected_count: int) -> list:
    if labels is None or len(labels) != expected_count:
        raise ValueError(
            "boundary_gdpo requires answerability_label for every completion "
            f"({0 if labels is None else len(labels)}/{expected_count})"
        )

    labels = [str(label) for label in labels]
    invalid = [label for label in labels if label not in {"answerable", "unanswerable"}]
    if invalid:
        raise ValueError(
            "boundary_gdpo requires answerability_label values to be exactly "
            f"'answerable' or 'unanswerable'; got {invalid[:3]}"
        )
    return labels


def _coerce_bool(value) -> bool:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no", ""}:
            return False
    return bool(value)


def _validate_boundary_metadata(sample_types, docs_supported, labels, expected_count: int) -> list:
    if sample_types is None or len(sample_types) != expected_count:
        raise ValueError(
            "boundary_gdpo requires sample_type='boundary' for every completion "
            f"({0 if sample_types is None else len(sample_types)}/{expected_count})"
        )
    invalid_types = [str(value) for value in sample_types if str(value) != "boundary"]
    if invalid_types:
        raise ValueError(f"boundary_gdpo requires sample_type='boundary'; got {invalid_types[:3]}")

    if docs_supported is None or len(docs_supported) != expected_count:
        raise ValueError(
            "boundary_gdpo requires docs_supported metadata for every completion "
            f"({0 if docs_supported is None else len(docs_supported)}/{expected_count})"
        )
    docs_supported = [_coerce_bool(value) for value in docs_supported]
    inconsistent = [
        (label, flag)
        for label, flag in zip(labels, docs_supported)
        if (label == "answerable" and not flag) or (label == "unanswerable" and flag)
    ]
    if inconsistent:
        raise ValueError(
            "boundary_gdpo docs_supported metadata must be True for answerable and "
            f"False for unanswerable samples; got {inconsistent[:3]}"
        )
    return docs_supported


def _mean(values) -> float:
    if not values:
        return 0.0
    return float(np.mean([float(value) for value in values]))


def _mean_masked(values, mask) -> float:
    valid_values = [float(value) for value, include in zip(values, mask) if include]
    return _mean(valid_values)


def _emit_mode_reward_metrics(reward_mode: str, metrics: dict) -> None:
    if (
        REWARD_DEBUG_CONFIG.get("enabled")
        and REWARD_DEBUG_CONFIG.get("calls", 0) < REWARD_DEBUG_CONFIG.get("max_batches", 0)
    ):
        REWARD_DEBUG_CONFIG["calls"] = REWARD_DEBUG_CONFIG.get("calls", 0) + 1
        metric_text = " ".join(
            f"{name}={value:.4f}" if isinstance(value, float) else f"{name}={value}"
            for name, value in metrics.items()
        )
        print(f"[reward-debug] mode={reward_mode} {metric_text}", flush=True)

    try:
        import wandb
    except ImportError:
        return

    if getattr(wandb, "run", None) is None:
        return

    log_dict = {"sub_rewards/mode": reward_mode}
    for name, value in metrics.items():
        log_dict[f"sub_rewards/{reward_mode}/{name}"] = value
    wandb.log(log_dict, commit=False)


def _build_reward_config(args) -> dict:
    return {
        "normalize_per_component": args.normalize_rewards,
        "group_size": args.num_generations,
        "reward_mode": getattr(args, "reward_mode", "current"),
        "disable_reward_anneal": getattr(args, "disable_reward_anneal", False),
        "anneal_start_step": getattr(args, "anneal_start_step", ANNEAL_START_STEP),
        "anneal_decay_steps": getattr(args, "anneal_decay_steps", ANNEAL_DECAY_STEPS),
    }


def _is_peft_adapter_path(model_path: str) -> bool:
    return os.path.isfile(os.path.join(str(model_path), "adapter_config.json"))


def _apply_lora_if_needed(model, fast_language_model_cls):
    if hasattr(model, "peft_config"):
        return model, False
    model = fast_language_model_cls.get_peft_model(
        model,
        r=LORA_R,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=LORA_ALPHA,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    return model, True


def _completion_text(completion) -> str:
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        return completion[0].get("content", "")
    if isinstance(completion, dict):
        return completion.get("content", "")
    return str(completion)


def _completion_answer(text: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else text


def _sample_from_reward_batch(
    i: int,
    prompts,
    docs,
    short_answers,
    long_answer_with_citation=None,
    sample_type=None,
    answerability_label=None,
    docs_supported=None,
) -> dict:
    sample = {
        "docs": docs[i],
        "short_answers": short_answers[i],
        "question": "",
        "long_answer_with_citation": long_answer_with_citation[i] if long_answer_with_citation is not None else "",
        "sample_type": sample_type[i] if sample_type is not None else "",
        "answerability_label": answerability_label[i] if answerability_label is not None else "",
        "docs_supported": bool(docs_supported[i]) if docs_supported is not None else False,
    }
    user_msg = prompts[i][-1]["content"] if prompts[i] else ""
    q_match = re.match(r"Question:\s*(.+?)\s*\n", user_msg)
    sample["question"] = q_match.group(1) if q_match else user_msg
    return sample


def _boundary_gdpo_cache_key(prompts, completions, kwargs) -> tuple:
    trainer_state = kwargs.get("trainer_state")
    step = trainer_state.global_step if trainer_state is not None else 0
    completion_fingerprint = tuple(_completion_text(completion) for completion in completions)
    reward_mode = REWARD_CONFIG.get("reward_mode", "boundary_gdpo")
    return id(prompts), id(completions), len(completions), step, reward_mode, completion_fingerprint


def _compute_boundary_gdpo_components(
    prompts,
    completions,
    docs,
    short_answers,
    long_answer_with_citation=None,
    sample_type=None,
    answerability_label=None,
    docs_supported=None,
    **kwargs,
) -> dict[str, list[float]]:
    labels = _validate_boundary_answerability_labels(answerability_label, len(completions))
    _validate_boundary_metadata(sample_type, docs_supported, labels, len(completions))

    def _worker(i):
        text = _completion_text(completions[i])
        sample = _sample_from_reward_batch(
            i,
            prompts,
            docs,
            short_answers,
            long_answer_with_citation=long_answer_with_citation,
            sample_type=sample_type,
            answerability_label=answerability_label,
            docs_supported=docs_supported,
        )
        try:
            return _compute_single_reward(text, sample, **kwargs)
        except Exception as e:
            print(f"[reward] Error sample_index={i}: {type(e).__name__}: {e}", flush=True)
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(_worker, range(len(completions))))

    raw_qa = [r[1] for r in results]
    raw_logic = [r[2] for r in results]
    skip_cite_mask = [r[7] for r in results]
    annealed_cite_rec = [r[11] for r in results]
    annealed_cite_prec = [r[12] for r in results]

    full_completion_texts = [_completion_text(completion) for completion in completions]
    completion_answers = [_completion_answer(text) for text in full_completion_texts]
    answerable_mask = [str(label) == "answerable" for label in labels]
    refused = [is_refusal_answer(answer) for answer in completion_answers]
    qa_mask = [answerable_mask[i] and not refused[i] for i in range(len(completions))]
    routes = [
        classify_pareto_route(is_answerable=answerable_mask[i], refused=refused[i])
        for i in range(len(completions))
    ]
    gate = [
        pareto_gate_reward(
            routes[i],
            qa_reward=raw_qa[i],
            cite_rec=annealed_cite_rec[i] if annealed_cite_rec[i] is not None else 0.0,
            cite_prec=annealed_cite_prec[i] if annealed_cite_prec[i] is not None else 0.0,
        )
        for i in range(len(completions))
    ]
    reward_mode = REWARD_CONFIG.get("reward_mode", "boundary_gdpo")
    if reward_mode == "constrained_pareto_gdpo":
        answer_quality_mask = [
            qa_mask[i] and routes[i] == "answerable_answer" and raw_qa[i] > 0.0
            for i in range(len(completions))
        ]
        citation_mask = [
            answer_quality_mask[i]
            and not skip_cite_mask[i]
            and (
                (annealed_cite_rec[i] or 0.0) > 0.0
                or (annealed_cite_prec[i] or 0.0) > 0.0
            )
            for i in range(len(completions))
        ]
        logic_mask = answer_quality_mask
        refusal_mask = [routes[i] == "unanswerable_refusal" for i in range(len(completions))]
        refusal_metric_mask = [not answerable_mask[i] for i in range(len(completions))]
    else:
        citation_mask = [qa_mask[i] and not skip_cite_mask[i] for i in range(len(completions))]
        logic_mask = qa_mask
        refusal_mask = [not answerable_mask[i] for i in range(len(completions))]
        refusal_metric_mask = refusal_mask

    components = {
        "gate": gate,
        "answerability": [
            answerability_reward(is_answerable=answerable_mask[i], refused=refused[i])
            for i in range(len(completions))
        ],
        "answerable_boundary": [
            answerability_reward(is_answerable=True, refused=refused[i]) if answerable_mask[i] else 0.0
            for i in range(len(completions))
        ],
        "refusal_boundary": [
            answerability_reward(is_answerable=False, refused=refused[i]) if not answerable_mask[i] else 0.0
            for i in range(len(completions))
        ],
        "qa": [
            float(raw_qa[i]) if qa_mask[i] and routes[i] == "answerable_answer" else 0.0
            for i in range(len(completions))
        ],
        "cite_rec": [
            float(annealed_cite_rec[i]) if citation_mask[i] and annealed_cite_rec[i] is not None else 0.0
            for i in range(len(completions))
        ],
        "cite_prec": [
            float(annealed_cite_prec[i]) if citation_mask[i] and annealed_cite_prec[i] is not None else 0.0
            for i in range(len(completions))
        ],
        "logic": [
            float(raw_logic[i]) if logic_mask[i] else 0.0
            for i in range(len(completions))
        ],
        "refusal": [
            1.0 if refusal_mask[i] else 0.0
            for i in range(len(completions))
        ],
        "length": [
            length_reward(completion_answers[i], refused=refused[i])
            for i in range(len(completions))
        ],
        "answer_length": [
            length_reward(completion_answers[i], refused=refused[i]) if answerable_mask[i] else 0.0
            for i in range(len(completions))
        ],
        "refusal_length": [
            length_reward(completion_answers[i], refused=refused[i]) if not answerable_mask[i] else 0.0
            for i in range(len(completions))
        ],
    }

    _emit_mode_reward_metrics(
        reward_mode,
        {
            "gate": _mean(components["gate"]),
            "answerability": _mean(components["answerability"]),
            "answerable_boundary": _mean_masked(components["answerable_boundary"], answerable_mask),
            "refusal_boundary": _mean_masked(
                components["refusal_boundary"],
                [not value for value in answerable_mask],
            ),
            "qa": _mean(components["qa"]),
            "cite_rec": _mean_masked(components["cite_rec"], citation_mask),
            "cite_prec": _mean_masked(components["cite_prec"], citation_mask),
            "refusal": _mean_masked(components["refusal"], refusal_metric_mask),
            "logic": _mean_masked(components["logic"], logic_mask),
            "length": _mean(components["length"]),
            "answer_length": _mean_masked(components["answer_length"], answerable_mask),
            "refusal_length": _mean_masked(
                components["refusal_length"],
                [not value for value in answerable_mask],
            ),
            "refused_rate": _mean([1.0 if value else 0.0 for value in refused]),
            "answerable_label_rate": _mean([1.0 if value else 0.0 for value in answerable_mask]),
            "unanswerable_label_rate": _mean([0.0 if value else 1.0 for value in answerable_mask]),
        },
    )
    return components


def _get_boundary_gdpo_components(
    prompts,
    completions,
    docs,
    short_answers,
    long_answer_with_citation=None,
    sample_type=None,
    answerability_label=None,
    docs_supported=None,
    **kwargs,
) -> dict[str, list[float]]:
    cache_key = _boundary_gdpo_cache_key(prompts, completions, kwargs)
    if BOUNDARY_GDPO_COMPONENT_CACHE["key"] == cache_key:
        return BOUNDARY_GDPO_COMPONENT_CACHE["components"]

    components = _compute_boundary_gdpo_components(
        prompts,
        completions,
        docs,
        short_answers,
        long_answer_with_citation=long_answer_with_citation,
        sample_type=sample_type,
        answerability_label=answerability_label,
        docs_supported=docs_supported,
        **kwargs,
    )
    BOUNDARY_GDPO_COMPONENT_CACHE["key"] = cache_key
    BOUNDARY_GDPO_COMPONENT_CACHE["components"] = components
    return components


def _make_boundary_gdpo_reward_func(component_name: str, function_name: str):
    def reward_func(
        prompts,
        completions,
        docs,
        short_answers,
        long_answer_with_citation=None,
        sample_type=None,
        answerability_label=None,
        docs_supported=None,
        **kwargs,
    ):
        components = _get_boundary_gdpo_components(
            prompts,
            completions,
            docs,
            short_answers,
            long_answer_with_citation=long_answer_with_citation,
            sample_type=sample_type,
            answerability_label=answerability_label,
            docs_supported=docs_supported,
            **kwargs,
        )
        return components[component_name]

    reward_func.__name__ = function_name
    return reward_func


def build_boundary_gdpo_reward_funcs():
    return [
        _make_boundary_gdpo_reward_func("answerability", "boundary_answerability_reward"),
        _make_boundary_gdpo_reward_func("qa", "boundary_qa_reward"),
        _make_boundary_gdpo_reward_func("cite_rec", "boundary_cite_rec_reward"),
        _make_boundary_gdpo_reward_func("cite_prec", "boundary_cite_prec_reward"),
        _make_boundary_gdpo_reward_func("logic", "boundary_logic_reward"),
        _make_boundary_gdpo_reward_func("length", "boundary_length_reward"),
    ]


def build_masked_boundary_gdpo_reward_funcs():
    return [
        _make_boundary_gdpo_reward_func("answerable_boundary", "masked_answerable_boundary_reward"),
        _make_boundary_gdpo_reward_func("refusal_boundary", "masked_refusal_boundary_reward"),
        _make_boundary_gdpo_reward_func("qa", "masked_qa_reward"),
        _make_boundary_gdpo_reward_func("cite_rec", "masked_cite_rec_reward"),
        _make_boundary_gdpo_reward_func("cite_prec", "masked_cite_prec_reward"),
        _make_boundary_gdpo_reward_func("logic", "masked_logic_reward"),
        _make_boundary_gdpo_reward_func("answer_length", "masked_answer_length_reward"),
        _make_boundary_gdpo_reward_func("refusal_length", "masked_refusal_length_reward"),
    ]


def build_constrained_pareto_gdpo_reward_funcs():
    return [
        _make_boundary_gdpo_reward_func("gate", "pareto_gate_reward"),
        _make_boundary_gdpo_reward_func("qa", "pareto_qa_reward"),
        _make_boundary_gdpo_reward_func("cite_rec", "pareto_cite_rec_reward"),
        _make_boundary_gdpo_reward_func("cite_prec", "pareto_cite_prec_reward"),
        _make_boundary_gdpo_reward_func("refusal", "pareto_refusal_reward"),
        _make_boundary_gdpo_reward_func("logic", "pareto_logic_reward"),
        _make_boundary_gdpo_reward_func("length", "pareto_length_reward"),
    ]


def _boundary_citation_weights(args) -> tuple[float, float]:
    cite_rec_weight = getattr(args, "boundary_cite_rec_weight", None)
    cite_prec_weight = getattr(args, "boundary_cite_prec_weight", None)
    if cite_rec_weight is None and cite_prec_weight is None:
        return args.boundary_citation_weight / 2.0, args.boundary_citation_weight / 2.0
    return float(cite_rec_weight or 0.0), float(cite_prec_weight or 0.0)


def build_boundary_gdpo_reward_weights(args) -> list[float]:
    boundary_logic_weight = getattr(args, "boundary_logic_weight", None)
    if boundary_logic_weight is None:
        boundary_logic_weight = getattr(args, "format_weight", 0.05)
    cite_rec_weight, cite_prec_weight = _boundary_citation_weights(args)
    if args.reward_mode == "constrained_pareto_gdpo":
        return [
            args.pareto_gate_weight,
            args.qa_weight,
            cite_rec_weight,
            cite_prec_weight,
            args.pareto_refusal_weight,
            boundary_logic_weight,
            args.length_weight,
        ]
    if args.reward_mode == "boundary_gdpo_masked":
        return [
            args.answerability_weight,
            args.refusal_weight,
            args.qa_weight,
            cite_rec_weight,
            cite_prec_weight,
            boundary_logic_weight,
            args.answer_length_weight,
            args.refusal_length_weight,
        ]
    return [
        args.answerability_weight,
        args.qa_weight,
        cite_rec_weight,
        cite_prec_weight,
        boundary_logic_weight,
        args.length_weight,
    ]


GDPO_REWARD_MODES = {"boundary_gdpo", "boundary_gdpo_masked", "constrained_pareto_gdpo"}
GDPO_ONLY_REWARD_MODES = {"boundary_gdpo_masked", "constrained_pareto_gdpo"}


def _validate_reward_mode_args(args) -> None:
    if args.apply_gdpo and args.reward_mode not in GDPO_REWARD_MODES:
        raise ValueError(
            "--apply-gdpo is currently supported only with --reward-mode boundary_gdpo "
            "or boundary_gdpo_masked or constrained_pareto_gdpo"
        )
    if not args.apply_gdpo and args.reward_mode in GDPO_ONLY_REWARD_MODES:
        raise ValueError(f"--reward-mode {args.reward_mode} requires --apply-gdpo")


def _apply_reward_mode_scale_overrides(args) -> None:
    if args.reward_mode == "boundary_gdpo" and not args.apply_gdpo and args.scale_rewards != "none":
        print(
            f"[config] {args.reward_mode} uses internal group-normalized components; "
            f"overriding scale_rewards={args.scale_rewards!r} to 'none'."
        )
        args.scale_rewards = "none"


def logic_guided_reward(
    prompts,
    completions,
    docs,
    short_answers,
    long_answer_with_citation=None,
    sample_type=None,
    answerability_label=None,
    docs_supported=None,
    **kwargs,
):
    """
    TRL-compatible reward function. Computes rewards in parallel via
    ThreadPoolExecutor since Nemo parsing is CPU-bound.
    Also logs per-sub-reward means to wandb.
    """
    reward_mode = REWARD_CONFIG.get("reward_mode", "current")
    boundary_labels = None
    if reward_mode == "boundary_gdpo":
        boundary_labels = _validate_boundary_answerability_labels(answerability_label, len(completions))
        _validate_boundary_metadata(sample_type, docs_supported, boundary_labels, len(completions))

    def _completion_text(completion) -> str:
        if isinstance(completion, list) and completion and isinstance(completion[0], dict):
            return completion[0].get("content", "")
        if isinstance(completion, dict):
            return completion.get("content", "")
        return str(completion)

    def _worker(i):
        completion = completions[i]
        text = _completion_text(completion)
        # Pass raw docs dicts so reward.py can format them for
        # generate_long_answer_for_item (original Nemo+Qwen pipeline)
        sample = {
            "docs": docs[i],
            "short_answers": short_answers[i],
            "question": "",
            "long_answer_with_citation": long_answer_with_citation[i] if long_answer_with_citation is not None else "",
        }
        sample["sample_type"] = sample_type[i] if sample_type is not None else ""
        sample["answerability_label"] = answerability_label[i] if answerability_label is not None else ""
        sample["docs_supported"] = bool(docs_supported[i]) if docs_supported is not None else False
        user_msg = prompts[i][-1]["content"] if prompts[i] else ""
        q_match = re.match(r"Question:\s*(.+?)\s*\n", user_msg)
        sample["question"] = q_match.group(1) if q_match else user_msg
        try:
            return _compute_single_reward(text, sample, **kwargs)
        except Exception as e:
            print(f"[reward] Error sample_index={i}: {type(e).__name__}: {e}", flush=True)
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(_worker, range(len(completions))))

    # Raw component values (before group normalization and annealing)
    raw_qa = [r[1] for r in results]
    raw_logic = [r[2] for r in results]
    raw_feature = [r[3] for r in results]
    raw_cite_rec = [r[4] for r in results]
    raw_cite_prec = [r[5] for r in results]
    raw_illegal = [r[6] for r in results]
    skip_cite_mask = [r[7] for r in results]  # True -> correct refusal, citation neutral
    anneal_values = [r[8] for r in results]
    # Annealed values (applied before group normalization)
    annealed_logic = [r[9] for r in results]
    annealed_feature = [r[10] for r in results]
    annealed_cite_rec = [r[11] for r in results]
    annealed_cite_prec = [r[12] for r in results]
    raw_doc_refusal = [r[13] for r in results]
    raw_citation_missing_penalty = [r[14] for r in results]
    raw_gt_refusal_correct = [r[15] for r in results]
    raw_gt_refusal_wrong = [r[16] for r in results]

    # --- Per-component group min-max normalization ---------------------------
    # GRPOTrainer later does its own (total - group_mean) / group_std.
    # By normalizing each component to [-1, 1] inside each group first, we ensure
    # every reward component has the same intra-group scale, preventing one
    # component from dominating due to larger variance.
    G = REWARD_CONFIG.get("group_size", 4)
    do_norm = REWARD_CONFIG.get("normalize_per_component", True)

    def _group_minmax(values, group_size):
        arr = np.array(values, dtype=np.float32)
        if len(arr) % group_size != 0 or len(arr) == 0:
            return arr.tolist()
        arr = arr.reshape(-1, group_size)
        mins = arr.min(axis=1, keepdims=True)
        maxs = arr.max(axis=1, keepdims=True)
        ranges = maxs - mins
        # Map to [-1, 1]; constant groups become 0 so they contribute no gradient
        normed = np.where(ranges > 1e-6, 2.0 * (arr - mins) / ranges - 1.0, 0.0)
        return normed.flatten().tolist()

    def _group_minmax_masked(values, group_size, mask):
        """Min-max over only valid (mask=True) entries; masked entries get 0 (neutral)."""
        arr = np.array(values, dtype=np.float32)
        m = np.array(mask, dtype=bool)
        if len(arr) % group_size != 0 or len(arr) == 0:
            return arr.tolist()
        arr = arr.reshape(-1, group_size)
        m = m.reshape(-1, group_size)
        out = np.zeros_like(arr)
        for g in range(arr.shape[0]):
            valid = arr[g][m[g]]
            if len(valid) == 0 or (valid.max() - valid.min()) < 1e-6:
                continue
            min_v, max_v = valid.min(), valid.max()
            rng = max_v - min_v
            out[g][m[g]] = 2.0 * (valid - min_v) / rng - 1.0
        return out.flatten().tolist()

    # Replace None placeholders (correct refusals) with 0.0 before numpy conversion;
    # the mask ensures they are excluded from min/max and receive neutral 0 afterwards.
    _raw_cite_rec = [v if v is not None else 0.0 for v in raw_cite_rec]
    _raw_cite_prec = [v if v is not None else 0.0 for v in raw_cite_prec]
    _annealed_cite_rec = [v if v is not None else 0.0 for v in annealed_cite_rec]
    _annealed_cite_prec = [v if v is not None else 0.0 for v in annealed_cite_prec]

    full_completion_texts = [_completion_text(completion) for completion in completions]
    completion_answers = []
    for text in full_completion_texts:
        m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        completion_answers.append(m.group(1).strip() if m else text)

    if reward_mode == "qa_citation":
        valid_cite_mask = [not x for x in skip_cite_mask]
        qa_values = group_minmax(raw_qa, G) if do_norm else raw_qa
        cite_rec_values = (
            group_minmax_masked(_annealed_cite_rec, G, valid_cite_mask)
            if do_norm
            else _annealed_cite_rec
        )
        cite_prec_values = (
            group_minmax_masked(_annealed_cite_prec, G, valid_cite_mask)
            if do_norm
            else _annealed_cite_prec
        )
        logic_values = group_minmax(annealed_logic, G) if do_norm else annealed_logic
        feature_values = group_minmax(annealed_feature, G) if do_norm else annealed_feature
        qa_weight = REWARD_WEIGHTS["qa"] if REWARD_ENABLED["qa"] else 0.0
        cite_rec_weight = REWARD_WEIGHTS["cite_rec"] if REWARD_ENABLED["cite_rec"] else 0.0
        cite_prec_weight = REWARD_WEIGHTS["cite_prec"] if REWARD_ENABLED["cite_prec"] else 0.0
        logic_weight = REWARD_WEIGHTS["logic"] if REWARD_ENABLED["logic"] else 0.0
        feature_weight = REWARD_WEIGHTS["feature"] if REWARD_ENABLED["feature"] else 0.0
        scores = []
        for i in range(len(completions)):
            score = qa_weight * qa_values[i]
            if not skip_cite_mask[i]:
                score += cite_rec_weight * cite_rec_values[i] + cite_prec_weight * cite_prec_values[i]
            score += logic_weight * logic_values[i] + feature_weight * feature_values[i]
            scores.append(score)
        _emit_mode_reward_metrics(
            reward_mode,
            {
                "total": _mean(scores),
                "qa": _mean(qa_values),
                "cite_rec": _mean_masked(cite_rec_values, valid_cite_mask),
                "cite_prec": _mean_masked(cite_prec_values, valid_cite_mask),
                "logic": _mean(logic_values),
                "feature": _mean(feature_values),
                "skip_cite_rate": _mean([1.0 if x else 0.0 for x in skip_cite_mask]),
            },
        )
        return scores

    if reward_mode == "boundary_gdpo":
        labels = boundary_labels
        answerable_mask = [str(label) == "answerable" for label in labels]
        refused = [is_refusal_answer(answer) for answer in completion_answers]
        answerability = [
            answerability_reward(is_answerable=answerable_mask[i], refused=refused[i])
            for i in range(len(completions))
        ]
        qa_mask = [answerable_mask[i] and not refused[i] for i in range(len(completions))]
        citation_mask = [qa_mask[i] and not skip_cite_mask[i] for i in range(len(completions))]
        citation_values = [
            (float(_annealed_cite_rec[i]) + float(_annealed_cite_prec[i])) / 2.0
            for i in range(len(completions))
        ]
        components = {
            "answerability": group_minmax(answerability, G),
            "qa": group_minmax_masked(raw_qa, G, qa_mask),
            "citation": group_minmax_masked(citation_values, G, citation_mask),
            "logic": group_minmax_masked(raw_logic, G, qa_mask),
            "length": group_minmax(
                [length_reward(completion_answers[i], refused=refused[i]) for i in range(len(completions))],
                G,
            ),
        }
        scores = weighted_sum(
            components,
            {
                "answerability": 0.35,
                "qa": 0.35,
                "citation": 0.20,
                "logic": 0.05,
                "length": 0.05,
            },
        )
        _emit_mode_reward_metrics(
            reward_mode,
            {
                "total": _mean(scores),
                "answerability": _mean(components["answerability"]),
                "qa": _mean(components["qa"]),
                "citation": _mean(components["citation"]),
                "logic": _mean_masked(components["logic"], qa_mask),
                "length": _mean(components["length"]),
                "refused_rate": _mean([1.0 if value else 0.0 for value in refused]),
                "answerable_label_rate": _mean([1.0 if value else 0.0 for value in answerable_mask]),
                "unanswerable_label_rate": _mean([0.0 if value else 1.0 for value in answerable_mask]),
            },
        )
        return scores

    sub_qa = _group_minmax(raw_qa, G) if do_norm else raw_qa
    # Group normalization uses ANNEALED values for process rewards
    sub_logic = _group_minmax(annealed_logic, G) if do_norm else annealed_logic
    sub_feature = _group_minmax(annealed_feature, G) if do_norm else annealed_feature
    # Citation: mask out correct refusals so they don't become -1 in normalization
    sub_cite_rec = _group_minmax_masked(_annealed_cite_rec, G, [not x for x in skip_cite_mask]) if do_norm else _annealed_cite_rec
    sub_cite_prec = _group_minmax_masked(_annealed_cite_prec, G, [not x for x in skip_cite_mask]) if do_norm else _annealed_cite_prec
    sub_illegal = _group_minmax(raw_illegal, G) if do_norm else raw_illegal
    sub_doc_refusal = _group_minmax(raw_doc_refusal, G) if do_norm else raw_doc_refusal

    # Re-assemble total reward from (possibly normalized) components
    doc_refusal_pos = sum(1 for v in raw_doc_refusal if v > 0)
    doc_refusal_neg = sum(1 for v in raw_doc_refusal if v < 0)
    doc_refusal_nonzero = doc_refusal_pos + doc_refusal_neg
    gt_refusal_correct = sum(1 for v in raw_gt_refusal_correct if v > 0)
    gt_refusal_wrong = sum(1 for v in raw_gt_refusal_wrong if v > 0)
    gt_refusal_total = gt_refusal_correct + gt_refusal_wrong
    skip_cite_count = sum(skip_cite_mask)
    scores = []
    for i in range(len(completions)):
        s = 0.0
        if REWARD_ENABLED["qa"]:
            s += REWARD_WEIGHTS["qa"] * sub_qa[i]
        if REWARD_ENABLED["logic"]:
            s += REWARD_WEIGHTS["logic"] * sub_logic[i]
        if REWARD_ENABLED["feature"]:
            s += REWARD_WEIGHTS["feature"] * sub_feature[i]
        # Skip citation for correct refusals (neutral contribution)
        if REWARD_ENABLED["cite_rec"] and not skip_cite_mask[i]:
            s += REWARD_WEIGHTS["cite_rec"] * sub_cite_rec[i]
        if REWARD_ENABLED["cite_prec"] and not skip_cite_mask[i]:
            s += REWARD_WEIGHTS["cite_prec"] * sub_cite_prec[i]
        if REWARD_ENABLED["illegal"]:
            s += REWARD_WEIGHTS["illegal"] * sub_illegal[i]
        # Document-driven refusal signal is annealed with a minimum floor.
        s += _doc_refusal_weight(anneal_values[i]) * sub_doc_refusal[i]
        s += raw_citation_missing_penalty[i]
        scores.append(s)
    # ------------------------------------------------------------------------

    debug_cfg = REWARD_DEBUG_CONFIG
    if debug_cfg.get("enabled") and debug_cfg.get("calls", 0) < debug_cfg.get("max_batches", 0):
        debug_cfg["calls"] = debug_cfg.get("calls", 0) + 1

        def _stats(values):
            valid = [v for v in values if v is not None]
            if not valid:
                return "mean=0.0000 std=0.0000 min=0.0000 max=0.0000"
            arr = np.array(valid, dtype=np.float32)
            return (
                f"mean={float(arr.mean()):.4f} std={float(arr.std()):.4f} "
                f"min={float(arr.min()):.4f} max={float(arr.max()):.4f}"
            )

        print(
            f"[reward-debug] batch={debug_cfg['calls']} n={len(scores)} "
            f"group_size={G} normalize={do_norm} anneal_mean={float(np.mean(anneal_values)):.4f}",
            flush=True,
        )
        print(f"[reward-debug] scores {_stats(scores)}", flush=True)
        print(
            "[reward-debug] "
            f"qa {_stats(raw_qa)} | logic_ann {_stats(annealed_logic)} | "
            f"feature_ann {_stats(annealed_feature)} | cite_rec_ann {_stats(annealed_cite_rec)} | "
            f"cite_prec_ann {_stats(annealed_cite_prec)} | doc_refusal {_stats(raw_doc_refusal)} | "
            f"citation_missing_penalty {_stats(raw_citation_missing_penalty)} | "
            f"doc_refusal_pos={doc_refusal_pos} doc_refusal_neg={doc_refusal_neg} "
            f"doc_refusal_nonzero={doc_refusal_nonzero}/{len(raw_doc_refusal)} "
            f"gt_refusal_correct={gt_refusal_correct} gt_refusal_wrong={gt_refusal_wrong} "
            f"gt_refusal_total={gt_refusal_total}/{len(raw_doc_refusal)} "
            f"skip_cite={skip_cite_count}/{len(skip_cite_mask)}",
            flush=True,
        )
        for i in range(min(debug_cfg.get("max_samples", 0), len(scores))):
            completion = completions[i]
            text = completion[0]["content"] if isinstance(completion[0], dict) else str(completion)
            snippet = re.sub(r"\s+", " ", text[:240]).strip()
            print(
                f"[reward-debug] sample={i} score={scores[i]:.4f} qa={raw_qa[i]:.4f} "
                f"logic_ann={annealed_logic[i]:.4f} feature_ann={annealed_feature[i]:.4f} "
                f"cite_rec_ann={annealed_cite_rec[i]} cite_prec_ann={annealed_cite_prec[i]} "
                f"doc_refusal={raw_doc_refusal[i]:.4f} "
                f"gt_refusal_correct={raw_gt_refusal_correct[i]:.0f} "
                f"gt_refusal_wrong={raw_gt_refusal_wrong[i]:.0f} "
                f"citation_missing_penalty={raw_citation_missing_penalty[i]:.4f} "
                f"skip_cite={skip_cite_mask[i]} text={snippet!r}",
                flush=True,
            )

    # Log raw + annealed sub-reward means to wandb (commit=False so they share the same step)
    # GRPOTrainer handles its own group normalization; we only log raw values here.
    import wandb
    if getattr(wandb, "run", None) is not None:
        log_dict = {"sub_rewards/total": float(np.mean(scores))}
        log_dict["sub_rewards/anneal"] = float(np.mean(anneal_values))
        if REWARD_ENABLED["qa"]:
            log_dict["sub_rewards/qa_raw"] = float(np.mean(raw_qa))
        if REWARD_ENABLED["logic"]:
            log_dict["sub_rewards/logic_raw"] = float(np.mean(raw_logic))
            log_dict["sub_rewards/logic_annealed"] = float(np.mean(annealed_logic))
        if REWARD_ENABLED["feature"]:
            log_dict["sub_rewards/feature_raw"] = float(np.mean(raw_feature))
            log_dict["sub_rewards/feature_annealed"] = float(np.mean(annealed_feature))
        if REWARD_ENABLED["cite_rec"]:
            # Filter out None values (correct refusals) before computing mean
            valid_cite_rec = [v for v in raw_cite_rec if v is not None]
            log_dict["sub_rewards/cite_rec_raw"] = float(np.mean(valid_cite_rec)) if valid_cite_rec else 0.0
            valid_annealed_cite_rec = [v for v in annealed_cite_rec if v is not None]
            log_dict["sub_rewards/cite_rec_annealed"] = float(np.mean(valid_annealed_cite_rec)) if valid_annealed_cite_rec else 0.0
        if REWARD_ENABLED["cite_prec"]:
            valid_cite_prec = [v for v in raw_cite_prec if v is not None]
            log_dict["sub_rewards/cite_prec_raw"] = float(np.mean(valid_cite_prec)) if valid_cite_prec else 0.0
            valid_annealed_cite_prec = [v for v in annealed_cite_prec if v is not None]
            log_dict["sub_rewards/cite_prec_annealed"] = float(np.mean(valid_annealed_cite_prec)) if valid_annealed_cite_prec else 0.0
        if REWARD_ENABLED["illegal"]:
            log_dict["sub_rewards/illegal_raw"] = float(np.mean(raw_illegal))
        # Log document-driven refusal signal
        valid_doc_refusal = [v for v in raw_doc_refusal if v != 0.0]
        log_dict["sub_rewards/doc_refusal_raw"] = float(np.mean(valid_doc_refusal)) if valid_doc_refusal else 0.0
        log_dict["sub_rewards/doc_refusal_weight"] = float(np.mean([_doc_refusal_weight(v) for v in anneal_values]))
        log_dict["sub_rewards/citation_missing_penalty"] = float(np.mean(raw_citation_missing_penalty))
        n_rewards = max(1, len(raw_doc_refusal))
        log_dict["sub_rewards/doc_refusal_pos_rate"] = doc_refusal_pos / n_rewards
        log_dict["sub_rewards/doc_refusal_neg_rate"] = doc_refusal_neg / n_rewards
        log_dict["sub_rewards/doc_refusal_nonzero_rate"] = doc_refusal_nonzero / n_rewards
        log_dict["sub_rewards/gt_refusal_correct_rate"] = gt_refusal_correct / n_rewards
        log_dict["sub_rewards/gt_refusal_wrong_rate"] = gt_refusal_wrong / n_rewards
        log_dict["sub_rewards/gt_refusal_total_rate"] = gt_refusal_total / n_rewards
        log_dict["sub_rewards/skip_cite_rate"] = skip_cite_count / n_rewards
        wandb.log(log_dict, commit=False)

    return scores


# ── Dataset preparation ──────────────────────────────────────────────────────

def load_grpo_dataset(data_path: str):
    """
    Load the training dataset and convert to TRL-compatible format.

    TRL GRPOTrainer expects:
      - 'prompt' column: list of chat messages
      - other columns are forwarded to reward functions via **kwargs
    """
    from datasets import Dataset

    with open(data_path, "r") as f:
        raw_data = json.load(f)

    records = []
    for d in raw_data:
        # Convert ranked_passages (str list) → docs (dict list)
        passages = d.get("ranked_passages", [])[:5]
        answers_found = d.get("answers_found", [])
        # Pad/truncate answers_found to match passages length
        if isinstance(answers_found, list):
            answers_found = (answers_found + [0] * len(passages))[:len(passages)]
        else:
            answers_found = [0] * len(passages)

        if passages and isinstance(passages[0], str):
            docs = [{"title": "", "text": p, "answers_found": af} for p, af in zip(passages, answers_found)]
        elif passages and isinstance(passages[0], dict):
            docs = [{**p, "answers_found": af} for p, af in zip(passages, answers_found)]
        else:
            docs = []

        # Build user content (same format as grpo_rollout.format_user_content)
        docs_text = "\n".join([
            f"Title: {doc.get('title', '')} Content: {doc.get('text', '')}"
            for doc in docs
        ])
        user_content = f"Question: {d['question']} \n Documents: {docs_text}"

        # Build chat-format prompt
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        records.append({
            "prompt": prompt,
            "docs": docs,
            "short_answers": d.get("short_answers", []),
            "long_answer_with_citation": d.get("long_answer_with_citation", ""),
            "sample_type": d.get("sample_type", ""),
            "answerability_label": d.get("answerability_label", ""),
            "docs_supported": bool(d.get("docs_supported", False)),
        })

    return Dataset.from_list(records)


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Unsloth GRPO training with freely configurable rewards")
    # Reward on/off switches (BooleanOptionalAction adds --use-X and --no-use-X automatically)
    parser.add_argument("--use-qa", action=argparse.BooleanOptionalAction, default=True, help="Enable QA reward")
    parser.add_argument("--use-cite-rec", action=argparse.BooleanOptionalAction, default=True, help="Enable citation recall")
    parser.add_argument("--use-cite-prec", action=argparse.BooleanOptionalAction, default=True, help="Enable citation precision")
    parser.add_argument("--use-logic", action=argparse.BooleanOptionalAction, default=True, help="Enable logic reward")
    parser.add_argument("--use-feature", action=argparse.BooleanOptionalAction, default=True, help="Enable feature extraction reward")
    parser.add_argument("--use-illegal", action=argparse.BooleanOptionalAction, default=False, help="Enable illegal predicate reward")
    # Reward weights (only matter when the corresponding switch is ON)
    parser.add_argument("--qa-weight", type=float, default=0.55, help="QA reward weight")
    parser.add_argument("--cite-rec-weight", type=float, default=0.125, help="Citation recall weight")
    parser.add_argument("--cite-prec-weight", type=float, default=0.125, help="Citation precision weight")
    parser.add_argument("--logic-weight", type=float, default=0.12, help="Logic reward weight")
    parser.add_argument("--feature-weight", type=float, default=0.08, help="Feature extraction reward weight")
    parser.add_argument("--illegal-weight", type=float, default=0.00, help="Illegal predicate reward weight")
    parser.add_argument("--answerability-weight", type=float, default=0.45, help="Boundary GDPO answerability weight")
    parser.add_argument("--refusal-weight", type=float, default=0.80, help="Masked Boundary GDPO refusal-boundary weight")
    parser.add_argument("--pareto-gate-weight", type=float, default=0.15, help="Constrained Pareto-GDPO gate component weight")
    parser.add_argument("--pareto-refusal-weight", type=float, default=0.20, help="Constrained Pareto-GDPO refusal component weight")
    parser.add_argument("--boundary-citation-weight", type=float, default=0.15, help="Boundary GDPO citation total weight")
    parser.add_argument("--boundary-cite-rec-weight", type=float, default=None, help="Boundary GDPO citation recall weight override")
    parser.add_argument("--boundary-cite-prec-weight", type=float, default=None, help="Boundary GDPO citation precision weight override")
    parser.add_argument("--boundary-logic-weight", type=float, default=None, help="Boundary GDPO logic reward weight")
    parser.add_argument("--format-weight", type=float, default=0.05, help="Deprecated: fallback for --boundary-logic-weight")
    parser.add_argument("--length-weight", type=float, default=0.05, help="Boundary GDPO answer length weight")
    parser.add_argument("--answer-length-weight", type=float, default=0.05, help="Masked Boundary GDPO answer length weight")
    parser.add_argument("--refusal-length-weight", type=float, default=0.10, help="Masked Boundary GDPO refusal length weight")
    # Training hyper-params
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--num-generations", type=int, default=16, help="GRPO group size (G)")
    parser.add_argument("--max-steps", type=int, default=1400, help="Max training steps")
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--temp", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--beta", type=float, default=0.001, help="KL penalty coefficient for GRPO")
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=0.6,
        help="GPU memory utilization reserved by colocated vLLM",
    )
    parser.add_argument(
        "--vllm-tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size for colocated vLLM generation",
    )
    parser.add_argument(
        "--vllm-enable-sleep-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Let colocated vLLM sleep between generation calls to reduce training memory pressure",
    )
    parser.add_argument(
        "--fast-inference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Unsloth fast inference during model loading",
    )
    parser.add_argument(
        "--unsloth-vllm-standby",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Unsloth vLLM standby mode during fast-inference model loading",
    )
    parser.add_argument("--normalize-rewards", action=argparse.BooleanOptionalAction, default=False, help="Per-component group min-max normalization")
    parser.add_argument("--disable-reward-anneal", action="store_true", help="Keep citation, logic, and feature rewards unannealed")
    parser.add_argument("--anneal-start-step", type=int, default=ANNEAL_START_STEP, help="Step where process reward annealing starts")
    parser.add_argument("--anneal-decay-steps", type=int, default=ANNEAL_DECAY_STEPS, help="Exponential decay length for process reward annealing")
    parser.add_argument("--apply-gdpo", action="store_true", help="Use official GDPO per-reward advantage normalization")
    parser.add_argument(
        "--reward-mode",
        choices=["current", "qa_citation", "boundary_gdpo", "boundary_gdpo_masked", "constrained_pareto_gdpo"],
        default="current",
        help=(
            "Reward assembly mode: current mixed reward, QA/citation-only Stage 1, "
            "Boundary GDPO Stage 2, masked Boundary GDPO with answerable/refusal reward split, "
            "or constrained Pareto-GDPO"
        ),
    )
    parser.add_argument(
        "--scale-rewards",
        choices=["group", "batch", "none"],
        default="group",
        help="TRL GRPO reward scaling. Ignored by --apply-gdpo because GDPO computes advantages directly.",
    )
    parser.add_argument("--save-steps", type=int, default=SAVE_STEPS, help="Checkpoint save interval")
    parser.add_argument("--wandb-name", type=str, default="grpo-sft-v3.1", help="WandB run name")
    parser.add_argument("--cuda", type=str, default=None, help="CUDA device index (overrides env var)")
    parser.add_argument("--output-dir", type=str, default=SAVE_PATH, help="Output directory for checkpoints and final model")
    parser.add_argument("--model-path", type=str, default=BASE_MODEL_PATH, help="Path to SFT checkpoint (base model for GRPO)")
    parser.add_argument("--data-path", type=str, default=DATA_PATH, help="Path to GRPO training dataset")
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to a Trainer checkpoint to resume optimizer, scheduler, RNG, and trainer state",
    )
    parser.add_argument("--debug-reward", action="store_true", help="Print reward batch diagnostics to stdout")
    parser.add_argument("--debug-reward-batches", type=int, default=3, help="Number of reward batches to print when --debug-reward is set")
    parser.add_argument("--debug-reward-samples", type=int, default=2, help="Number of completion snippets per debug reward batch")
    return parser.parse_args()


def main():
    args = parse_args()

    # Respect CUDA_VISIBLE_DEVICES env var unless --cuda is explicitly passed
    if args.cuda is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    elif "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    if args.unsloth_vllm_standby:
        os.environ["UNSLOTH_VLLM_STANDBY"] = "1"
    print(f"[config] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        import torch

        if torch.cuda.is_available():
            torch.cuda.set_device(int(local_rank))
            print(f"[config] LOCAL_RANK={local_rank} cuda_device={torch.cuda.current_device()}")

    _validate_reward_mode_args(args)
    _apply_reward_mode_scale_overrides(args)

    # Update global reward configuration from CLI
    global REWARD_WEIGHTS, REWARD_ENABLED
    REWARD_ENABLED = {
        "cite_rec":  args.use_cite_rec,
        "cite_prec": args.use_cite_prec,
        "qa":        args.use_qa,
        "logic":     args.use_logic,
        "feature":   args.use_feature,
        "illegal":   args.use_illegal,
    }
    REWARD_WEIGHTS = {
        "cite_rec":  args.cite_rec_weight,
        "cite_prec": args.cite_prec_weight,
        "qa":        args.qa_weight,
        "logic":     args.logic_weight,
        "feature":   args.feature_weight,
        "illegal":   args.illegal_weight,
    }
    print(f"[config] Reward enabled: {REWARD_ENABLED}")
    print(f"[config] Reward weights: {REWARD_WEIGHTS}")
    boundary_gdpo_reward_weights = build_boundary_gdpo_reward_weights(args)
    if args.apply_gdpo:
        print("[config] Applying official GDPO advantage normalization")
        print(f"[config] Boundary GDPO reward weights: {boundary_gdpo_reward_weights}")

    # Update reward computation config
    global REWARD_CONFIG
    REWARD_CONFIG = _build_reward_config(args)
    print(f"[config] Reward config: {REWARD_CONFIG}")

    global REWARD_DEBUG_CONFIG
    REWARD_DEBUG_CONFIG = {
        "enabled": args.debug_reward,
        "max_batches": max(0, args.debug_reward_batches),
        "max_samples": max(0, args.debug_reward_samples),
        "calls": 0,
    }
    print(f"[config] Reward debug: {REWARD_DEBUG_CONFIG}")

    from unsloth import FastLanguageModel, is_bfloat16_supported
    from trl import GRPOConfig, GRPOTrainer
    from vllm import SamplingParams
    import wandb

    reward_funcs = [logic_guided_reward]
    trainer_reward_weights = None
    TrainerClass = GRPOTrainer
    trainer_extra_kwargs = {}
    if args.apply_gdpo:
        from gdpo_trainer import get_official_gdpo_trainer_class

        if args.reward_mode == "boundary_gdpo_masked":
            reward_funcs = build_masked_boundary_gdpo_reward_funcs()
        elif args.reward_mode == "constrained_pareto_gdpo":
            reward_funcs = build_constrained_pareto_gdpo_reward_funcs()
        else:
            reward_funcs = build_boundary_gdpo_reward_funcs()
        trainer_reward_weights = boundary_gdpo_reward_weights
        TrainerClass = get_official_gdpo_trainer_class(GRPOTrainer)
        trainer_extra_kwargs["apply_gdpo"] = True

    # ── Device setup ──────────────────────────────────────────────────────
    import torch.distributed as dist

    is_distributed = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
    is_main = int(os.environ.get("RANK", 0)) == 0 if is_distributed else True

    # ── Load model ────────────────────────────────────────────────────────
    model_path = args.model_path
    if is_main:
        print(f"Loading model from: {model_path}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,
        fast_inference=args.fast_inference,
        max_lora_rank=LORA_R,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        unsloth_vllm_standby=args.unsloth_vllm_standby,
        local_files_only=True,
    )

    # ── Apply LoRA ────────────────────────────────────────────────────────
    model, initialized_new_lora = _apply_lora_if_needed(model, FastLanguageModel)
    if is_main:
        if initialized_new_lora:
            print("[config] Initialized a fresh LoRA adapter")
        else:
            print("[config] Loaded existing PEFT adapter; skipping fresh LoRA initialization")

    # ── Dataset ───────────────────────────────────────────────────────────
    dataset = load_grpo_dataset(args.data_path)
    if is_main:
        print(f"Loading dataset from: {args.data_path}")
        print(f"Dataset loaded: {len(dataset)} samples")

    # ── vLLM sampling params ──────────────────────────────────────────────
    vllm_sampling_params = SamplingParams(
        min_p=0.1,
        top_p=1.0,
        top_k=-1,
        seed=3407,
        stop=[tokenizer.eos_token],
        include_stop_str_in_output=True,
    )

    # ── GRPOConfig ────────────────────────────────────────────────────────
    training_args = GRPOConfig(
        use_vllm=True,
        vllm_mode="colocate",
        vllm_sampling_params=vllm_sampling_params,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        vllm_enable_sleep_mode=args.vllm_enable_sleep_mode,
        learning_rate=args.lr,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_torch_fused",
        logging_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        temperature=args.temp,
        beta=args.beta,
        loss_type="bnpo",
        scale_rewards=args.scale_rewards,
        reward_weights=trainer_reward_weights,
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=MAX_COMPLETION_LENGTH,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        max_grad_norm=0.1,
        report_to="wandb",
        output_dir=args.output_dir,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        seed=3407,
        ddp_find_unused_parameters=False if is_distributed else None,
    )
    # Local TRL lacks the official GDPO config field; keep the value on args so
    # OfficialGDPOTrainer also works with an official-style config object.
    training_args.apply_gdpo = args.apply_gdpo

    # ── wandb ─────────────────────────────────────────────────────────────
    if is_main:
        wandb.init(
            project="logic-guided-aqa",
            name=args.wandb_name,
            config={
                "model": model_path,
                "lora_r": LORA_R,
                "num_generations": args.num_generations,
                "max_prompt_length": MAX_PROMPT_LENGTH,
                "max_completion_length": MAX_COMPLETION_LENGTH,
                "reward_enabled": REWARD_ENABLED,
                "reward_weights": REWARD_WEIGHTS,
                "reward_normalize": args.normalize_rewards,
                "reward_mode": args.reward_mode,
                "apply_gdpo": args.apply_gdpo,
                "scale_rewards": args.scale_rewards,
                "trainer_reward_weights": trainer_reward_weights,
                "save_steps": args.save_steps,
                "lr": args.lr,
                "grad_accum": args.grad_accum,
                "temperature": args.temp,
                "beta": args.beta,
                "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization,
                "vllm_tensor_parallel_size": args.vllm_tensor_parallel_size,
                "vllm_enable_sleep_mode": args.vllm_enable_sleep_mode,
                "fast_inference": args.fast_inference,
                "unsloth_vllm_standby": args.unsloth_vllm_standby,
            },
            tags=["grpo", "unsloth", "single-gpu"],
        )

    # ── Training config (saved with every checkpoint and final model) ──────
    training_config = {
        "model_path": model_path,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "max_seq_length": MAX_SEQ_LENGTH,
        "max_prompt_length": MAX_PROMPT_LENGTH,
        "max_completion_length": MAX_COMPLETION_LENGTH,
        "num_generations": args.num_generations,
        "max_steps": args.max_steps,
        "grad_accum": args.grad_accum,
        "learning_rate": args.lr,
        "temperature": args.temp,
        "beta": args.beta,
        "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization,
        "vllm_tensor_parallel_size": args.vllm_tensor_parallel_size,
        "vllm_enable_sleep_mode": args.vllm_enable_sleep_mode,
        "fast_inference": args.fast_inference,
        "unsloth_vllm_standby": args.unsloth_vllm_standby,
        "reward_normalize": args.normalize_rewards,
        "reward_mode": args.reward_mode,
        "apply_gdpo": args.apply_gdpo,
        "scale_rewards": args.scale_rewards,
        "trainer_reward_weights": trainer_reward_weights,
        "reward_func_names": [func.__name__ for func in reward_funcs],
        "save_steps": args.save_steps,
        "reward_enabled": REWARD_ENABLED,
        "reward_weights": REWARD_WEIGHTS,
        "resume_from_checkpoint": args.resume_from_checkpoint,
        "model_path_is_peft_adapter": _is_peft_adapter_path(model_path),
        "initialized_new_lora": initialized_new_lora,
    }

    from transformers import TrainerCallback

    class SaveConfigCallback(TrainerCallback):
        """Callback to save training_config.json at every checkpoint."""
        def on_save(self, args, state, control, **kwargs):
            if state.is_world_process_zero:
                checkpoint_folder = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
                os.makedirs(checkpoint_folder, exist_ok=True)
                config_path = os.path.join(checkpoint_folder, "training_config.json")
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(training_config, f, indent=2, ensure_ascii=False)
                print(f"[save] Config saved to {config_path}")
            return control

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = TrainerClass(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        callbacks=[SaveConfigCallback()],
        **trainer_extra_kwargs,
    )

    if is_main:
        print("Starting GRPO training...")

    resume_checkpoint = args.resume_from_checkpoint or None
    if is_main and resume_checkpoint:
        print(f"[resume] Resuming Trainer state from {resume_checkpoint}")

    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # ── Save ──────────────────────────────────────────────────────────────
    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        config_path = os.path.join(args.output_dir, "training_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(training_config, f, indent=2, ensure_ascii=False)
        print(f"[save] Config saved to {config_path}")
        print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
