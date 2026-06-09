"""
run_evaluation.py
=================
一站式评估入口：将 eval_result_construct + Trust-Eval 合并到一个文件中。

用法：
    python train_and_evaluation/run_evaluation.py \
        --inference_result  <推理结果 JSON，含 long_answer_with_nemo_guidance 字段> \
        --ground_truth      <含 qa_pairs / annotations / docs 的原始测试集 JSON> \
        --output_dir        <结果输出目录，默认 ./results> \
        [--model_name       <checkpoint 完整路径，用于结果文件命名>]
        [--save_per_sample  <是否保存每个样本的 citation 指标，默认 True>]

流程：
    Step 1  构造 eval 格式文件（等价于 eval_result_construct.py）
    Step 2  运行 Trust-Eval，在计算 citation 指标的同时保存每个样本的 citation_rec/citation_prec
"""

import argparse
import copy
import itertools
import json
import os
import re
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nltk
import numpy as np
import requests
import yaml
from nltk import sent_tokenize

# Trust-Eval 模块目录（相对本文件的固定位置）
TRUST_EVAL_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "trust-align", "trust_eval", "trust_eval")
)
# eval_config.yaml 与本文件同目录
EVAL_CONFIG_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_config.yaml")
AUTOAIS_URLS = [
    url.strip()
    for url in os.environ.get("AUTOAIS_URLS", "").split(",")
    if url.strip()
]
_autoais_cycle = itertools.cycle(AUTOAIS_URLS) if AUTOAIS_URLS else None
AUTOAIS_SESSION = requests.Session()
AUTOAIS_SESSION.trust_env = False
DEFAULT_AUTOAIS_MODEL = os.environ.get("AUTOAIS_MODEL_PATH", "")


# ──────────────────────────────────────────────────────────────────────────────
# Step 1: 构造 Trust-Eval 所需格式（eval_result_construct.py 逻辑）
# ──────────────────────────────────────────────────────────────────────────────

def build_eval_file(inference_result_path: str, ground_truth_path: str, output_path: str) -> str:
    """
    合并推理结果与 ground-truth，输出 Trust-Eval 所需格式：
        {"data": [ {question, docs, qa_pairs, annotations, output, ...}, ... ]}
    返回输出文件路径。
    """
    with open(inference_result_path, "r", encoding="utf-8") as f:
        infer_data = json.load(f)

    with open(ground_truth_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    # ground_truth 支持裸列表和 {"data": [...]} 两种格式
    if isinstance(gt_data, dict) and "data" in gt_data:
        gt_data = gt_data["data"]

    gt_by_question = {item["question"]: item for item in gt_data}

    new_data = []
    for item in infer_data:
        question = item.get("question", "")
        predict_answer = item.get("long_answer_with_nemo_guidance", "")
        if not predict_answer or predict_answer == ". ":
            predict_answer = "I apologize, but I couldn't find an answer."

        if question in gt_by_question:
            merged = dict(gt_by_question[question])
            merged["output"] = predict_answer
            merged["original_response"] = item.get("original_response", "")
            new_data.append(merged)

    result = {"data": new_data}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[Step 1] eval 格式文件已写入：{output_path}  ({len(new_data)} 条)")
    return output_path


# ──────────────────────────────────────────────────────────────────────────────
# Step 2: Trust-Eval + Per-sample citation metrics（一体化计算）
# ──────────────────────────────────────────────────────────────────────────────

def format_document(doc: Dict[str, Any]) -> str:
    """Format a document for AutoAIS evaluation."""
    if "title" in doc and "text" in doc:
        return f"Title: {doc['title']}\nContent: {doc['text']}"
    elif "phrase" in doc and "sent" in doc:
        return f"Phrase: {doc['phrase']}\nSentence: {doc['sent']}"
    else:
        return str(doc)


def remove_citations(text: str) -> str:
    """Remove citations from text."""
    return re.sub(r'\[\d+\]', '', text)


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    import string
    text = answer.lower()
    text = " ".join(text.split())
    return text


def _looks_like_refusal_text(output: str) -> bool:
    normalized = normalize_answer(output or "")
    return (
        normalized.startswith("[false]")
        or normalized.startswith("false")
        or "not enough information" in normalized
        or "couldn't find an answer" in normalized
        or "could not find an answer" in normalized
    )


def _autoais_health_url(nli_url: str) -> str:
    return nli_url.rsplit("/", 1)[0] + "/health"


def _autoais_service_available(timeout: int = 10) -> bool:
    for url in AUTOAIS_URLS:
        try:
            response = AUTOAIS_SESSION.get(_autoais_health_url(url), timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            if payload.get("status") == "ok" and payload.get("model_loaded"):
                return True
        except Exception:
            continue
    return False


@lru_cache(maxsize=20000)
def _run_nli_autoais_http(passage: str, claim: str) -> int:
    if _autoais_cycle is None:
        raise ValueError("Set AUTOAIS_URLS to one or more AutoAIS /nli endpoints")
    first_url = next(_autoais_cycle)
    urls_to_try = [first_url] + [url for url in AUTOAIS_URLS if url != first_url]

    for url in urls_to_try:
        try:
            response = AUTOAIS_SESSION.post(
                url,
                json={"premise": passage, "hypothesis": claim},
                timeout=30,
            )
            response.raise_for_status()
            return int(response.json().get("result", 0))
        except Exception:
            continue
    return 0


def _run_nli_autoais(passage: str, claim: str, autoais_model, autoais_tokenizer) -> int:
    """Run NLI inference for AutoAIS."""
    if autoais_model is None or autoais_tokenizer is None:
        return _run_nli_autoais_http(passage, claim)

    import torch

    input_text = f"premise: {passage} hypothesis: {claim}"
    input_ids = autoais_tokenizer(input_text, return_tensors="pt").input_ids.to(
        autoais_model.device
    )

    with torch.inference_mode():
        outputs = autoais_model.generate(input_ids, max_new_tokens=10)
    result = autoais_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return 1 if result == "1" else 0


def _is_refusal(item: Dict[str, Any], refusal_flag: str, refusal_threshold: int) -> bool:
    """Check if the model refused to answer."""
    from fuzzywuzzy import fuzz
    if _looks_like_refusal_text(item.get("output", "")):
        return True
    return (
        fuzz.partial_ratio(
            normalize_answer(refusal_flag), normalize_answer(item["output"])
        )
        > refusal_threshold
    )


def compute_citation_metrics_with_per_sample(
    data: List[Dict[str, Any]],
    args: Any,
    is_qampari: bool = False,
    at_most_citations: Optional[int] = None,
    autoais_model=None,
    autoais_tokenizer=None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Compute AutoAIS score with per-sample citation recall/precision.
    
    Returns:
        - Aggregate metrics dict
        - Per-sample results list with citation_rec and citation_prec
    """
    from logging_config import logger
    from tqdm import tqdm

    logger.info("Running AutoAIS with per-sample metrics...")

    regular_ais_scores = []
    regular_ais_scores_prec = []
    answered_ais_scores = []
    answered_ais_scores_prec = []

    sent_total = 0
    sent_mcite = 0
    sent_mcite_support = 0
    sent_mcite_overcite = 0
    autoais_log = []
    
    # Per-sample results
    per_sample_results = []
    
    for item_idx, item in enumerate(tqdm(data, desc="Computing citation score")):
        # Initialize per-sample result
        sample_result = {
            "question": item.get("question", ""),
            "output": item.get("output", ""),
            "original_response": item.get("original_response", ""),
            "citation_rec": 0.0,
            "citation_prec": 0.0,
        }
        sample_autoais_log = []

        # Get sentences by using NLTK
        if is_qampari:
            sents = [
                item["question"] + " " + x.strip()
                for x in item["output"].rstrip().rstrip(".").rstrip(",").split(",")
            ]
        else:
            sents = sent_tokenize(item["output"])
        if len(sents) == 0:
            per_sample_results.append(sample_result)
            continue
        target_sents = [remove_citations(sent).strip() for sent in sents]

        entail = 0
        entail_prec = 0
        total_citations = 0
        for sent_id, sent in enumerate(sents):
            target_sent = target_sents[sent_id]
            joint_entail = -1

            # Find references
            ref = [
                int(r[1:]) - 1 for r in re.findall(r"\[\d+", sent)
            ]

            # No citations
            if len(ref) == 0:
                joint_entail = 0
            # Citations out of range
            elif any([ref_id >= len(item["docs"]) for ref_id in ref]):
                joint_entail = 0
            else:
                if at_most_citations is not None:
                    ref = ref[:at_most_citations]
                total_citations += len(ref)
                joint_passage = "\n".join(
                    [format_document(item["docs"][psgs_id]) for psgs_id in ref]
                )

            # Calculate the recall score
            if joint_entail == -1:
                joint_entail = _run_nli_autoais(joint_passage, target_sent, 
                                               autoais_model, autoais_tokenizer)
                sample_autoais_log.append(
                    {
                        "question": item["question"],
                        "output": item["output"],
                        "claim": sent,
                        "passage": [joint_passage],
                        "model_type": "NLI",
                        "model_output": joint_entail,
                    }
                )
                autoais_log.append(sample_autoais_log[-1])

            entail += joint_entail
            if len(ref) > 1:
                sent_mcite += 1

            # Calculate the precision score if applicable
            if joint_entail and len(ref) > 1:
                sent_mcite_support += 1
                for psgs_id in ref:
                    passage = format_document(item["docs"][psgs_id])
                    nli_result = _run_nli_autoais(passage, target_sent, 
                                                  autoais_model, autoais_tokenizer)

                    if not nli_result:
                        subset_exclude = copy.deepcopy(ref)
                        subset_exclude.remove(psgs_id)
                        passage = "\n".join(
                            [format_document(item["docs"][pid]) for pid in subset_exclude]
                        )
                        nli_result = _run_nli_autoais(passage, target_sent,
                                                      autoais_model, autoais_tokenizer)
                        subset_coverage = np.bitwise_or.reduce(
                            [item["docs"][pid]["answers_found"] for pid in subset_exclude]
                        )
                        contained = False
                        for i in range(len(subset_coverage)):
                            if (
                                subset_coverage[i] == 1
                                and item["docs"][psgs_id]["answers_found"][i] == 1
                            ):
                                contained = True
                                break

                        if nli_result and (not contained):
                            sent_mcite_overcite += 1
                        else:
                            entail_prec += 1
                    else:
                        entail_prec += 1
            else:
                entail_prec += joint_entail

        sent_total += len(sents)
        
        # Compute per-sample recall and precision
        sample_result["citation_rec"] = entail / len(sents) if len(sents) > 0 else 0.0
        sample_result["citation_prec"] = entail_prec / total_citations if total_citations > 0 else 0.0
        sample_result["autoais_log"] = sample_autoais_log
        per_sample_results.append(sample_result)

        # Add to aggregate scores
        regular_ais_scores.append(entail / len(sents))
        regular_ais_scores_prec.append(entail_prec / total_citations if total_citations > 0 else 0.0)

        # Answered data - use local _is_refusal function with config parameters
        refusal = _is_refusal(item, args.refusal_flag, args.refusal_threshold)
        if not refusal:
            answered_ais_scores.append(entail / len(sents))
            answered_ais_scores_prec.append(entail_prec / total_citations if total_citations > 0 else 0.0)

    if sent_mcite > 0 and sent_mcite_support > 0:
        print(
            "Among all sentences, %.2f%% have multiple citations, among which %.2f%% are supported "
            "by the joint set, among which %.2f%% overcite."
            % (
                100 * sent_mcite / sent_total,
                100 * sent_mcite_support / sent_mcite,
                100 * sent_mcite_overcite / sent_mcite_support,
            )
        )

    regular_recall = 100 * np.mean(regular_ais_scores)
    regular_precision = 100 * np.mean(regular_ais_scores_prec)
    regular_f1_score = (
        2 * (regular_precision * regular_recall) / (regular_precision + regular_recall)
        if (regular_precision + regular_recall) > 0
        else 0
    )

    answered_recall = 100 * np.mean(answered_ais_scores if len(answered_ais_scores) != 0 else 0)
    answered_precision = 100 * np.mean(answered_ais_scores_prec if len(answered_ais_scores_prec) != 0 else 0)
    answered_f1_score = (
        2 * (answered_precision * answered_recall) / (answered_precision + answered_recall)
        if (answered_precision + answered_recall) > 0
        else 0
    )

    result = {
        "regular_citation_rec": regular_recall,
        "regular_citation_prec": regular_precision,
        "regular_citation_f1": regular_f1_score,
        "answered_citation_rec": answered_recall,
        "answered_citation_prec": answered_precision,
        "answered_citation_f1": answered_f1_score,
    }
    
    return result, per_sample_results


def run_trust_eval(eval_file: str, trust_output_path: str,
                   per_sample_output_path: Optional[str] = None,
                   save_per_sample: bool = True,
                   autoais_model_path: str = DEFAULT_AUTOAIS_MODEL,
                   data_type: str = "asqa",
                   eval_type: Optional[str] = None) -> dict:
    """
    运行 Trust-Eval 评估，同时计算每个样本的 citation 指标：
        1. 将 eval_config.yaml 中的 eval_file 字段临时覆盖为当前 eval_file
        2. EvaluationConfig.from_yaml() -> Evaluator -> compute_metrics()
        3. 在计算 citation 指标时，同时保存每个样本的 citation_rec 和 citation_prec
    """
    if TRUST_EVAL_DIR not in sys.path:
        sys.path.insert(0, TRUST_EVAL_DIR)

    from config import EvaluationConfig
    from evaluator import Evaluator
    from logging_config import logger

    with open(EVAL_CONFIG_YAML, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)

    cfg_dict["eval_file"] = eval_file
    cfg_dict["data_type"] = data_type
    if eval_type is not None:
        cfg_dict["eval_type"] = eval_type

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as tmp:
        yaml.dump(cfg_dict, tmp, allow_unicode=True)
        tmp_yaml_path = tmp.name

    try:
        evaluation_config = EvaluationConfig.from_yaml(yaml_path=tmp_yaml_path)
        evaluation_config.autoais_model = autoais_model_path
        # Override eval_type after from_yaml to prevent __post_init__ from overwriting it
        if eval_type is not None:
            evaluation_config.eval_type = eval_type
        logger.info(evaluation_config)
        evaluator = Evaluator(evaluation_config)
        
        # Compute basic metrics (correctness scores)
        evaluator.compute_metrics(correctness=True, citations=False)

        # Compute citation metrics with per-sample results
        # Note: Always compute citations since this is the main purpose of this script
        logger.info("Computing citation scores with per-sample metrics...")
        logger.info(f"compute_citations config value: {evaluation_config.compute_citations}")

        autoais_model = None
        autoais_tokenizer = None
        if _autoais_service_available():
            logger.info(f"Using AutoAIS HTTP service: {AUTOAIS_URLS}")
        else:
            logger.info("AutoAIS HTTP service unavailable; loading local AutoAIS model.")
            from auto_ais_loader import get_autoais_model_and_tokenizer
            autoais_model, autoais_tokenizer = get_autoais_model_and_tokenizer(evaluation_config)

        # Compute citation metrics with per-sample results
        citation_results, per_sample_results = compute_citation_metrics_with_per_sample(
            evaluator.eval_data,
            evaluation_config,
            is_qampari=evaluation_config.is_qampari,
            at_most_citations=evaluation_config.at_most_citations,
            autoais_model=autoais_model,
            autoais_tokenizer=autoais_tokenizer,
        )

        # Add citation results to evaluator.result
        evaluator.result.update(citation_results)

        # Compute trust score
        from metrics import compute_trust_score
        evaluator.result = compute_trust_score(evaluator.result, evaluation_config)

        # Save per-sample results if requested
        if save_per_sample and per_sample_output_path:
            logger.info(f"Saving per-sample citation metrics to {per_sample_output_path}...")
            logger.info(f"per_sample_results length: {len(per_sample_results)}")
            try:
                with open(per_sample_output_path, "w", encoding="utf-8") as f:
                    json.dump(per_sample_results, f, ensure_ascii=False, indent=2)
                logger.info(f"Successfully saved per-sample results to {per_sample_output_path}")
            except Exception as e:
                logger.error(f"Failed to save per-sample results: {e}")
        
        evaluator.save_results(output_path=trust_output_path)
    finally:
        os.unlink(tmp_yaml_path)

    print(f"[Step 2] Trust-Eval 结果已写入：{trust_output_path}")
    print(f"         指标摘要：{evaluator.result}")
    
    if save_per_sample and per_sample_output_path:
        print(f"         Per-sample 指标已写入：{per_sample_output_path}")
    
    return evaluator.result


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="一站式评估脚本：eval_result_construct + Trust-Eval + Per-sample citation metrics"
    )
    parser.add_argument(
        "--inference_result", required=True,
        help="推理结果 JSON 文件路径，每条数据需含 question 和 long_answer_with_nemo_guidance 字段",
    )
    parser.add_argument(
        "--ground_truth", required=True,
        help="原始测试集 JSON 文件路径（含 qa_pairs / annotations / docs 字段）",
    )
    parser.add_argument(
        "--output_dir", default="./results",
        help="结果文件输出目录（默认 ./results）"
    )
    parser.add_argument(
        "--model_name",
        help="checkpoint 完整路径，用于结果文件命名（取最后一级目录名拼接到输出文件名中）",
        default="model"
    )
    parser.add_argument(
        "--save_per_sample", type=str, default="true",
        help="是否保存每个样本的 citation recall/precision 指标（true/false）"
    )
    parser.add_argument(
        "--autoais_model",
        default=DEFAULT_AUTOAIS_MODEL,
        help="AutoAIS 模型路径"
    )
    parser.add_argument(
        "--data_type",
        default="asqa",
        choices=["asqa", "qampari", "eli5"],
        help="数据集类型，决定 eval_type 和 is_qampari（默认 asqa）"
    )
    parser.add_argument(
        "--eval_type",
        default=None,
        choices=["em", "cm"],
        help="覆盖 eval_config.yaml 中的 eval_type（em=exact match, cm=claims-based）"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 从推理结果文件名派生输出文件名前缀，可选拼接 checkpoint 名
    stem = Path(args.inference_result).stem
    if args.model_name:
        ckpt_name = Path(args.model_name).name
        stem = f"{stem}_{ckpt_name}"

    # Step 1: 构造 eval 格式文件
    eval_file = str(output_dir / f"{stem}_eval_format.json")
    build_eval_file(args.inference_result, args.ground_truth, eval_file)

    # Step 2: Trust-Eval + Per-sample citation metrics（一体化计算）
    trust_output = str(output_dir / f"{stem}_trust_eval_result.json")
    per_sample_output = str(output_dir / f"{stem}_per_sample_citation.json") if args.save_per_sample.lower() == "true" else None

    print(f"DEBUG: per_sample_output = {per_sample_output}")
    print(f"DEBUG: save_per_sample = {args.save_per_sample.lower() == 'true'}")

    run_trust_eval(
        eval_file,
        trust_output,
        per_sample_output_path=per_sample_output,
        save_per_sample=(args.save_per_sample.lower() == "true"),
        autoais_model_path=args.autoais_model,
        data_type=args.data_type,
        eval_type=args.eval_type
    )

    print("\n[完成] 所有评估结果已保存到:", args.output_dir)


if __name__ == "__main__":
    main()
