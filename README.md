# DeLoG Training, Inference, and Evaluation Package

This repository contains a reproducible SFT -> GRPO -> inference -> evaluation
pipeline for logic-guided long-form QA with citations.

Model checkpoints, evaluation sets, generated outputs, logs, and API credentials
are not included.

## Repository Contents

Included data:

| File | Purpose | Size |
| --- | --- | --- |
| `data_sft.json` | SFT training data | 4,331 samples |
| `data_grpo.json` | GRPO training data | 1,800 samples |

Main entrypoints:

| File | Purpose |
| --- | --- |
| `train_sft.sh` | Shell wrapper for SFT training |
| `train_grpo.sh` | Shell wrapper for GRPO training |
| `train_sft.py` | DeepSpeed SFT implementation |
| `train_grpo.py` | Unsloth + TRL GRPO implementation |
| `train_and_evaluation/sft_batch_inference_vllm.py` | vLLM batch inference |
| `train_and_evaluation/symbolic_answer_from_existing_results.py` | Symbolic long-answer synthesis from saved model outputs |
| `train_and_evaluation/autoais_server.py` | AutoAIS NLI HTTP service |
| `train_and_evaluation/run_evaluation.py` | Evaluation wrapper for Trust-Eval and per-sample citation metrics |

Vendored code:

| Directory | Purpose |
| --- | --- |
| `unsloth-main/` | Local Unsloth source used by GRPO training |
| `trust-align/` | Local Trust-Eval code used by evaluation |

## External Artifacts

You must provide these paths locally:

| Variable or argument | Required for | Description |
| --- | --- | --- |
| `MODEL_PATH` or `--model_path` | SFT | Base instruct model checkpoint |
| `MODEL_PATH` or `--model-path` | GRPO | SFT checkpoint used as the GRPO base model |
| `--model_path` | Inference | SFT/base model loaded by vLLM |
| `--adapter_path` | Inference | Optional GRPO LoRA adapter |
| `AUTOAIS_MODEL_PATH` | AutoAIS server | Local AutoAIS NLI model checkpoint |
| `AUTOAIS_URLS` | Evaluation | One or more AutoAIS `/nli` service endpoints |
| `--input_path` / `--ground_truth` | Inference/evaluation | Evaluation dataset, not included in this repository |

The reward and symbolic-answer code also imports `nmo_python`. Install the
Datalog/Nemo runtime that provides this package before running GRPO rewards or
symbolic synthesis.

## Install

```bash
cd <repo>
pip install -r requirements.txt
```

The original misspelled `requirments.txt` is kept for compatibility and has the
same contents as `requirements.txt`.

## API Configuration

No API URLs or keys are stored in the repository. Set them only through
environment variables when using API-backed symbolic answer generation or reward
components.

```bash
export LONG_ANSWER_QWEN_BASE_URL=<provider-base-url>
export LONG_ANSWER_QWEN_API_KEY=<provider-api-key>
export LONG_ANSWER_CLAUDE_BASE_URL=<provider-base-url>
export LONG_ANSWER_CLAUDE_API_KEY=<provider-api-key>
export LOCAL_OPENAI_BASE_URL=<local-openai-compatible-url>
export LOCAL_OPENAI_API_KEY=<local-api-key>
export LOCAL_OPENAI_MODEL=<local-model-name>
```

## SFT

Run from the repository root:

```bash
GPUS=0,1 \
NUM_GPUS=2 \
MODEL_PATH=<local-base-model> \
SAVE_PATH=saves/sft-model \
bash train_sft.sh
```

The wrapper runs:

```bash
deepspeed --num_gpus "$NUM_GPUS" \
  train_sft.py \
  --model_path "$MODEL_PATH" \
  --data_path data_sft.json \
  --save_path "$SAVE_PATH" \
  --output_mode think_cite_plan_answer
```

Useful environment overrides:

| Variable | Default |
| --- | --- |
| `DEEPSPEED` | `deepspeed` |
| `GPUS` | `0,1` |
| `NUM_GPUS` | `2` |
| `DATA_PATH` | `data_sft.json` |
| `SAVE_PATH` | `saves/sft-best-asqa-repro` |
| `OUTPUT_MODE` | `think_cite_plan_answer` |

`OUTPUT_MODE` supports `think_cite_plan_answer`, `think_answer`, and
`answer_only`.

## GRPO

Run after producing an SFT checkpoint:

```bash
GPU=0 \
MODEL_PATH=saves/sft-model \
OUTPUT_DIR=saves/grpo-model \
WANDB_NAME=grpo-doc-refusal-reward-fix \
MAX_STEPS=1800 \
NUM_GENERATIONS=16 \
SAVE_STEPS=300 \
bash train_grpo.sh
```

The wrapper runs:

```bash
python train_grpo.py \
  --model-path "$MODEL_PATH" \
  --data-path data_grpo.json \
  --output-dir "$OUTPUT_DIR" \
  --wandb-name "$WANDB_NAME" \
  --max-steps "$MAX_STEPS" \
  --num-generations "$NUM_GENERATIONS" \
  --save-steps "$SAVE_STEPS" \
  --debug-reward \
  --debug-reward-batches "$MAX_STEPS" \
  --debug-reward-samples 0
```

Useful environment overrides:

| Variable | Default |
| --- | --- |
| `PYTHON` | `python` |
| `GPU` | `0` |
| `MODEL_PATH` | `saves/sft-best-asqa-repro` |
| `DATA_PATH` | `data_grpo.json` |
| `OUTPUT_DIR` | `saves/grpo-doc-refusal-reward-fix-20260522` |
| `WANDB_NAME` | `grpo-doc-refusal-reward-fix-20260522` |
| `MAX_STEPS` | `1800` |
| `NUM_GENERATIONS` | `16` |
| `SAVE_STEPS` | `300` |

## Inference

Run vLLM batch inference:

```bash
CUDA_VISIBLE_DEVICES=0 python \
  train_and_evaluation/sft_batch_inference_vllm.py \
  --model_path saves/sft-model \
  --adapter_path saves/grpo-model \
  --input_path data/eval_dataset.json \
  --output_path results/inference.json \
  --dataset_type asqa \
  --num_docs 5 \
  --output_mode think_cite_plan_answer
```

If the checkpoint is a fully merged model instead of a LoRA adapter, omit
`--adapter_path`.

Supported `--dataset_type` values are `asqa`, `qampari`, `eli5`, and
`expertqa`. Supported `--output_mode` values are `think_cite_plan_answer`,
`think_answer`, and `answer_only`.

## Symbolic Answer Synthesis

To convert existing model outputs into symbolically grounded long answers:

```bash
python train_and_evaluation/symbolic_answer_from_existing_results.py \
  --input_path results/inference.json \
  --output_path results/symbolic_inference.json \
  --response_field original_response \
  --num_docs 5 \
  --symbolic_llm_provider qwen \
  --symbolic_llm_model <provider-model-name>
```

This command requires the corresponding `LONG_ANSWER_*` API environment
variables unless `--symbolic_llm_base_url` and `--symbolic_llm_api_key` are
provided explicitly.

## AutoAIS Server

Start the AutoAIS NLI service:

```bash
AUTOAIS_MODEL_PATH=<local-autoais-model> \
CUDA_VISIBLE_DEVICES=0 python \
  train_and_evaluation/autoais_server.py \
  --host 0.0.0.0 \
  --port 8001
```

The service exposes:

| Endpoint | Purpose |
| --- | --- |
| `POST /nli` | Body: `{"premise": "...", "hypothesis": "..."}` |
| `GET /health` | Health check |

## Evaluation

Run evaluation after inference:

```bash
AUTOAIS_URLS=<autoais-nli-url> \
python train_and_evaluation/run_evaluation.py \
  --inference_result results/inference.json \
  --ground_truth data/eval_dataset.json \
  --output_dir results/eval_output \
  --model_name grpo-model \
  --data_type asqa \
  --eval_type em
```

`AUTOAIS_URLS` accepts a comma-separated list of `/nli` endpoints. The evaluator
round-robins across them.

Outputs are written under `--output_dir`:

| Output | Description |
| --- | --- |
| `*_eval_format.json` | Inference and ground truth merged into Trust-Eval format |
| `*_trust_eval_result.json` | Aggregate Trust-Eval result |
| `*_per_sample_citation.json` | Per-sample citation recall and precision |

## Notes for Reproduction

- Run commands from the repository root unless explicitly stated otherwise.
- The scripts use relative paths by default.
- Large model checkpoints and evaluation sets are intentionally not bundled.
- `unsloth-main/` and `trust-align/` are included as local source dependencies.
- GRPO reward computation can call external LLM APIs and AutoAIS, depending on
  enabled reward components and configuration.
