import argparse
import json
from pathlib import Path
from typing import Any

from inference_logic_guid_aqa import (
    SYMBOLIC_REFUSAL_ANSWER,
    compute_cite_pattern,
    generate_long_answer_for_item,
)
from sft_batch_inference_vllm import extract_think_process


def format_documents(item: dict[str, Any], num_docs: int) -> list[str]:
    docs = item.get("docs", [])[:num_docs]
    formatted = []
    for i, doc in enumerate(docs):
        if isinstance(doc, dict):
            formatted.append(f"[D{i+1}]: {doc.get('title', '')}\t\t{doc.get('text', '')}")
        else:
            formatted.append(f"[D{i+1}]: {doc}")
    return formatted


def convert_item(
    item: dict[str, Any],
    response_field: str = "original_response",
    num_docs: int = 5,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    llm_base_url: str | None = None,
    llm_api_key: str | None = None,
    llm_timeout: int | None = None,
) -> dict[str, Any]:
    if response_field not in item:
        raise KeyError(f"Missing response field: {response_field}")

    result = dict(item)
    full_output = item.get(response_field) or ""
    think_process = extract_think_process(full_output)
    cite_pattern = compute_cite_pattern(think_process) if think_process else []

    if cite_pattern:
        long_answer = generate_long_answer_for_item(
            cite_pattern,
            item["question"],
            format_documents(item, num_docs),
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
            llm_timeout=llm_timeout,
        )
    else:
        long_answer = SYMBOLIC_REFUSAL_ANSWER

    result["think_process"] = think_process
    result["cite_pattern"] = cite_pattern
    result["symbolic_eval"] = True
    result["symbolic_source_response_field"] = response_field
    result["symbolic_llm_provider"] = llm_provider
    result["symbolic_llm_model"] = llm_model
    result["long_answer_with_nemo_guidance"] = long_answer
    return result


def convert_file(
    input_path: str | Path,
    output_path: str | Path,
    response_field: str = "original_response",
    num_docs: int = 5,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    llm_base_url: str | None = None,
    llm_api_key: str | None = None,
    llm_timeout: int | None = None,
    limit: int | None = None,
) -> int:
    input_path = Path(input_path)
    output_path = Path(output_path)
    data = json.loads(input_path.read_text())
    if limit is not None:
        data = data[:limit]

    converted = []
    partial_path = output_path.with_suffix(output_path.suffix + ".partial.jsonl")
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    refusal_count = 0
    answer_count = 0
    with partial_path.open("w", encoding="utf-8") as partial_file:
        for idx, item in enumerate(data, start=1):
            result = convert_item(
                item,
                response_field=response_field,
                num_docs=num_docs,
                llm_provider=llm_provider,
                llm_model=llm_model,
                llm_base_url=llm_base_url,
                llm_api_key=llm_api_key,
                llm_timeout=llm_timeout,
            )
            converted.append(result)
            if result.get("long_answer_with_nemo_guidance") == SYMBOLIC_REFUSAL_ANSWER:
                refusal_count += 1
            else:
                answer_count += 1
            partial_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            partial_file.flush()
            if idx % 10 == 0 or idx == len(data):
                print(
                    f"converted {idx}/{len(data)} "
                    f"answered={answer_count} refusal={refusal_count} "
                    f"partial={partial_path}",
                    flush=True,
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(converted, ensure_ascii=False, indent=2))
    return len(converted)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build symbolic-eval answers from existing inference JSON results."
    )
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--response_field", default="original_response")
    parser.add_argument("--num_docs", type=int, default=5)
    parser.add_argument("--symbolic_llm_provider", default="qwen", choices=["qwen", "gpt", "claude", "openai"])
    parser.add_argument("--symbolic_llm_model", default=None)
    parser.add_argument("--symbolic_llm_base_url", default=None)
    parser.add_argument("--symbolic_llm_api_key", default=None)
    parser.add_argument("--symbolic_llm_timeout", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = convert_file(
        input_path=args.input_path,
        output_path=args.output_path,
        response_field=args.response_field,
        num_docs=args.num_docs,
        llm_provider=args.symbolic_llm_provider,
        llm_model=args.symbolic_llm_model,
        llm_base_url=args.symbolic_llm_base_url,
        llm_api_key=args.symbolic_llm_api_key,
        llm_timeout=args.symbolic_llm_timeout,
        limit=args.limit,
    )
    print(f"saved {count} symbolic items to {args.output_path}")


if __name__ == "__main__":
    main()
