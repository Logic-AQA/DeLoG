import json
import re
import os
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import openai

# Global executor for LLM API calls (max 16 concurrent)
_QWEN_EXECUTOR = ThreadPoolExecutor(max_workers=16)
from nmo_python import load_string, NemoEngine

DEFAULT_QWEN_MODEL = "qwen3-max-2026-01-23"

DEFAULT_QWEN_BASE_URL = os.environ.get("LONG_ANSWER_QWEN_BASE_URL")
DEFAULT_QWEN_API_KEY = os.environ.get("LONG_ANSWER_QWEN_API_KEY")
DEFAULT_OPENAI_CLAUDE_BASE_URL = os.environ.get("LONG_ANSWER_CLAUDE_BASE_URL")
DEFAULT_OPENAI_CLAUDE_API_KEY = os.environ.get("LONG_ANSWER_CLAUDE_API_KEY")

SYMBOLIC_REFUSAL_ANSWER = (
    "[FALSE], according to the given documents, there not enough information "
    "to accurately answer the question."
)

local_url = os.environ.get("LOCAL_OPENAI_BASE_URL")
local_api_key = os.environ.get("LOCAL_OPENAI_API_KEY")
local_model = os.environ.get("LOCAL_OPENAI_MODEL", "default")

# Initialize OpenAI clients
qwen_client = (
    openai.OpenAI(api_key=DEFAULT_QWEN_API_KEY, base_url=DEFAULT_QWEN_BASE_URL)
    if DEFAULT_QWEN_API_KEY and DEFAULT_QWEN_BASE_URL
    else None
)
local_client = (
    openai.OpenAI(api_key=local_api_key, base_url=local_url)
    if local_api_key and local_url
    else None
)

def run_inference(question, formated_documents):
    """Call local OpenAI API for inference"""
    if local_client is None:
        raise ValueError("Set LOCAL_OPENAI_BASE_URL and LOCAL_OPENAI_API_KEY before calling run_inference")
    Instruction = "Your task is to generate a Datalog-style logic programing process between <think></think> and a long-form answer between <answer></answer> with citation [*] for a given question and documents"

    inputs = "Question: {question} \n Documents: {documents}".format(
        question=question,
        documents="\n".join(formated_documents)
    )

    messages = [
        {"role": "system", "content": Instruction},
        {"role": "user", "content": inputs}
    ]

    response = local_client.chat.completions.create(
        model=local_model,
        messages=messages,
        temperature=0.0,
        max_tokens=4096,
        timeout=300
    )

    return response.choices[0].message.content

def _normalize_provider(provider: str | None) -> str:
    provider_name = (provider or os.environ.get("LONG_ANSWER_PROVIDER", "qwen")).strip().lower()
    if provider_name in {"openai", "gpt", "chatgpt"}:
        return "gpt"
    if provider_name in {"claude", "anthropic"}:
        return "claude"
    return "qwen"


def _resolve_llm_config(
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: int | None = None,
):
    provider_name = _normalize_provider(provider)
    env_prefix = {
        "qwen": "LONG_ANSWER_QWEN",
        "gpt": "LONG_ANSWER_OPENAI",
        "claude": "LONG_ANSWER_CLAUDE",
    }[provider_name]

    resolved_model = model or os.environ.get(f"{env_prefix}_MODEL")
    resolved_base_url = base_url or os.environ.get(f"{env_prefix}_BASE_URL")
    resolved_api_key = api_key or os.environ.get(f"{env_prefix}_API_KEY")
    resolved_timeout = timeout or int(os.environ.get(f"{env_prefix}_TIMEOUT", "60"))

    if provider_name == "qwen":
        resolved_model = resolved_model or DEFAULT_QWEN_MODEL
        resolved_base_url = resolved_base_url or DEFAULT_QWEN_BASE_URL
        resolved_api_key = resolved_api_key or DEFAULT_QWEN_API_KEY
    elif provider_name in {"gpt", "claude"}:
        resolved_base_url = resolved_base_url or DEFAULT_OPENAI_CLAUDE_BASE_URL
        resolved_api_key = resolved_api_key or DEFAULT_OPENAI_CLAUDE_API_KEY

    if not resolved_model:
        raise ValueError(f"Missing model for provider={provider_name}")
    if not resolved_base_url or not resolved_api_key:
        raise ValueError(f"Missing base_url/api_key for provider={provider_name}")

    return provider_name, resolved_model, resolved_base_url, resolved_api_key, resolved_timeout


@lru_cache(maxsize=32)
def _get_openai_client(api_key: str, base_url: str):
    return openai.OpenAI(api_key=api_key, base_url=base_url)


@lru_cache(maxsize=10000)
def query_answer_llm(
    provider: str,
    prompt: str,
    max_tokens: int = 1,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: int | None = None,
):
    provider_name, resolved_model, resolved_base_url, resolved_api_key, resolved_timeout = _resolve_llm_config(
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
    )
    client = _get_openai_client(resolved_api_key, resolved_base_url)
    last_error = None
    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=resolved_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                timeout=resolved_timeout,
                max_completion_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as exc:
            last_error = exc
            if attempt == 4:
                break
            time.sleep(min(2 ** attempt, 8))
    raise last_error


def query_qwen(prompt, max_tokens=1):
    return query_answer_llm("qwen", prompt, max_tokens=max_tokens)

   

def compute_cite_pattern(logic_program: str):
    """
    Run Nemo reasoning for a single logic program and build cite_pattern.
    Each call creates its own NemoEngine — safe to call from multiple threads.
    """
    try:
        engine = NemoEngine(load_string(logic_program))
        engine.reason()
        raw_results = [[str(x) for x in row] for row in engine.result("valid_result")]
    except BaseException:
        return []

    findall_groups = {}
    clean = lambda s: str(s).strip('"') if isinstance(s, (str, int, float)) else s

    def normalize_result(row):
        if len(row) == 5:
            cl, branch, doc, attr, val = row
        elif len(row) == 4:
            cl = ""
            branch, doc, attr, val = row
        elif len(row) == 3:
            cl = ""
            branch, attr, val = row
            doc = "N/A"
        else:
            return None
        branch = clean(branch)
        if not str(branch).startswith("action_"):
            return None
        return {
            "cl": clean(cl),
            "branch": branch,
            "doc": clean(doc),
            "attr": clean(attr),
            "val": clean(val),
        }

    for row in raw_results:
        result = normalize_result(row)
        if result is None:
            continue
        b_clean = result["branch"]
        if b_clean not in findall_groups:
            findall_groups[b_clean] = []
        findall_groups[b_clean].append(result)

    final_plan = []
    seen_facts = set()
    for branch, docs_list in findall_groups.items():
        docs_list.sort(key=lambda x: (x["doc"], x["attr"]))
        if branch in ["action_redundancy", "action_atomic"]:
            for d in docs_list:
                fact_key = (d['attr'],d['val'])
                if fact_key not in seen_facts:
                    seen_facts.add(fact_key)
                    final_plan.append({
                        "branch": branch,
                        "content": f"context: {d['attr']} [{d['doc']}]",
                        "raw_fact": {
                            "attribute": d['attr'],
                            "value": d['val']
                        }
                    })
                
        elif branch == "action_interleaved":
            segments = []
            for dct in docs_list:
                segments.append({
                    "content": f"context: {dct['attr']} [{dct['doc']}]",
                    "raw_fact": {
                        "attribute": dct['attr'],
                        "value": dct['val']
                    }
                })
            final_plan.append({
                "branch": branch,
                "segments": segments
            })
        elif branch == "action_composite":
            main_doc = docs_list[0]["doc"]
            fact_list = []
            for dct in docs_list:
                fact_list.append({
                    "attribute": dct['attr'],
                    "value": dct['val']
                })
            final_plan.append({
                "branch": branch,
                "doc": main_doc,
                "factList": fact_list
            })
        elif branch == "action_missing":
            final_plan.append({
                "branch": branch,
                "content": "No answer found"
            })

    return final_plan

def generate_long_answer_for_item(
    final_plan,
    question,
    documents,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    llm_base_url: str | None = None,
    llm_api_key: str | None = None,
    llm_timeout: int | None = None,
):
    """Worker for pattern_to_long_form_answer: generate long answer with citation for one item.
    Returns tuple (index, long_answer_with_citation)."""
    pattern_prompt = """You are synthesizing one sentence for a non-empty cite plan.

The current cite-plan step is already selected from grounded evidence. Your job is to verbalize that grounded step as exactly one factual sentence.

Requirements:
1. Answer the question only with grounded facts from the provided document text.
2. Use the provided raw_attribute and raw_value explicitly.
3. Generate exactly one complete sentence for this cite-plan step.
4. Do not mention citation markers, brackets, or document ids in the sentence.
5. Do not add extra facts, hedging, or explanations not directly supported by the document.
6. Do not output refusal text, uncertainty text, or "not enough information". The cite-plan is non-empty, so you must verbalize the cited fact(s).
7. Do not answer with attribute labels only. Write a natural sentence that directly addresses the question using the grounded fact(s).

Input:
Question: {question}
Document: {document}
Raw_attribute: {raw_attribute}
Raw_value: {raw_value}

Output:
One grounded sentence only.
"""
    try:
        cite_pattern = final_plan
        actionable_patterns = [
            pattern for pattern in cite_pattern
            if pattern.get("branch") != "action_missing"
        ]
        if not actionable_patterns:
            return SYMBOLIC_REFUSAL_ANSWER

        def _clean_sentence(response: str) -> str:
            text = (response or "").strip()
            text = text.replace("**Output**:", "").replace("Output:", "").strip()
            text = re.sub(r"^\s*Output\s*", "", text, flags=re.IGNORECASE).strip()
            text = text.splitlines()[0].strip() if text else ""
            if not text:
                return ""
            if text == SYMBOLIC_REFUSAL_ANSWER:
                return text
            text = re.sub(r"\s+", " ", text).strip()
            text = re.sub(r"\[[^\]]+\]", "", text).strip()
            text = text.rstrip(". ")
            return text + "."

        def _generate_step_sentence(answer_generation_prompt: str) -> str:
            last_refusal = None
            for _ in range(3):
                response = query_answer_llm(
                    llm_provider or "qwen",
                    answer_generation_prompt,
                    max_tokens=1024,
                    model=llm_model,
                    base_url=llm_base_url,
                    api_key=llm_api_key,
                    timeout=llm_timeout,
                )
                cleaned = _clean_sentence(response)
                if cleaned and cleaned != SYMBOLIC_REFUSAL_ANSWER:
                    return cleaned
                last_refusal = cleaned or SYMBOLIC_REFUSAL_ANSWER
            raise RuntimeError(f"LLM refused cite-plan step after retries: {last_refusal}")

        def _process_pattern(args):
            idx, pattern = args
            branch = pattern.get('branch')
            if branch in ('action_atomic', 'action_redundancy'):
                document_id = re.findall(r"\[d(\d+)\]", pattern['content'])[0]
                document = documents[int(document_id)-1]
                raw_attribute = pattern['raw_fact']["attribute"]
                raw_value = pattern['raw_fact']["value"]
                answer_generation_prompt = pattern_prompt.format(question=question, document=document, raw_attribute=raw_attribute, raw_value=raw_value)
                cleaned = _generate_step_sentence(answer_generation_prompt)
                fragment = cleaned.rstrip(".") + "[" + str(document_id) + "]. "
                return idx, fragment

            elif branch == 'action_interleaved':
                segments = pattern.get('segments', [])
                raw_attributes, raw_values, used_documents, document_ids = [], [], [], []
                for segment in segments:
                    document_id = re.findall(r"\[d(\d+)\]", segment['content'])[0]
                    document_ids.append(document_id)
                    document = documents[int(document_id)-1]
                    used_documents.append(document)
                    raw_attributes.append(segment['raw_fact']["attribute"])
                    raw_values.append(segment['raw_fact']["value"])
                answer_generation_prompt = pattern_prompt.format(question=question, document="\n".join(used_documents), raw_attribute=";".join(raw_attributes), raw_value=";".join(raw_values))
                response = _generate_step_sentence(answer_generation_prompt)
                cite_string = "".join("[" + str(i) + "]" for i in document_ids)
                fragment = response.rstrip(".") + cite_string + ". "
                return idx, fragment

            elif branch == 'action_composite':
                document_id = re.findall(r"d(\d+)", pattern['doc'])[0]
                if isinstance(document_id, list):
                    return idx, ""
                document = documents[int(document_id)-1]
                raw_attributes, raw_values = [], []
                for fact in pattern.get('factList', []):
                    raw_attributes.append(fact["attribute"])
                    raw_values.append(fact["value"])
                answer_generation_prompt = pattern_prompt.format(question=question, document=document, raw_attribute=";".join(raw_attributes), raw_value=";".join(raw_values))
                response = _generate_step_sentence(answer_generation_prompt)
                fragment = response.rstrip(".") + "[" + document_id + "]. "
                return idx, fragment

            elif branch == 'action_missing':
                return idx, ""
            return idx, ""

        results = list(_QWEN_EXECUTOR.map(_process_pattern, enumerate(cite_pattern)))
        results.sort(key=lambda x: x[0])
        long_form_answer = "".join(r[1] for r in results).strip()
        if not long_form_answer:
            return SYMBOLIC_REFUSAL_ANSWER
        return long_form_answer
    except Exception:
        import traceback
        print(traceback.print_exc())
        return SYMBOLIC_REFUSAL_ANSWER
    
def _process_single_item(idx, data):
    """Process a single item: inference + logic plan + long answer"""
    try:
        question = data['question']
        docs = data['docs'][:5]
        documents = ["[D"+str(i+1)+"]: "+data['docs'][i]['title']+"\t\t"+data['docs'][i]["text"] for i in range(len(docs))]

        # Run inference
        response = run_inference(question, documents)

        # Extract logic process (backward-compatible with V3.1 format)
        m = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
        if m:
            logic_process = m.group(1)
        else:
            m = re.search(r"</think>", response, re.DOTALL)
            if not m:
                raise ValueError("No </think> block in response")
            logic_process = response[:m.start()]

        # Compute cite pattern
        logic_plan = compute_cite_pattern(logic_process)

        # Generate long answer
        long_answer = generate_long_answer_for_item(logic_plan, question, documents)

        # Add results to data
        data['original_response'] = response
        data['long_answer_with_nemo_guidance'] = long_answer

        return idx, data
    except Exception as e:
        import traceback
        print(traceback.print_exc())
        print(f"Error processing item {idx}: {e}")
        return idx, None

def main_inference(num_workers=8, output_path=None):
    """Multi-threaded inference with streaming output"""
    input_path = os.environ.get("INFERENCE_INPUT_PATH")
    if not input_path:
        raise ValueError("Set INFERENCE_INPUT_PATH before calling main_inference")
    if output_path is None:
        output_path = os.environ.get("INFERENCE_OUTPUT_PATH", "results/inference.json")
    cache_path = output_path + ".cache"

    with open(input_path, "r") as rf:
        json_data = json.load(rf)

    # Build cache of processed questions
    processed_questions = set()
    old_results = []

    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as cf:
                for line in cf:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        q = json.loads(line)
                    except Exception:
                        q = line
                    processed_questions.add(q)
        except Exception:
            processed_questions = set()
    else:
        if os.path.exists(output_path):
            try:
                with open(output_path, "r", encoding="utf-8") as rf:
                    old_results = json.load(rf)
                for item in old_results:
                    q = item.get("question")
                    if q:
                        processed_questions.add(q)
            except Exception:
                old_results = []

    # Filter pending items
    scheduled = set()
    pending_items = []
    for i, d in enumerate(json_data):
        q = d.get('question')
        if not q:
            continue
        if q in processed_questions or q in scheduled:
            continue
        scheduled.add(q)
        pending_items.append((i, d))

    # Stream write results
    with open(output_path, "w", encoding="utf-8") as out_f, \
         open(cache_path, "a", encoding="utf-8") as cache_f:

        out_f.write('[\n')
        first_written = False
        written_questions = set()

        # Prewrite old results
        if old_results:
            for item in old_results:
                q = item.get("question")
                if not q or q in written_questions:
                    continue
                if first_written:
                    out_f.write(',\n')
                json.dump(item, out_f, ensure_ascii=False, indent=2)
                first_written = True
                written_questions.add(q)

        # Multi-threaded processing
        from tqdm import tqdm as _tqdm
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_process_single_item, idx, d): (idx, d) for idx, d in pending_items}

            with _tqdm(total=len(futures), desc="Inference (multi-threaded)") as pbar:
                for future in futures:
                    try:
                        idx, result_data = future.result(timeout=600)
                        if result_data is not None:
                            q = result_data.get('question')
                            if q and q not in written_questions:
                                if first_written:
                                    out_f.write(',\n')
                                json.dump(result_data, out_f, ensure_ascii=False, indent=2)
                                out_f.flush()
                                first_written = True
                                written_questions.add(q)
                                processed_questions.add(q)
                                cache_f.write(json.dumps(q, ensure_ascii=False) + "\n")
                                cache_f.flush()
                    except Exception as e:
                        print(f"Error in thread: {e}")
                    finally:
                        pbar.update(1)

        out_f.write('\n]')

def _process_single_item_sft(idx, data):
    """Process a single SFT item: concatenate instruction+input and run inference"""
    try:
        if local_client is None:
            raise ValueError("Set LOCAL_OPENAI_BASE_URL and LOCAL_OPENAI_API_KEY before calling SFT inference")
        instruction = data.get('instruction', '')
        input_text = data.get('input', '')

        # Concatenate instruction and input
        # combined_prompt = instruction + "\n" + input_text

        # Call local API
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": input_text}
        ]

        response = local_client.chat.completions.create(
            model=local_model,
            messages=messages,
            temperature=0.7,
            max_tokens=4096,
            timeout=300
        )

        result_text = response.choices[0].message.content

        # Add result to data
        data['generated_output'] = result_text

        return idx, data
    except Exception as e:
        print(f"Error processing item {idx}: {e}")
        return idx, None

def main_inference_sft(num_workers=4):
    """Multi-threaded inference for SFT dataset with streaming output"""
    input_path = os.environ.get("SFT_INFERENCE_INPUT_PATH")
    output_path = os.environ.get("SFT_INFERENCE_OUTPUT_PATH", "results/sft_inference.json")
    if not input_path:
        raise ValueError("Set SFT_INFERENCE_INPUT_PATH before calling main_inference_sft")
    cache_path = output_path + ".cache"

    with open(input_path, "r", encoding="utf-8") as rf:
        json_data = json.load(rf)

    # Build cache of processed questions
    processed_questions = set()
    old_results = []

    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as cf:
                for line in cf:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        q = json.loads(line)
                    except Exception:
                        q = line
                    processed_questions.add(q)
        except Exception:
            processed_questions = set()
    else:
        if os.path.exists(output_path):
            try:
                with open(output_path, "r", encoding="utf-8") as rf:
                    old_results = json.load(rf)
                for item in old_results:
                    q = item.get("input")
                    if q:
                        processed_questions.add(q)
            except Exception:
                old_results = []

    # Filter pending items
    scheduled = set()
    pending_items = []
    for i, d in enumerate(json_data):
        q = d.get('input')
        if not q:
            continue
        if q in processed_questions or q in scheduled:
            continue
        scheduled.add(q)
        pending_items.append((i, d))

    # Stream write results
    with open(output_path, "w", encoding="utf-8") as out_f, \
         open(cache_path, "a", encoding="utf-8") as cache_f:

        out_f.write('[\n')
        first_written = False
        written_questions = set()

        # Prewrite old results
        if old_results:
            for item in old_results:
                q = item.get("input")
                if not q or q in written_questions:
                    continue
                if first_written:
                    out_f.write(',\n')
                json.dump(item, out_f, ensure_ascii=False, indent=2)
                first_written = True
                written_questions.add(q)

        # Multi-threaded processing
        from tqdm import tqdm as _tqdm
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_process_single_item_sft, idx, d): (idx, d) for idx, d in pending_items}

            with _tqdm(total=len(futures), desc="SFT Inference (multi-threaded)") as pbar:
                for future in futures:
                    try:
                        idx, result_data = future.result(timeout=600)
                        if result_data is not None:
                            q = result_data.get('input')
                            if q and q not in written_questions:
                                if first_written:
                                    out_f.write(',\n')
                                json.dump(result_data, out_f, ensure_ascii=False, indent=2)
                                out_f.flush()
                                first_written = True
                                written_questions.add(q)
                                processed_questions.add(q)
                                cache_f.write(json.dumps(q, ensure_ascii=False) + "\n")
                                cache_f.flush()
                    except Exception as e:
                        print(f"Error in thread: {e}")
                    finally:
                        pbar.update(1)

        out_f.write('\n]')

if __name__ == "__main__":
    # Get number of workers from environment or use default
    num_workers_env = os.getenv("NUM_WORKERS")
    num_workers = int(num_workers_env) if (num_workers_env and num_workers_env.isdigit()) else 8

    print(f"Starting inference with {num_workers} worker threads")
    # Uncomment the function you want to run:
    main_inference(num_workers=num_workers)
    # main_inference_sft(num_workers=num_workers)
