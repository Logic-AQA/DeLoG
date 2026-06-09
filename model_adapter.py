"""
Model Adapter — Model-Family-Agnostic Utilities
================================================
Abstracts away Qwen3-specific assumptions (hardcoded think token IDs,
hardcoded assistant prefixes) so the same training/evaluation scripts
can run on Llama-3.x and Qwen2.5 models.

Supported families:
  - qwen3    (Qwen3-4B-Thinking, native <think> token IDs)
  - qwen2.5  (Qwen2.5-Instruct, <|im_start|> template, no native think)
  - llama3   (Llama-3.x-Instruct, <|start_header_id|> template, no native think)
"""

from typing import List, Tuple

# Qwen3 native think token IDs
_QWEN3_THINK_TOKEN_ID     = 151667
_QWEN3_THINK_END_TOKEN_ID = 151668


def detect_model_family(tokenizer) -> str:
    """Detect model family from tokenizer properties."""
    # Qwen3 has <think> as a special added token
    added = getattr(tokenizer, "added_tokens_encoder", {})
    if "<think>" in added and added["<think>"] == _QWEN3_THINK_TOKEN_ID:
        return "qwen3"

    # Fall back to chat template inspection
    tpl = getattr(tokenizer, "chat_template", "") or ""
    if "start_header_id" in tpl:
        return "llama3"
    if "im_start" in tpl:
        return "qwen2.5"

    return "unknown"


def get_assistant_prefix(tokenizer) -> str:
    """Return the assistant-turn prefix string for the model family.

    This is the text we must append after apply_chat_template(..., add_generation_prompt=False)
    to start the assistant response.
    """
    family = detect_model_family(tokenizer)
    if family == "qwen3":
        return "<|im_start|>assistant\n"
    if family == "qwen2.5":
        return "<|im_start|>assistant\n"
    if family == "llama3":
        return "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return ""


def build_chat_prompt(tokenizer, system_prompt: str, user_content: str) -> str:
    """Build a chat-format prompt compatible with the model's template.

    For Qwen3 we disable generation_prompt and manually append the assistant
    prefix (avoiding the automatic <think>\n that Qwen3's template injects).
    For Qwen2.5 and Llama3 we let apply_chat_template generate the assistant
    prefix natively.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    family = detect_model_family(tokenizer)

    if family == "qwen3":
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            add_special_tokens=False,
        )
        prompt += get_assistant_prefix(tokenizer)
        return prompt

    # Qwen2.5 / Llama3 — let the template add the assistant prefix
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    )
    return prompt


def find_think_boundaries(
    input_ids: List[int], tokenizer, prompt_len: int
) -> Tuple[int, int]:
    """Find <think> ... </think> boundaries in the response portion.

    Returns (think_start, think_end) where both are indices into input_ids.
    think_start == -1 means no <think> tag was found.
    think_end defaults to len(input_ids) if </think> is missing.
    """
    family = detect_model_family(tokenizer)

    if family == "qwen3":
        # Search by special token IDs (fast, exact)
        think_start = -1
        for i in range(prompt_len, len(input_ids)):
            if input_ids[i] == _QWEN3_THINK_TOKEN_ID:
                think_start = i
                break

        think_end = len(input_ids)
        if think_start != -1:
            for i in range(think_start + 1, len(input_ids)):
                if input_ids[i] == _QWEN3_THINK_END_TOKEN_ID:
                    think_end = i
                    break
        return think_start, think_end

    # For Llama3 / Qwen2.5: <think> is plain text, not a special token.
    # We decode the response portion and search for the string literals,
    # then map back to token indices via tokenizer.encode().
    response_ids = input_ids[prompt_len:]
    if not response_ids:
        return -1, len(input_ids)

    response_text = tokenizer.decode(response_ids)
    ts = response_text.find("<think>")
    if ts == -1:
        return -1, len(input_ids)

    # Map string positions back to token-level indices.
    # We encode the text *before* <think> and count tokens.
    prefix_text = response_text[:ts]
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    think_start = prompt_len + len(prefix_ids)

    te = response_text.find("</think>")
    if te != -1:
        mid_text = response_text[:te]
        mid_ids = tokenizer.encode(mid_text, add_special_tokens=False)
        think_end = prompt_len + len(mid_ids)
    else:
        think_end = len(input_ids)

    return think_start, think_end
