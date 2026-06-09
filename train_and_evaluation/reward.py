from nmo_python import load_string, NemoEngine
import openai
import os

qwen_url = os.environ.get("LONG_ANSWER_QWEN_BASE_URL")
qwen_api_key = os.environ.get("LONG_ANSWER_QWEN_API_KEY")

qwen_client = (
    openai.OpenAI(api_key=qwen_api_key, base_url=qwen_url)
    if qwen_api_key and qwen_url
    else None
)

import re
import copy
from functools import lru_cache

import numpy as np

from alec_eval import _run_nli_autoais, remove_citations, sent_tokenize



def format_document(doc):
    """Format a document dict (or raw string) into a plain text string for NLI."""
    if isinstance(doc, dict):
        if "title" in doc and "text" in doc:
            return f"Title: {doc['title']}\nContent: {doc['text']}"
        elif "phrase" in doc and "sent" in doc:
            return f"Phrase: {doc['phrase']}\nSentence: {doc['sent']}"
        else:
            return str(doc)
    return str(doc)


def check_nemo_logic_dead_ends(datalog_str):
    # 1. 预处理：去掉注释，统一空格
    clean_code = re.sub(r'%.*', '', datalog_str)

    # 2. 提取【定义集】 (Definitions) - 逻辑的来源
    # 包括：事实、规则头、@import
    defined = set()

    # A. 匹配事实和规则头 (出现在 :- 之前，或直接以 . 结尾)
    # 正则：匹配行首或句号后的单词，且后面跟着 '('
    # 使用 finditer 结合逻辑位置判断更准
    potential_defs = re.finditer(r'([a-z0-9_]+)\s*\(', clean_code)
    for match in potential_defs:
        name = match.group(1)
        start_pos = match.start()
        # 简单判定：如果在该谓词之后，先遇到 ':-' 或在遇到下一个谓词前遇到了 '.'
        # 则认为它是定义（左侧）
        remaining = clean_code[match.end():]
        first_sep = re.search(r'(\.|:-)', remaining)
        if first_sep and first_sep.group(1) in ('.', ':-'):
            # 排除掉规则体内部的情况（稍微粗略但对标准 Datalog 有效）
            # 更严谨的做法是按句号分割查看
            pass

    # 更稳妥的办法：按句号分割每一条声明
    statements = clean_code.split('.')
    for stmt in statements:
        stmt = stmt.strip()
        if not stmt: continue

        # 处理 @import
        if stmt.startswith('@import'):
            m = re.search(r'@import\s+([a-z0-9_]+)', stmt)
            if m: defined.add(m.group(1))
            continue

        # 处理规则和事实
        if ':-' in stmt:
            head_part = stmt.split(':-')[0]
            m = re.search(r'([a-z0-9_]+)\s*\(', head_part)
            if m: defined.add(m.group(1))
        else:
            m = re.search(r'([a-z0-9_]+)\s*\(', stmt)
            if m: defined.add(m.group(1))

    # 3. 提取【使用集】 (Usages) - 逻辑的消耗
    used = set()
    for stmt in statements:
        stmt = stmt.strip()
        if not stmt: continue

        # 规则体中的调用
        if ':-' in stmt:
            body_part = stmt.split(':-')[1]
            calls = re.findall(r'([a-z0-9_]+)\s*\(', body_part)
            used.update(calls)

        # 指令中的调用
        if stmt.startswith('@output'):
            out_m = re.findall(r'@output\s+([a-z0-9_]+)', stmt)
            used.update(out_m)

    # 4. 过滤内置函数
    built_ins = {'count', 'sum', 'min', 'max', 'avg', 'group_by', 'shared', 'true', 'false'}

    # 5. 计算断裂点：使用了，但没有任何地方定义
    dead_ends = used - defined - built_ins

    return sorted(list(dead_ends))

@lru_cache(maxsize=10000)
def query_qwen(prompt, max_tokens=1):
    if qwen_client is None:
        raise ValueError("Set LONG_ANSWER_QWEN_BASE_URL and LONG_ANSWER_QWEN_API_KEY before calling query_qwen")
    response = qwen_client.chat.completions.create(
        model="qwen3-max-2026-01-23",
        messages=[
            {"role":"user", "content":prompt}
        ],
        temperature=0,
        timeout=60,
        max_completion_tokens=max_tokens
    )
    return response.choices[0].message.content

def _normalize_text(text: str) -> str:
    """Lowercase, remove punctuation, normalize whitespace."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def is_refusal_ground_truth(sample: dict) -> bool:
    """Check whether the ground-truth label is a refusal answer.

    Uses the `long_answer_with_citation` field (if present); falls back to
    checking `short_answers` for the legacy '[FALSE]' token.
    """
    la = sample.get("long_answer_with_citation", "")
    if isinstance(la, str) and la.strip().startswith("[FALSE]"):
        return True
    short_answers = sample.get("short_answers", [])
    if isinstance(short_answers, list) and any("[FALSE]" in str(sa) for sa in short_answers):
        return True
    return False


def docs_support_answer(sample: dict) -> bool:
    """Check whether the top-5 retrieved documents contain enough information
    to answer the question (based on answers_found metadata).

    Returns False when none of the top-5 docs has any answers_found,
    indicating the model *should* refuse.
    """
    docs = sample.get("docs", [])
    if not docs:
        return False
    top5 = docs[:5]
    found_list = []
    for doc in top5:
        af = doc.get("answers_found", False)
        if isinstance(af, bool):
            found_list.append(int(af))
        elif isinstance(af, list):
            found_list.append(af)
        else:
            found_list.append(int(bool(af)))

    if found_list and isinstance(found_list[0], list):
        supported = np.bitwise_or.reduce(found_list)
        return any(supported)
    else:
        return any(found_list)


def qa_reward(long_answer, sample):
    """
    Compute QA reward: fraction of short answers found in the generated long answer.
    Accepts pre-computed long_answer to avoid redundant LLM calls.

    For refusal ground-truth samples, the model receives a positive reward (1.0)
    only when it explicitly emits the '[FALSE]' refusal flag in the generated
    answer; otherwise it receives 0.0.
    """
    if not long_answer or long_answer == ". ":
        return 0.0

    # Refusal branch: reward correct refusal, penalise wrongful answers
    if is_refusal_ground_truth(sample):
        return 1.5 if "[FALSE]" in long_answer else -0.5

    # Normal (answerable) branch
    # Penalise over-refusal: answering a normal question with [FALSE] is worse
    # than getting the answer wrong, because it destroys citation reward.
    if "[FALSE]" in long_answer:
        return -0.5

    try:
        short_answers = sample['short_answers']
        if not short_answers:
            return 0.0
        norm_answer = _normalize_text(long_answer)
        reward = 0
        for sa in short_answers:
            if _normalize_text(sa) in norm_answer:
                reward += 1
        return reward / len(short_answers) if reward > 0 else 0.0
    except Exception:
        return 0.0


def qa_reward_smooth(long_answer, sample):
    """
    Smoothed QA reward using token-level overlap instead of hard substring match.
    Returns a continuous value in [0, 1], reducing variance for GRPO advantage estimation.

    For refusal ground-truth samples, falls back to a binary signal (1.0 if the
    generated answer contains '[FALSE]', else 0.0) because token overlap is
    meaningless for refusal utterances.

    Strategy (answerable samples):
      - For each short answer, compute token overlap ratio (Jaccard-like) with long answer.
      - If exact substring match exists, score is 1.0 for that answer.
      - Otherwise score = |tokens_sa ∩ tokens_ans| / |tokens_sa| (recall-oriented).
      - Final reward averaged over all short answers.
    """
    if not long_answer or long_answer == ". ":
        return 0.0

    # Refusal branch: binary signal
    if is_refusal_ground_truth(sample):
        return 1.5 if "[FALSE]" in long_answer else 0.0

    # Normal (answerable) branch
    try:
        short_answers = sample.get('short_answers', [])
        if not short_answers:
            return 0.0
        norm_answer = _normalize_text(long_answer)
        ans_tokens = set(norm_answer.split())
        total_score = 0.0
        for sa in short_answers:
            norm_sa = _normalize_text(sa)
            if not norm_sa:
                continue
            # Exact match bonus
            if norm_sa in norm_answer:
                total_score += 1.0
                continue
            sa_tokens = set(norm_sa.split())
            if not sa_tokens:
                continue
            overlap = len(ans_tokens & sa_tokens)
            score = min(overlap / len(sa_tokens), 1.0)
            total_score += score
        return total_score / len(short_answers)
    except Exception:
        return 0.0


def citation_reward(long_answer, sample):
    """
    Compute citation F1 reward (AutoAIS-based recall × precision).
    Accepts pre-computed long_answer to avoid redundant LLM calls.
    """
    rec, prec = citation_recall_precision(long_answer, sample)
    if rec == 0 or prec == 0:
        return 0
    return 2 * rec * prec / (rec + prec)


def citation_recall_precision(long_answer, sample):
    """
    Compute citation recall and precision separately (AutoAIS-based).
    Returns (recall, precision).
    Accepts pre-computed long_answer to avoid redundant LLM calls.
    """
    if not long_answer:
        return 0.0, 0.0

    documents = sample['docs']
    try:
        if long_answer == ". ":
            return 0.0, 0.0

        # Refusal ground-truth branch -------------------------------------------------
        if is_refusal_ground_truth(sample):
            if "[FALSE]" in long_answer:
                # Correct refusal: return None so the caller can treat citation as
                # "missing" (neutral in group normalization) instead of pulling the
                # sample to the minimum via min-max scaling.
                return None, None
            else:
                # Wrongly answered a refusal question: compute citation normally
                # so that hallucinated / unsupported citations lower the reward.
                pass
        # -----------------------------------------------------------------------------

        sentences = sent_tokenize(long_answer)
        target_sents = [remove_citations(sent).strip() for sent in sentences]

        entail = 0
        entail_prec = 0
        total_citations = 0

        for sent_id, sent in enumerate(sentences):
            target_sent = target_sents[sent_id]
            joint_entail = -1  # Undecided

            # Find references
            ref = [int(r[1:])-1 for r in re.findall(r"\[\d+", sent)]  # In text citation id starts from 1

            if len(ref) == 0:
                # No citations
                joint_entail = 0
            elif any([ref_id >= len(documents) for ref_id in ref]):
                # Citations out of range
                joint_entail = 0
            else:
                ref = ref[:5]
                total_citations += len(ref)
                # Fix: use format_document() so dict-type docs are properly serialized
                joint_passage = '\n'.join([format_document(documents[psgs_id]) for psgs_id in ref])

            # If not directly rejected by citation format error, calculate the recall score
            if joint_entail == -1:
                joint_entail = _run_nli_autoais(joint_passage, target_sent)

            entail += joint_entail

            # calculate the precision score if applicable
            if joint_entail and len(ref) > 1:
                # Precision check: did the model cite any unnecessary documents?
                for psgs_id in ref:
                    # condition A
                    passage = format_document(documents[psgs_id])
                    nli_result = _run_nli_autoais(passage, target_sent)

                    # condition B
                    if not nli_result:
                        subset_exclude = copy.deepcopy(ref)
                        subset_exclude.remove(psgs_id)
                        passage = '\n'.join([format_document(documents[pid]) for pid in subset_exclude])
                        nli_result = _run_nli_autoais(passage, target_sent)
                        if nli_result:  # psgs_id is not necessary
                            pass  # over-citation detected but not penalized in current implementation
                        else:
                            entail_prec += 1
                    else:
                        entail_prec += 1
            else:
                entail_prec += joint_entail

        try:
            citation_recall = entail / len(sentences)
            citation_precision = entail_prec / total_citations if total_citations > 0 else 0
        except:
            return 0.0, 0.0
        return float(citation_recall), float(citation_precision)
    except:
        return 0.0, 0.0


def logic_reward(logic_process):
    """
    Reward for syntactic validity and logical completeness of the Nemo program.
    +1 if the program parses without error (grammar OK).
    +1 if there are no dead-end predicates (logic is complete / no broken references).
    """
    reward = 0
    try:
        load_string(logic_process)
        reward += 1
        if not check_nemo_logic_dead_ends(logic_process):
            reward += 1
        return reward
    except:
        return reward


def feature_extract_reward(logic_process, sample):
    pattern = r'^feature\("[^\d"]*(\d+)[^"]*",\s*"([^"]+)",\s*"([^"]+)"\)\.'
    matches = re.findall(pattern, logic_process, re.MULTILINE)
    docs = sample['docs']
    reward = 0
    contained = []
    for match in matches:
        ids, _, value = match
        if ids and int(ids) <= 5:
            # Fix: use format_document to get a string representation for 'in' check
            doc_text = format_document(docs[int(ids)-1])
            if value in doc_text:
                contained.append(1)
    if len(contained) == 0 or len(matches) == 0:
        return 0
    reward = len(contained) / len(matches)
    return reward

def illegal_predicate_reward(logic_process):
    pattern = r'^valid_[^.]+\.'
    reward = 0
    Flag = True
    matches = re.findall(pattern, logic_process, re.MULTILINE)
    for match in matches:
        if "valid_result" not in match:
            Flag = False
            break
    if Flag:
        reward += 1
    return reward

def _extract_answer_from_completion(text: str) -> str:
    """Extract content inside <answer>...</answer>; fallback to post-</think> content."""
    m = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m2 = re.search(r'</think>(.*)', text, re.DOTALL)
    if m2:
        return m2.group(1).strip()
    return text.strip()


def answer_length_penalty(long_answer: str, target_words: int = 100, max_words: int = 200) -> float:
    """
    Penalize answers that are too long.
    Thresholds calibrated against the original successful run
    (answered_length ~83 words).  Repro-v5-no-cite averaged ~53 words,
    so this mainly prevents future runs from bloating past the sweet spot.
    - 1.0  if word_count <= target_words
    - linear decay from 1.0 -> 0.5 if target_words < word_count <= max_words
    - 0.5  if word_count > max_words
    """
    word_count = len(long_answer.split())
    if word_count <= target_words:
        return 1.0
    if word_count >= max_words:
        return 0.5
    return 1.0 - 0.5 * (word_count - target_words) / (max_words - target_words)


def partial_reward(logic_process, sample, completion_text: str = None):
    """
    Compute all reward components except citation_reward.
    Returns (qa_reward_value, logic_reward_value, feature_extract_reward_value,
             illegal_predicate_reward_value, long_answer).

    **2026-05-08 update**: When completion_text is provided, extract the answer
    directly from the model's raw output (<answer>...</answer>) instead of
    synthesizing via Nemo + Qwen API. A length penalty is applied to discourage
    overly long answers.
    """
    # Compute logic / feature / illegal rewards from logic_process (no Nemo needed)
    logic_reward_value = logic_reward(logic_process)
    feature_extract_reward_value = feature_extract_reward(logic_process, sample)
    illegal_predicate_reward_value = illegal_predicate_reward(logic_process)

    # Extract long_answer from model-generated completion if available
    long_answer = None
    if completion_text is not None:
        try:
            long_answer = _extract_answer_from_completion(completion_text)
            if not long_answer or long_answer == ". ":
                long_answer = None
        except Exception:
            long_answer = None

    # If the model failed to generate <answer>, do NOT fall back to ground truth
    # (that silently rewards format errors). Return zero QA so the policy learns
    # to produce the proper tag.
    if long_answer is None and completion_text is not None:
        qa_reward_value = 0.0
        return qa_reward_value, logic_reward_value, feature_extract_reward_value, illegal_predicate_reward_value, None

    # Fallback to Nemo + Qwen API only if extraction failed AND no completion_text was provided
    if long_answer is None:
        try:
            from utils_qa import compute_cite_pattern, generate_long_answer_for_item
            logic_plan = compute_cite_pattern(logic_process)
            question = sample.get('question', '')
            docs = sample.get('docs', [])
            if docs and isinstance(docs[0], dict):
                documents = [
                    f"[D{i+1}]: {doc.get('title', '')}\t\t{doc.get('text', '')}"
                    for i, doc in enumerate(docs)
                ]
            elif docs and isinstance(docs[0], str):
                documents = docs
            else:
                documents = []
            if logic_plan and question and documents:
                long_answer = generate_long_answer_for_item(logic_plan, question, documents)
        except Exception:
            long_answer = None

    # QA reward on the extracted/generated long answer, with length penalty
    if long_answer is not None and long_answer:
        raw_qa = qa_reward(long_answer, sample)
        # Skip length penalty for refusal samples: a concise refusal is desired,
        # so we should not penalise short answers that correctly contain [FALSE].
        if is_refusal_ground_truth(sample):
            qa_reward_value = raw_qa
        else:
            length_pen = answer_length_penalty(long_answer)
            qa_reward_value = raw_qa * length_pen
    else:
        qa_reward_value = 0.0

    # Penalise wrong refusal harshly: if a normal sample outputs [FALSE],
    # the logic / feature / illegal rewards are meaningless.
    if (
        not is_refusal_ground_truth(sample)
        and long_answer is not None
        and "[FALSE]" in long_answer
    ):
        logic_reward_value = 0.0
        feature_extract_reward_value = 0.0
        illegal_predicate_reward_value = 0.0

    return qa_reward_value, logic_reward_value, feature_extract_reward_value, illegal_predicate_reward_value, long_answer


def total_reward(logic_process, sample):
    """
    Compute all reward components for one sample (convenience wrapper).
    """
    qa_reward_value, logic_reward_value, feature_extract_reward_value, illegal_predicate_reward_value, long_answer = partial_reward(logic_process, sample)
    citation_reward_value = citation_reward(long_answer, sample)
    return qa_reward_value, logic_reward_value, feature_extract_reward_value, illegal_predicate_reward_value, citation_reward_value
