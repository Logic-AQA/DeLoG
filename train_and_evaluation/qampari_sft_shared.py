import re
from typing import Any


FINAL_ANSWER_PREFIX = "Final answer:"
QAMPARI_INSTRUCTION = (
    "For this Qampari question, identify every answer entity that is directly supported by the documents. "
    "Only include entities that satisfy the relation asked by the question; do not list related entities, "
    "document titles, dates, organizations, or attributes unless they are the requested answer entity. "
    "Write one concise evidence sentence, then end with a 'Final answer:' line containing only the cited entity list. "
    "If the documents do not support any answer entity, output the refusal answer exactly."
)
REFUSAL_ANSWER = (
    "[FALSE], according to the given documents, there not enough information "
    "to accurately answer the question."
)
REFUSAL_LOGIC_PROGRAM = """
lib("q").

insufficient_evidence("query_context").

valid_result("query_context", "insufficient_evidence", "N/A", "N/A", "No answer found") :-
    insufficient_evidence("query_context").
""".strip()
TRANSITION_PHRASE = "Based on this plan, generate the answer:"


def normalize_answer(text: str) -> str:
    text = re.sub(r"\[\d+\]", "", str(text))
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text.rstrip(".,;:")


def unique_answers(short_answers: list[Any]) -> list[str]:
    seen = set()
    result = []
    for answer in short_answers or []:
        value = str(answer).strip()
        key = normalize_answer(value)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def docs_from_ranked_passages(ranked_passages: list[Any]) -> list[dict[str, str]]:
    docs = []
    for idx, passage in enumerate(ranked_passages or [], start=1):
        if isinstance(passage, dict):
            title = passage.get("title") or f"Document {idx}"
            content = passage.get("text") or passage.get("content") or ""
        else:
            title = f"Document {idx}"
            content = str(passage)
        docs.append({"title": title, "text": content})
    return docs


def clean_entity_value(value: Any) -> str:
    value = str(value or "").strip().strip('"').strip("'").strip(".")
    return value.replace("_", " ").strip()


def split_entities(raw_value: Any) -> list[str]:
    value = clean_entity_value(raw_value)
    if not value:
        return []

    entities = []
    for part in [item.strip() for item in value.split(",") if item.strip()]:
        lower_part = part.lower()
        if lower_part.startswith("and "):
            part = part[4:].strip()
        elif lower_part.startswith("or "):
            part = part[3:].strip()

        split_made = False
        for separator in (" vs ", " and "):
            if separator in part:
                subparts = [item.strip() for item in part.split(separator) if item.strip()]
                if len(subparts) == 2 and all(len(item.split()) <= 3 for item in subparts):
                    entities.extend(subparts)
                    split_made = True
                break
        if not split_made and part:
            entities.append(part)

    return [entity for entity in entities if normalize_answer(entity)]


def doc_ids_from_value(value: Any) -> list[int]:
    ids: set[int] = set()
    if isinstance(value, int):
        ids.add(value)
    elif isinstance(value, str):
        bracketed_matches = re.findall(r"\[d(\d+)\]", value)
        if bracketed_matches:
            ids.update(int(match) for match in bracketed_matches)
        else:
            match = re.fullmatch(r"d?(\d+)", value.strip())
            if match:
                ids.add(int(match.group(1)))
    elif isinstance(value, list | tuple | set):
        for item in value:
            ids.update(doc_ids_from_value(item))
    return sorted(doc_id for doc_id in ids if doc_id > 0)


def add_entity_citations(
    entity_map: dict[str, dict[str, Any]],
    raw_value: Any,
    doc_ids: list[int],
) -> None:
    if not doc_ids:
        return
    for entity in split_entities(raw_value):
        key = normalize_answer(entity)
        if not key:
            continue
        if key not in entity_map:
            entity_map[key] = {"entity": entity, "doc_ids": []}
        merged_ids = sorted(set(entity_map[key]["doc_ids"]) | set(doc_ids))
        entity_map[key]["doc_ids"] = merged_ids


def extract_entity_citations(item: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entity_map: dict[str, dict[str, Any]] = {}
    for pattern in item.get("cite_pattern") or []:
        branch = pattern.get("branch")

        if branch == "action_atomic":
            doc_ids = doc_ids_from_value(pattern.get("content", ""))
            raw_value = (pattern.get("raw_fact") or {}).get("value", "")
            add_entity_citations(entity_map, raw_value, doc_ids)

        elif branch == "action_convergence":
            for segment_key, segment_doc_ids in (pattern.get("segments") or {}).items():
                parts = str(segment_key).split("++")
                raw_value = parts[1] if len(parts) >= 2 else segment_key
                add_entity_citations(entity_map, raw_value, doc_ids_from_value(segment_doc_ids))

        elif branch == "action_interleaved":
            for segment in pattern.get("segments") or []:
                doc_ids = doc_ids_from_value(segment.get("content", ""))
                raw_value = (segment.get("raw_fact") or {}).get("value", "")
                add_entity_citations(entity_map, raw_value, doc_ids)

        elif branch == "action_composite":
            doc_ids = doc_ids_from_value(pattern.get("doc", ""))
            for fact in pattern.get("factList") or []:
                add_entity_citations(entity_map, fact.get("value", ""), doc_ids)

    return entity_map


def citation_ids_from_pattern(item: dict[str, Any]) -> set[int]:
    ids: set[int] = set()
    for entry in extract_entity_citations(item).values():
        ids.update(entry["doc_ids"])
    return ids


def remaining_docs_exclude_answers(docs: list[dict[str, str]], answers: list[str]) -> bool:
    combined = normalize_answer(" ".join(doc.get("text") or doc.get("content", "") for doc in docs))
    for answer in answers:
        key = normalize_answer(answer)
        if key and key in combined:
            return False
    return True


def support_doc_indices(item: dict[str, Any]) -> set[int]:
    return {doc_id - 1 for doc_id in citation_ids_from_pattern(item) if doc_id > 0}


def interleave_uniformly(base: list[dict[str, Any]], inserts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not inserts:
        return list(base)
    interval = len(base) / len(inserts)
    merged = []
    insert_idx = 0
    for base_idx, sample in enumerate(base):
        merged.append(sample)
        while insert_idx < len(inserts) and (base_idx + 1) >= (insert_idx + 1) * interval:
            merged.append(inserts[insert_idx])
            insert_idx += 1
    while insert_idx < len(inserts):
        merged.append(inserts[insert_idx])
        insert_idx += 1
    return merged
