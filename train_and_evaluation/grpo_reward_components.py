from typing import Dict, List, Sequence


REFUSAL_FLAG = "I apologize, but I couldn't find an answer to your question in the search results."


def is_refusal_answer(text: str) -> bool:
    if not isinstance(text, str):
        return False
    stripped = text.strip()
    return "[FALSE]" in stripped or stripped == REFUSAL_FLAG


def answerability_reward(
    *,
    is_answerable: bool,
    refused: bool,
    answerable_answer_reward: float = 1.0,
    answerable_refusal_penalty: float = -2.0,
    unanswerable_refusal_reward: float = 0.5,
    unanswerable_answer_penalty: float = -1.0,
) -> float:
    if is_answerable:
        return answerable_refusal_penalty if refused else answerable_answer_reward
    return unanswerable_refusal_reward if refused else unanswerable_answer_penalty


def classify_pareto_route(*, is_answerable: bool, refused: bool) -> str:
    if is_answerable and refused:
        return "answerable_refusal"
    if is_answerable and not refused:
        return "answerable_answer"
    if not is_answerable and refused:
        return "unanswerable_refusal"
    return "unanswerable_answer"


def pareto_gate_reward(
    route: str,
    *,
    qa_reward: float,
    cite_rec: float,
    cite_prec: float,
    hard_penalty: float = -1.0,
    qa_missing_penalty: float = -0.75,
    citation_missing_penalty: float = -0.5,
    pass_reward: float = 0.5,
) -> float:
    if route in {"answerable_refusal", "unanswerable_answer"}:
        return hard_penalty
    if route == "unanswerable_refusal":
        return pass_reward
    if route != "answerable_answer":
        raise ValueError(f"unknown pareto route: {route}")

    if float(qa_reward) <= 0.0:
        return qa_missing_penalty
    if float(cite_rec) <= 0.0 and float(cite_prec) <= 0.0:
        return citation_missing_penalty
    return pass_reward


def group_minmax(values: Sequence[float], group_size: int) -> List[float]:
    float_values = [float(value) for value in values]
    if _invalid_group_shape(float_values, group_size):
        return float_values

    normalized = []
    for start in range(0, len(float_values), group_size):
        normalized.extend(_normalize_group(float_values[start : start + group_size]))
    return normalized


def group_minmax_masked(
    values: Sequence[float],
    group_size: int,
    mask: Sequence[bool],
) -> List[float]:
    float_values = [float(value) for value in values]
    if _invalid_group_shape(float_values, group_size) or len(float_values) != len(mask):
        return float_values

    normalized = [0.0] * len(float_values)
    bool_mask = [bool(item) for item in mask]
    for start in range(0, len(float_values), group_size):
        end = start + group_size
        valid_indexes = [idx for idx in range(start, end) if bool_mask[idx]]
        valid_values = [float_values[idx] for idx in valid_indexes]
        valid_normalized = _normalize_group(valid_values)
        for idx, value in zip(valid_indexes, valid_normalized):
            normalized[idx] = value
    return normalized


def length_reward(
    text: str,
    *,
    refused: bool,
    answer_target_words: int = 120,
    answer_max_words: int = 220,
) -> float:
    del refused
    word_count = len(str(text).split())
    if word_count <= answer_target_words:
        return 1.0
    if word_count >= answer_max_words:
        return -1.0

    span = answer_max_words - answer_target_words
    if span <= 0:
        return -1.0
    return 1.0 - (2.0 * (word_count - answer_target_words) / span)


def weighted_sum(
    components: Dict[str, Sequence[float]],
    weights: Dict[str, float],
) -> List[float]:
    lengths = {len(values) for values in components.values()}
    if not lengths:
        return []
    if len(lengths) != 1:
        raise ValueError("component lengths differ")

    count = lengths.pop()
    totals = [0.0] * count
    for name, weight in weights.items():
        if name not in components:
            continue
        for idx, value in enumerate(components[name]):
            totals[idx] += float(weight) * float(value)
    return [round(total, 12) for total in totals]


def update_constrained_pareto_weights(
    weights: Dict[str, float],
    *,
    metrics: Dict[str, float],
    targets: Dict[str, float],
) -> Dict[str, float]:
    updated = {name: float(value) for name, value in weights.items()}

    if metrics.get("regular_citation_f1", 0.0) < targets.get("regular_citation_f1", 0.0):
        updated["citation_rec"] = updated.get("citation_rec", 0.0) + 0.03
        updated["citation_prec"] = updated.get("citation_prec", 0.0) + 0.02
        updated["refusal"] = updated.get("refusal", 0.0) - 0.02

    refusal_low = (
        metrics.get("refuse_f1", 100.0) < targets.get("refuse_f1", 0.0)
        or metrics.get("refuse_prec", 100.0) < targets.get("refuse_prec", 0.0)
    )
    answered_ratio = metrics.get("answered_ratio", 0.0)
    if refusal_low and answered_ratio > targets.get("answered_ratio_max", 100.0):
        updated["refusal"] = updated.get("refusal", 0.0) + 0.05
        updated["qa"] = updated.get("qa", 0.0) - 0.02

    if (
        metrics.get("answerable_f1", 100.0) < targets.get("answerable_f1", 0.0)
        or metrics.get("calib_claims_nli_f1", 100.0) < targets.get("calib_claims_nli_f1", 0.0)
    ):
        updated["qa"] = updated.get("qa", 0.0) + 0.05
        updated["refusal"] = updated.get("refusal", 0.0) - 0.03

    if answered_ratio < targets.get("answered_ratio_min", 0.0):
        updated["refusal"] = updated.get("refusal", 0.0) - 0.05
        updated["qa"] = updated.get("qa", 0.0) + 0.03
    elif answered_ratio > targets.get("answered_ratio_max", 100.0):
        updated["refusal"] = updated.get("refusal", 0.0) + 0.05

    return _normalize_constrained_weights(_clamp_constrained_weights(updated))


def _clamp_constrained_weights(weights: Dict[str, float]) -> Dict[str, float]:
    clamped = dict(weights)
    clamped["qa"] = min(0.60, max(0.35, clamped.get("qa", 0.0)))
    clamped["refusal"] = min(0.35, max(0.10, clamped.get("refusal", 0.0)))
    clamped["logic"] = min(0.06, max(0.02, clamped.get("logic", 0.0)))
    clamped["length"] = min(0.04, max(0.01, clamped.get("length", 0.0)))
    clamped["citation_rec"] = max(0.0, clamped.get("citation_rec", 0.0))
    clamped["citation_prec"] = max(0.0, clamped.get("citation_prec", 0.0))

    citation_total = clamped.get("citation_rec", 0.0) + clamped.get("citation_prec", 0.0)
    if citation_total <= 0.0:
        clamped["citation_rec"] = 0.12
        clamped["citation_prec"] = 0.08
    else:
        target_total = min(0.40, max(0.20, citation_total))
        rec_share = clamped.get("citation_rec", 0.0) / citation_total
        clamped["citation_rec"] = target_total * rec_share
        clamped["citation_prec"] = target_total * (1.0 - rec_share)
    return clamped


def _normalize_constrained_weights(weights: Dict[str, float]) -> Dict[str, float]:
    citation_total = weights.get("citation_rec", 0.0) + weights.get("citation_prec", 0.0)
    rec_share = weights.get("citation_rec", 0.0) / citation_total if citation_total > 0.0 else 0.6
    grouped = {
        "qa": float(weights.get("qa", 0.45)),
        "citation": float(citation_total),
        "refusal": float(weights.get("refusal", 0.20)),
        "logic": float(weights.get("logic", 0.03)),
        "length": float(weights.get("length", 0.02)),
    }
    bounds = {
        "qa": (0.35, 0.60),
        "citation": (0.20, 0.40),
        "refusal": (0.10, 0.35),
        "logic": (0.02, 0.06),
        "length": (0.01, 0.04),
    }

    for name, (lower, upper) in bounds.items():
        grouped[name] = min(upper, max(lower, grouped[name]))

    total = sum(grouped.values())
    if total > 1.0:
        excess = total - 1.0
        capacity = {
            name: grouped[name] - bounds[name][0]
            for name in grouped
            if grouped[name] > bounds[name][0]
        }
        capacity_total = sum(capacity.values())
        if capacity_total > 0.0:
            for name, room in capacity.items():
                grouped[name] -= excess * room / capacity_total
    elif total < 1.0:
        deficit = 1.0 - total
        capacity = {
            name: bounds[name][1] - grouped[name]
            for name in grouped
            if grouped[name] < bounds[name][1]
        }
        capacity_total = sum(capacity.values())
        if capacity_total > 0.0:
            for name, room in capacity.items():
                grouped[name] += deficit * room / capacity_total

    return {
        "qa": grouped["qa"],
        "citation_rec": grouped["citation"] * rec_share,
        "citation_prec": grouped["citation"] * (1.0 - rec_share),
        "refusal": grouped["refusal"],
        "logic": grouped["logic"],
        "length": grouped["length"],
    }


def _invalid_group_shape(values: Sequence[float], group_size: int) -> bool:
    return group_size <= 0 or not values or len(values) % group_size != 0


def _normalize_group(values: Sequence[float]) -> List[float]:
    if not values:
        return []

    group_min = min(values)
    group_max = max(values)
    if group_max - group_min <= 1e-6:
        return [0.0] * len(values)

    scale = group_max - group_min
    return [((value - group_min) / scale) * 2.0 - 1.0 for value in values]
