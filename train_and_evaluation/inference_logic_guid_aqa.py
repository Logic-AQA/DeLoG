"""
Compatibility shim: original inference_logic_guid_aqa.py was renamed/merged into utils_qa.py.
This module re-exports the functions expected by run_inference.py.
"""
from utils_qa import (
    SYMBOLIC_REFUSAL_ANSWER,
    compute_cite_pattern,
    generate_long_answer_for_item,
)

__all__ = [
    "SYMBOLIC_REFUSAL_ANSWER",
    "compute_cite_pattern",
    "generate_long_answer_for_item",
]
