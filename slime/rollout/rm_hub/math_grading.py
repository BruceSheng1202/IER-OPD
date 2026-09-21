"""Shared boxed-answer grading and benchmark-specific equivalent answers."""

from functools import lru_cache
import json
from pathlib import Path
from typing import Any


@lru_cache(maxsize=1)
def load_equivalent_answers() -> dict:
    path = Path(__file__).with_name("hmmt_equivalent_answers.json")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def equivalent_answers(metadata: dict[str, Any] | None) -> list[str]:
    """Resolve both top-level and nested dataset question identifiers."""
    metadata = metadata if isinstance(metadata, dict) else {}
    task = load_equivalent_answers().get(metadata.get("data_source", ""), {})
    index = metadata.get("index")
    if index is None and isinstance(metadata.get("extra_info"), dict):
        index = metadata["extra_info"].get("index")
    question = str(index).rsplit("-", 1)[-1] if index is not None else ""
    answers = []
    for key in (f"Q{question}", question):
        for entry in task.get(key, {}).get("equivalents", []):
            if isinstance(entry, dict) and "answer" in entry:
                answers.append(entry["answer"])
    return answers


def grade_math_answer(response: str, label: Any, metadata: dict | None = None) -> int:
    """Use the same final-box extraction and symbolic verifier as training eval."""
    from .math_utils import grade_answer_verl

    if grade_answer_verl(response, label):
        return 1
    return int(any(grade_answer_verl(response, answer) for answer in equivalent_answers(metadata)))
