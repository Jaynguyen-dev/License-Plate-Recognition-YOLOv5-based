"""Rule-based canonical plate validation and conservative corrections."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from .constants import VOCAB
from .label_utils import canonicalize_label


DIGIT_CONFUSIONS = {
    "O": "0",
    "Q": "0",
    "I": "1",
    "L": "1",
    "Z": "2",
    "S": "5",
    "B": "8",
    "G": "6",
}


@dataclass
class PostprocessResult:
    text: str
    low_confidence: bool
    corrected: bool
    matched_pattern: Optional[str] = None


def char_kind(char: str) -> str:
    if char.isdigit():
        return "D"
    if char in VOCAB:
        return "A"
    return "X"


def label_pattern(label: str) -> str:
    return "".join(char_kind(ch) for ch in canonicalize_label(label))


def learn_patterns(labels: Iterable[str], min_count: int = 1) -> List[str]:
    counts = {}
    for label in labels:
        pattern = label_pattern(label)
        if pattern:
            counts[pattern] = counts.get(pattern, 0) + 1
    return sorted([p for p, count in counts.items() if count >= min_count])


def pattern_to_regex(pattern: str) -> re.Pattern[str]:
    parts = []
    for kind in pattern:
        if kind == "D":
            parts.append(r"\d")
        elif kind == "A":
            parts.append(r"[A-ZĐ]")
        else:
            parts.append(r".")
    return re.compile("^" + "".join(parts) + "$")


class PlateValidator:
    def __init__(self, patterns: Sequence[str]):
        self.patterns = list(patterns)
        self._regexes = [(pattern, pattern_to_regex(pattern)) for pattern in self.patterns]

    @classmethod
    def from_labels(cls, labels: Iterable[str], min_count: int = 1) -> "PlateValidator":
        return cls(learn_patterns(labels, min_count=min_count))

    def match(self, text: str) -> Optional[str]:
        text = canonicalize_label(text)
        for pattern, regex in self._regexes:
            if regex.match(text):
                return pattern
        return None

    def correct(self, text: str) -> PostprocessResult:
        text = canonicalize_label(text)
        matched = self.match(text)
        if matched is not None:
            return PostprocessResult(text=text, low_confidence=False, corrected=False, matched_pattern=matched)

        for pattern in self.patterns:
            if len(pattern) != len(text):
                continue
            chars = list(text)
            changed = False
            for idx, kind in enumerate(pattern):
                if kind == "D" and chars[idx] in DIGIT_CONFUSIONS:
                    chars[idx] = DIGIT_CONFUSIONS[chars[idx]]
                    changed = True
            if not changed:
                continue
            candidate = "".join(chars)
            if pattern_to_regex(pattern).match(candidate):
                return PostprocessResult(
                    text=candidate,
                    low_confidence=False,
                    corrected=True,
                    matched_pattern=pattern,
                )

        return PostprocessResult(text=text, low_confidence=True, corrected=False, matched_pattern=None)
