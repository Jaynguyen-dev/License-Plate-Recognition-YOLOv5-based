"""OCR evaluation metrics."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(
                min(
                    prev[j] + 1,
                    cur[j - 1] + 1,
                    prev[j - 1] + (0 if ca == cb else 1),
                )
            )
        prev = cur
    return prev[-1]


def exact_match_rate(pairs: Sequence[Tuple[str, str]]) -> float:
    if not pairs:
        return 0.0
    return sum(1 for pred, target in pairs if pred == target) / float(len(pairs))


def character_error_rate(pairs: Sequence[Tuple[str, str]]) -> float:
    total_chars = sum(len(target) for _, target in pairs)
    if total_chars == 0:
        return 0.0
    return sum(levenshtein(pred, target) for pred, target in pairs) / float(total_chars)


def summarize_pairs(pairs: Sequence[Tuple[str, str]]) -> Dict[str, float]:
    return {
        "count": float(len(pairs)),
        "exact_match": exact_match_rate(pairs),
        "cer": character_error_rate(pairs),
    }


def grouped_summary(rows: Iterable[Dict[str, str]], group_field: str = "layout") -> List[Dict[str, str]]:
    groups: Dict[str, List[Tuple[str, str]]] = {}
    for row in rows:
        group = row.get(group_field, "")
        groups.setdefault(group, []).append((row.get("prediction", ""), row.get("target", "")))
    out = []
    for group, pairs in sorted(groups.items()):
        metrics = summarize_pairs(pairs)
        out.append(
            {
                group_field: group,
                "count": str(int(metrics["count"])),
                "exact_match": f"{metrics['exact_match']:.6f}",
                "cer": f"{metrics['cer']:.6f}",
            }
        )
    return out
