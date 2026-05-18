import re
from typing import Tuple


def canonicalize_label(s: str) -> str:
    if s is None:
        return ""
    s = s.strip().upper()
    return re.sub(r"[\s.\-]+", "", s)


def split_two_line(raw_label: str) -> Tuple[str, str]:
    if not raw_label or raw_label.strip() == "":
        return "", ""
    parts = raw_label.strip().split(None, 1)
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[1]


def split_two_line_canonical(raw_label: str) -> Tuple[str, str]:
    top, bottom = split_two_line(raw_label)
    return canonicalize_label(top), canonicalize_label(bottom)


def source_family(source: str, image_name: str) -> str:
    stem = image_name.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    if "_" in stem:
        return f"{source}:{stem.split('_', 1)[0]}"
    return f"{source}:{stem}"


def split_key_from_name(image_name: str) -> str:
    stem = image_name.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    if stem.endswith("_lp"):
        stem = stem[:-3]
    return stem
