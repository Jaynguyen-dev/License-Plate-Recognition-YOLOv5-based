"""CTC vocabulary encoding and greedy decoding."""

from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np

from .constants import BLANK_INDEX, CHAR_TO_IDX, IDX_TO_CHAR
from .label_utils import canonicalize_label


def encode_label(label: str) -> List[int]:
    encoded = []
    for char in canonicalize_label(label):
        if char not in CHAR_TO_IDX:
            raise ValueError(f"Character {char!r} is not in the OCR vocabulary")
        encoded.append(CHAR_TO_IDX[char])
    return encoded


def decode_indices(indices: Iterable[int], blank: int = BLANK_INDEX) -> str:
    chars = []
    prev = None
    for idx in indices:
        idx = int(idx)
        if idx != blank and idx != prev:
            chars.append(IDX_TO_CHAR[idx])
        prev = idx
    return "".join(chars)


def greedy_decode(logits_or_indices, blank: int = BLANK_INDEX) -> List[str]:
    """Decode CTC emissions.

    Accepts either a sequence of class indices with shape ``T`` or ``T x B`` or
    emissions with shape ``T x B x C`` / ``B x T x C``. The output is always a
    list, one decoded string per batch item.
    """

    values = logits_or_indices
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    else:
        values = np.asarray(values)

    if values.ndim == 1:
        return [decode_indices(values, blank=blank)]
    if values.ndim == 2:
        # Treat T x B as class indices when values look integer-like.
        if np.issubdtype(values.dtype, np.integer):
            return [decode_indices(values[:, b], blank=blank) for b in range(values.shape[1])]
        return [decode_indices(np.argmax(values, axis=1), blank=blank)]
    if values.ndim == 3:
        # Common PyTorch CTC convention is T x B x C.
        if values.shape[1] <= values.shape[0]:
            best = np.argmax(values, axis=2)
            return [decode_indices(best[:, b], blank=blank) for b in range(best.shape[1])]
        best = np.argmax(values, axis=2)
        return [decode_indices(best[b, :], blank=blank) for b in range(best.shape[0])]
    raise ValueError(f"Unsupported CTC tensor shape: {values.shape}")


def flatten_targets(labels: Sequence[str]) -> List[int]:
    out: List[int] = []
    for label in labels:
        out.extend(encode_label(label))
    return out
