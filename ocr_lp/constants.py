"""Shared constants for the license-plate OCR pipeline."""

BLANK_INDEX = 0
VOCAB = "0123456789ABCDEFGHIJKLMNOPQRSTUVXYZĐ"
IDX_TO_CHAR = ["<blank>"] + list(VOCAB)
CHAR_TO_IDX = {char: idx for idx, char in enumerate(IDX_TO_CHAR)}
LAYOUT_TO_IDX = {"one_line": 0, "two_line": 1}
IDX_TO_LAYOUT = {idx: label for label, idx in LAYOUT_TO_IDX.items()}
