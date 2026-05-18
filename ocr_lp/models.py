"""PyTorch model definitions for layout classification and CRNN OCR."""

from __future__ import annotations

from .constants import IDX_TO_CHAR
from .torch_utils import require_torch


def _nn():
    torch = require_torch()
    return torch, torch.nn


class LayoutCNN:
    def __new__(cls, num_classes: int = 2):
        torch, nn = _nn()

        class _LayoutCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    nn.Conv2d(128, 192, kernel_size=3, padding=1),
                    nn.BatchNorm2d(192),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((1, 1)),
                )
                self.classifier = nn.Linear(192, num_classes)

            def forward(self, x):
                x = self.features(x)
                x = x.flatten(1)
                return self.classifier(x)

        return _LayoutCNN()


class CRNN:
    def __new__(cls, num_classes: int = len(IDX_TO_CHAR), hidden_size: int = 256):
        torch, nn = _nn()

        class _CRNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.cnn = nn.Sequential(
                    nn.Conv2d(1, 64, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(128, 256, 3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d((2, 1), (2, 1)),
                    nn.Conv2d(256, 512, 3, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(512, 512, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d((2, 1), (2, 1)),
                    nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(0, 1)),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                )
                self.rnn = nn.LSTM(
                    input_size=512,
                    hidden_size=hidden_size,
                    num_layers=2,
                    bidirectional=True,
                    dropout=0.1,
                )
                self.classifier = nn.Linear(hidden_size * 2, num_classes)

            def forward(self, x):
                features = self.cnn(x)
                if features.shape[2] != 1:
                    features = features.mean(dim=2, keepdim=True)
                features = features.squeeze(2).permute(2, 0, 1)
                sequence, _ = self.rnn(features)
                logits = self.classifier(sequence)
                return logits.log_softmax(dim=2)

            @staticmethod
            def feature_lengths(widths):
                return torch.clamp(widths // 4, min=1)

        return _CRNN()
