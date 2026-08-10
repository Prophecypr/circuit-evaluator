"""CRNN-v2 architecture and CTC decoding."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch
from torch import nn


class CRNNv2(nn.Module):
    """CNN feature extractor followed by two bidirectional LSTM layers."""

    def __init__(self, num_classes: int, hidden_size: int = 256) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.hidden_size = int(hidden_size)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.AdaptiveAvgPool2d((1, None)),
        )
        self.rnn = nn.LSTM(
            512,
            self.hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )
        self.fc = nn.Linear(self.hidden_size * 2, self.num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.cnn(images)
        batch, channels, height, width = features.shape
        sequence = features.reshape(batch, channels * height, width).permute(0, 2, 1)
        sequence, _ = self.rnn(sequence)
        return self.fc(sequence)


def greedy_decode(indices: Iterable[int], index_to_char: Mapping[int, str]) -> str:
    """Collapse repeats and CTC blanks (index zero)."""
    result: list[str] = []
    previous = 0
    for raw_index in indices:
        index = int(raw_index)
        if index != 0 and index != previous:
            result.append(index_to_char.get(index, ""))
        previous = index
    return "".join(result)


def decode_logits(logits: torch.Tensor, chars: str) -> list[str]:
    """Greedy-decode a ``[B,T,C]`` logits tensor."""
    index_to_char = {index + 1: char for index, char in enumerate(chars)}
    best = logits.argmax(dim=-1).detach().cpu().tolist()
    return [greedy_decode(indices, index_to_char) for indices in best]
