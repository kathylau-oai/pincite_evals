from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

import yaml

from .schema import DatasetItem, Packet


def load_packet(path: str | Path) -> Packet:
    data = yaml.safe_load(Path(path).read_text())
    return Packet.model_validate(data)


def iter_dataset(path: str | Path) -> Iterator[DatasetItem]:
    p = Path(path)
    with p.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield DatasetItem.model_validate(json.loads(line))
