from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


class COCODataLoader:
    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.annotations_dir = self.data_dir / "annotations"
        self.images_dir = {
            "train": self.data_dir / "train2014",
            "val": self.data_dir / "val2014",
        }

    def _annotation_file(self, split: str) -> Path:
        split = split.lower()
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")
        return self.annotations_dir / f"captions_{split}2014.json"

    def load_annotations(self, split: str = "train") -> pd.DataFrame:
        annotation_file = self._annotation_file(split)
        with annotation_file.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        images = {image["id"]: image["file_name"] for image in payload.get("images", [])}
        records = []

        for annotation in payload.get("annotations", []):
            image_id = annotation["image_id"]
            file_name = images.get(image_id)
            records.append(
                {
                    "image_id": image_id,
                    "file_name": file_name,
                    "image_path": self.images_dir[split.lower()] / file_name if file_name else None,
                    "caption": annotation["caption"],
                }
            )

        return pd.DataFrame(records)

    def load_split(self, split: str = "train") -> pd.DataFrame:
        return self.load_annotations(split=split)
