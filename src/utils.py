"""Consolidated utilities for data loading, image processing, and preprocessing"""

from PIL import Image
from src.config import Config
from pathlib import Path
from shutil import copy2

class DataLoader:
    """Load and manage datasets"""
    @staticmethod
    def load_images(mode="sample", extensions=(".png", ".jpg", ".jpeg")):
        root = Config.get_raw_data_path(mode)
        if not root.exists():
            raise FileNotFoundError(f"Data folder not found: {root}")

        images = []
        for path in root.rglob("*"):
            if path.suffix.lower() in extensions:
                images.append(Image.open(path))
        return images

def build_sample_raw_dataset(source_root, target_root, images_per_class=10, extensions=(".png", ".jpg", ".jpeg")):
    source_root = Path(source_root)
    target_root = Path(target_root)
    target_root.mkdir(parents=True, exist_ok=True)

    for class_dir in source_root.iterdir():
        if not class_dir.is_dir():
            continue

        dest_class_dir = target_root / class_dir.name
        dest_class_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for file_path in sorted(class_dir.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                copy2(file_path, dest_class_dir / file_path.name)
                count += 1
                if count >= images_per_class:
                    break


class ImageUtils:
    """Image processing operations (load, normalize, augment, denoise)"""
    pass


class PreprocessingPipeline:
    """Image preprocessing pipeline"""
    pass