import os
from pathlib import Path
from dotenv import load_dotenv

class Config:
    load_dotenv()
    BASE_DIR = Path(__file__).resolve().parent.parent

    DATA_PATH = Path(os.getenv("DATA_PATH") or BASE_DIR / "data")
    MODELS_PATH = BASE_DIR / "src/models"
    RESULTS_PATH = BASE_DIR / "notebooks/figures"

    RANDOM_SEED = 42

    @staticmethod
    def get_raw_data_path(mode="full"):
        if mode == "full":
            return Config.DATA_PATH / "raw"
        elif mode == "sample":
            return Config.BASE_DIR / "data" / "raw"
        else:
            raise ValueError(f"Unknown mode: {mode}")