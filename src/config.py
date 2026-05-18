from pathlib import Path

class Config:
    BASE_DIR = Path(__file__).parent.parent.parent
    DATA_PATH = BASE_DIR / 'data'
    MODELS_PATH = BASE_DIR / 'models'
    RESULTS_PATH = BASE_DIR / 'reports'
    RANDOM_SEED = 42
