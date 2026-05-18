# Leyenda

## Setup (Python)

1. Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

2. Create a .env file in the project root with your local data path:

```env
DATA_PATH=C:\path\to\your\data
```

## Data

The project includes a small sample dataset in `data/raw/` for quick tests and development.

For the full dataset refer to Setup (2.) and see `data/README.md` for details.