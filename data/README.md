# Data Directory

This folder contains local project data organized by processing stage.

## `raw/`
stores the original, unmodified input data.

- **In-repo sample:** A small subset (N images per class) for quick tests
- **Full dataset:** Lives externally, referenced via `DATA_PATH` environment variable
- Never edit these files in place
## `processed/`
stores data generated from `raw/` after deterministic transformations.

Examples:
- denoised images
- resized/normalized variants

Guidelines:
- Keep each transformation variant in a clearly named subfolder (e.g. `denoise_v1`, `gaussian_sigma0.1`)

## Notes
- Small sample data may be stored in-repo for quick tests.
- Large/full datasets should not be committed to Git.
- Never overwrite files in `raw/`; write all derived outputs to `processed/`.