# Data Directory

This folder contains local project data organized by processing stage.

## `raw/`
`raw/` stores the original, unmodified input data.

- Source-of-truth and must remain immutable (do not edit in place)
- If the full dataset is large, it can live outside the repository and be referenced via `DATA_PATH`

## `processed/`
`processed/` stores data generated from `raw/` after deterministic transformations.

Examples:
- denoised images
- resized/normalized variants

Guidelines:
- Keep each transformation variant in a clearly named subfolder (e.g. `denoise_v1`, `gaussian_sigma0.1`)

## Notes
- Small sample data may be stored in-repo for quick tests.
- Large/full datasets should not be committed to Git.
- Never overwrite files in `raw/`; write all derived outputs to `processed/`.