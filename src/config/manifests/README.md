# Model Manifests

This directory contains manifest files for different Z-Image model variants.

## Purpose

Manifest files list all required files for each model, optionally with MD5 checksums for integrity verification.

## File Naming Convention

- `z-image-turbo.txt` - Z-Image Turbo model
- Custom models: `{model-name}.txt`

## Format

### Standard Format (with MD5 - Recommended)

```txt
# Z-Image Model Manifest
# Format: <md5hash>  <filepath>
# Generated automatically - DO NOT edit manually

5e3226ed72a9a4a080f2a4ca78b98ddc  model_index.json
ca682fcc6c5a94cf726b7187e64b9411  scheduler/scheduler_config.json
1e97eb35d9d0b6aa60c58a8df8d7d99a  text_encoder/config.json
30b85686b9a9b002e012494fadc027cb  text_encoder/model-00001-of-00003.safetensors
...
```

**Verification Behavior:**
- `verify=False`: Default, only checks file existence, ignores MD5 (fast)
- `verify=True`: Checks existence AND verifies MD5 checksums (thorough)

## Usage

The manifest file is automatically selected based on the model directory name:

```python
# Auto-detects manifest from "Z-Image-Turbo" -> uses z-image-turbo.txt
model_path = ensure_model_weights("ckpts/Z-Image-Turbo")

# Explicit manifest
model_path = ensure_model_weights("ckpts/Z-Image-Turbo", manifest_name="z-image-turbo.txt")
```

## Generating Manifests

Use the provided tool to generate manifests:

```bash
# Generate with MD5 checksums (auto-saves to this directory)
python -m src.tools.generate_manifest ckpts/Z-Image-Turbo

# Generate without checksums (faster, not recommended)
python -m src.tools.generate_manifest ckpts/Z-Image-Turbo --no-checksums

# With verbose output
python -m src.tools.generate_manifest ckpts/Z-Image-Turbo --verbose

# Custom output path
python -m src.tools.generate_manifest ckpts/Z-Image-Turbo --output custom.txt
```

## Available Manifests

- **z-image-turbo.txt** - Z-Image Turbo model
