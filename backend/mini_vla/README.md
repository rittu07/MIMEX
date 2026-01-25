
# Mini-VLA

A minimal implementation of a Vision-Language-Action (VLA) model from scratch.

## Structure
- `models/`: Contains the VLA architecture (Encoders, Fusion, Diffusion Head, Policy).
- `scripts/`: Scripts for data collection, training, and testing.

## Usage

### 1. Collect Data
Requires `metaworld`. Use `--mock` to generate dummy data for testing without `metaworld`.
```bash
python -m backend.mini_vla.scripts.collect_data --env-name push-v3 --episodes 10 --mock --output-path data/push_v3.npz
```

### 2. Train
Train the VLA on the collected dataset.
```bash
python -m backend.mini_vla.scripts.train --dataset-path data/push_v3.npz --epochs 50 --batch-size 64
```

### 3. Test
Test the trained model.
```bash
python -m backend.mini_vla.scripts.test --checkpoint checkpoints/model.pt --env-name push-v3 --save-video
```
