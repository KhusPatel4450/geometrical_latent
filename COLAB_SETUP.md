# Colab Training Setup

## Every Session (Cells 1-3)

### Cell 1: Mount Drive & Environment
```python
import os
from google.colab import drive
drive.mount('/content/drive')

os.environ["DATASET_DIR"] = "/content/drive/MyDrive/temporal_straightening_data"

# Cache dinov2 so checkpoints can be loaded
import torch
torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=False)
```

### Cell 2: Install Dependencies
```python
!pip install hydra-core==1.3.2 omegaconf wandb accelerate torchjd einops
```

### Cell 3: Clone Repo (first time only)
```python
!git clone https://github.com/<your-repo>/geometrical_latent /content/geometrical_latent
```

---

## Running Experiments

All commands resume automatically from checkpoint if one exists in the `hydra.run.dir`.
Prepend `PYTHONPATH="/root/.cache/torch/hub/facebookresearch_dinov2_main:$PYTHONPATH"` to every run command so checkpoints load correctly.

### MOO: AMTL min
```python
!cd /content/geometrical_latent && PYTHONPATH="/root/.cache/torch/hub/facebookresearch_dinov2_main:$PYTHONPATH" python train.py training.moo.enabled=true training.moo.algorithm=amtl training.moo.amtl_scale_mode=min 'training.moo.objectives=[prediction,straightening,second_difference]' 'hydra.run.dir=/content/drive/MyDrive/temporal_straightening_data/checkpoints/moo_amtl_min'
```

### MOO: AMTL median
```python
!cd /content/geometrical_latent && PYTHONPATH="/root/.cache/torch/hub/facebookresearch_dinov2_main:$PYTHONPATH" python train.py training.moo.enabled=true training.moo.algorithm=amtl training.moo.amtl_scale_mode=median 'training.moo.objectives=[prediction,straightening,second_difference]' 'hydra.run.dir=/content/drive/MyDrive/temporal_straightening_data/checkpoints/moo_amtl_median'
```

### MOO: AMTL rmse
```python
!cd /content/geometrical_latent && PYTHONPATH="/root/.cache/torch/hub/facebookresearch_dinov2_main:$PYTHONPATH" python train.py training.moo.enabled=true training.moo.algorithm=amtl training.moo.amtl_scale_mode=rmse 'training.moo.objectives=[prediction,straightening,second_difference]' 'hydra.run.dir=/content/drive/MyDrive/temporal_straightening_data/checkpoints/moo_amtl_rmse'
```

### MOO: UPGrad
```python
!cd /content/geometrical_latent && PYTHONPATH="/root/.cache/torch/hub/facebookresearch_dinov2_main:$PYTHONPATH" python train.py training.moo.enabled=true training.moo.algorithm=upgrad 'training.moo.objectives=[prediction,straightening,second_difference]' 'hydra.run.dir=/content/drive/MyDrive/temporal_straightening_data/checkpoints/moo_upgrad'
```

### MOO: CAGrad
```python
!cd /content/geometrical_latent && PYTHONPATH="/root/.cache/torch/hub/facebookresearch_dinov2_main:$PYTHONPATH" python train.py training.moo.enabled=true training.moo.algorithm=cagrad 'training.moo.objectives=[prediction,straightening,second_difference]' 'hydra.run.dir=/content/drive/MyDrive/temporal_straightening_data/checkpoints/moo_cagrad'
```

### Baseline: Temporal Straightening
```python
!cd /content/geometrical_latent && PYTHONPATH="/root/.cache/torch/hub/facebookresearch_dinov2_main:$PYTHONPATH" python train.py training.straighten=cos1e-1 'hydra.run.dir=/content/drive/MyDrive/temporal_straightening_data/checkpoints/temporal_straightening'
```

### Baseline: Second Difference
```python
!cd /content/geometrical_latent && PYTHONPATH="/root/.cache/torch/hub/facebookresearch_dinov2_main:$PYTHONPATH" python train.py training.straighten=2nd1e-1 'hydra.run.dir=/content/drive/MyDrive/temporal_straightening_data/checkpoints/second_difference'
```

### Baseline: Distance Consistency
```python
!cd /content/geometrical_latent && PYTHONPATH="/root/.cache/torch/hub/facebookresearch_dinov2_main:$PYTHONPATH" python train.py training.straighten=dist1e-1 'hydra.run.dir=/content/drive/MyDrive/temporal_straightening_data/checkpoints/distance_consistency'
```

### Combined Manual Regularizers
```python
!cd /content/geometrical_latent && PYTHONPATH="/root/.cache/torch/hub/facebookresearch_dinov2_main:$PYTHONPATH" python train.py training.regularizers.straightening.enabled=true training.regularizers.straightening.lambda=0.1 training.regularizers.second_difference.enabled=true training.regularizers.second_difference.lambda=0.1 'hydra.run.dir=/content/drive/MyDrive/temporal_straightening_data/checkpoints/combined_manual'
```

---

## Running Evaluation (MPC)

### Eval: AMTL min (MPC)
```python
!cd /content/geometrical_latent && PYTHONPATH="/root/.cache/torch/hub/facebookresearch_dinov2_main:$PYTHONPATH" python plan.py \
  --config-name plan_gd_mpc \
  ckpt_base_path=/content/drive/MyDrive/temporal_straightening_data/checkpoints/moo_amtl_min/checkpoints \
  model_name=moo_amtl_min_f5_h3_p1 \
  'hydra.run.dir=/content/drive/MyDrive/temporal_straightening_data/checkpoints/moo_amtl_min/eval_mpc'
```

For other experiments swap `moo_amtl_min` with the experiment name (e.g. `moo_amtl_median`, `moo_upgrad`, etc.) in both `ckpt_base_path` and `model_name`.

---

## Notes

- Checkpoints save to Google Drive automatically — they persist across Colab resets
- Training resumes from the last saved epoch automatically (no extra flags needed)
- If you get `ModuleNotFoundError: No module named 'dinov2'` — Cell 1 fixes this via `PYTHONPATH`
- If you get `weights_only` pickle error — the checkpoint is from an incompatible environment; delete it and restart
- `~46 min/epoch` on A100, `~20 epochs` total per experiment
