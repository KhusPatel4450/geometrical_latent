# Geometry-Aware Latent Space Regularization for Planning

**Thesis:** Comparing geometry-aware regularization methods for latent planning in reinforcement learning.

**Author:** Khush Patel  
**Mentor:** Rasa Khosrowshahli  
**Supervisor:** Beatrice Ombuki-Berman  
**Date:** January 2026

---

## Overview

This repository contains the experimental code for a comparative study of geometry-aware latent space regularizers. Building on the temporal straightening framework from [Wang et al., 2026](https://arxiv.org/abs/2603.12231), we implement and compare multiple regularization objectives to understand which geometric properties most benefit downstream planning.

### Research Questions

1. How does latent-space geometry influence planning performance in model-based RL?
2. How does temporal straightening compare with alternative geometry-aware regularization strategies?
3. In what task settings does local straightening improve planning, and where does it become insufficient?

---

## Compared Regularizers

| Regularizer | Config | Geometric Property |
|-------------|--------|-------------------|
| Prediction-only baseline | `training.straighten=False` | None |
| VCReg baseline | `training.vcreg=True` | None (non-geometric) |
| Temporal straightening | `training.straighten=cos1e-1` | Directional alignment |
| Second-difference smoothness | `training.straighten=2nd1e-1` | Acceleration magnitude |
| Distance consistency | `training.straighten=dist1e-1` | Step-size uniformity |

---

## Installation

```bash
git clone https://github.com/KhusPatel4450/geometrical_latent.git
cd geometrical_latent
conda env create -f environment.yaml
conda activate ts
```

### Dataset

Download the PointMaze dataset from [OSF](https://osf.io/bmw48/) and set the environment variable:

```bash
export DATASET_DIR=/path/to/data
```

---

## Training

### Base command

```bash
python train.py env=point_maze encoder=scratch_resnet training.epochs=20 training.batch_size=16
```

### Running all experimental conditions

```bash
# 1. Prediction-only baseline
python train.py env=point_maze encoder=scratch_resnet training.straighten=False training.vcreg=False

# 2. VCReg baseline
python train.py env=point_maze encoder=scratch_resnet training.straighten=False training.vcreg=True training.vcreg_std_coeff=0.1 training.vcreg_cov_coeff=0.1

# 3. Temporal straightening
python train.py env=point_maze encoder=scratch_resnet training.straighten=cos1e-1

# 4. Second-difference smoothness (NEW)
python train.py env=point_maze encoder=scratch_resnet training.straighten=2nd1e-1

# 5. Distance consistency (NEW)
python train.py env=point_maze encoder=scratch_resnet training.straighten=dist1e-1
```

### Encoder options

- `encoder=scratch_resnet` — ResNet trained from scratch (faster)
- `encoder=dino_channel` — Frozen DINOv2 + channel projector (matches original paper)

For DINOv2 with aggregation, use `aggcos1e-1` instead of `cos1e-1`.

---

## Evaluation Metrics

The following metrics are logged to Weights & Biases during validation:

| Metric | Symbol | Description | Expected Best |
|--------|--------|-------------|---------------|
| Cosine alignment | `val_C_avg` | Directional consistency (↑ better) | Temporal straightening |
| Curvature | `val_kappa_avg` | Second-difference magnitude (↓ better) | Second-difference |
| Step-size std | `val_sigma_step` | Temporal parameterization (↓ better) | Distance consistency |

---

## Planning

After training, evaluate planning performance:

```bash
python plan.py --config-name plan_gd.yaml ckpt_base_path=<ckpt_root> model_name=<model_name>
```

---

## Project Structure

```
geometrical_latent/
├── train.py                 # Main training script
├── plan.py                  # Planning evaluation
├── models/
│   └── visual_world_model.py  # Model with regularizers and metrics
├── conf/
│   ├── train.yaml           # Training config
│   └── env/                 # Environment configs
└── datasets/                # Data loading
```

---

## New Contributions

This repository extends the original temporal straightening codebase with:

1. **Second-difference smoothness regularizer** — Penalizes acceleration magnitude: `||z_{t+2} - 2z_{t+1} + z_t||²`

2. **Distance consistency regularizer** — Penalizes step-size variation: `(||v_{t+1}|| - ||v_t||)²`

3. **Geometric evaluation metrics** — C_avg, κ_avg, σ_step logged during training

---

## Acknowledgements

This repository is adapted from the [temporal-straightening](https://github.com/agentic-learning-ai-lab/temporal-straightening) codebase by Wang et al. We thank the authors for their excellent open-source implementation.

---

## References

```bibtex
@article{wang2026temporal_straightening,
  title={Temporal Straightening for Latent Planning},
  author={Wang, Ying and Bounou, Oumayma and Zhou, Gaoyue and Balestriero, Randall and Rudner, Tim GJ and LeCun, Yann and Ren, Mengye},
  journal={arXiv preprint arXiv:2603.12231},
  year={2026}
}

@inproceedings{bardes2022vicreg,
  title={VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning},
  author={Bardes, Adrien and Ponce, Jean and LeCun, Yann},
  booktitle={ICLR},
  year={2022}
}
```
