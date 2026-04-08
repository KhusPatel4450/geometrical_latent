# Plan: Multi-Objective Optimization (MOO) with TorchJD

## Context

The thesis compares geometry-aware regularizers (temporal straightening, second-difference, distance consistency) for latent planning in RL. Currently, only ONE regularizer can be active at a time via `training.straighten=cos1e-1`. This change adds:
1. **Combined regularizers** — multiple regularizers with manual lambda weighting
2. **MOO via TorchJD** — automatic gradient-based balancing without manual lambda tuning
3. **Backward compatibility** — old `training.straighten` format still works

---

## Files to Modify

| File | What changes |
|------|-------------|
| [models/visual_world_model.py](models/visual_world_model.py) | Add `compute_regularizer_losses()` method; modify `forward()` to support returning individual losses |
| [train.py](train.py) | Add MOO branch in training loop; add TorchJD backward call; add combined regularizer logic |
| [conf/train.yaml](conf/train.yaml) | Add `training.regularizers` and `training.moo` config sections |
| [environment.yaml](environment.yaml) | Add `torchjd` dependency |
| [run_all_experiments.py](run_all_experiments.py) | Add new experiment entries for combined + MOO conditions |

---

## Step 1: Update Config (`conf/train.yaml`)

Add new sections under `training:` (after existing keys at ~line 50):

```yaml
training:
  # ... existing keys (straighten, vcreg, etc.) stay unchanged ...

  # New: combined regularizers with manual lambda
  regularizers:
    straightening:
      enabled: false
      lambda: 0.1
    second_difference:
      enabled: false
      lambda: 0.1
    distance_consistency:
      enabled: false
      lambda: 0.1

  # New: MOO with TorchJD
  moo:
    enabled: false
    algorithm: upgrad  # upgrad, cagrad, mgda, nash
    objectives:
      - prediction
      - straightening
      - second_difference
```

**Key decision:** `training.straighten` (old format) and `training.regularizers` / `training.moo` (new format) are mutually exclusive. If `straighten` is set to a string value, it takes priority (backward compat). If `straighten=False` and `regularizers` or `moo` are configured, the new system activates.

---

## Step 2: Modify `VWorldModel` ([visual_world_model.py](models/visual_world_model.py))

### 2a. Add `compute_regularizer_losses()` method (~after line 346)

```python
def compute_regularizer_losses(self, z):
    """Compute all regularizer losses individually (no scaling applied).
    Returns dict of {name: scalar_loss_tensor}."""
    feats = self.visual_only(z)
    losses = {}
    if feats.shape[1] >= 3:
        # Cosine straightening
        v1 = feats[:, 1:-1] - feats[:, :-2]
        v2 = feats[:, 2:] - feats[:, 1:-1]
        losses["straightening"] = self._cos_curvature(v1, v2)
        # Second difference
        losses["second_difference"] = self.second_difference_loss(feats)
        # Distance consistency
        losses["distance_consistency"] = self.distance_consistency_loss(feats)
    else:
        dev = z.device
        losses["straightening"] = torch.tensor(0.0, device=dev)
        losses["second_difference"] = torch.tensor(0.0, device=dev)
        losses["distance_consistency"] = torch.tensor(0.0, device=dev)
    return losses
```

### 2b. Modify `forward()` to expose individual losses for MOO

Add a `return_individual_losses` flag (default False) to `forward()`. When True, return the individual loss tensors (prediction, each regularizer) in `loss_components` as live tensors (not detached), so the caller (train.py) can run TorchJD backward on them.

In the current flow (lines 402-421), loss components are already tracked. The key change: when MOO is active, do NOT sum regularizer losses into `loss` inside forward(). Instead, return them separately and let train.py handle the backward pass.

Add new init params to `__init__`:
- `self.use_moo = False` — set by train.py after model creation
- `self.regularizer_config = None` — dict of {name: {enabled, lambda}} for combined mode

Modify forward() lines 417-421:
```python
# Replace the single straighten block with:
if self.use_moo or self.regularizer_config:
    reg_losses = self.compute_regularizer_losses(z)
    for name, val in reg_losses.items():
        loss_components[f"reg_{name}"] = val

    if not self.use_moo and self.regularizer_config:
        # Combined mode: add weighted sum
        for name, cfg in self.regularizer_config.items():
            if cfg["enabled"] and name in reg_losses:
                loss = loss + reg_losses[name] * cfg["lambda"]
                loss_components[f"reg_{name}_scaled"] = reg_losses[name] * cfg["lambda"]
elif self.straighten and self.straighten_scale > 0:
    # Legacy single-regularizer mode (backward compat)
    feats = self.visual_only(z)
    curvature_loss = self.total_curvature(feats, mode=self.curvature_mode)
    loss = loss + curvature_loss * self.straighten_scale
    loss_components["curvature_loss_used_for_training"] = curvature_loss
```

---

## Step 3: Modify Training Loop ([train.py](train.py))

### 3a. Configure model after instantiation (~line 394)

After `self.model = hydra.utils.instantiate(...)`, add:

```python
# Configure MOO / combined regularizers
moo_cfg = self.cfg.training.get("moo", {})
reg_cfg = self.cfg.training.get("regularizers", {})
use_old_straighten = isinstance(self.cfg.training.get("straighten", False), str)

if not use_old_straighten:
    if moo_cfg.get("enabled", False):
        self.model.use_moo = True
        self.moo_enabled = True
        self.moo_algorithm = moo_cfg.get("algorithm", "upgrad")
        self.moo_objectives = list(moo_cfg.get("objectives", []))
        log.info(f"MOO enabled: algorithm={self.moo_algorithm}, objectives={self.moo_objectives}")
    elif any(v.get("enabled", False) for v in reg_cfg.values() if isinstance(v, dict)):
        self.model.regularizer_config = {
            k: {"enabled": v.get("enabled", False), "lambda": v.get("lambda", 0.1)}
            for k, v in reg_cfg.items() if isinstance(v, dict)
        }
        self.moo_enabled = False
        log.info(f"Combined regularizers: {self.model.regularizer_config}")
    else:
        self.moo_enabled = False
else:
    self.moo_enabled = False
```

### 3b. Modify training loop `train()` method (~lines 597-616)

Replace the backward/step section:

```python
z_out, visual_out, visual_reconstructed, loss, loss_components = self.model(obs, act)

self.encoder_optimizer.zero_grad()
if decoder_active:
    self.decoder_optimizer.zero_grad()
if self.cfg.has_predictor:
    self.predictor_optimizer.zero_grad()
    self.action_encoder_optimizer.zero_grad()

if self.moo_enabled:
    # MOO backward with TorchJD
    from torchjd import backward as jd_backward
    from torchjd.aggregation import UPGrad, CAGrad, MGDA, NashMTL

    # Build list of objective losses
    moo_losses = []
    for obj_name in self.moo_objectives:
        if obj_name == "prediction":
            moo_losses.append(loss_components["z_loss"])
        else:
            key = f"reg_{obj_name}"
            if key in loss_components:
                moo_losses.append(loss_components[key])

    # Select aggregator
    agg_map = {"upgrad": UPGrad, "cagrad": CAGrad, "mgda": MGDA, "nash": NashMTL}
    aggregator = agg_map[self.moo_algorithm]()

    # Collect trainable parameters (exclude decoder — it has its own isolated loss)
    moo_params = list(self.encoder.parameters()) + list(self.predictor.parameters()) + \
                 list(self.action_encoder.parameters()) + list(self.proprio_encoder.parameters())
    moo_params = [p for p in moo_params if p.requires_grad]

    # TorchJD backward
    jd_backward(moo_losses, moo_params, aggregator)

    # Decoder loss backward separately (it uses z.detach(), independent)
    if decoder_active and "decoder_loss_reconstructed" in loss_components:
        # Need to backward decoder loss through decoder params only
        decoder_loss = loss_components.get("decoder_loss_reconstructed")
        if decoder_loss is not None and decoder_loss.requires_grad:
            decoder_loss.backward()
else:
    # Standard backward (old path — covers legacy straighten, combined regularizers, and baselines)
    self.accelerator.backward(loss)

# Step all optimizers (unchanged)
if self.model.train_encoder:
    self.encoder_optimizer.step()
if decoder_active:
    self.decoder_optimizer.step()
if self.cfg.has_predictor and self.model.train_predictor:
    self.predictor_optimizer.step()
    self.action_encoder_optimizer.step()
```

### 3c. Accelerator compatibility note

When MOO is enabled, we bypass `self.accelerator.backward(loss)` and call `torchjd.backward()` directly. This means:
- **Mixed precision**: TorchJD handles autocast internally; gradients are float32 regardless
- **Distributed**: For single-GPU (Colab/local), this is fine. For multi-GPU, we'd need `accelerator.reduce` on gradients after TorchJD backward — but this project currently runs single-GPU only

### 3d. Enhanced wandb logging

After the optimizer step, log individual regularizer losses:
```python
# Already logged via loss_components, but add MOO-specific info
if self.moo_enabled:
    loss_components["moo_algorithm"] = self.moo_algorithm
    loss_components["moo_num_objectives"] = len(moo_losses)
```

---

## Step 4: Add TorchJD Dependency

In [environment.yaml](environment.yaml), add under pip dependencies:
```yaml
- torchjd
```

Note: For CAGrad/NashMTL, user may need `pip install torchjd[cagrad]` or `torchjd[nash_mtl]`. UPGrad and MGDA work with base install.

---

## Step 5: Update `run_all_experiments.py`

Add new experiment entries for the combined and MOO conditions:

```python
EXPERIMENTS = {
    # ... existing experiments ...
    "combined_manual": {
        "name": "combined_manual",
        "args": "training.regularizers.straightening.enabled=true training.regularizers.straightening.lambda=0.1 training.regularizers.second_difference.enabled=true training.regularizers.second_difference.lambda=0.1",
    },
    "moo_upgrad": {
        "name": "moo_upgrad",
        "args": "training.moo.enabled=true training.moo.algorithm=upgrad 'training.moo.objectives=[prediction,straightening,second_difference]'",
    },
    "moo_cagrad": {
        "name": "moo_cagrad",
        "args": "training.moo.enabled=true training.moo.algorithm=cagrad 'training.moo.objectives=[prediction,straightening,second_difference]'",
    },
}
```

---

## Step 6: Backward Compatibility Check

The old format must still work. Verification:
- `training.straighten=cos1e-1` → `use_old_straighten=True` → `moo_enabled=False` → model uses legacy curvature path (lines 417-421, unchanged)
- `training.straighten=False` with no regularizers/moo → baseline, no regularizers
- `training.straighten=False training.vcreg=True` → VCReg baseline, unchanged path
- Hydra `hydra.run.dir` template references `${training.straighten}` → still resolves since key is unchanged

---

## Verification Plan

1. **Backward compat**: Run `python train.py training.straighten=cos1e-1 training.epochs=1` — should train identically to before
2. **Combined mode**: Run `python train.py training.regularizers.straightening.enabled=true training.regularizers.second_difference.enabled=true training.epochs=1` — check wandb logs show both reg losses
3. **MOO mode**: Run `python train.py training.moo.enabled=true training.moo.algorithm=upgrad 'training.moo.objectives=[prediction,straightening]' training.epochs=1` — check training runs without error, gradients flow
4. **Logging**: Verify wandb shows `train_reg_straightening`, `train_reg_second_difference`, `train_reg_distance_consistency` as separate logged values
5. **Baseline**: Run `python train.py training.straighten=False training.epochs=1` — should still work as prediction-only baseline

---

## Risk Mitigation

- **TorchJD + Accelerator**: MOO bypasses Accelerator's backward. Safe for single-GPU. If multi-GPU needed later, add gradient sync manually.
- **Decoder isolation**: Decoder loss uses `z.detach()` so it never conflicts with MOO objectives. Handled separately.
- **Hydra config resolution**: New config keys have defaults (`enabled: false`, `moo.enabled: false`) so they don't break existing commands.
- **Import guards**: TorchJD imports are inside the `if self.moo_enabled:` block, so it's only required when MOO is actually used.
