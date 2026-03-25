# Claude Code Prompt: Adding Evaluation Metrics

## Context

I'm working on my thesis comparing geometry-aware latent space regularizers for RL. The training regularizers (second-difference and distance-consistency) are already implemented. Now I need to add evaluation metrics that measure the geometric properties of learned latent trajectories during validation. These metrics should be computed and logged to wandb.

---

## Metrics to Implement (from thesis Section 5.6)

### 1. C_avg — Average Cosine Similarity (Directional Alignment)

**Formula from thesis (Section 5.6.3):**
```
C_avg = (1 / (T-1)) * Σ_{t=1}^{T-1} [v_t^T * v_{t+1} / (||v_t||_2 * ||v_{t+1}||_2)]
```
where `v_t = z_{t+1} - z_t` is the latent transition vector.

**Interpretation:** Higher values indicate stronger directional alignment between consecutive latent transitions. This metric should be **highest for temporal straightening**.

**PyTorch implementation:**
```python
def compute_cosine_alignment(features):
    """
    Compute average cosine similarity between consecutive velocity vectors.
    
    Args:
        features: Tensor of shape (B, T, D) or (B, T, P, D) - latent representations over time
    
    Returns:
        Scalar: mean cosine similarity (higher = more directionally aligned)
    """
    # Handle 4D tensors by flattening patches
    if features.dim() == 4:
        B, T, P, D = features.shape
        features = features.view(B, T, P * D)
    
    if features.shape[1] < 3:
        return torch.tensor(0.0, device=features.device)
    
    # Velocity vectors: v_t = z_{t+1} - z_t
    v = features[:, 1:] - features[:, :-1]  # (B, T-1, D)
    
    # Consecutive velocity pairs
    v1 = v[:, :-1]  # v_t
    v2 = v[:, 1:]   # v_{t+1}
    
    # Cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(v1, v2, dim=-1)  # (B, T-2)
    
    return cos_sim.mean()
```

---

### 2. κ_avg — Average Curvature (Second-Difference Magnitude)

**Formula from thesis (Section 5.6.3):**
```
κ_avg = (1 / (T-1)) * Σ_{t=1}^{T-1} ||z_{t+2} - 2*z_{t+1} + z_t||_2
```

**Interpretation:** Lower values indicate less local bending/curvature in latent trajectories. This metric should be **lowest for second-difference smoothness**.

**PyTorch implementation:**
```python
def compute_curvature(features):
    """
    Compute average discrete curvature (second-difference magnitude).
    
    Args:
        features: Tensor of shape (B, T, D) or (B, T, P, D)
    
    Returns:
        Scalar: mean curvature (lower = smoother trajectories)
    """
    # Handle 4D tensors by flattening patches
    if features.dim() == 4:
        B, T, P, D = features.shape
        features = features.view(B, T, P * D)
    
    if features.shape[1] < 3:
        return torch.tensor(0.0, device=features.device)
    
    # Second difference: z_{t+2} - 2*z_{t+1} + z_t
    second_diff = features[:, 2:] - 2 * features[:, 1:-1] + features[:, :-2]  # (B, T-2, D)
    
    # L2 norm of second difference
    curvature = second_diff.norm(dim=-1)  # (B, T-2)
    
    return curvature.mean()
```

---

### 3. σ_step — Step-Size Consistency (Standard Deviation of Step Sizes)

**Formula from thesis (Section 5.6.3):**
```
σ_step = std({||v_t||_2}_{t=1}^{T-1})
```

**Interpretation:** Lower values indicate more uniform temporal parameterization (consistent step sizes). This metric should be **lowest for distance consistency**.

**PyTorch implementation:**
```python
def compute_step_size_std(features):
    """
    Compute standard deviation of step sizes (transition magnitudes).
    
    Args:
        features: Tensor of shape (B, T, D) or (B, T, P, D)
    
    Returns:
        Scalar: std of step sizes (lower = more uniform parameterization)
    """
    # Handle 4D tensors by flattening patches
    if features.dim() == 4:
        B, T, P, D = features.shape
        features = features.view(B, T, P * D)
    
    if features.shape[1] < 2:
        return torch.tensor(0.0, device=features.device)
    
    # Velocity vectors: v_t = z_{t+1} - z_t
    v = features[:, 1:] - features[:, :-1]  # (B, T-1, D)
    
    # Step sizes (L2 norms)
    step_sizes = v.norm(dim=-1)  # (B, T-1)
    
    # Standard deviation across time (per batch), then mean across batch
    step_std = step_sizes.std(dim=-1).mean()
    
    return step_std
```

---

## Where to Add These Metrics

### Option A: Add to `train.py` validation loop (recommended)

Find the validation/evaluation section in `train.py`. After computing the validation loss, add:

```python
# Compute geometric metrics on validation data
with torch.no_grad():
    # Get latent features from encoder
    # features shape should be (B, T, D) or (B, T, P, D)
    
    c_avg = compute_cosine_alignment(features)
    kappa_avg = compute_curvature(features)
    sigma_step = compute_step_size_std(features)
    
    # Log to wandb
    wandb.log({
        'val/C_avg': c_avg.item(),
        'val/kappa_avg': kappa_avg.item(),
        'val/sigma_step': sigma_step.item(),
    })
```

### Option B: Add as methods to `VWorldModel` class

Add to `models/visual_world_model.py`:

```python
def compute_geometric_metrics(self, features):
    """
    Compute all geometric diagnostic metrics.
    
    Args:
        features: Tensor of shape (B, T, D) or (B, T, P, D)
    
    Returns:
        dict with keys: 'C_avg', 'kappa_avg', 'sigma_step'
    """
    # Handle 4D tensors
    if features.dim() == 4:
        B, T, P, D = features.shape
        features = features.view(B, T, P * D)
    
    metrics = {}
    
    if features.shape[1] >= 3:
        # C_avg: cosine alignment
        v = features[:, 1:] - features[:, :-1]
        v1, v2 = v[:, :-1], v[:, 1:]
        cos_sim = torch.nn.functional.cosine_similarity(v1, v2, dim=-1)
        metrics['C_avg'] = cos_sim.mean().item()
        
        # kappa_avg: curvature
        second_diff = features[:, 2:] - 2 * features[:, 1:-1] + features[:, :-2]
        metrics['kappa_avg'] = second_diff.norm(dim=-1).mean().item()
    else:
        metrics['C_avg'] = 0.0
        metrics['kappa_avg'] = 0.0
    
    if features.shape[1] >= 2:
        # sigma_step: step size std
        v = features[:, 1:] - features[:, :-1]
        step_sizes = v.norm(dim=-1)
        metrics['sigma_step'] = step_sizes.std(dim=-1).mean().item()
    else:
        metrics['sigma_step'] = 0.0
    
    return metrics
```

---

## Implementation Steps

1. **Inspect the codebase first** — Find where:
   - Validation/evaluation happens in `train.py`
   - Latent features are computed (likely via `self.model.visual_only()` or similar)
   - Metrics are logged to wandb

2. **Add the metric computation functions** — Either as standalone functions or methods

3. **Integrate into validation loop** — Call the metrics on validation batches

4. **Log to wandb** — Use these exact names for consistency:
   - `val/C_avg` — cosine alignment (↑ better for straightening)
   - `val/kappa_avg` — curvature (↓ better for 2nd-diff)
   - `val/sigma_step` — step size std (↓ better for dist-consistency)

---

## Expected Behavior After Implementation

When training completes, wandb should show:

| Condition | C_avg (↑) | κ_avg (↓) | σ_step (↓) |
|-----------|-----------|-----------|------------|
| Prediction-only | low | high | high |
| VCReg | low | high | high |
| Temporal straightening | **highest** | medium | medium |
| Second-difference | high | **lowest** | medium |
| Distance-consistency | low | medium | **lowest** |

---

## Verification

After implementation, run a quick test:

```bash
python train.py env=point_maze encoder=scratch_resnet training.epochs=1 training.batch_size=16 training.straighten=False
```

Check wandb logs — you should see `val/C_avg`, `val/kappa_avg`, and `val/sigma_step` appearing.

---

## Notes

- These are **diagnostic metrics** — they verify each regularizer achieves its intended geometric effect
- The **primary evaluation metric** (Success Rate) requires running the planning evaluation separately after training
- Make sure to compute metrics on the **encoded features** (output of encoder), not on raw observations
- Handle both 3D `(B, T, D)` and 4D `(B, T, P, D)` tensor shapes (P = patches for ViT encoders)
