from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_config import COORD_VOCAB_START_IDX


def compute_continuous_coordinate_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pos_indices_list: list[torch.Tensor],
    coord_vocab_start_idx: int = COORD_VOCAB_START_IDX,
    num_bins: int = 1001,
) -> torch.Tensor:
    """
    Computes soft-argmax expected continuous Smooth-L1 coordinate refinement loss for box localization.
    Bridges discrete token classification with differentiable continuous spatial alignment.
    Fixed Off-by-8 vocabulary index offset to correct <0> coordinate token starting index at 151677.
    """
    batch_size = logits.shape[0]
    device = logits.device
    dtype = logits.dtype

    coord_slice = slice(coord_vocab_start_idx, coord_vocab_start_idx + num_bins)
    bin_values = torch.arange(num_bins, device=device, dtype=torch.float32)

    losses = []
    for pos_indices in pos_indices_list:
        if pos_indices is None:
            continue
        pos_indices = pos_indices.to(device=device)
        valid_mask = pos_indices >= 0
        if not valid_mask.any():
            continue
        
        valid_batch = torch.arange(batch_size, device=device)[valid_mask]
        valid_pos = pos_indices[valid_mask]
        
        target_token_ids = labels[valid_batch, valid_pos]
        # Ignore positions where label is masked (-100) or outside vocabulary coordinates
        target_valid = (target_token_ids >= coord_vocab_start_idx) & (target_token_ids < coord_vocab_start_idx + num_bins)
        if not target_valid.any():
            continue
            
        b_idx = valid_batch[target_valid]
        p_idx = valid_pos[target_valid]
        
        # modeling_qwen2.py scores shift_logits[..., :-1] against shift_labels[..., 1:],
        # so the slot at position p is predicted by the logits at position p - 1.
        # Indexing the logits at p reads the prediction for the NEXT slot.
        coord_logits = logits[b_idx, p_idx - 1, coord_slice]
        probs = F.softmax(coord_logits.float(), dim=-1)
        pred_cont = torch.sum(probs * bin_values, dim=-1) / (num_bins - 1.0)
        
        target_cont = (labels[b_idx, p_idx] - coord_vocab_start_idx).float().clamp(0, num_bins - 1) / (num_bins - 1.0)
        
        l1_loss = F.smooth_l1_loss(pred_cont, target_cont, beta=0.01)
        losses.append(l1_loss.to(dtype=dtype))

    if not losses:
        return torch.tensor(0.0, device=device, dtype=dtype)
    return torch.stack(losses).mean()


def compute_circular_consistency_loss(
    logits: torch.Tensor,
    sin_pos_indices: torch.Tensor,
    cos_pos_indices: torch.Tensor,
    coord_vocab_start_idx: int = COORD_VOCAB_START_IDX,
    num_bins: int = 1001,
) -> torch.Tensor:
    """
    Computes the differentiable circular consistency loss L_circle = ((s_hat)^2 + (c_hat)^2 - 1.0)^2.
    Fixed Off-by-8 vocabulary index offset to correct <0> coordinate token starting index at 151677.
    """
    batch_size = logits.shape[0]
    device = logits.device
    dtype = logits.dtype

    sin_pos_indices = sin_pos_indices.to(device=device)
    cos_pos_indices = cos_pos_indices.to(device=device)
    coord_slice = slice(coord_vocab_start_idx, coord_vocab_start_idx + num_bins)
    
    batch_indices = torch.arange(batch_size, device=device)
    # Shifted-CE alignment: slot p is predicted by the logits at p - 1.
    sin_logits = logits[batch_indices, sin_pos_indices - 1, coord_slice]
    cos_logits = logits[batch_indices, cos_pos_indices - 1, coord_slice]

    sin_probs = F.softmax(sin_logits.float(), dim=-1)
    cos_probs = F.softmax(cos_logits.float(), dim=-1)

    bin_values = torch.arange(num_bins, device=device, dtype=torch.float32)
    sin_exp_bin = torch.sum(sin_probs * bin_values, dim=-1)
    cos_exp_bin = torch.sum(cos_probs * bin_values, dim=-1)

    # Convert bin index [0, 1000] into continuous range [-1.0, 1.0]
    sin_hat = (sin_exp_bin / ((num_bins - 1) / 2.0)) - 1.0
    cos_hat = (cos_exp_bin / ((num_bins - 1) / 2.0)) - 1.0

    circle_error = (sin_hat**2 + cos_hat**2 - 1.0)**2
    return circle_error.mean().to(dtype=dtype)


def compute_cosine_directional_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    sin_pos_indices: torch.Tensor,
    cos_pos_indices: torch.Tensor,
    w_pos_indices: torch.Tensor | None = None,
    h_pos_indices: torch.Tensor | None = None,
    coord_vocab_start_idx: int = COORD_VOCAB_START_IDX,
    num_bins: int = 1001,
    aspect_ratio_threshold: float = 1.05,
    image_aspect: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Computes Cosine Directional Distance Loss with Near-Square Angle Stability Filter.
    Resolves 100% multimodality in rotational angles while preventing gradient noise from near-square objects.
    """
    batch_size = logits.shape[0]
    device = logits.device
    dtype = logits.dtype

    sin_pos_indices = sin_pos_indices.to(device=device)
    cos_pos_indices = cos_pos_indices.to(device=device)
    coord_slice = slice(coord_vocab_start_idx, coord_vocab_start_idx + num_bins)
    
    batch_indices = torch.arange(batch_size, device=device)
    
    sin_logits = logits[batch_indices, sin_pos_indices - 1, coord_slice]
    cos_logits = logits[batch_indices, cos_pos_indices - 1, coord_slice]
    
    sin_target_ids = labels[batch_indices, sin_pos_indices] - coord_vocab_start_idx
    cos_target_ids = labels[batch_indices, cos_pos_indices] - coord_vocab_start_idx
    
    valid = (sin_target_ids >= 0) & (sin_target_ids < num_bins) & (cos_target_ids >= 0) & (cos_target_ids < num_bins)
    if not valid.any():
        return torch.tensor(0.0, device=device, dtype=dtype)
        
    sin_probs = F.softmax(sin_logits[valid].float(), dim=-1)
    cos_probs = F.softmax(cos_logits[valid].float(), dim=-1)
    
    bin_values = torch.arange(num_bins, device=device, dtype=torch.float32)
    sin_exp_bin = torch.sum(sin_probs * bin_values, dim=-1)
    cos_exp_bin = torch.sum(cos_probs * bin_values, dim=-1)
    
    sin_hat = (sin_exp_bin / ((num_bins - 1) / 2.0)) - 1.0
    cos_hat = (cos_exp_bin / ((num_bins - 1) / 2.0)) - 1.0
    
    sin_gt = (sin_target_ids[valid].float() / ((num_bins - 1) / 2.0)) - 1.0
    cos_gt = (cos_target_ids[valid].float() / ((num_bins - 1) / 2.0)) - 1.0

    # Near-Square Angle Stability Filter
    if w_pos_indices is not None and h_pos_indices is not None:
        w_pos = w_pos_indices.to(device=device)[valid]
        h_pos = h_pos_indices.to(device=device)[valid]
        w_t = (labels[batch_indices[valid], w_pos] - coord_vocab_start_idx).float()
        h_t = (labels[batch_indices[valid], h_pos] - coord_vocab_start_idx).float()
        if image_aspect is not None:
            # w_t, h_t encode w/W and h/H; scaling w_t by W/H makes the ratio the
            # physical aspect ratio w/h instead of the image-normalized one.
            w_t = w_t * image_aspect.to(device=device, dtype=torch.float32)[valid]
        
        max_dim = torch.maximum(w_t, h_t)
        min_dim = torch.maximum(torch.minimum(w_t, h_t), torch.tensor(1e-5, device=device))
        aspect_ratio = max_dim / min_dim
        
        # Zero-out angle direction gradient for near-square geometries (AR < 1.05)
        stability_mask = (aspect_ratio >= aspect_ratio_threshold).float()
    else:
        stability_mask = torch.ones_like(sin_hat)

    if stability_mask.sum() == 0:
        return torch.tensor(0.0, device=device, dtype=dtype)

    norm_pred = torch.sqrt(sin_hat**2 + cos_hat**2 + 1e-7)
    norm_gt = torch.sqrt(sin_gt**2 + cos_gt**2 + 1e-7)
    cosine_sim = (sin_hat * sin_gt + cos_hat * cos_gt) / (norm_pred * norm_gt)
    
    dir_loss = (1.0 - cosine_sim) * stability_mask
    return (dir_loss.sum() / stability_mask.sum()).to(dtype=dtype)


class OBBLossWrapper(nn.Module):
    def __init__(
        self,
        lambda_circle: float = 0.1, # 10x increased to compensate low grad norm
        lambda_center: float = 5.0,  # 2x boosted as Center error is proven #1 bottleneck
        lambda_direction: float = 1.0, # New Cosine Direction Angle Loss
        coord_vocab_start_idx: int = COORD_VOCAB_START_IDX, # Correctly verified <0> vocabulary ID
    ):
        super().__init__()
        self.lambda_circle = lambda_circle
        self.lambda_center = lambda_center
        self.lambda_direction = lambda_direction
        self.coord_vocab_start_idx = coord_vocab_start_idx

    def forward(
        self,
        outputs,
        labels: torch.Tensor | None = None,
        sin_pos_indices: torch.Tensor | None = None,
        cos_pos_indices: torch.Tensor | None = None,
        cx_pos_indices: torch.Tensor | None = None,
        cy_pos_indices: torch.Tensor | None = None,
        w_pos_indices: torch.Tensor | None = None,
        h_pos_indices: torch.Tensor | None = None,
        image_aspect: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        base_loss = outputs.loss
        if base_loss is None:
            raise ValueError("Outputs object does not contain an initial CE loss.")
            
        metrics = {"loss_ce": float(base_loss.item())}
        total_loss = base_loss

        # 1. Apply continuous coordinate spatial refinement loss (Center & Scale)
        if self.lambda_center > 0.0 and labels is not None and hasattr(outputs, "logits") and outputs.logits is not None:
            spatial_indices = [idx for idx in [cx_pos_indices, cy_pos_indices, w_pos_indices, h_pos_indices] if idx is not None]
            if spatial_indices:
                l_center = compute_continuous_coordinate_loss(
                    outputs.logits,
                    labels=labels,
                    pos_indices_list=spatial_indices,
                    coord_vocab_start_idx=self.coord_vocab_start_idx,
                )
                total_loss = total_loss + self.lambda_center * l_center
                metrics["loss_center"] = float(l_center.item())
            else:
                metrics["loss_center"] = 0.0
        else:
            metrics["loss_center"] = 0.0

        # 2. Apply circular consistency loss (s_hat^2 + c_hat^2 - 1)^2
        if self.lambda_circle > 0 and sin_pos_indices is not None and cos_pos_indices is not None and hasattr(outputs, "logits") and outputs.logits is not None:
            l_circle = compute_circular_consistency_loss(
                outputs.logits,
                sin_pos_indices=sin_pos_indices,
                cos_pos_indices=cos_pos_indices,
                coord_vocab_start_idx=self.coord_vocab_start_idx,
            )
            total_loss = total_loss + self.lambda_circle * l_circle
            metrics["loss_circle"] = float(l_circle.item())
        else:
            metrics["loss_circle"] = 0.0

        # 3. Apply Cosine Direction Angle Loss with Near-Square Filter
        if self.lambda_direction > 0 and labels is not None and sin_pos_indices is not None and cos_pos_indices is not None and hasattr(outputs, "logits") and outputs.logits is not None:
            l_dir = compute_cosine_directional_loss(
                outputs.logits,
                labels=labels,
                sin_pos_indices=sin_pos_indices,
                cos_pos_indices=cos_pos_indices,
                w_pos_indices=w_pos_indices,
                h_pos_indices=h_pos_indices,
                coord_vocab_start_idx=self.coord_vocab_start_idx,
                image_aspect=image_aspect,
            )
            total_loss = total_loss + self.lambda_direction * l_dir
            metrics["loss_direction"] = float(l_dir.item())
        else:
            metrics["loss_direction"] = 0.0
            
        metrics["loss_total"] = float(total_loss.item())
        return total_loss, metrics
