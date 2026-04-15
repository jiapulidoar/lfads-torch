import torch

def get_ci_transformer_attn_weights(bi_transformer):
    """
    Extract attention projection weights from both directions of the
    BidirectionalTransformerEncoder.

    Each TransformerEncoderLayer has:
      - self_attn.in_proj_weight  : (3*H, H)  fused Q, K, V projections
      - self_attn.out_proj.weight : (H, H)    output projection
    """
    weights = []
    for enc in (bi_transformer.fwd_enc, bi_transformer.bwd_enc):
        for layer in enc.transformer.layers:
            weights.append(layer.self_attn.in_proj_weight)
            weights.append(layer.self_attn.out_proj.weight)
    return weights

def compute_l2_penalty_transformer(lfads, hps):
      # ── IC encoder: GRU recurrent kernels (unchanged) ─────────────────────
    recurrent_kernels_and_weights = [
        (lfads.encoder.ic_enc.fwd_gru.cell.weight_hh, hps.l2_ic_enc_scale),
        (lfads.encoder.ic_enc.bwd_gru.cell.weight_hh, hps.l2_ic_enc_scale),
        (lfads.decoder.rnn.cell.gen_cell.weight_hh,    hps.l2_gen_scale),
    ]
    if lfads.use_con:
        recurrent_kernels_and_weights.append(
            (lfads.decoder.rnn.cell.con_cell.weight_hh, hps.l2_con_scale)
        )

    # ── CI encoder: Transformer attention weights (new) ───────────────────
    transformer_kernels_and_weights = []
    if lfads.use_con:
        ci_attn_weights = get_ci_transformer_attn_weights(lfads.encoder.ci_enc)
        for w in ci_attn_weights:
            transformer_kernels_and_weights.append((w, hps.l2_ci_enc_scale))

    # ── Recurrent penalty ─────────────────────────────────────────────────
    recurrent_penalty = 0.0
    recurrent_size = 0
    for kernel, scale in recurrent_kernels_and_weights:
        if scale > 0:
            recurrent_penalty += scale * 0.5 * torch.norm(kernel, 2) ** 2
            recurrent_size += kernel.numel()
    recurrent_penalty /= recurrent_size + 1e-8

    # ── Transformer attention penalty ─────────────────────────────────────
    transformer_penalty = 0.0
    transformer_size = 0
    for kernel, scale in transformer_kernels_and_weights:
        if scale > 0:
            # Frobenius norm: natural L2 generalisation for matrices
            transformer_penalty += scale * 0.5 * torch.norm(kernel, "fro") ** 2
            transformer_size += kernel.numel()
    transformer_penalty /= transformer_size + 1e-8

    # ── Reconstruction penalty (unchanged) ────────────────────────────────
    recon_penalty = 0.0
    for recon in lfads.recon:
        if hasattr(recon, "compute_l2"):
            recon_penalty += recon.compute_l2()

    return recurrent_penalty + transformer_penalty + recon_penalty


def compute_l2_penalty(lfads, hps):
    recurrent_kernels_and_weights = [
        (lfads.encoder.ic_enc.fwd_gru.cell.weight_hh, hps.l2_ic_enc_scale),
        (lfads.encoder.ic_enc.bwd_gru.cell.weight_hh, hps.l2_ic_enc_scale),
        (lfads.decoder.rnn.cell.gen_cell.weight_hh, hps.l2_gen_scale),
    ]
    if lfads.use_con:
        recurrent_kernels_and_weights.extend(
            [
                (lfads.encoder.ci_enc.fwd_gru.cell.weight_hh, hps.l2_ci_enc_scale),
                (lfads.encoder.ci_enc.bwd_gru.cell.weight_hh, hps.l2_ci_enc_scale),
                (lfads.decoder.rnn.cell.con_cell.weight_hh, hps.l2_con_scale),
            ]
        )
    # Add recurrent penalty
    recurrent_penalty = 0.0
    recurrent_size = 0
    for kernel, weight in recurrent_kernels_and_weights:
        if weight > 0:
            recurrent_penalty += weight * 0.5 * torch.norm(kernel, 2) ** 2
            recurrent_size += kernel.numel()
    recurrent_penalty /= recurrent_size + 1e-8
    # Add recon penalty if applicable
    recon_penalty = 0.0
    for recon in lfads.recon:
        if hasattr(recon, "compute_l2"):
            recon_penalty += recon.compute_l2()
    return recurrent_penalty + recon_penalty
