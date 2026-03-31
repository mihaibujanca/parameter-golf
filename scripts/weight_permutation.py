#!/usr/bin/env python3
"""
weight_permutation.py — Lossless weight reordering for better compression.

Sorts MLP hidden neurons by per-row quantization scale, grouping rows with
similar dynamic range. Applied AFTER quantization, BEFORE compression.
The network is mathematically identical — no indices needed, no overhead.

Importable functions:
  - permute_mlp_qobj(): sort MLP neurons in quantized state dict
  - permute_attn_qobj(): sort attention heads in quantized state dict (minimal effect)

Usage:
    from scripts.weight_permutation import permute_mlp_qobj
    qobj = permute_mlp_qobj(qobj)  # mutates in place, returns same dict
"""
from __future__ import annotations

import re
import numpy as np


def permute_mlp_qobj(qobj: dict) -> dict:
    """Sort MLP hidden neurons by per-row scale in the quantized dict.

    For each MLP layer:
      - fc.weight rows sorted by scale (groups similar dynamic range)
      - fc.weight scales reordered to match
      - proj.weight columns reordered with same permutation
      (proj scales are per output-row, unaffected by column permutation)

    Returns the same qobj dict (mutated in place).
    """
    quantized = qobj['quantized']
    scales = qobj['scales']

    fc_keys = sorted(k for k in quantized if re.fullmatch(r"blocks\.\d+\.mlp\.fc\.weight", k))
    for fc_key in fc_keys:
        layer = re.search(r"blocks\.(\d+)\.", fc_key).group(1)
        proj_key = f"blocks.{layer}.mlp.proj.weight"
        if proj_key not in quantized:
            continue

        fc_q = quantized[fc_key]    # int8 [hidden, dim]
        fc_s = scales[fc_key]       # float16 [hidden]
        proj_q = quantized[proj_key]  # int8 [dim, hidden]

        # Sort by per-row scale (primary), then by row L1 norm (tiebreaker)
        sort_key = np.array(fc_s, dtype=np.float32)
        P = np.argsort(sort_key, kind='stable')

        quantized[fc_key] = np.ascontiguousarray(fc_q[P, :])
        scales[fc_key] = np.ascontiguousarray(fc_s[P])
        quantized[proj_key] = np.ascontiguousarray(proj_q[:, P])

    return qobj


def permute_attn_qobj(qobj: dict, n_q_heads: int = 8, n_kv_heads: int = 4) -> dict:
    """Sort attention heads by K-row scale within GQA groups.

    Typically saves <2KB — included for completeness but not recommended.
    """
    quantized = qobj['quantized']
    scales = qobj['scales']
    passthrough = qobj.get('passthrough', {})

    q_per_kv = n_q_heads // n_kv_heads

    q_keys = sorted(k for k in quantized if re.fullmatch(r"blocks\.\d+\.attn\.c_q\.weight", k))
    for q_key in q_keys:
        layer = re.search(r"blocks\.(\d+)\.", q_key).group(1)
        k_key = f"blocks.{layer}.attn.c_k.weight"
        v_key = f"blocks.{layer}.attn.c_v.weight"
        p_key = f"blocks.{layer}.attn.proj.weight"
        gain_key = f"blocks.{layer}.attn.q_gain"
        if not all(k in quantized for k in [k_key, v_key, p_key]):
            continue

        head_dim = quantized[q_key].shape[0] // n_q_heads

        # Sort KV groups by combined K scale norm
        k_s = np.array(scales[k_key], dtype=np.float32)
        kv_norms = np.array([
            np.linalg.norm(k_s[g * head_dim:(g + 1) * head_dim])
            for g in range(n_kv_heads)
        ])
        Pk = np.argsort(kv_norms)

        # Within each KV group, sort Q heads by Q scale norm
        q_s = np.array(scales[q_key], dtype=np.float32)
        q_row_idx = []
        for orig_kv in Pk:
            q_pair = [orig_kv * q_per_kv + qi for qi in range(q_per_kv)]
            pair_norms = [np.linalg.norm(q_s[h * head_dim:(h + 1) * head_dim]) for h in q_pair]
            for h in [q_pair[i] for i in np.argsort(pair_norms)]:
                q_row_idx.extend(range(h * head_dim, (h + 1) * head_dim))

        q_row_idx = np.array(q_row_idx)
        kv_row_idx = np.concatenate([np.arange(g * head_dim, (g + 1) * head_dim) for g in Pk])

        for key, idx in [(q_key, q_row_idx), (k_key, kv_row_idx), (v_key, kv_row_idx)]:
            quantized[key] = np.ascontiguousarray(quantized[key][idx, :])
            scales[key] = np.ascontiguousarray(scales[key][idx])
        quantized[p_key] = np.ascontiguousarray(quantized[p_key][:, q_row_idx])

        # q_gain is per-Q-head
        if gain_key in passthrough:
            q_head_order = q_row_idx[::head_dim] // head_dim
            passthrough[gain_key] = np.ascontiguousarray(
                np.array(passthrough[gain_key])[q_head_order]
            )

    return qobj
