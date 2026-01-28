import numpy as np

import mindspore as ms
from mindspore import mint, ops

DTYPE_FP16_MIN = float(np.finfo(np.float16).min)

def _scaled_dot_product_attention_native(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, dtype=None):
    # attn calculation with fp32
    if attn_mask is not None:
        if attn_mask.dtype == ms.bool_:
            attn_mask = attn_mask.to(ms.float32)
            attn_mask = attn_mask.masked_fill((1 - attn_mask).to(ms.bool_), DTYPE_FP16_MIN)
        attn_mask = attn_mask.to(query.dtype)

        attn_weight = mint.nn.functional.softmax(
            mint.matmul(query, mint.swapaxes(key, -2, -1)) / (query.shape[-1] ** 0.5) + attn_mask,
            dim=-1,
        ).to(query.dtype)
    else:
        L, S = query.shape[-2], key.shape[-2]
        attn_bias = mint.zeros((L, S), dtype=query.dtype)
        if is_causal:
            # assert attn_mask is None
            temp_mask = mint.ones((L, S), dtype=ms.bool_).tril(diagonal=0)
            attn_bias = ops.masked_fill(attn_bias, mint.logical_not(temp_mask), DTYPE_FP16_MIN)
            attn_bias = attn_bias.to(query.dtype)

        attn_weight = mint.nn.functional.softmax(
            mint.matmul(query, mint.swapaxes(key, -2, -1)) / (query.shape[-1] ** 0.5) + attn_bias,
            dim=-1,
        ).to(query.dtype)

    attn_weight = mint.nn.Dropout(p=dropout_p)(attn_weight)
    out = mint.matmul(attn_weight, value)

    return out

def _scaled_dot_product_attention_fp32(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, dtype=None):
    scale_factor = 1.0 / (query.shape[-1] ** 0.5)
    attn_scores = mint.matmul(query, mint.swapaxes(key, -2, -1)) * scale_factor

    if attn_mask is not None or is_causal:
        L, S = query.shape[-2], key.shape[-2]
        
        if attn_mask is not None:
            if attn_mask.dtype == ms.bool_:
                attn_mask = attn_mask.to(ms.float32)
                attn_mask = mint.where(attn_mask == 0, -1e9, 0.0)
            attn_mask = attn_mask.to(attn_scores.dtype)
            attn_scores += attn_mask
        elif is_causal:
            causal_mask = mint.tril(mint.ones((L, S), dtype=attn_scores.dtype))
            attn_scores = mint.where(causal_mask == 0, -1e9, attn_scores)

    attn_weight = mint.nn.functional.softmax(attn_scores, dim=-1)

    if dropout_p > 0.0:
        attn_weight = mint.nn.Dropout(p=dropout_p)(attn_weight)

    out = mint.matmul(attn_weight, value)

    return out

def _scaled_dot_product_attention_op(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, dtype=None):
    if attn_mask is None and is_causal:
        L, S = query.shape[-2], key.shape[-2]
        attn_mask = mint.ones((L, S), dtype=ms.bool_).tril(diagonal=0)

    num_heads = query.shape[1]
    scale_value = 1 / (query.shape[-1] ** 0.5)
    inner_precise = 0  # int(query.dtype == ms.float16)
    input_layout = "BNSD"

    out = ops.operations.nn_ops.PromptFlashAttention(
        num_heads=num_heads, scale_value=scale_value, input_layout=input_layout, inner_precise=inner_precise
    )(query.to(ms.float16), key.to(ms.float16), value.to(ms.float16), attn_mask)

    return out
