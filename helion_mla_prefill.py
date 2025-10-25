import torch
import helion
import dataclasses
import math
from typing import Tuple, Callable
from helion._testing import DEVICE, run_example
from utils import TestParam, Testcase, generate_testcase

import helion.language as hl

def _torch_mla_prefill(q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor, p: TestParam, sm_scale: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    def log2sumexp2(a: torch.Tensor, dim: int) -> torch.Tensor:
        return torch.logsumexp(a * math.log(2), dim=dim) * math.log2(math.e)

    assert p.b == 1
    indices = indices[0, :, 0, :]  # [s_q, topk]
    invalid_indices_mask = (indices < 0) | (indices >= p.s_kv)
    qs = q[0, :, :, :].float()  # [s_q, h_q, d_qk]
    kvs = kv[0, :, 0, :].float()  # [s_kv, d_qk]

    kvs = torch.index_select(kvs, 0, indices.masked_fill(invalid_indices_mask, 0).flatten()).view(p.s_q, p.topk, p.d_qk)  # [s_q, topk, d_qk]
    attn_score = qs @ kvs.transpose(1, 2)    # [s_q, h_q, topk]
    attn_score.masked_fill_(invalid_indices_mask.unsqueeze(1), float('-inf'))
    attn_score *= sm_scale * math.log2(math.e)
    # max_logits = torch.max(attn_score, dim=-1)[0]   # [s_q, h_q]
    lse = log2sumexp2(attn_score, dim=-1)   # [s_q, h_q]
    attn_score = torch.exp2(attn_score - lse.unsqueeze(-1))   # [s_q, h_q, topk]
    result = attn_score @ kvs[:, :, :p.d_v]
    return result

def torch_mla_prefill(q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor, p: TestParam, sm_scale: float) -> Callable[[], torch.Tensor]:
    return lambda: _torch_mla_prefill(q, kv, indices, p, sm_scale)

@helion.kernel()
def _helion_mla_prefill(q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor, p: TestParam, sm_scale: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Inputs
    indices = indices[0, :, 0, :]  # [b, s_q, h_kv, top_k] -> [s_q, topk]
    b, seq_q, h_q, d_qk = q.shape
    b_, seq_kv, h_kv, d_qk_ = kv.shape
    d_v = p.d_v
    s_kv = p.s_kv
    seq_q_, topk = indices.shape

    assert b == b_, f"Batch size mismatch {b} != {b_}"
    assert b == 1, f"Batch size must be 1 for testing"
    assert d_qk == d_qk_ , f"d_qk size mismatch {d_qk} != {d_qk_}"
    assert seq_q == seq_q_ , f"seq_q size mismatch {seq_q} != {seq_q_}"
    assert h_kv == 1, f"KV Heads must be 1"

    kv = kv[0, :, 0, :]            # [b, s_kv, h_kv, d_qk] -> [s_kv, d_qk]
    sm_scale = sm_scale * 1.44269504

    # Outputs
    # max_logits = torch.empty([s_q, h_q], dtype=torch.float8_e4m3fn, device=q.device)
    # lse = torch.empty([s_q, h_q], dtype=torch.float8_e4m3fn, device=q.device)
    kvs = torch.empty([seq_q, topk, d_qk], dtype=torch.float8_e4m3fn, device=kv.device)
    attn_score = torch.empty([seq_q, h_q, topk], dtype=torch.float8_e4m3fn, device=q.device)
    
    # TODO: remove
    # invalid_indices_mask = (indices < 0) | (indices >= s_kv)        # [s_q, topk]
    # mask = indices.masked_fill(invalid_indices_mask, 0).flatten()   # [s_q * topk]
    # kvs = torch.index_select(kv, 0, mask)                           # [s_q * topk, d_qk]
    # kvs = kvs.view(p.s_q, p.topk, p.d_qk)                           # [s_q, topk (smaller s_kv), d_qk]

    for tile_q in hl.tile(seq_q):
        for tile_topk in hl.tile(topk):
            indices_tile = indices[tile_q, tile_topk]
            # indices_valid = (indices_tile < 0) | (indices_tile >= s_kv) 
            kvs[tile_q, tile_topk] = kv[indices_tile]
            # kvs[tile_q, tile_topk] = hl.load(kv, [indices_tile], indices_valid)

    for tile_q in hl.tile(seq_q):
        m_i = hl.full([tile_q], float("-inf"), dtype=torch.float32)
        l_i = hl.full([tile_q], 0.0, dtype=torch.float32)
        acc = hl.zeros([tile_q, d_qk], dtype=torch.float32)
        for tile_k in hl.tile(seq_kv):
            k_tile = kv[0, tile_k, 0, :]        # [tile_k, d_qk], h_kv = 1
            k_tile_t = k_tile.transpose(0, 1)   # [d_qk, tile_k]
            for tile_hq in hl.tile(h_q):
                q_tile = q[0, tile_q, tile_hq, :]      # [tile_q, d_qk] (MAYBE WRONG)

                qk = hl.dot(q_tile, k_tile_t)

                qk_scaled = qk * sm_scale
                qk_max = torch.amax(qk_scaled, dim=-1)

                m_new = torch.maximum(m_i, qk_max)
                qk_shifted = qk_scaled - m_new[:, None]

                p = torch.exp2(qk_shifted)
                l_ij = torch.sum(p, dim=-1)

                alpha = torch.exp2(m_i - m_new)
                l_i = l_i * alpha + l_ij
                acc = acc * alpha[:, None]

                v_tile = kv[tile_k, 0, :d_v] # fix here
                v_t = v_tile.t()
                p_fp8 = p.to(v.dtype)

                acc = hl.dot(p_fp8, v_t, acc=acc)

                m_i = m_new
            acc = acc / l_i[:, None]
            out[tile_q, tile_hq, :] = acc.to(torch.float8_e4m3fn)

    return out

def helion_mla_prefill(q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor, p: TestParam, sm_scale: float) -> Callable[[], torch.Tensor]:
    return lambda: _helion_mla_prefill(q, kv, indices, p, sm_scale)


if __name__ == "__main__":
    assert DEVICE.type == "cuda", "Requires CUDA device"

    p = TestParam(1, 256, 256, 256, h_q=128)
    
    # [
    #     TestParam(1, s_q, s_kv, topk, h_q=128)
    #     for s_q in [256]
    #     for s_kv in [256]
    #     for topk in [256]     
    # ]

    torch.cuda.empty_cache()
    assert p.b == 1
    
    q, kv, indices = generate_testcase(p)
    sm_scale = 1 / math.sqrt(p.d_qk)

    helion_fn = helion_mla_prefill(q, kv, indices, p, sm_scale)
    pytorch_fn = torch_mla_prefill(q, kv, indices, p, sm_scale)
    run_example(
        helion_fn,
        pytorch_fn,
        (),
        atol=0.1,
        rtol=0.1
    )
    