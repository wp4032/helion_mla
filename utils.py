import dataclasses
import torch
import random
from helion._testing import DEVICE

@dataclasses.dataclass
class TestParam:
    b: int
    s_q: int
    s_kv: int
    topk: int
    h_q: int = 128
    h_kv: int = 1
    d_qk: int = 576
    d_v: int = 512
    seed: int = 0
    check_correctness: bool = True
    benchmark: bool = True

@dataclasses.dataclass
class Testcase:
    t: TestParam
    q: torch.Tensor
    kv: torch.Tensor
    indices: torch.Tensor

def generate_testcase(t: TestParam) -> Testcase:
    torch.manual_seed(t.seed)
    torch.cuda.manual_seed(t.seed)
    random.seed(t.seed)
    q = torch.randn((t.b, t.s_q, t.h_q, t.d_qk), dtype=torch.bfloat16, device=DEVICE) / 10
    kv = torch.randn((t.b, t.s_kv, t.h_kv, t.d_qk), dtype=torch.bfloat16, device=DEVICE) / 10

    q.clamp_(-10, 10)
    kv.clamp_(-10, 10)

    indices = torch.full((t.b, t.s_q, t.h_kv, t.topk), t.s_kv, dtype=torch.int32, device=DEVICE)
    for b in range(t.b):
        for s in range(t.s_q):
            for h in range(t.h_kv):
                # NOTE We use the following method to generate indices so that most indices lies within [s_kv-20000, s_kv), which is more realistic for sparse attention
                near_mask = torch.randint(0, 32, (min(t.topk, t.s_kv),)) < 31
                cur_indices = torch.randperm(t.s_kv)[:t.topk]
                cur_indices[near_mask] = torch.randint(max(0, t.s_kv - 20000), t.s_kv - 1, (near_mask.sum().item(),))
                if len(cur_indices) < t.topk:
                    cur_indices = torch.cat([cur_indices, torch.full((t.topk - len(cur_indices),), 2147480000)])
                cur_indices = cur_indices[torch.randperm(t.topk)]
                indices[b, s, h] = cur_indices
    indices = indices.to(q.device)

    return q, kv, indices
    # return Testcase(
    #     t=t,
    #     q=q,
    #     kv=kv,
    #     indices=indices
    # )