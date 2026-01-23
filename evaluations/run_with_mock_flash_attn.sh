#!/bin/bash

# Script to run evaluation with mock flash_attn

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=1
export HF_TRANSFORMERS_TRUST_REMOTE_CODE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Run with mock flash_attn
python3 -c "
import sys
import types

# Mock flash_attn with basic CPU implementation
class MockFlashAttn:
    @staticmethod
    def flash_attn_func(*args, **kwargs):
        import torch
        import torch.nn.functional as F
        
        if len(args) < 3:
            raise ValueError('flash_attn_func needs at least 3 arguments')
            
        q, k, v = args[:3]
        if not isinstance(q, torch.Tensor) or not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor):
            raise TypeError('flash_attn_func expects torch.Tensor inputs')
            
        # Basic attention implementation
        scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, v)
    
    @staticmethod
    def flash_attn_varlen_func(*args, **kwargs):
        return MockFlashAttn.flash_attn_func(*args, **kwargs)

mock_module = types.ModuleType('flash_attn')
mock_module.flash_attn_func = MockFlashAttn.flash_attn_func
mock_module.flash_attn_varlen_func = MockFlashAttn.flash_attn_varlen_func

sys.modules['flash_attn'] = mock_module

print('✓ Mock flash_attn initialized')

# Now run the actual evaluation
exec(open('eval_llada_mini_parallel_bench.sh').read())
"
