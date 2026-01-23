
import sys
import types

# Mock flash_attn
class MockFlashAttn:
    @staticmethod
    def flash_attn_func(*args, **kwargs):
        import torch
        import torch.nn.functional as F
        q, k, v = args[:3]
        scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, v)
    
    @staticmethod
    def flash_attn_varlen_func(*args, **kwargs):
        return MockFlashAttn.flash_attn_func(*args, **kwargs)

mock_module = types.ModuleType("flash_attn")
mock_module.flash_attn_func = MockFlashAttn.flash_attn_func
mock_module.flash_attn_varlen_func = MockFlashAttn.flash_attn_varlen_func
sys.modules["flash_attn"] = mock_module

# Import and run the actual script
import subprocess
result = subprocess.run([sys.executable, "eval_llada_mini_parallel_bench.sh"], 
                       capture_output=True, text=True)
print("STDOUT:")
print(result.stdout)
print("STDERR:") 
print(result.stderr)
print(f"Return code: {result.returncode}")
