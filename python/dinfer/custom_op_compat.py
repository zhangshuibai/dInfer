from __future__ import annotations

import torch


def _build_custom_op_shim():
    class _CustomOpShim(torch.nn.Module):
        """
        Compatibility shim for old SGLang `CustomOp` API.

        Newer SGLang versions expose `register_custom_op` but may not provide a
        `CustomOp` base class. This shim preserves the interface used by dInfer.
        """

        def __init__(self, *args, **kwargs):
            super().__init__()

        def enter_torch_compile(self, *args, **kwargs):
            return None

        def leave_torch_compile(self, *args, **kwargs):
            return None

        def forward(self, *args, **kwargs):
            tensor_arg = None
            for x in args:
                if isinstance(x, torch.Tensor):
                    tensor_arg = x
                    break
            if tensor_arg is None:
                for _, v in kwargs.items():
                    if isinstance(v, torch.Tensor):
                        tensor_arg = v
                        break

            if tensor_arg is not None:
                if tensor_arg.is_cuda and hasattr(self, "forward_cuda"):
                    return self.forward_cuda(*args, **kwargs)
                if hasattr(tensor_arg, "is_npu") and tensor_arg.is_npu and hasattr(self, "forward_npu"):
                    return self.forward_npu(*args, **kwargs)
                if tensor_arg.device.type == "cpu" and hasattr(self, "forward_cpu"):
                    return self.forward_cpu(*args, **kwargs)

            if hasattr(self, "forward_native"):
                return self.forward_native(*args, **kwargs)
            raise NotImplementedError("CustomOp shim requires forward_* or forward_native implementation.")

    return _CustomOpShim


try:
    from sglang.srt.custom_op import CustomOp as CustomOp  # old API
except Exception:
    try:
        import sglang.srt.utils.custom_op as _custom_op_mod  # new module path

        CustomOp = getattr(_custom_op_mod, "CustomOp", None)
        if CustomOp is None:
            CustomOp = _build_custom_op_shim()
    except Exception:
        CustomOp = _build_custom_op_shim()

