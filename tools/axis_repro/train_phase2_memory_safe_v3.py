"""Production memory-safe Phase-II entry point."""
from __future__ import annotations
import types
from . import model_utils
from . import train_phase2_memory_safe as base

_plain_build = base._plain_build

def _build():
    model = _plain_build(); llm=model.axis.model
    llm.gradient_checkpointing_enable(); llm.enable_input_require_grads(); llm.config.use_cache=False
    llm.get_input_embeddings().register_forward_hook(lambda _m,_i,out: out.clone())
    model.axis.forward = types.MethodType(base._axis_forward, model.axis)
    # HF checkpointing is gated by module.training; Qwen2 attention_dropout=0.
    llm.eval = types.MethodType(lambda self: self, llm)
    return model

model_utils.build_model = _build
from . import train_phase2_ddp as implementation  # noqa: E402
implementation.build_model = _build

if __name__ == "__main__":
    implementation.main()
