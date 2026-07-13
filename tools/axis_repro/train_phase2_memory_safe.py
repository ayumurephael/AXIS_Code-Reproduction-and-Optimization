"""Exact Phase-II objective with memory-safe checkpointed vocabulary loss."""
from __future__ import annotations
import types
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from . import model_utils

_plain_build = model_utils.build_model

def _axis_forward(self, local_embeddings, time_series, questions, answers,
                  start_indices, end_indices, return_logits=False, ablation_mode=None):
    if return_logits:
        raise ValueError("memory-safe training forward does not materialize full logits")
    ids, mask, labels, _ = self.generate_input_ids_and_labels(
        questions, answers, time_series, start_indices, end_indices, ablation_mode)
    device = self.get_device(); ids=ids.to(device); mask=mask.to(device); labels=labels.to(device)
    embeds = self.get_hint_embeddings(ids, local_embeddings, start_indices, end_indices)
    hidden = self.model.model(inputs_embeds=embeds, attention_mask=mask,
                              use_cache=False, return_dict=True).last_hidden_state
    hidden, targets = hidden[:, :-1, :], labels[:, 1:]
    valid = (targets != -100).sum().clamp_min(1)
    total = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, hidden.size(1), 64):
        h = hidden[:, start:start+64, :]
        y = targets[:, start:start+64]
        def chunk_loss(hh, yy):
            logits = self.model.lm_head(hh).float()
            return F.cross_entropy(logits.reshape(-1, logits.size(-1)), yy.reshape(-1),
                                   ignore_index=-100, reduction="sum")
        total = total + checkpoint(chunk_loss, h, y, use_reentrant=False)
    return total / valid

def _build():
    model = _plain_build(); llm=model.axis.model
    llm.gradient_checkpointing_enable(); llm.enable_input_require_grads(); llm.config.use_cache=False
    llm.get_input_embeddings().register_forward_hook(lambda _m,_i,out: out.clone())
    model.axis.forward = types.MethodType(_axis_forward, model.axis)
    return model

model_utils.build_model = _build
from . import train_phase2_ddp as implementation  # noqa: E402
implementation.build_model = _build

if __name__ == "__main__":
    implementation.main()
