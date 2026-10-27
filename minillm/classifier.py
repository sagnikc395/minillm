"""Turning the pretrained GPT into a spam classifier (book ch. 6.5-6.6).

The base model predicts one of 50257 tokens; a spam classifier predicts one of
two labels. Three things change, and only three:

1. every pretrained parameter is frozen -- we are finetuning, not retraining,
   and ~124M parameters on ~1000 training rows would overfit immediately,
2. `out_head` is replaced by a fresh `nn.Linear(emb_dim, num_classes)`, which
   is trainable because it is brand new,
3. the last transformer block and the final layer norm are unfrozen again, so
   the representation feeding the new head can still adapt.
"""

import torch
from torch import nn


from minillm.model import GPTModel

NUM_CLASSES = 2


def prepare_for_classification(
    model: GPTModel, num_classes: int = NUM_CLASSES, tune_last_block: bool = True
) -> GPTModel:
    """Freeze `model`, swap in a `num_classes` output head, return it in place.

    With `tune_last_block=False` only the new head trains -- a useful ablation,
    and noticeably worse in the book's results.
    """
    # the freeze comes first so the modules replaced/unfrozen below keep
    # requires_grad=True: a parameter created after this loop is unaffected.
    for param in model.parameters():
        param.requires_grad = False

    # replace the output layer which originally mapped the layer inputs to 50,257 dims to the size of the vocabulary.

    torch.manual_seed(123)
    model.out_head = nn.Linear(model.cfg.emb_dim, num_classes)

    # to make the final LayerNorm and last transformer block trainable, we set their respective requires_grad to True
    if tune_last_block:
        for param in model.trf_blocks[-1].parameters():
            param.requires_grad = True
        for param in model.final_norm.parameters():
            param.requires_grad = True

    return model


def trainable_parameters(model: nn.Module) -> int:
    """How many parameters the optimizer will actually update."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
