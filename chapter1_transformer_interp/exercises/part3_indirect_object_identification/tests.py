import sys, os
from pathlib import Path

import torch as t
import solutions
from typing import Tuple, List, Callable
from transformer_lens import (
    HookedTransformer,
    HookedTransformerConfig,
    FactoredMatrix,
    ActivationCache,
)


def test_logits_to_ave_logit_diff(logits_to_ave_logit_diff: Callable):

    batch = 4
    seq = 5
    d_vocab = 6
    logits = t.randn(batch, seq, d_vocab)
    answer_tokens = t.randint(0, d_vocab, (batch, 2))

    actual = logits_to_ave_logit_diff(logits, answer_tokens, per_prompt=True)
    # expected = solutions.logits_to_ave_logit_diff(logits, answer_tokens, per_prompt=True)
    final_logits = logits[:, -1, :]
    answer_logits = final_logits.gather(dim=-1, index=answer_tokens)
    correct_logits, incorrect_logits = answer_logits.unbind(dim=-1)
    expected = correct_logits - incorrect_logits
    t.testing.assert_close(actual, expected)

    actual = logits_to_ave_logit_diff(logits, answer_tokens)
    # expected = solutions.logits_to_ave_logit_diff(logits, answer_tokens)
    t.testing.assert_close(actual, expected.mean())

    print("All tests in `test_logits_to_ave_logit_diff` passed!")
