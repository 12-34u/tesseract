"""Hand-checked tests for token accuracy and sequence exact-match accuracy."""

import pytest
import torch

from evaluation.metrics import evaluate, exact_match_accuracy, token_accuracy, token_errors_per_sequence
from models.tesseract import TesseractModel


def seqs(rows):
    return torch.tensor(rows, dtype=torch.long)


def test_identical_sequence_is_exact_match():
    pred, target = seqs([[1, 0, 1, 1]]), seqs([[1, 0, 1, 1]])
    assert exact_match_accuracy(pred, target) == 100.0
    assert token_accuracy(pred, target) == 100.0


def test_one_wrong_token_fails_exact_match():
    pred, target = seqs([[1, 0, 0, 1]]), seqs([[1, 0, 1, 1]])
    assert exact_match_accuracy(pred, target) == 0.0
    assert token_accuracy(pred, target) == 75.0


def test_mixed_batch_by_hand():
    pred = seqs([[1, 0, 1, 1], [1, 0, 0, 1], [0, 0, 0, 0], [1, 1, 1, 1]])
    target = seqs([[1, 0, 1, 1], [1, 0, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
    # Sequences fully correct: rows 0 and 3 → 2/4. Tokens correct: 4+3+0+4 = 11/16.
    assert exact_match_accuracy(pred, target) == 50.0
    assert token_accuracy(pred, target) == pytest.approx(68.75)
    assert token_errors_per_sequence(pred, target).tolist() == [0, 1, 4, 0]


def test_single_error_in_length_32_sequence():
    target = torch.zeros(1, 32, dtype=torch.long)
    pred = target.clone()
    pred[0, 17] = 1
    assert exact_match_accuracy(pred, target) == 0.0
    assert token_accuracy(pred, target) == pytest.approx(100.0 * 31 / 32)


@pytest.mark.parametrize(
    "pred, target",
    [
        (torch.zeros(2, 4, dtype=torch.long), torch.zeros(2, 5, dtype=torch.long)),
        (torch.zeros(4, dtype=torch.long), torch.zeros(4, dtype=torch.long)),
        (torch.zeros(0, 4, dtype=torch.long), torch.zeros(0, 4, dtype=torch.long)),
    ],
)
def test_invalid_shapes_raise(pred, target):
    with pytest.raises(ValueError):
        exact_match_accuracy(pred, target)
    with pytest.raises(ValueError):
        token_accuracy(pred, target)


def test_evaluate_disables_dropout_and_restores_mode():
    torch.manual_seed(0)
    model = TesseractModel(vocab_size=4, d_model=16, num_heads=2, d_ff=32, max_seq_len=8,
                           num_recursive_steps=2, dropout=0.5)
    model.train()
    x = torch.randint(0, 4, (8, 8))
    first, second = evaluate(model, x, x), evaluate(model, x, x)
    assert torch.equal(first.predictions, second.predictions)
    assert model.training, "evaluate() must restore the previous train/eval mode"
