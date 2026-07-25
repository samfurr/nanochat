"""StateHead-specific correctness checks for the CORE evaluator."""

import torch
import torch.nn.functional as F

from nanochat.core_eval import forward_model, stack_sequences
from nanochat.statehead import StateHead, StateHeadConfig


def make_model():
    torch.manual_seed(20260725)
    model = StateHead(
        StateHeadConfig(
            sequence_len=32,
            vocab_size=64,
            n_layer=2,
            n_head=4,
            n_embd=32,
            scan_chunk_size=4,
            scan_backend="pytorch",
            value_embeddings=True,
        )
    )
    model.init_weights()
    with torch.no_grad():
        model.smear_lambda.fill_(0.5)
        for block in model.transformer.h:
            block.state_bank.out_proj.weight.normal_(std=0.05)
            block.state_bank.initial_state.normal_(std=0.05)
    model.eval()
    return model


def evaluator_scores(model, sequences, start_indices, end_indices, pad_token_id=0):
    input_ids = stack_sequences(sequences, pad_token_id)
    losses, _ = forward_model(model, input_ids)
    return torch.stack([
        losses[row, start - 1:end - 1].mean()
        for row, (start, end) in enumerate(zip(start_indices, end_indices))
    ])


def unpadded_scores(model, sequences, start_indices, end_indices):
    scores = []
    for sequence, start, end in zip(sequences, start_indices, end_indices):
        input_ids = torch.tensor(sequence).unsqueeze(0)
        logits, _, _ = model.forward_with_state(input_ids, scan_impl="sequential")
        targets = input_ids[:, start:end]
        answer_logits = logits[:, start - 1:end - 1]
        scores.append(F.cross_entropy(
            answer_logits.reshape(-1, answer_logits.size(-1)),
            targets.reshape(-1),
        ))
    return torch.stack(scores)


def shared_prompt_state_scores(model, sequences, answer_start):
    """Score each candidate from a separately cloned common prompt state."""
    common_prompt = sequences[0][:answer_start]
    assert all(sequence[:answer_start] == common_prompt for sequence in sequences)
    prompt_ids = torch.tensor(common_prompt).unsqueeze(0)
    prompt_logits, prompt_states, prompt_prev = model.forward_with_state(
        prompt_ids,
        scan_impl="sequential",
    )

    scores = []
    for sequence in sequences:
        answer = sequence[answer_start:]
        candidate_logits = [prompt_logits[:, -1:]]
        if len(answer) > 1:
            continuation_inputs = torch.tensor(
                sequence[answer_start:-1]
            ).unsqueeze(0)
            continuation_logits, _, _ = model.forward_with_state(
                continuation_inputs,
                states=[state.clone() for state in prompt_states],
                prev_embedding=prompt_prev.clone(),
                scan_impl="sequential",
            )
            candidate_logits.append(continuation_logits)
        logits = torch.cat(candidate_logits, dim=1)
        targets = torch.tensor(answer).unsqueeze(0)
        scores.append(F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        ))
    return torch.stack(scores)


def test_batched_padded_scores_match_independent_unpadded_oracle():
    model = make_model()
    sequences = [
        [1, 5, 8, 13, 21],
        [1, 5, 8, 34, 35, 36, 37],
        [1, 5, 8, 55, 56],
        [1, 5, 8, 60, 61, 62],
    ]
    starts = [3] * len(sequences)
    ends = [len(sequence) for sequence in sequences]

    actual = evaluator_scores(model, sequences, starts, ends)
    expected = unpadded_scores(model, sequences, starts, ends)
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)


def test_candidate_order_cannot_change_scores():
    model = make_model()
    sequences = [
        [2, 3, 4, 10, 11],
        [2, 3, 4, 20, 21, 22, 23],
        [2, 3, 4, 30, 31, 32],
        [2, 3, 4, 40, 41],
    ]
    starts = [3] * len(sequences)
    ends = [len(sequence) for sequence in sequences]
    expected = evaluator_scores(model, sequences, starts, ends)

    permutation = [2, 0, 3, 1]
    permuted = evaluator_scores(
        model,
        [sequences[index] for index in permutation],
        [starts[index] for index in permutation],
        [ends[index] for index in permutation],
    )
    restored = torch.empty_like(permuted)
    for new_index, old_index in enumerate(permutation):
        restored[old_index] = permuted[new_index]
    torch.testing.assert_close(restored, expected, rtol=3e-5, atol=3e-6)


def test_answer_only_loss_consumes_prompt_and_reuses_no_candidate_state():
    model = make_model()
    sequences = [
        [7, 8, 9, 10, 11, 12],
        [7, 8, 9, 20, 21],
        [7, 8, 9, 30, 31, 32, 33],
    ]
    starts = [3] * len(sequences)
    ends = [len(sequence) for sequence in sequences]

    actual = evaluator_scores(model, sequences, starts, ends)
    expected = shared_prompt_state_scores(model, sequences, answer_start=3)
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)


def test_right_padding_cannot_change_scored_prefix():
    model = make_model()
    sequence = [3, 5, 7, 11, 13]
    padded = sequence + [0, 0, 0, 0]

    true_logits, true_states, _ = model.forward_with_state(
        torch.tensor(sequence).unsqueeze(0),
        scan_impl="sequential",
    )
    padded_logits, padded_states, _ = model.forward_with_state(
        torch.tensor(padded).unsqueeze(0),
        scan_impl="sequential",
    )

    torch.testing.assert_close(
        padded_logits[:, :len(sequence)],
        true_logits,
        rtol=3e-5,
        atol=3e-6,
    )
    assert any(
        not torch.allclose(padded_state, true_state)
        for padded_state, true_state in zip(padded_states, true_states)
    ), "the test must exercise padding that changes discarded recurrent state"

    score = evaluator_scores(model, [padded], [3], [len(sequence)])
    oracle = unpadded_scores(model, [sequence], [3], [len(sequence)])
    torch.testing.assert_close(score, oracle, rtol=3e-5, atol=3e-6)
