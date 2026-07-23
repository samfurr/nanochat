"""Correctness tests for the reference MLP-free StateHead implementation."""

import copy
from dataclasses import asdict

import pytest
import torch
import torch.nn.functional as F

import nanochat.checkpoint_manager as checkpoint_manager
from nanochat.checkpoint_manager import build_model, save_checkpoint
from nanochat.common import COMPUTE_DTYPE
from nanochat.gpt import GPT, GPTConfig
from nanochat.statehead import (
    StateHead,
    StateHeadBank,
    StateHeadConfig,
    statehead_scan_parallel,
    statehead_scan_sequential,
)


SEQUENCE_LENGTHS = [1, 2, 3, 7, 63, 64, 65, 127, 128, 129, 2048]


def make_scan_inputs(batch_size, sequence_len, n_head=2, head_dim=3, dtype=torch.float32):
    generator = torch.Generator().manual_seed(1000 + batch_size + sequence_len)
    shape = (batch_size, sequence_len, n_head, head_dim)
    a = torch.sigmoid(torch.randn(shape, generator=generator)).to(dtype)
    u = (torch.sigmoid(torch.randn(shape, generator=generator)) * torch.tanh(torch.randn(shape, generator=generator))).to(dtype)
    o = torch.sigmoid(torch.randn(shape, generator=generator)).to(dtype)
    initial_state = torch.randn(batch_size, n_head, head_dim, generator=generator).to(dtype)
    return a, u, o, initial_state


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("sequence_len", SEQUENCE_LENGTHS)
def test_scan_forward_parity(batch_size, sequence_len):
    inputs = make_scan_inputs(batch_size, sequence_len)
    sequential = statehead_scan_sequential(*inputs)
    parallel = statehead_scan_parallel(*inputs, chunk_size=64)
    torch.testing.assert_close(parallel[0], sequential[0], rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(parallel[1], sequential[1], rtol=2e-5, atol=2e-6)


def test_scan_bfloat16_forward_parity():
    inputs = make_scan_inputs(2, 65, dtype=torch.bfloat16)
    sequential = statehead_scan_sequential(*inputs)
    parallel = statehead_scan_parallel(*inputs, chunk_size=64)
    torch.testing.assert_close(parallel[0], sequential[0], rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(parallel[1], sequential[1], rtol=2e-2, atol=2e-2)


def _scan_grads(scan, tensors, weights):
    y, final_state = scan(*tensors)
    loss = (y * weights[0]).sum() + (final_state * weights[1]).sum()
    return torch.autograd.grad(loss, tensors)


def test_scan_gradient_parity():
    base = make_scan_inputs(2, 65)
    sequential_inputs = tuple(t.clone().requires_grad_() for t in base)
    parallel_inputs = tuple(t.clone().requires_grad_() for t in base)
    generator = torch.Generator().manual_seed(99)
    weights = (
        torch.randn(base[0].shape, generator=generator),
        torch.randn(base[-1].shape, generator=generator),
    )
    sequential_grads = _scan_grads(statehead_scan_sequential, sequential_inputs, weights)
    parallel_grads = _scan_grads(
        lambda a, u, o, state: statehead_scan_parallel(a, u, o, state, chunk_size=64),
        parallel_inputs,
        weights,
    )
    for actual, expected in zip(parallel_grads, sequential_grads):
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)


def test_bank_parameter_and_input_gradient_parity():
    torch.manual_seed(7)
    config = StateHeadConfig(sequence_len=17, vocab_size=64, n_layer=1, n_head=2, n_embd=8, scan_chunk_size=8)
    bank = StateHeadBank(config)
    with torch.no_grad():
        bank.gate.weight.normal_(std=0.1)
        bank.out_proj.weight.normal_(std=0.1)
        bank.gate_bias.normal_(std=0.1)
        bank.initial_state.normal_(std=0.1)
    x_data = torch.randn(2, 17, 8)
    output_weight = torch.randn_like(x_data)
    state_weight = torch.randn(2, 2, 4)

    def grads(scan_impl):
        x = x_data.clone().requires_grad_()
        y, final_state = bank(x, scan_impl=scan_impl)
        loss = (y * output_weight).sum() + (final_state * state_weight).sum()
        return torch.autograd.grad(loss, (x, *bank.parameters()))

    sequential_grads = grads("sequential")
    parallel_grads = grads("parallel")
    for actual, expected in zip(parallel_grads, sequential_grads):
        torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-5)


def test_auto_scan_uses_reference_backend_off_cuda():
    torch.manual_seed(8)
    config = StateHeadConfig(
        sequence_len=17,
        vocab_size=64,
        n_layer=1,
        n_head=2,
        n_embd=8,
        scan_chunk_size=8,
    )
    bank = StateHeadBank(config)
    with torch.no_grad():
        bank.gate.weight.normal_(std=0.1)
        bank.out_proj.weight.normal_(std=0.1)
    x = torch.randn(2, 17, 8)
    auto = bank(x, scan_impl="auto")
    reference = bank(x, scan_impl="parallel")
    torch.testing.assert_close(auto[0], reference[0], rtol=0, atol=0)
    torch.testing.assert_close(auto[1], reference[1], rtol=0, atol=0)


def test_native_cuda_scan_rejects_cpu_tensors():
    from nanochat.statehead_cuda import statehead_scan_cuda

    gates = torch.randn(1, 3, 4, 2, 4)
    state = torch.randn(1, 2, 4)
    with pytest.raises(ValueError, match="requires CUDA tensors"):
        statehead_scan_cuda(gates, state)


def test_native_cuda_fake_dispatch_shapes():
    from torch._subclasses.fake_tensor import FakeTensorMode
    from nanochat.statehead_cuda import _statehead_scan_forward

    with FakeTensorMode():
        gates = torch.empty(2, 65, 4, 3, 8, device="cuda", dtype=torch.bfloat16)
        state = torch.empty(2, 3, 8, device="cuda", dtype=torch.bfloat16)
        y, final_state, chunk_initials, activated_gates = _statehead_scan_forward(
            gates, state, 64
        )
    assert y.shape == (2, 65, 3, 8)
    assert y.dtype == torch.bfloat16
    assert final_state.shape == state.shape
    assert chunk_initials.shape == (2, 2, 3, 8)
    assert chunk_initials.dtype == torch.float32
    assert activated_gates.shape == gates.shape
    assert activated_gates.dtype == gates.dtype


def _raw_gate_scan_reference(gates, initial_state):
    a_logits, b_logits, c_logits, o_logits = gates.unbind(dim=2)
    return statehead_scan_sequential(
        torch.sigmoid(a_logits),
        torch.sigmoid(b_logits) * torch.tanh(c_logits),
        torch.sigmoid(o_logits),
        initial_state,
    )


def _reverse_chunk_carries(a, o, grad_y, grad_final_state, chunk_size):
    """Reference the affine reverse summaries used by the native CUDA v2 path."""
    sequence_len = a.size(1)
    n_chunks = (sequence_len + chunk_size - 1) // chunk_size
    chunk_a = []
    chunk_u = []
    for chunk in range(n_chunks):
        start = chunk * chunk_size
        stop = min(start + chunk_size, sequence_len)
        transition_a = torch.ones_like(grad_final_state)
        transition_u = torch.zeros_like(grad_final_state)
        for time in range(stop - 1, start - 1, -1):
            transition_u = a[:, time] * (
                transition_u + grad_y[:, time] * o[:, time]
            )
            transition_a = a[:, time] * transition_a
        chunk_a.append(transition_a)
        chunk_u.append(transition_u)

    carry = grad_final_state
    chunk_carries = [None] * n_chunks
    for chunk in range(n_chunks - 1, -1, -1):
        chunk_carries[chunk] = carry
        carry = chunk_a[chunk] * carry + chunk_u[chunk]
    return torch.stack(chunk_carries, dim=1), carry


@pytest.mark.parametrize(
    ("sequence_len", "chunk_size"),
    [(1, 1), (31, 16), (32, 16), (33, 16), (65, 32), (129, 64), (2048, 32)],
)
def test_reverse_chunk_affine_summary_matches_direct_recurrence(
    sequence_len, chunk_size
):
    generator = torch.Generator().manual_seed(7050 + sequence_len)
    shape = (2, sequence_len, 2, 4)
    a = torch.sigmoid(torch.randn(shape, generator=generator))
    o = torch.sigmoid(torch.randn(shape, generator=generator))
    grad_y = torch.randn(shape, generator=generator)
    grad_final_state = torch.randn(2, 2, 4, generator=generator)

    actual_carries, actual_initial = _reverse_chunk_carries(
        a, o, grad_y, grad_final_state, chunk_size
    )

    n_chunks = (sequence_len + chunk_size - 1) // chunk_size
    expected_carries = [None] * n_chunks
    expected_initial = grad_final_state
    for chunk in range(n_chunks - 1, -1, -1):
        expected_carries[chunk] = expected_initial
        start = chunk * chunk_size
        stop = min(start + chunk_size, sequence_len)
        for time in range(stop - 1, start - 1, -1):
            expected_initial = a[:, time] * (
                expected_initial + grad_y[:, time] * o[:, time]
            )

    torch.testing.assert_close(
        actual_carries,
        torch.stack(expected_carries, dim=1),
        rtol=3e-5,
        atol=3e-6,
    )
    torch.testing.assert_close(
        actual_initial,
        expected_initial,
        rtol=3e-5,
        atol=3e-6,
    )


CUDA_SCAN_CASES = [
    (1, 1),
    (31, 16),
    (32, 16),
    (33, 16),
    (63, 32),
    (64, 32),
    (65, 32),
    (129, 64),
    (2048, 32),
    (2048, 64),
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="native scan requires CUDA")
@pytest.mark.parametrize(("sequence_len", "chunk_size"), CUDA_SCAN_CASES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_native_cuda_forward_matches_reference(sequence_len, chunk_size, dtype):
    from nanochat.statehead_cuda import statehead_scan_cuda

    generator = torch.Generator(device="cuda").manual_seed(7000 + sequence_len)
    gates = torch.randn(
        2, 4, sequence_len, 3, 8, generator=generator, device="cuda", dtype=dtype
    ).transpose(1, 2).contiguous()
    initial_state = torch.randn(
        2, 3, 8, generator=generator, device="cuda", dtype=dtype
    )
    actual = statehead_scan_cuda(gates, initial_state, chunk_size=chunk_size)
    expected = _raw_gate_scan_reference(gates, initial_state)
    tolerance = 2e-2 if dtype == torch.bfloat16 else 3e-5
    torch.testing.assert_close(actual[0], expected[0], rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(actual[1], expected[1], rtol=tolerance, atol=tolerance)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="native scan requires CUDA")
@pytest.mark.parametrize(
    ("sequence_len", "chunk_size"),
    [(1, 1), (31, 16), (33, 16), (65, 32), (129, 64), (2048, 32), (2048, 64)],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_native_cuda_gradient_matches_reference(sequence_len, chunk_size, dtype):
    from nanochat.statehead_cuda import statehead_scan_cuda

    generator = torch.Generator(device="cuda").manual_seed(7100 + sequence_len)
    base_gates = torch.randn(
        2, sequence_len, 4, 2, 4, generator=generator, device="cuda", dtype=dtype
    )
    base_state = torch.randn(
        2, 2, 4, generator=generator, device="cuda", dtype=dtype
    )
    output_weight = torch.randn(
        2, sequence_len, 2, 4, generator=generator, device="cuda", dtype=dtype
    )
    state_weight = torch.randn(
        2, 2, 4, generator=generator, device="cuda", dtype=dtype
    )

    def gradients(scan):
        gates = base_gates.detach().clone().requires_grad_()
        initial_state = base_state.detach().clone().requires_grad_()
        y, final_state = scan(gates, initial_state)
        loss = (y * output_weight).sum() + (final_state * state_weight).sum()
        return torch.autograd.grad(loss, (gates, initial_state))

    actual = gradients(
        lambda gates, state: statehead_scan_cuda(gates, state, chunk_size)
    )
    expected = gradients(_raw_gate_scan_reference)
    if dtype == torch.bfloat16:
        rtol, atol = 3e-2, 3e-2
    else:
        rtol, atol = 5e-4, 5e-5
    torch.testing.assert_close(actual[0], expected[0], rtol=rtol, atol=atol)
    torch.testing.assert_close(actual[1], expected[1], rtol=rtol, atol=atol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="native scan requires CUDA")
def test_native_cuda_scan_compiles_fullgraph():
    from nanochat.statehead_cuda import statehead_scan_cuda

    gates = torch.randn(1, 65, 4, 2, 8, device="cuda", dtype=torch.bfloat16)
    initial_state = torch.randn(1, 2, 8, device="cuda", dtype=torch.bfloat16)

    def loss_fn(raw_gates, state):
        y, final_state = statehead_scan_cuda(raw_gates, state, 64)
        return y.float().square().mean() + final_state.float().square().mean()

    compiled = torch.compile(loss_fn, dynamic=False, fullgraph=True)
    loss = compiled(gates.requires_grad_(), initial_state.requires_grad_())
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(gates.grad).all()
    assert torch.isfinite(initial_state.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="native scan requires CUDA")
def test_compiled_native_cuda_full_model_loss_and_gradients_match_pytorch():
    from nanochat.statehead_cuda import preload_statehead_cuda

    preload_statehead_cuda()
    torch.manual_seed(7200)
    reference_config = StateHeadConfig(
        sequence_len=65,
        vocab_size=128,
        n_layer=2,
        n_head=4,
        n_embd=32,
        scan_chunk_size=64,
        scan_backend="pytorch",
    )
    native_config = copy.deepcopy(reference_config)
    native_config.scan_backend = "cuda"
    reference_model = StateHead(reference_config).cuda()
    reference_model.init_weights()
    with torch.no_grad():
        reference_model.smear_lambda.fill_(0.4)
        for block in reference_model.transformer.h:
            block.state_bank.out_proj.weight.normal_(std=0.05)
            block.state_bank.initial_state.normal_(std=0.05)
    native_model = StateHead(native_config).cuda()
    native_model.init_weights()
    native_model.load_state_dict(reference_model.state_dict())

    inputs = torch.randint(0, reference_config.vocab_size, (2, 65), device="cuda")
    targets = torch.roll(inputs, shifts=-1, dims=1)

    def run_compiled(model):
        def logits_and_loss(idx, target):
            logits = model(idx)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target.view(-1),
            )
            return logits, loss

        compiled = torch.compile(logits_and_loss, dynamic=False, fullgraph=True)
        logits, loss = compiled(inputs, targets)
        loss.backward()
        gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
        }
        return logits.detach(), loss.detach(), gradients

    expected_logits, expected_loss, expected_gradients = run_compiled(reference_model)
    actual_logits, actual_loss, actual_gradients = run_compiled(native_model)
    if COMPUTE_DTYPE == torch.bfloat16:
        forward_rtol, forward_atol = 3e-2, 3e-2
        gradient_rtol, gradient_atol = 6e-2, 6e-3
    else:
        forward_rtol, forward_atol = 2e-4, 2e-5
        gradient_rtol, gradient_atol = 3e-3, 3e-4
    torch.testing.assert_close(
        actual_logits,
        expected_logits,
        rtol=forward_rtol,
        atol=forward_atol,
    )
    torch.testing.assert_close(
        actual_loss,
        expected_loss,
        rtol=forward_rtol,
        atol=forward_atol,
    )
    assert actual_gradients.keys() == expected_gradients.keys()
    for name in actual_gradients:
        torch.testing.assert_close(
            actual_gradients[name],
            expected_gradients[name],
            rtol=gradient_rtol,
            atol=gradient_atol,
            msg=lambda message, parameter_name=name: f"{parameter_name}: {message}",
        )


def test_statehead_fp8_filter_matches_large_linear_recipe():
    from nanochat.fp8 import Float8Linear, convert_to_float8_training, is_float8_linear_eligible

    config = StateHeadConfig(
        sequence_len=8,
        vocab_size=256,
        n_layer=1,
        n_head=1,
        n_embd=128,
    )
    model = StateHead(config)
    flops_before = model.estimate_flops()
    convert_to_float8_training(model, module_filter_fn=is_float8_linear_eligible)
    converted = [name for name, module in model.named_modules() if isinstance(module, Float8Linear)]
    assert converted == [
        "transformer.h.0.state_bank.gate",
        "transformer.h.0.state_bank.out_proj",
        "lm_head",
    ]
    assert model.estimate_flops() == flops_before


@pytest.mark.parametrize("split", [1, 63, 64, 65, 128])
def test_segmentation_property(split):
    a, u, o, initial_state = make_scan_inputs(2, 129)
    full_y, full_state = statehead_scan_parallel(a, u, o, initial_state, chunk_size=64)
    left_y, split_state = statehead_scan_parallel(
        a[:, :split], u[:, :split], o[:, :split], initial_state, chunk_size=64
    )
    right_y, segmented_state = statehead_scan_parallel(
        a[:, split:], u[:, split:], o[:, split:], split_state, chunk_size=64
    )
    segmented_y = torch.cat((left_y, right_y), dim=1)
    torch.testing.assert_close(segmented_y, full_y, rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(segmented_state, full_state, rtol=3e-5, atol=3e-6)


def test_batch_independence():
    inputs = make_scan_inputs(2, 65)
    batched_y, batched_state = statehead_scan_parallel(*inputs)
    individual = [
        statehead_scan_parallel(*(tensor[i:i + 1] for tensor in inputs))
        for i in range(2)
    ]
    expected_y = torch.cat([result[0] for result in individual], dim=0)
    expected_state = torch.cat([result[1] for result in individual], dim=0)
    torch.testing.assert_close(batched_y, expected_y)
    torch.testing.assert_close(batched_state, expected_state)


def make_statehead(n_layer=2):
    config = StateHeadConfig(
        sequence_len=8,
        vocab_size=64,
        n_layer=n_layer,
        n_head=4,
        n_embd=32,
        scan_chunk_size=4,
    )
    model = StateHead(config)
    model.init_weights()
    return model


def test_training_rows_reset_to_learned_initial_state():
    torch.manual_seed(11)
    model = make_statehead()
    row = torch.randint(0, model.config.vocab_size, (2, 8))
    preceding_row = torch.randint(0, model.config.vocab_size, (2, 8))
    expected = model(row)
    model(preceding_row)
    actual = model(row)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_prefill_matches_token_by_token_decode_with_smear():
    torch.manual_seed(12)
    model = make_statehead()
    with torch.no_grad():
        model.smear_lambda.fill_(0.5)
        for block in model.transformer.h:
            block.state_bank.out_proj.weight.normal_(std=0.05)
    tokens = torch.randint(0, model.config.vocab_size, (2, 8))
    full_logits, full_states, full_prev = model.forward_with_state(tokens)

    states = None
    prev_embedding = None
    logits = []
    for t in range(tokens.size(1)):
        step_logits, states, prev_embedding = model.forward_with_state(
            tokens[:, t:t + 1], states=states, prev_embedding=prev_embedding
        )
        logits.append(step_logits)
    decode_logits = torch.cat(logits, dim=1)
    reduced_precision = full_states[0].dtype in (torch.float16, torch.bfloat16)
    rtol, atol = (2e-2, 2e-2) if reduced_precision else (3e-5, 3e-6)
    torch.testing.assert_close(decode_logits, full_logits, rtol=rtol, atol=atol)
    for actual, expected in zip(states, full_states):
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    torch.testing.assert_close(prev_embedding, full_prev)


class _TokenizerStub:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    def get_vocab_size(self):
        return self.vocab_size


@pytest.mark.parametrize(
    "model_type",
    [None, "gpt", "statehead"],
    ids=["legacy-gpt", "gpt", "statehead"],
)
def test_checkpoint_round_trip(tmp_path, monkeypatch, model_type):
    torch.manual_seed(13)
    if model_type in (None, "gpt"):
        config = GPTConfig(
            sequence_len=8, vocab_size=64, n_layer=1, n_head=1,
            n_kv_head=1, n_embd=32, window_pattern="L",
        )
        model = GPT(config)
    else:
        config = StateHeadConfig(
            sequence_len=8, vocab_size=64, n_layer=1, n_head=4,
            n_embd=32, scan_chunk_size=4,
        )
        model = StateHead(config)
    model.init_weights()
    model.eval()
    tokens = torch.randint(0, config.vocab_size, (2, 8))
    expected = model(tokens)
    metadata = {"step": 1, "model_config": asdict(config)}
    if model_type is not None:
        metadata["model_type"] = model_type
    save_checkpoint(tmp_path, 1, model.state_dict(), None, metadata)
    monkeypatch.setattr(checkpoint_manager, "get_tokenizer", lambda: _TokenizerStub(config.vocab_size))
    loaded, _, loaded_metadata = build_model(tmp_path, 1, torch.device("cpu"), phase="eval")
    actual = loaded(tokens)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert loaded_metadata.get("model_type", "gpt") == (model_type or "gpt")


def test_meta_device_initialization():
    config = StateHeadConfig(
        sequence_len=8, vocab_size=64, n_layer=2, n_head=4,
        n_embd=32, scan_chunk_size=4,
    )
    with torch.device("meta"):
        model = StateHead(config)
    assert all(parameter.is_meta for parameter in model.parameters())
    model.to_empty(device="cpu")
    model.init_weights()
    assert all(not parameter.is_meta for parameter in model.parameters())
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())


def test_optimizer_partition_is_exact():
    model = make_statehead()
    optimizer = model.setup_optimizer()
    grouped = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert len(grouped) == len({id(parameter) for parameter in grouped})
    assert {id(parameter) for parameter in grouped} == {id(parameter) for parameter in trainable}
    muon_ids = {
        id(parameter)
        for group in optimizer.param_groups if group["kind"] == "muon"
        for parameter in group["params"]
    }
    for block in model.transformer.h:
        assert id(block.state_bank.gate_bias) not in muon_ids
        assert id(block.state_bank.initial_state) not in muon_ids
        assert id(block.state_bank.gate.weight) in muon_ids
        assert id(block.state_bank.out_proj.weight) in muon_ids


def test_tiny_batch_overfit():
    torch.manual_seed(14)
    model = make_statehead(n_layer=1)
    inputs = torch.arange(8).repeat(4, 1)
    targets = torch.roll(inputs, shifts=-1, dims=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
    losses = []
    for _ in range(40):
        optimizer.zero_grad(set_to_none=True)
        loss = model(inputs, targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert torch.isfinite(torch.tensor(losses)).all()
    assert losses[-1] < 0.5 * losses[0], f"loss did not substantially decrease: {losses[0]} -> {losses[-1]}"


def test_eager_and_compiled_outputs_and_gradients_match():
    torch.manual_seed(15)
    eager_model = make_statehead()
    with torch.no_grad():
        eager_model.smear_lambda.fill_(0.4)
        for block in eager_model.transformer.h:
            block.state_bank.out_proj.weight.normal_(std=0.05)
    compiled_base = copy.deepcopy(eager_model)
    compiled_model = torch.compile(
        compiled_base,
        backend="aot_eager",
        dynamic=False,
        fullgraph=True,
    )
    inputs = torch.randint(0, eager_model.config.vocab_size, (2, 8))
    targets = torch.roll(inputs, shifts=-1, dims=1)

    eager_logits = eager_model(inputs)
    compiled_logits = compiled_model(inputs)
    torch.testing.assert_close(compiled_logits, eager_logits, rtol=1e-5, atol=1e-6)

    eager_loss = eager_model(inputs, targets)
    compiled_loss = compiled_model(inputs, targets)
    eager_loss.backward()
    compiled_loss.backward()
    torch.testing.assert_close(compiled_loss, eager_loss, rtol=1e-5, atol=1e-6)
    eager_grads = dict(eager_model.named_parameters())
    compiled_grads = dict(compiled_base.named_parameters())
    assert eager_grads.keys() == compiled_grads.keys()
    for name in eager_grads:
        actual = compiled_grads[name].grad
        expected = eager_grads[name].grad
        assert actual is not None and expected is not None, f"missing gradient for {name}"
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)


def _fixed_seed_loss_curve(seed, steps=20):
    torch.manual_seed(seed)
    model = make_statehead(n_layer=1)
    inputs = torch.arange(8).repeat(4, 1)
    targets = torch.roll(inputs, shifts=-1, dims=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
    losses = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = model(inputs, targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach())
    return torch.stack(losses)


def test_fixed_seed_loss_curve_is_repeatable():
    first = _fixed_seed_loss_curve(1337)
    second = _fixed_seed_loss_curve(1337)
    torch.testing.assert_close(second, first, rtol=0, atol=0)
    assert first[-1] < first[0]


def test_naive_generation_path_runs_without_engine_cache():
    torch.manual_seed(16)
    model = make_statehead(n_layer=1)
    generated = list(model.generate([0, 1], max_tokens=4, temperature=0))
    assert len(generated) == 4
    assert all(0 <= token < model.config.vocab_size for token in generated)
