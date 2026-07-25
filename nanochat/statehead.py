"""Correctness-first, MLP-free StateHead language model."""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import COMPUTE_DTYPE, print0
from nanochat.gpt import Linear, has_ve, norm
from nanochat.optim import MuonAdamW


@dataclass
class StateHeadConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_embd: int = 768
    scan_chunk_size: int = 64
    scan_backend: str = "auto"
    retention_bias: float = 2.0
    learned_initial_state: bool = True
    value_embeddings: bool = False

    @property
    def head_dim(self):
        assert self.n_embd % self.n_head == 0
        return self.n_embd // self.n_head


def _scan_tensors(a, u, o, initial_state):
    """Promote recurrence math to fp32 for reduced-precision activations."""
    scan_dtype = torch.float32 if a.dtype in (torch.float16, torch.bfloat16) else a.dtype
    return (
        a.to(scan_dtype),
        u.to(scan_dtype),
        o.to(scan_dtype),
        initial_state.to(scan_dtype),
    )


def statehead_scan_sequential(a, u, o, initial_state):
    """Simple recurrent oracle. Shapes are [B, T, H, Dh] and [B, H, Dh]."""
    a_scan, u_scan, o_scan, state = _scan_tensors(a, u, o, initial_state)
    outputs = []
    for t in range(a.size(1)):
        state = a_scan[:, t] * state + u_scan[:, t]
        outputs.append(o_scan[:, t] * state)
    y = torch.stack(outputs, dim=1)
    return y.to(a.dtype), state.to(initial_state.dtype)


def _associative_prefix(a, u, dim):
    """Inclusive prefix scan of affine transitions using functional doubling."""
    length = a.size(dim)
    offset = 1
    while offset < length:
        prefix = [slice(None)] * a.ndim
        suffix = [slice(None)] * a.ndim
        prefix[dim] = slice(None, -offset)
        suffix[dim] = slice(offset, None)
        prefix = tuple(prefix)
        suffix = tuple(suffix)

        a_left, u_left = a[prefix], u[prefix]
        a_right, u_right = a[suffix], u[suffix]
        composed_a = a_right * a_left
        composed_u = a_right * u_left + u_right

        unchanged = [slice(None)] * a.ndim
        unchanged[dim] = slice(None, offset)
        unchanged = tuple(unchanged)
        a = torch.cat((a[unchanged], composed_a), dim=dim)
        u = torch.cat((u[unchanged], composed_u), dim=dim)
        offset *= 2
    return a, u


def statehead_scan_parallel(a, u, o, initial_state, chunk_size=64):
    """Chunked associative StateHead scan implemented in functional PyTorch."""
    assert a.shape == u.shape == o.shape
    assert a.ndim == 4
    assert initial_state.shape == (a.size(0), a.size(2), a.size(3))
    assert a.size(1) > 0
    assert chunk_size > 0

    output_dtype = a.dtype
    state_dtype = initial_state.dtype
    a, u, o, initial_state = _scan_tensors(a, u, o, initial_state)
    batch_size, sequence_len, n_head, head_dim = a.shape
    n_chunks = (sequence_len + chunk_size - 1) // chunk_size
    padded_len = n_chunks * chunk_size
    pad_len = padded_len - sequence_len
    if pad_len:
        pad_shape = (batch_size, pad_len, n_head, head_dim)
        a = torch.cat((a, torch.ones(pad_shape, dtype=a.dtype, device=a.device)), dim=1)
        u = torch.cat((u, torch.zeros(pad_shape, dtype=u.dtype, device=u.device)), dim=1)
        o = torch.cat((o, torch.zeros(pad_shape, dtype=o.dtype, device=o.device)), dim=1)

    # Prefix transitions within every chunk.
    chunk_shape = (batch_size, n_chunks, chunk_size, n_head, head_dim)
    a_chunks = a.reshape(chunk_shape)
    u_chunks = u.reshape(chunk_shape)
    prefix_a, prefix_u = _associative_prefix(a_chunks, u_chunks, dim=2)

    # Prefix scan chunk summaries to obtain the state entering every chunk.
    chunk_a = prefix_a[:, :, -1]
    chunk_u = prefix_u[:, :, -1]
    chunks_prefix_a, chunks_prefix_u = _associative_prefix(chunk_a, chunk_u, dim=1)
    if n_chunks == 1:
        chunk_initial = initial_state[:, None]
    else:
        prior_states = (
            chunks_prefix_a[:, :-1] * initial_state[:, None]
            + chunks_prefix_u[:, :-1]
        )
        chunk_initial = torch.cat((initial_state[:, None], prior_states), dim=1)

    states = prefix_a * chunk_initial[:, :, None] + prefix_u
    states = states.reshape(batch_size, padded_len, n_head, head_dim)
    y = (o * states)[:, :sequence_len]
    final_state = states[:, sequence_len - 1]
    return y.to(output_dtype), final_state.to(state_dtype)


class StateHeadBank(nn.Module):
    def __init__(self, config, use_value_embedding=False):
        super().__init__()
        if config.scan_backend not in ("auto", "pytorch", "cuda"):
            raise ValueError(f"Unknown StateHead scan backend: {config.scan_backend}")
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_embd = config.n_embd
        self.chunk_size = config.scan_chunk_size
        self.scan_backend = config.scan_backend
        self.retention_bias = config.retention_bias
        self.gate = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.gate_bias = nn.Parameter(torch.zeros(4 * config.n_embd))
        self.out_proj = Linear(config.n_embd, config.n_embd, bias=False)
        self.ve_gate = (
            Linear(12, config.n_head, bias=False) if use_value_embedding else None
        )
        initial_state = torch.zeros(config.n_head, config.head_dim)
        if config.learned_initial_state:
            self.initial_state = nn.Parameter(initial_state)
        else:
            self.register_buffer("initial_state", initial_state)

    def fresh_state(self, batch_size, device, dtype):
        return self.initial_state.to(device=device, dtype=dtype).unsqueeze(0).expand(
            batch_size, -1, -1
        )

    def forward(self, x, state=None, scan_impl=None, value_embedding=None):
        batch_size, sequence_len, n_embd = x.shape
        if state is None:
            state = self.fresh_state(batch_size, x.device, x.dtype)
        if (value_embedding is None) != (self.ve_gate is None):
            raise ValueError(
                "value_embedding must be provided exactly for StateHead banks "
                "configured with a value embedding"
            )
        candidate_residual = None
        if value_embedding is not None:
            if value_embedding.shape != x.shape:
                raise ValueError(
                    f"value_embedding shape {value_embedding.shape} must match x shape {x.shape}"
                )
            value_embedding = value_embedding.view(
                batch_size, sequence_len, self.n_head, self.head_dim
            )
            value_gate = 3 * torch.sigmoid(self.ve_gate(x[..., :12]))
            candidate_residual = value_gate.unsqueeze(-1) * value_embedding
        gate_logits = self.gate(x)
        if scan_impl is None:
            scan_impl = self.scan_backend
        if scan_impl == "auto":
            scan_impl = "cuda" if gate_logits.is_cuda else "parallel"
        elif scan_impl == "pytorch":
            scan_impl = "parallel"
        if scan_impl == "cuda":
            from nanochat.statehead_cuda import statehead_scan_cuda

            gate_logits = gate_logits.view(
                batch_size, sequence_len, 4, self.n_head, self.head_dim
            )
            y, final_state = statehead_scan_cuda(
                gate_logits,
                state,
                self.chunk_size,
                gate_bias=self.gate_bias.to(x.dtype),
                candidate_residual=candidate_residual,
            )
            y = y.reshape(batch_size, sequence_len, n_embd)
            return self.out_proj(y), final_state

        gates = gate_logits + self.gate_bias.to(x.dtype)
        gates = gates.view(batch_size, sequence_len, 4, self.n_head, self.head_dim)
        a_logits, b_logits, c_logits, o_logits = gates.unbind(dim=2)
        a = torch.sigmoid(a_logits)
        candidate = torch.tanh(c_logits)
        if candidate_residual is not None:
            candidate = candidate + candidate_residual
        u = torch.sigmoid(b_logits) * candidate
        o = torch.sigmoid(o_logits)
        if scan_impl == "parallel":
            y, final_state = statehead_scan_parallel(a, u, o, state, self.chunk_size)
        elif scan_impl == "sequential":
            y, final_state = statehead_scan_sequential(a, u, o, state)
        else:
            raise ValueError(f"Unknown StateHead scan implementation: {scan_impl}")
        y = y.reshape(batch_size, sequence_len, n_embd)
        return self.out_proj(y), final_state


class StateHeadBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.state_bank = StateHeadBank(
            config,
            use_value_embedding=config.value_embeddings
            and has_ve(layer_idx, config.n_layer),
        )

    def forward(self, x, state=None, scan_impl=None, value_embedding=None):
        delta, next_state = self.state_bank(
            norm(x),
            state,
            scan_impl=scan_impl,
            value_embedding=value_embedding,
        )
        return x + delta, next_state


class StateHead(nn.Module):
    """MLP-free recurrent LM preserving nanochat's architecture-neutral scaffold."""

    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__()
        assert config.n_embd >= 24, "nanochat token smearing requires n_embd >= 24"
        self.config = config
        padded_vocab_size = (
            (config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to
        ) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([
                StateHeadBlock(config, layer_idx)
                for layer_idx in range(config.n_layer)
            ]),
        })
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        self.smear_gate = Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))
        self.value_embeds = nn.ModuleDict({
            str(layer_idx): nn.Embedding(padded_vocab_size, config.n_embd)
            for layer_idx in range(config.n_layer)
            if config.value_embeddings and has_ve(layer_idx, config.n_layer)
        })

    @torch.no_grad()
    def init_weights(self):
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        n_embd = self.config.n_embd
        bound = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            bank = block.state_bank
            torch.nn.init.uniform_(bank.gate.weight, -bound, bound)
            torch.nn.init.zeros_(bank.out_proj.weight)
            torch.nn.init.zeros_(bank.gate_bias)
            bank.gate_bias[:n_embd].fill_(self.config.retention_bias)
            torch.nn.init.zeros_(bank.initial_state)
            if bank.ve_gate is not None:
                torch.nn.init.uniform_(bank.ve_gate.weight, 0.0, 0.02)

        for value_embedding in self.value_embeds.values():
            torch.nn.init.uniform_(value_embedding.weight, -bound, bound)

        n_layer = self.config.n_layer
        for i in range(n_layer):
            self.resid_lambdas[i] = 1.15 - (0.10 * i / max(n_layer - 1, 1))
            self.x0_lambdas[i] = 0.20 - (0.15 * i / max(n_layer - 1, 1))
        torch.nn.init.zeros_(self.smear_lambda)
        torch.nn.init.constant_(self.backout_lambda, 0.2)
        torch.nn.init.uniform_(self.smear_gate.weight, 0.0, 0.02)
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)
            for value_embedding in self.value_embeds.values():
                value_embedding.to(dtype=COMPUTE_DTYPE)

    def get_device(self):
        return self.transformer.wte.weight.device

    def num_matmul_params(self):
        return sum(m.weight.numel() for m in self.modules() if isinstance(m, nn.Linear))

    def estimate_flops(self):
        # Projection FLOPs use nanochat's forward+backward convention. The extra
        # term reports the elementwise recurrence without implying wall-clock speed.
        recurrence_flops = 9 * self.config.n_layer * self.config.n_embd
        return 6 * self.num_matmul_params() + recurrence_flops

    def recurrent_state_bytes(self, batch_size=1, dtype=None):
        dtype = COMPUTE_DTYPE if dtype is None else dtype
        return batch_size * self.config.n_layer * self.config.n_embd * dtype.itemsize

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        banks = [block.state_bank for block in self.transformer.h]
        statehead_matrices = sum(
            bank.gate.weight.numel()
            + bank.out_proj.weight.numel()
            + (bank.ve_gate.weight.numel() if bank.ve_gate is not None else 0)
            for bank in banks
        )
        statehead_vectors = sum(
            bank.gate_bias.numel()
            + (bank.initial_state.numel() if bank.initial_state.requires_grad else 0)
            for bank in banks
        )
        scalars = (
            self.resid_lambdas.numel()
            + self.x0_lambdas.numel()
            + self.smear_gate.weight.numel()
            + self.smear_lambda.numel()
            + self.backout_lambda.numel()
        )
        total = (
            wte
            + value_embeds
            + lm_head
            + statehead_matrices
            + statehead_vectors
            + scalars
        )
        assert total == sum(p.numel() for p in self.parameters())
        return {
            "wte": wte,
            "value_embeds": value_embeds,
            "lm_head": lm_head,
            "statehead_matrices": statehead_matrices,
            "statehead_vectors": statehead_vectors,
            "transformer_matrices": statehead_matrices,
            "scalars": scalars,
            "total": total,
        }

    def setup_optimizer(
        self,
        unembedding_lr=0.004,
        embedding_lr=0.2,
        matrix_lr=0.02,
        weight_decay=0.0,
        scalar_lr=0.5,
    ):
        model_dim = self.config.n_embd
        banks = [block.state_bank for block in self.transformer.h]
        matrix_params = [
            parameter
            for bank in banks
            for parameter in (
                bank.gate.weight,
                bank.out_proj.weight,
                None if bank.ve_gate is None else bank.ve_gate.weight,
            )
            if parameter is not None
        ]
        vector_params = [p for bank in banks for p in (bank.gate_bias, bank.initial_state) if p.requires_grad]
        embedding_params = list(self.transformer.wte.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        smear_params = [self.smear_gate.weight, self.smear_lambda, self.backout_lambda]
        all_groups = (
            matrix_params + vector_params + embedding_params + lm_head_params
            + value_embeds_params + resid_params + x0_params + smear_params
        )
        trainable = [p for p in self.parameters() if p.requires_grad]
        assert len(all_groups) == len(trainable)
        assert {id(p) for p in all_groups} == {id(p) for p in trainable}

        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")
        param_groups = [
            dict(kind="adamw", params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind="adamw", params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind="adamw", params=vector_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.0),
            dict(kind="adamw", params=resid_params, lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
            dict(kind="adamw", params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
            dict(kind="adamw", params=smear_params, lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        if value_embeds_params:
            param_groups.insert(2, dict(
                kind="adamw",
                params=value_embeds_params,
                lr=embedding_lr * dmodel_lr_scale * 0.5,
                betas=(0.8, 0.995),
                eps=1e-10,
                weight_decay=0.01,
            ))
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind="muon", params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay,
            ))
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward_with_state(self, idx, states=None, prev_embedding=None, scan_impl=None):
        """Forward with explicit recurrent state, used for correctness checks and naive decode."""
        batch_size, sequence_len = idx.shape
        if states is not None:
            assert len(states) == self.config.n_layer
        x = norm(self.transformer.wte(idx).to(COMPUTE_DTYPE))
        next_prev_embedding = x[:, -1:, :]
        if prev_embedding is None:
            if sequence_len > 1:
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
                x = torch.cat((x[:, :1], x[:, 1:] + gate * x[:, :-1]), dim=1)
        else:
            previous = torch.cat((prev_embedding, x[:, :-1]), dim=1)
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, :, :24]))
            x = x + gate * previous

        x0 = x
        backout_layer = self.config.n_layer // 2
        x_backout = None
        next_states = []
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            state = None if states is None else states[i]
            value_embedding = (
                self.value_embeds[str(i)](idx).to(x.dtype)
                if str(i) in self.value_embeds
                else None
            )
            x, next_state = block(
                x,
                state,
                scan_impl=scan_impl,
                value_embedding=value_embedding,
            )
            next_states.append(next_state)
            if i == backout_layer:
                x_backout = x
        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        x = norm(x)
        softcap = 15
        logits = self.lm_head(x)[..., :self.config.vocab_size].float()
        logits = softcap * torch.tanh(logits / softcap)
        return logits, next_states, next_prev_embedding

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction="mean"):
        if kv_cache is not None:
            raise NotImplementedError("StateHead engine cache integration is deferred beyond Phase 1")
        logits, _, _ = self.forward_with_state(idx)
        if targets is None:
            return logits
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
            reduction=loss_reduction,
        )

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """Naive full-prefix generation; fixed-size engine caching is a later phase."""
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device).manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        for _ in range(max_tokens):
            logits = self.forward(ids)[:, -1]
            if top_k is not None and top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("inf")
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            yield next_ids.item()
