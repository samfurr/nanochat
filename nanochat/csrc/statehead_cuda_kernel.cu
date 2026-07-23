#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <vector>

namespace {

constexpr int kForwardThreads = 256;
constexpr int kBackwardThreads = 128;
constexpr int kMaxChunkSize = 64;

__device__ __forceinline__ float sigmoidf(float value) {
  return 1.0f / (1.0f + expf(-value));
}

template <typename scalar_t>
__global__ void statehead_activate_gates_kernel(
    const scalar_t* __restrict__ gates,
    const scalar_t* __restrict__ gate_bias,
    scalar_t* __restrict__ activated_gates,
    int64_t sequence_len,
    int64_t gate_stride) {
  const int64_t state =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (state >= gate_stride) {
    return;
  }

  const int64_t token =
      static_cast<int64_t>(blockIdx.z) * sequence_len + blockIdx.y;
  const int64_t gate_base = token * 4 * gate_stride + state;
  activated_gates[gate_base] = static_cast<scalar_t>(
      sigmoidf(
          static_cast<float>(gates[gate_base]) +
          static_cast<float>(gate_bias[state])));
  activated_gates[gate_base + gate_stride] = static_cast<scalar_t>(
      sigmoidf(
          static_cast<float>(gates[gate_base + gate_stride]) +
          static_cast<float>(gate_bias[gate_stride + state])));
  activated_gates[gate_base + 2 * gate_stride] = static_cast<scalar_t>(
      tanhf(
          static_cast<float>(gates[gate_base + 2 * gate_stride]) +
          static_cast<float>(gate_bias[2 * gate_stride + state])));
  activated_gates[gate_base + 3 * gate_stride] = static_cast<scalar_t>(
      sigmoidf(
          static_cast<float>(gates[gate_base + 3 * gate_stride]) +
          static_cast<float>(gate_bias[3 * gate_stride + state])));
}

template <typename scalar_t>
__device__ __forceinline__ float load_gate(
    const scalar_t* gates,
    int64_t batch,
    int64_t time,
    int64_t gate,
    int64_t head,
    int64_t dim,
    int64_t sequence_len,
    int64_t n_head,
    int64_t head_dim) {
  const int64_t index =
      ((((batch * sequence_len + time) * 4 + gate) * n_head + head) *
       head_dim + dim);
  return static_cast<float>(gates[index]);
}

template <typename scalar_t>
__global__ void statehead_chunk_summary_kernel(
    const scalar_t* __restrict__ gates,
    float* __restrict__ chunk_a,
    float* __restrict__ chunk_u,
    int64_t total_states,
    int64_t sequence_len,
    int64_t n_head,
    int64_t head_dim,
    int64_t chunk_size,
    int64_t n_chunks) {
  const int64_t state_index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (state_index >= total_states) {
    return;
  }

  const int64_t chunk = blockIdx.y;
  const int64_t dim = state_index % head_dim;
  const int64_t head = (state_index / head_dim) % n_head;
  const int64_t batch = state_index / (n_head * head_dim);
  const int64_t start = chunk * chunk_size;
  const int64_t stop =
      start + chunk_size < sequence_len ? start + chunk_size : sequence_len;

  float transition_a = 1.0f;
  float transition_u = 0.0f;
  for (int64_t time = start; time < stop; ++time) {
    const float a = load_gate(
        gates, batch, time, 0, head, dim, sequence_len, n_head, head_dim);
    const float b = load_gate(
        gates, batch, time, 1, head, dim, sequence_len, n_head, head_dim);
    const float c = load_gate(
        gates, batch, time, 2, head, dim, sequence_len, n_head, head_dim);
    transition_u = a * transition_u + b * c;
    transition_a = a * transition_a;
  }

  const int64_t summary_index =
      ((batch * n_chunks + chunk) * n_head + head) * head_dim + dim;
  chunk_a[summary_index] = transition_a;
  chunk_u[summary_index] = transition_u;
}

template <typename scalar_t>
__global__ void statehead_chunk_boundary_kernel(
    const float* __restrict__ chunk_a,
    const float* __restrict__ chunk_u,
    const scalar_t* __restrict__ initial_state,
    float* __restrict__ chunk_initials,
    scalar_t* __restrict__ final_state,
    int64_t total_states,
    int64_t n_head,
    int64_t head_dim,
    int64_t n_chunks) {
  const int64_t state_index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (state_index >= total_states) {
    return;
  }

  const int64_t dim = state_index % head_dim;
  const int64_t head = (state_index / head_dim) % n_head;
  const int64_t batch = state_index / (n_head * head_dim);
  float state = static_cast<float>(initial_state[state_index]);
  for (int64_t chunk = 0; chunk < n_chunks; ++chunk) {
    const int64_t summary_index =
        ((batch * n_chunks + chunk) * n_head + head) * head_dim + dim;
    chunk_initials[summary_index] = state;
    state = chunk_a[summary_index] * state + chunk_u[summary_index];
  }
  final_state[state_index] = static_cast<scalar_t>(state);
}

template <typename scalar_t>
__global__ void statehead_output_kernel(
    const scalar_t* __restrict__ gates,
    const float* __restrict__ chunk_initials,
    scalar_t* __restrict__ output,
    int64_t total_states,
    int64_t sequence_len,
    int64_t n_head,
    int64_t head_dim,
    int64_t chunk_size,
    int64_t n_chunks) {
  const int64_t state_index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (state_index >= total_states) {
    return;
  }

  const int64_t chunk = blockIdx.y;
  const int64_t dim = state_index % head_dim;
  const int64_t head = (state_index / head_dim) % n_head;
  const int64_t batch = state_index / (n_head * head_dim);
  const int64_t summary_index =
      ((batch * n_chunks + chunk) * n_head + head) * head_dim + dim;
  const int64_t start = chunk * chunk_size;
  const int64_t stop =
      start + chunk_size < sequence_len ? start + chunk_size : sequence_len;
  float state = chunk_initials[summary_index];

  for (int64_t time = start; time < stop; ++time) {
    const float a = load_gate(
        gates, batch, time, 0, head, dim, sequence_len, n_head, head_dim);
    const float b = load_gate(
        gates, batch, time, 1, head, dim, sequence_len, n_head, head_dim);
    const float c = load_gate(
        gates, batch, time, 2, head, dim, sequence_len, n_head, head_dim);
    const float o = load_gate(
        gates, batch, time, 3, head, dim, sequence_len, n_head, head_dim);
    state = a * state + b * c;
    const int64_t output_index =
        ((batch * sequence_len + time) * n_head + head) * head_dim + dim;
    output[output_index] = static_cast<scalar_t>(o * state);
  }
}

template <typename scalar_t>
__global__ void statehead_backward_chunk_summary_kernel(
    const scalar_t* __restrict__ gates,
    const scalar_t* __restrict__ grad_output,
    float* __restrict__ chunk_a,
    float* __restrict__ chunk_u,
    int64_t sequence_len,
    int64_t n_head,
    int64_t head_dim,
    int64_t chunk_size,
    int64_t n_chunks) {
  const int64_t chunk = blockIdx.x;
  const int64_t head = blockIdx.y;
  const int64_t batch = blockIdx.z;
  const int64_t start = chunk * chunk_size;
  const int64_t stop =
      start + chunk_size < sequence_len ? start + chunk_size : sequence_len;

  for (int64_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
    // Summarize the reverse recurrence as
    // carry_before = transition_a * carry_after + transition_u.
    float transition_a = 1.0f;
    float transition_u = 0.0f;
    for (int64_t time = stop - 1; time >= start; --time) {
      const float a = load_gate(
          gates, batch, time, 0, head, dim, sequence_len, n_head, head_dim);
      const float o = load_gate(
          gates, batch, time, 3, head, dim, sequence_len, n_head, head_dim);
      const int64_t output_index =
          ((batch * sequence_len + time) * n_head + head) * head_dim + dim;
      const float dy = static_cast<float>(grad_output[output_index]);
      transition_u = a * (transition_u + dy * o);
      transition_a = a * transition_a;
    }

    const int64_t summary_index =
        ((batch * n_chunks + chunk) * n_head + head) * head_dim + dim;
    chunk_a[summary_index] = transition_a;
    chunk_u[summary_index] = transition_u;
  }
}

template <typename scalar_t>
__global__ void statehead_backward_chunk_boundary_kernel(
    const float* __restrict__ chunk_a,
    const float* __restrict__ chunk_u,
    const scalar_t* __restrict__ grad_final_state,
    float* __restrict__ chunk_carries,
    scalar_t* __restrict__ grad_initial_state,
    int64_t n_head,
    int64_t head_dim,
    int64_t n_chunks) {
  const int64_t head = blockIdx.y;
  const int64_t batch = blockIdx.z;
  for (int64_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
    const int64_t state_index =
        (batch * n_head + head) * head_dim + dim;
    float carry = static_cast<float>(grad_final_state[state_index]);
    for (int64_t chunk = n_chunks - 1; chunk >= 0; --chunk) {
      const int64_t summary_index =
          ((batch * n_chunks + chunk) * n_head + head) * head_dim + dim;
      chunk_carries[summary_index] = carry;
      carry = chunk_a[summary_index] * carry + chunk_u[summary_index];
    }
    grad_initial_state[state_index] = static_cast<scalar_t>(carry);
  }
}

template <typename scalar_t>
__global__ void statehead_backward_chunk_grad_kernel(
    const scalar_t* __restrict__ gates,
    const float* __restrict__ chunk_initials,
    const float* __restrict__ chunk_carries,
    const scalar_t* __restrict__ grad_output,
    scalar_t* __restrict__ grad_gates,
    int64_t sequence_len,
    int64_t n_head,
    int64_t head_dim,
    int64_t chunk_size,
    int64_t n_chunks) {
  extern __shared__ float local_states[];
  const int64_t chunk = blockIdx.x;
  const int64_t head = blockIdx.y;
  const int64_t batch = blockIdx.z;
  const int64_t start = chunk * chunk_size;
  const int64_t stop =
      start + chunk_size < sequence_len ? start + chunk_size : sequence_len;

  // The production d12 shape has head_dim == blockDim.x. The loop retains
  // correctness for other head dimensions without changing the public API.
  for (int64_t dim_start = 0; dim_start < head_dim; dim_start += blockDim.x) {
    const int64_t dim = dim_start + threadIdx.x;
    const bool active = dim < head_dim;
    float chunk_initial = 0.0f;
    float carry = 0.0f;
    if (active) {
      const int64_t summary_index =
          ((batch * n_chunks + chunk) * n_head + head) * head_dim + dim;
      chunk_initial = chunk_initials[summary_index];
      carry = chunk_carries[summary_index];
      float replay = chunk_initial;
      for (int64_t time = start; time < stop; ++time) {
        const float a = load_gate(
            gates, batch, time, 0, head, dim, sequence_len, n_head, head_dim);
        const float b = load_gate(
            gates, batch, time, 1, head, dim, sequence_len, n_head, head_dim);
        const float c = load_gate(
            gates, batch, time, 2, head, dim, sequence_len, n_head, head_dim);
        replay = a * replay + b * c;
        local_states[(time - start) * blockDim.x + threadIdx.x] = replay;
      }
    }
    __syncthreads();

    if (active) {
      for (int64_t time = stop - 1; time >= start; --time) {
        const int64_t local_time = time - start;
        const float a = load_gate(
            gates, batch, time, 0, head, dim, sequence_len, n_head, head_dim);
        const float b = load_gate(
            gates, batch, time, 1, head, dim, sequence_len, n_head, head_dim);
        const float c = load_gate(
            gates, batch, time, 2, head, dim, sequence_len, n_head, head_dim);
        const float o = load_gate(
            gates, batch, time, 3, head, dim, sequence_len, n_head, head_dim);
        const float state =
            local_states[local_time * blockDim.x + threadIdx.x];
        const float previous_state = local_time == 0
            ? chunk_initial
            : local_states[(local_time - 1) * blockDim.x + threadIdx.x];
        const int64_t output_index =
            ((batch * sequence_len + time) * n_head + head) * head_dim + dim;
        const float dy = static_cast<float>(grad_output[output_index]);
        const float ds = carry + dy * o;

        const int64_t gate_base =
            ((((batch * sequence_len + time) * 4) * n_head + head) *
             head_dim + dim);
        grad_gates[gate_base] =
            static_cast<scalar_t>(ds * previous_state * a * (1.0f - a));
        grad_gates[gate_base + n_head * head_dim] =
            static_cast<scalar_t>(ds * c * b * (1.0f - b));
        grad_gates[gate_base + 2 * n_head * head_dim] =
            static_cast<scalar_t>(ds * b * (1.0f - c * c));
        grad_gates[gate_base + 3 * n_head * head_dim] =
            static_cast<scalar_t>(dy * state * o * (1.0f - o));
        carry = ds * a;
      }
    }
    __syncthreads();
  }
}

void check_inputs(
    const torch::Tensor& gates,
    const torch::Tensor& gate_bias,
    const torch::Tensor& initial_state,
    int64_t chunk_size) {
  TORCH_CHECK(gates.is_cuda(), "gates must be a CUDA tensor");
  TORCH_CHECK(gate_bias.is_cuda(), "gate_bias must be a CUDA tensor");
  TORCH_CHECK(initial_state.is_cuda(), "initial_state must be a CUDA tensor");
  TORCH_CHECK(
      gates.device() == gate_bias.device() &&
          gates.device() == initial_state.device(),
      "devices must match");
  TORCH_CHECK(gates.is_contiguous(), "gates must be contiguous");
  TORCH_CHECK(gate_bias.is_contiguous(), "gate_bias must be contiguous");
  TORCH_CHECK(initial_state.is_contiguous(), "initial_state must be contiguous");
  TORCH_CHECK(gates.dim() == 5, "gates must have shape [B, T, 4, H, Dh]");
  TORCH_CHECK(gates.size(2) == 4, "gates dimension 2 must have size 4");
  TORCH_CHECK(initial_state.dim() == 3, "initial_state must have shape [B, H, Dh]");
  TORCH_CHECK(gates.size(0) == initial_state.size(0), "batch sizes must match");
  TORCH_CHECK(gates.size(3) == initial_state.size(1), "head counts must match");
  TORCH_CHECK(gates.size(4) == initial_state.size(2), "head dimensions must match");
  TORCH_CHECK(
      gate_bias.numel() == 4 * gates.size(3) * gates.size(4),
      "gate_bias must have 4 * H * Dh elements");
  TORCH_CHECK(gates.size(1) > 0, "sequence length must be positive");
  TORCH_CHECK(gates.scalar_type() == gate_bias.scalar_type(), "gate_bias dtype mismatch");
  TORCH_CHECK(gates.scalar_type() == initial_state.scalar_type(), "dtypes must match");
  TORCH_CHECK(
      gates.scalar_type() == torch::kFloat ||
          gates.scalar_type() == torch::kHalf ||
          gates.scalar_type() == torch::kBFloat16,
      "supported dtypes are float32, float16, and bfloat16");
  TORCH_CHECK(chunk_size > 0, "chunk_size must be positive");
  TORCH_CHECK(
      chunk_size <= kMaxChunkSize,
      "native CUDA scan currently requires chunk_size <= ",
      kMaxChunkSize);
}

}  // namespace

std::vector<torch::Tensor> statehead_forward_cuda(
    torch::Tensor gates,
    torch::Tensor gate_bias,
    torch::Tensor initial_state,
    int64_t chunk_size) {
  check_inputs(gates, gate_bias, initial_state, chunk_size);
  const c10::cuda::CUDAGuard device_guard(gates.device());

  const int64_t batch = gates.size(0);
  const int64_t sequence_len = gates.size(1);
  const int64_t n_head = gates.size(3);
  const int64_t head_dim = gates.size(4);
  const int64_t total_states = batch * n_head * head_dim;
  const int64_t n_chunks = (sequence_len + chunk_size - 1) / chunk_size;

  auto output = torch::empty(
      {batch, sequence_len, n_head, head_dim}, gates.options());
  auto final_state = torch::empty_like(initial_state);
  auto activated_gates = torch::empty_like(gates);
  auto float_options = gates.options().dtype(torch::kFloat);
  auto chunk_a = torch::empty(
      {batch, n_chunks, n_head, head_dim}, float_options);
  auto chunk_u = torch::empty_like(chunk_a);
  auto chunk_initials = torch::empty_like(chunk_a);

  const int summary_blocks =
      static_cast<int>((total_states + kForwardThreads - 1) / kForwardThreads);
  const dim3 summary_grid(
      static_cast<unsigned int>(summary_blocks),
      static_cast<unsigned int>(n_chunks));
  const int state_blocks =
      static_cast<int>((total_states + kForwardThreads - 1) / kForwardThreads);
  const int64_t gate_stride = n_head * head_dim;
  const int gate_blocks = static_cast<int>(
      (gate_stride + kForwardThreads - 1) / kForwardThreads);
  TORCH_CHECK(
      sequence_len <= 65535 && batch <= 65535 && n_chunks <= 65535,
      "native CUDA scan grid dimensions exceed CUDA launch limits");
  const dim3 gate_grid(
      static_cast<unsigned int>(gate_blocks),
      static_cast<unsigned int>(sequence_len),
      static_cast<unsigned int>(batch));
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      gates.scalar_type(),
      "statehead_forward_cuda",
      [&] {
        statehead_activate_gates_kernel<scalar_t>
            <<<gate_grid, kForwardThreads, 0, stream>>>(
                gates.data_ptr<scalar_t>(),
                gate_bias.data_ptr<scalar_t>(),
                activated_gates.data_ptr<scalar_t>(),
                sequence_len,
                gate_stride);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        statehead_chunk_summary_kernel<scalar_t>
            <<<summary_grid, kForwardThreads, 0, stream>>>(
                activated_gates.data_ptr<scalar_t>(),
                chunk_a.data_ptr<float>(),
                chunk_u.data_ptr<float>(),
                total_states,
                sequence_len,
                n_head,
                head_dim,
                chunk_size,
                n_chunks);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        statehead_chunk_boundary_kernel<scalar_t>
            <<<state_blocks, kForwardThreads, 0, stream>>>(
                chunk_a.data_ptr<float>(),
                chunk_u.data_ptr<float>(),
                initial_state.data_ptr<scalar_t>(),
                chunk_initials.data_ptr<float>(),
                final_state.data_ptr<scalar_t>(),
                total_states,
                n_head,
                head_dim,
                n_chunks);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        statehead_output_kernel<scalar_t>
            <<<summary_grid, kForwardThreads, 0, stream>>>(
                activated_gates.data_ptr<scalar_t>(),
                chunk_initials.data_ptr<float>(),
                output.data_ptr<scalar_t>(),
                total_states,
                sequence_len,
                n_head,
                head_dim,
                chunk_size,
                n_chunks);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

  return {output, final_state, chunk_initials, activated_gates};
}

std::vector<torch::Tensor> statehead_backward_cuda(
    torch::Tensor gates,
    torch::Tensor chunk_initials,
    torch::Tensor grad_y,
    torch::Tensor grad_final_state,
    int64_t chunk_size) {
  TORCH_CHECK(gates.is_cuda(), "gates must be a CUDA tensor");
  TORCH_CHECK(chunk_initials.is_cuda(), "chunk_initials must be a CUDA tensor");
  TORCH_CHECK(grad_y.is_cuda(), "grad_y must be a CUDA tensor");
  TORCH_CHECK(grad_final_state.is_cuda(), "grad_final_state must be a CUDA tensor");
  TORCH_CHECK(
      gates.device() == chunk_initials.device() &&
          gates.device() == grad_y.device() &&
          gates.device() == grad_final_state.device(),
      "all tensors must be on the same CUDA device");
  TORCH_CHECK(gates.is_contiguous(), "gates must be contiguous");
  TORCH_CHECK(chunk_initials.is_contiguous(), "chunk_initials must be contiguous");
  TORCH_CHECK(grad_y.is_contiguous(), "grad_y must be contiguous");
  TORCH_CHECK(grad_final_state.is_contiguous(), "grad_final_state must be contiguous");
  TORCH_CHECK(chunk_size > 0 && chunk_size <= kMaxChunkSize, "invalid chunk_size");

  const int64_t batch = gates.size(0);
  const int64_t sequence_len = gates.size(1);
  const int64_t n_head = gates.size(3);
  const int64_t head_dim = gates.size(4);
  const int64_t total_states = batch * n_head * head_dim;
  const int64_t n_chunks = (sequence_len + chunk_size - 1) / chunk_size;
  TORCH_CHECK(
      chunk_initials.dim() == 4 && chunk_initials.size(0) == batch &&
          chunk_initials.size(1) == n_chunks &&
          chunk_initials.size(2) == n_head &&
          chunk_initials.size(3) == head_dim,
      "chunk_initials shape does not match gates");
  TORCH_CHECK(
      grad_y.dim() == 4 && grad_y.size(0) == batch &&
          grad_y.size(1) == sequence_len && grad_y.size(2) == n_head &&
          grad_y.size(3) == head_dim,
      "grad_y shape does not match gates");
  TORCH_CHECK(
      grad_final_state.dim() == 3 && grad_final_state.size(0) == batch &&
          grad_final_state.size(1) == n_head &&
          grad_final_state.size(2) == head_dim,
      "grad_final_state shape does not match gates");
  TORCH_CHECK(gates.scalar_type() == grad_y.scalar_type(), "grad_y dtype mismatch");
  TORCH_CHECK(
      gates.scalar_type() == grad_final_state.scalar_type(),
      "grad_final_state dtype mismatch");
  TORCH_CHECK(chunk_initials.scalar_type() == torch::kFloat, "chunk_initials must be float32");

  const c10::cuda::CUDAGuard device_guard(gates.device());
  auto grad_gates = torch::empty_like(gates);
  auto grad_initial_state = torch::empty_like(grad_final_state);
  auto float_options = gates.options().dtype(torch::kFloat);
  auto chunk_a = torch::empty(
      {batch, n_chunks, n_head, head_dim}, float_options);
  auto chunk_u = torch::empty_like(chunk_a);
  auto chunk_carries = torch::empty_like(chunk_a);
  const dim3 chunk_grid(
      static_cast<unsigned int>(n_chunks),
      static_cast<unsigned int>(n_head),
      static_cast<unsigned int>(batch));
  const dim3 state_grid(
      1,
      static_cast<unsigned int>(n_head),
      static_cast<unsigned int>(batch));
  const size_t shared_bytes =
      static_cast<size_t>(chunk_size) * kBackwardThreads * sizeof(float);
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      gates.scalar_type(),
      "statehead_backward_cuda",
      [&] {
        statehead_backward_chunk_summary_kernel<scalar_t>
            <<<chunk_grid, kBackwardThreads, 0, stream>>>(
                gates.data_ptr<scalar_t>(),
                grad_y.data_ptr<scalar_t>(),
                chunk_a.data_ptr<float>(),
                chunk_u.data_ptr<float>(),
                sequence_len,
                n_head,
                head_dim,
                chunk_size,
                n_chunks);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        statehead_backward_chunk_boundary_kernel<scalar_t>
            <<<state_grid, kBackwardThreads, 0, stream>>>(
                chunk_a.data_ptr<float>(),
                chunk_u.data_ptr<float>(),
                grad_final_state.data_ptr<scalar_t>(),
                chunk_carries.data_ptr<float>(),
                grad_initial_state.data_ptr<scalar_t>(),
                n_head,
                head_dim,
                n_chunks);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        statehead_backward_chunk_grad_kernel<scalar_t>
            <<<chunk_grid, kBackwardThreads, shared_bytes, stream>>>(
                gates.data_ptr<scalar_t>(),
                chunk_initials.data_ptr<float>(),
                chunk_carries.data_ptr<float>(),
                grad_y.data_ptr<scalar_t>(),
                grad_gates.data_ptr<scalar_t>(),
                sequence_len,
                n_head,
                head_dim,
                chunk_size,
                n_chunks);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

  return {grad_gates, grad_initial_state};
}
