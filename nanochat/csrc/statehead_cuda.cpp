#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> statehead_forward_cuda(
    torch::Tensor gates,
    torch::Tensor gate_bias,
    torch::Tensor initial_state,
    int64_t chunk_size);

std::vector<torch::Tensor> statehead_backward_cuda(
    torch::Tensor gates,
    torch::Tensor chunk_initials,
    torch::Tensor grad_y,
    torch::Tensor grad_final_state,
    int64_t chunk_size);

std::vector<torch::Tensor> statehead_forward_value_cuda(
    torch::Tensor gates,
    torch::Tensor gate_bias,
    torch::Tensor initial_state,
    torch::Tensor candidate_residual,
    int64_t chunk_size);

std::vector<torch::Tensor> statehead_backward_value_cuda(
    torch::Tensor gates,
    torch::Tensor chunk_initials,
    torch::Tensor candidate_residual,
    torch::Tensor grad_y,
    torch::Tensor grad_final_state,
    int64_t chunk_size);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "forward",
      &statehead_forward_cuda,
      "Fused StateHead forward scan (CUDA)");
  module.def(
      "backward",
      &statehead_backward_cuda,
      "Fused StateHead reverse scan (CUDA)");
  module.def(
      "forward_value",
      &statehead_forward_value_cuda,
      "Fused StateHead value-residual forward scan (CUDA)");
  module.def(
      "backward_value",
      &statehead_backward_value_cuda,
      "Fused StateHead value-residual reverse scan (CUDA)");
}
