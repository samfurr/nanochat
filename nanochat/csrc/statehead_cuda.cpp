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

std::vector<torch::Tensor> statehead_forward_projected_cuda(
    torch::Tensor x,
    torch::Tensor gate_weight,
    torch::Tensor gate_bias,
    torch::Tensor initial_state,
    int64_t n_head,
    int64_t chunk_size,
    int64_t projection_tile_rows);

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
      "forward_projected",
      &statehead_forward_projected_cuda,
      "Pipelined StateHead gate projection, activation, and scan (CUDA)");
}
