// Copyright (c) 2023, DeepLink.
#include <iostream>

#include <torch/torch.h>

void testTensorRelu(at::Tensor &self) {
  std::cout << self << std::endl;
  std::cout << torch::relu(self) << std::endl;
  std::cout << self << std::endl;

  std::cout << torch::relu_(self) << std::endl;
  std::cout << self << std::endl;
}

int main() {
  torch::Tensor tensor = torch::randn(10).cuda();
  testTensorRelu(tensor);

  torch::Tensor tensor_cpu = torch::randn(10);
  testTensorRelu(tensor_cpu);
  return 0;
}
