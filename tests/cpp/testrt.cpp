#include <torch/torch.h>
#include <iostream>


void testcopy() {
  torch::Device device(torch::kPrivateUse1);
  auto option1 = torch::dtype(c10::ScalarType::Double);

  torch::Tensor t1 = torch::ones({3, 6}, option1);
  t1 = t1.to(device);

  torch::Tensor t2 = torch::ones({3, 6}, option1);
  t2 = t2.to(device);
  auto t3 = t1.add(t2);
  auto ts = t3.isfinite();
  std::cout << ts << std::endl;
}

// need change to use gtest.
int main() {
  for(int i=0; i<3; i++) {
    // test1()
    // testCross1();
    testcopy();
  }
}