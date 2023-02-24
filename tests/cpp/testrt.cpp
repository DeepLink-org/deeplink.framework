#include <torch/torch.h>
#include <iostream>


void testcopy() {
  torch::Device device(torch::kPrivateUse1);
  auto option1 = torch::dtype(c10::ScalarType::Double);

  torch::Tensor t1 = torch::empty({3, 6}, option1);
  t1 = t1.to(device);

  torch::Tensor t2 = torch::empty({3, 6}, option1);
  t2 = t2.to(device);
  auto t3 = t1.add(t2);
  std::cout << t3 << std::endl;
}

void testCast() {
  _cast_Byte
}
// need change to use gtest.
int main() {
  for(int i=0; i<10; i++) {
    // test1()
    // testCross1();
    testcopy();
  }
}