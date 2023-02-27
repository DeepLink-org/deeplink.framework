#include <iostream>

#include <ATen/Tensor.h>
#include <ATen/ATen.h>

void testTensorAdd(const at::Tensor& lhs, const at::Tensor& rhs) {
    at::Tensor result = lhs + rhs;
    std::cout << lhs << std::endl;
    std::cout << rhs << std::endl;
    std::cout << result << std::endl;
}

int main() {
    at::Tensor t0 = at::randn({2, 2}).cuda();
    at::Tensor t1 = at::randn({2}).cuda();
    testTensorAdd(t0, t1);
    testTensorAdd(t0.cpu(), t1.cpu());
    return 0;
}