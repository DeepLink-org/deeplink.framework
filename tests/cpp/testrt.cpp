// Copyright (c) 2023, DeepLink.
#include <torch/torch.h>
#include <iostream>
#include <csrc_dipu/runtime/device/deviceapis.h>
#include <csrc_dipu/runtime/core/DIPUStream.h>

using namespace dipu;
void testcopy() {
  torch::Device device(torch::kPrivateUse1);
  auto option1 = torch::dtype(c10::ScalarType::Double);

  torch::Tensor t1 = torch::ones({3, 6}, option1);
  t1 = t1.to(device);
  std::cout << t1 << std::endl;

  torch::Tensor t2 = torch::ones({3, 6}, option1);
  t2 = t2.to(device);
  // auto t3 = t1.add(t2);
  auto ts = t2.isfinite();
  std::cout << ts << std::endl;
}

void testDeviceSwitch() {
  devapis::setDevice(2);
  torch::Device device(torch::kPrivateUse1);
  auto option1 = torch::dtype(c10::ScalarType::Float);
  torch::Tensor t1 = torch::ones({3, 6}, option1);
  t1 = t1.to(device);
  std::cout << t1 << std::endl;

  devapis::setDevice(0);
  auto stream2 = getDIPUStreamFromPool();
  stream2 = getDIPUStreamFromPool();
  stream2 = getDIPUStreamFromPool();
  setCurrentDIPUStream(stream2);

  device = torch::Device(torch::kPrivateUse1);
  torch::Tensor t2 = torch::ones({3, 6}, option1);
  t2 = t2.to(device);
  std::cout << t2 << std::endl;
}

void testStream1() {
  auto stream1 = dipu::getDefaultDIPUStream();
  deviceStream_t rawStream = stream1.rawstream();
  auto stream2 = getDIPUStreamFromPool();
  setCurrentDIPUStream(stream2);
  std::cout << "current stream =" << stream2.rawstream() << std::endl;

  auto stream3 = getCurrentDIPUStream();
  rawStream = stream3.rawstream();
  std::cout << "current stream =" << rawStream << std::endl;

}

// need change to use gtest.
int main() {
  for(int i=0; i<3; i++) {
    // testcopy();
    testDeviceSwitch();
    // testStream1();
  }
}