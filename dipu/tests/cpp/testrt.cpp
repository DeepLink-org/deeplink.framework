// Copyright (c) 2023, DeepLink.
#include <iostream>

#include <torch/torch.h>

#include <csrc_dipu/runtime/core/DIPUStream.h>
#include <csrc_dipu/runtime/devproxy/deviceproxy.h>

using namespace dipu;
void testcopy() {
  torch::Device device(DIPU_DEVICE_TYPE);
  auto option1 = torch::dtype(c10::ScalarType::Double);

  torch::Tensor t1 = torch::ones({3, 6}, option1);
  t1 = t1.to(device);
  std::cout << t1 << std::endl;

  auto option2 = torch::dtype(c10::ScalarType::Long);
  torch::Tensor t2 = torch::ones({3, 6}, option2);
  t2 = t2.to(device);
  std::cout << t2 << std::endl;

  // auto t3 = t1.add(t2);
  auto ts = t2.isfinite();
  std::cout << ts << std::endl;
}

void testDeviceSwitch() {
  devproxy::setDevice(2);
  torch::Device device(DIPU_DEVICE_TYPE);
  auto option1 = torch::dtype(c10::ScalarType::Float);
  torch::Tensor t1 = torch::ones({3, 6}, option1);
  t1 = t1.to(device);
  std::cout << t1 << std::endl;

  devproxy::setDevice(0);
  auto stream2 = getDIPUStreamFromPool();
  stream2 = getDIPUStreamFromPool();
  stream2 = getDIPUStreamFromPool();
  setCurrentDIPUStream(stream2);

  device = torch::Device(DIPU_DEVICE_TYPE);
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
  for (int i = 0; i < 3; i++) {
    // testcopy();
    testDeviceSwitch();
    // testStream1();
  }
}