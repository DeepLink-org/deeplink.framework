// Copyright (c) 2023, DeepLink.
#include "./helpfunc.hpp"

namespace dipu
{
bool isDeviceTensor(const at::Tensor &tensor)
{
  return tensor.unsafeGetTensorImpl()->device_type() == dipu::DIPU_DEVICE_TYPE;
}

int countOp()
{
  static const char *env_ptr = std::getenv("DIPU_COUNT_OP");
  static int count = 0;
  if (env_ptr != nullptr)
  {
    if (std::strcmp(env_ptr, "true") == 0 or std::strcmp(env_ptr, "True") == 0 or std::strcmp(env_ptr, "TRUE") == 0)
    {
      count += 1;
    }
  }
  return count;
}

std::set<std::string> opset;

void countOps(std::string name, int flag)
{
  const char *filename1 = "impl_Ops.txt";
  const char *filename2 = "fallback_Ops.txt";
  int opnum = countOp();

  if (opnum > 0)
  {
    if (opnum == 1)
    {
      std::remove(filename1);
      std::remove(filename2);
    }
    if (opset.find(name) == opset.end())
    {
      std::ofstream file;
      if (flag == 1)
      {
        file.open(filename1, std::ios::app);
      }
      else
      {
        file.open(filename2, std::ios::app);
      }
      file << name << std::endl;
      file.close();
      opset.insert(name);
    }
  }
}

} // end dipu
