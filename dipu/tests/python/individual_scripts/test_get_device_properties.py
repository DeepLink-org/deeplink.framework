import torch
import torch_dipu


def test_get_device_properties():
    # There is no need to call torch.cuda.set_device before call get_device_properties
    for device_index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(device_index)
        assert properties.name is not None
        assert properties.major >= 0
        assert properties.minor >= 0
        assert properties.total_memory >= (256 << 20)
        assert properties.multi_processor_count >= 0


if __name__ == "__main__":
    test_get_device_properties()
