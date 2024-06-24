from utils.test_in_subprocess import run_individual_test_cases


def test_set_allocator_settings():
    """currently only detecting errors"""
    import torch
    import torch_dipu

    torch.cuda.memory._set_allocator_settings("expandable_segments:True")


if __name__ == "__main__":
    run_individual_test_cases((test_set_allocator_settings,))
