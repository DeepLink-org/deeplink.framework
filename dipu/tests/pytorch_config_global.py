# Copyright (c) 2023, DeepLink.
'''
Global configurations of pytorch tests.

Vendor-specified configurations will be later merged from
`pytorch_config_${vendor}.py` at `pytorch_test_base.py`.
'''

DEFAULT_FLOATING_PRECISION = 1e-3

TEST_PRECISIONS = {}

DISABLED_TESTS_GLOBAL = {
    'TestConvolutionNNDeviceTypeDIPU': {
        'test_conv_backend',
        'test_.*cudnn.*',
    }
}
