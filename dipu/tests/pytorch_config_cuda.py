# Copyright (c) 2023, DeepLink.
from torch_dipu.testing._internal import common_utils
from tests.pytorch_config_mlu import DISABLED_TESTS_MLU
TEST_PRECISIONS = {
    # test_name : floating_precision,
    'test_pow_dipu_float32': 0.0035,
    'test_pow_dipu_float64': 0.0045,
    'test_var_neg_dim_dipu_bfloat16': 0.01,
    'test_sum_dipu_bfloat16': 0.1,
}

DISABLED_TESTS_CUDA = {
    # test_binary_ufuncs.py
    'TestBinaryUfuncsDIPU': {
        # random related, cuda stuck, need fix!
        'test_add_with_tail',
    },

    # test_torch.py
    'TestTorchDeviceTypeDIPU': {
        # not change inner err to expected type
        'test_nondeterministic_alert_NLLLoss',
        'test_nondeterministic_alert_interpolate_bilinear',
    },
   
    # test_type_promotion.py
    'TestTypePromotionDIPU':{
        # dipu do incorrect promotion and diopi aten check but fail.
        'test_result_type',
        'test_booleans',
    },
    # test_convolution.py
    'TestConvolutionNNDeviceTypeDIPU': {
        'test_conv1d_same_padding',
        'test_conv1d_valid_padding',
        'test_conv1d_same_padding_backward',
        'test_conv1d_valid_padding_backward',
        'test_conv1d_vs_scipy',
        'test_conv2d_same_padding',
        'test_conv2d_same_padding_backward',
        'test_conv2d_valid_padding',
        'test_conv2d_valid_padding_backward',
        'test_conv2d_vs_scipy',
        'test_conv3d_same_padding',
        'test_conv3d_valid_padding',
        'test_conv3d_vs_scipy',
        'test_conv3d_same_padding_backward',
        'test_conv3d_valid_padding_backward',
        'test_conv_double_backward_strided_with_3D_input_and_weight',
        'test_conv_empty_channel',
        'test_conv_noncontig_weights',
        'test_conv_transpose_with_output_size_and_no_batch_dim',
        'test_group_convTranspose_empty',
        'test_convTranspose_empty',
    },
}

# merge cuda disable with mlu
common_utils.merge_disable_dist(DISABLED_TESTS_CUDA, DISABLED_TESTS_MLU)

DISABLED_TESTS = common_utils.prepare_match_set(DISABLED_TESTS_CUDA)