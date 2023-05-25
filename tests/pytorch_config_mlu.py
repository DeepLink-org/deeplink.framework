# Copyright (c) 2023, DeepLink.
from torch_dipu.testing._internal import common_utils

TEST_PRECISIONS = {
    # test_name : floating_precision,
    'test_pow_dipu_float32': 0.0035,
    'test_pow_dipu_float64': 0.0045,
    'test_var_neg_dim_dipu_bfloat16': 0.01,
    'test_sum_dipu_bfloat16': 0.1,
}

DISABLED_TESTS_MLU = {
    # test_torch.py
    'TestTorchDeviceTypeDIPU': {
        # CRASHED
        'test_dim_function_empty_dipu',
        'test_nondeterministic_alert_AdaptiveAvgPool2d_dipu',
        'test_scalar_check_dipu',
        # ERROR
        'test_addcmul',
        'test_assertRaisesRegex_ignore_msg_non_native_device',
        'test_bernoulli_self',
        'test_broadcast',
        'test_cdist_norm',
        'test_cdist_norm_batch',
        'test_conv_transposed_backward_agnostic_to_memory_format',
        'test_copy_',
        'test_cov',
        'test_cumsum',
        'test_deepcopy',
        'test_diff',
        'test_dist',
        'test_index_copy',
        'test_index_reduce',
        'test_masked_select',
        'test_memory_format_empty_like',
        'test_nondeterministic_alert_CTCLoss',
        'test_nondeterministic_alert_EmbeddingBag_max',
        'test_nondeterministic_alert_cumsum',
        'test_pairwise_distance_empty',
        'test_put',
        'test_put_accumulate',
        'test_storage',
        'test_strides_propagation',
        'test_take',

        # 'FAIL'
        'test_cdist_non_contiguous',
        'test_cdist_non_contiguous_batch',
        'test_copy_all_dtypes_and_devices',
        'test_cpp_warnings_have_python_context',
        'test_cummax_discontiguous',
        'test_cummin_discontiguous',
        'test_is_set_to',
        'test_masked_select_discontiguous',
        'test_memory_format_clone',
        'test_memory_format_operators',
        'test_memory_format_type_shortcuts',
        'test_multinomial',
    },
    # test_view_ops.py
    'TestViewOpsDIPU': {
        'test_contiguous_nonview',
        'test_expand_as_view',
        'test_expand_view',
        'test_reshape_nonview',
        'test_unfold_view',
        'test_advanced_indexing_nonview_dipu',
        'test_as_strided_gradients_dipu',
        'test_unbind_dipu',
        'test_view_copy_dipu',
    },

    # test_indexing.py
    'TestIndexingDIPU': {
        'test_setitem_expansion_error',
        'test_setitem_scalars',
        'test_multiple_byte_mask',
        'test_empty_slice',
        'test_byte_tensor_assignment',
        'test_byte_mask',
        'test_byte_mask_accumulate',
        'test_bool_indices',
        'test_index_getitem_copy_bools_slices',
        'test_index_setitem_bools_slices',
        'test_getitem_scalars',
        'test_empty_ndim_index',
        'test_index_put_byte_indices_dipu',
        'test_index_put_accumulate_large_tensor_dipu',
        # ERROR
        'test_index_put_accumulate_large_tensor',
        'test_index_put_src_datatype',
        'test_index_scalar_with_bool_mask',
        'test_index_src_datatype',
        'test_int_assignment',
        'test_int_indices2d',
        'test_int_indices_broadcast',
        'test_invalid_index',
        'test_jit_indexing',
        'test_multiple_bool_indices',
        'test_multiple_int',
        'test_none',
        'test_out_of_bound_index',
        'test_set_item_to_scalar_tensor',
        'test_single_int',
        'test_step',
        'test_step_assignment',
        'test_take_along_dim',
        'test_take_along_dim_invalid',
        'test_variable_slicing',
        'test_zero_dim_index',
        # FAIL
        'test_index',
        'test_int_indices',
    },

    # test_indexing.py
    'NumpyTestsDIPU': {
        'test_trivial_fancy_out_of_bounds',
        'test_boolean_assignment_value_mismatch',
        'test_empty_tuple_index',
        'test_empty_fancy_index',
        'test_ellipsis_index',
        'test_boolean_indexing_alldims',
        'test_boolean_indexing_onedim',
        'test_boolean_indexing_twodim',
        'test_boolean_list_indexing',
        'test_single_bool_index',
        'test_broaderrors_indexing',
        'test_boolean_shape_mismatch',
        'test_boolean_indexing_weirdness',
        'test_boolean_indexing_weirdness_tensors',
    },

    # test_type_promotion.py
    'TestTypePromotionDIPU':{
        # CRASHED
        'test_bfloat16',
        # FAIL
        'test_alternate_result',
        'test_div_promotion',
        'test_div_promotion_inplace',
        # ERROR
        'test_comparison_ops_with_type_promotion',
        'test_complex_promotion',
        'test_complex_scalar_mult_tensor_promotion',
        'test_create_bool_tensors',
        'test_lt_with_type_promotion',
        'test_many_promotions',
    },

    # test_nn.py
    'TestNN':{
        # CRASHED
        'test_pdist_empty_col',
    },

    # test_ops_fwd_gradients.py
    'TestFwdGradientsDIPU': {
        # CRASHED
        'test_fn_fwgrad_bwgrad',
        'test_forward_mode_AD',
        # ERROR
        'test_inplace_forward_mode_AD'
    },

    # test_ops_gradients.py
    'TestBwdGradientsDIPU':{
        'test_fn_fail_gradgrad',
        'test_fn_grad',
        'test_fn_gradgrad',
        'test_inplace_grad',
        'test_inplace_gradgrad',
    },
    # test_ops.py

    # test_shape_ops.py
    'TestShapeOpsDIPU': {
        'test_clamp',
        'test_clamp_propagates_nans',
        'test_flip',
        'test_flip_errors',
        'test_flip_numpy',
        'test_fliplr',
        'test_flipud',
        'test_movedim',
        'test_movedim_invalid',
        'test_nonzero',
        'test_nonzero_astuple_out',
        'test_sparse_dense_dim',
    },
}

DISABLED_TESTS = common_utils.prepare_match_set(DISABLED_TESTS_MLU)
