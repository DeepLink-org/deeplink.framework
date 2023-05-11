from torch_dipu.testing._internal import common_utils

TEST_PRECISIONS = {
    # test_name : floating_precision,
    'test_pow_dipu_float32': 0.0035,
    'test_pow_dipu_float64': 0.0045,
    'test_var_neg_dim_dipu_bfloat16': 0.01,
    'test_sum_dipu_bfloat16': 0.1,
}

DISABLED_TESTS_MLU = {
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
    
    # test/test_view_ops.py
    'TestOldViewOpsDIPU': {
        'test_atleast',
        'test_broadcast_to',
        'test_contiguous',
        'test_flatten',
        'test_ravel',
        'test_reshape_view_semantics',
        'test_transpose_vs_numpy',
        'test_transposes',
        'test_transposes_errors'
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
        'test_index_put_accumulate_large_tensor_dipu'
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
    
    # test/nn/test_pooling.py
    'TestPoolingNNDeviceTypeDIPU': {
        'test_adaptive_pooling_max_nhwc',
        'test_maxpool3d_non_square_backward',
        'test_pooling_max_nhwc',
        'test_MaxPool1d_indices',
        'test_adaptive_pooling_max_nhwc',
        'test_maxpool_indices_no_batch_dim'
    },
    
    # test/nn/test_convolution.py
    'TestConvolutionNNDeviceTypeDIPU': {
        'test_Conv2d_backward_depthwise',
        'test_Conv2d_naive_groups',
        'test_conv1d_same_padding_backward',
        'test_conv1d_same_padding',
        'test_conv1d_valid_padding_backward',
        'test_conv1d_valid_padding',
        'test_conv2d_same_padding_backward',
        'test_conv2d_same_padding',
        'test_conv2d_valid_padding_backward',
        'test_conv2d_valid_padding',
        'test_conv3d_same_padding_backward',
        'test_conv3d_same_padding',
        'test_conv3d_valid_padding_backward',
        'test_conv3d_valid_padding',
        'test_conv_double_backward_strided_with_3D_input_and_weight',
        'test_conv_empty_channel',
        'test_conv_noncontig_weights_and_bias',
        'test_conv_noncontig_weights',
        'test_conv_transpose_with_output_size_and_no_batch_dim_ConvTranspose2d',
        'test_conv_transpose_with_output_size_and_no_batch_dim_ConvTranspose3d',
        'test_group_conv_empty',
        'test_conv_transpose_with_output_size_and_no_batch_dim',
    },
    
    # test/test_unary_ufuncs.py
    'TestUnaryUfuncsDIPU': {
        'test_float_domains'
    },
    
    # test/test_binary_ufuncs.py
    'TestBinaryUfuncsDIPU': {
        
    },
    
    # test/test_linalg.py
    'TestLinalgDIPU': {
        
    },
    
    # test/test_tensor_creation_ops.py
    'TestTensorCreationDIPU': {
        
    },
    
    # test/test_tensor_creation_ops.py
    'TestRandomTensorCreationDIPU': {
        
    },
    
    # test/test_tensor_creation_ops.py
    'TestLikeTensorCreationDIPU': {
        
    },
    
    # test/test_tensor_creation_ops.py
    'TestAsArrayDIPU': {
        
    },
    
    # test/test_nn.py
    'TestNNDeviceTypeDIPU': {
        
    },
    
    # test/test_shape_ops.py
    'TestShapeOpsDIPU': {
        
    },
    
    # test/test_reductions.py
    'TestReductionsDIPU': {
        # 'test_dim_default',
        'test_dim_default_keepdim',
        'test_dim_none',
        'test_dim_none_keepdim',
        'test_dim_single',
        'test_dim_single_keepdim',
        'test_dim_empty',
        'test_dim_empty_keepdim',
        'test_dim_multi',
        'test_dim_multi_keepdim',
        'test_dim_multi_unsorted',
        'test_dim_multi_unsorted_keepdim',
        'test_dim_multi_duplicate',
        'test_dim_multi_unsupported',
        'test_dim_offbounds',
        'test_dim_ndim_limit',
        'test_identity',
        'test_nan_policy_propagate',
        'test_nan_policy_omit',
        'test_result_dtype',
        'test_empty_tensor_empty_slice',
        'test_empty_tensor_nonempty_slice',
        'test_noncontiguous_innermost',
        'test_noncontiguous_outermost',
        'test_noncontiguous_all',
        'test_noncontiguous_transposed',
        'test_noncontiguous_expanded',
        'test_ref_scalar_input',
        'test_ref_small_input',
        'test_ref_large_input_1D',
        'test_ref_large_input_2D',
        'test_ref_large_input_64bit_indexing',
        'test_ref_duplicate_values',
        'test_ref_extremal_values',
        'test_var_unbiased',
        'test_var_stability',
        'test_sum_dim_reduction_uint8_overflow',
        'test_dim_reduction_less_than_64',
        'test_dim_reduction_lastdim',
        'test_logsumexp',
        'test_logcumsumexp_complex',
        'test_all_any_with_dim',
        'test_numpy_named_args',
        'test_std_dim',
        'test_var_dim',
        'test_logsumexp_dim',
        'test_prod_integer_upcast',
        'test_cumsum_integer_upcast',
        'test_cumprod_integer_upcast',
        'test_mode',
        'test_mode_large',
        'test_mode_boolean',
        'test_mode_wrong_dtype',
        'test_mode_wrong_device',
        'test_var_mean_some_dims',
        'test_amin',
        'test_amax',
        'test_aminmax',
        'test_bincount',
        'test_var_stability2',
        'test_prod_gpu',
        'test_prod',
        'test_prod_bool',
        'test_bucketization',
        'test_nansum',
        'test_count_nonzero',
        'test_nansum_vs_numpy',
        'test_nansum_complex',
        'test_nansum_out_dtype',
        'test_argminmax_multiple',
        'test_all_any_vs_numpy',
        'test_repeated_dim',
        'test_var',
        'test_var_large_input',
        'test_sum_noncontig',
        'test_minmax_illegal_dtype',
        'test_dim_arg_reduction_scalar',
        'test_dim_reduction',
        'test_reduction_split',
        'test_reduction_vectorize_along_input_corner',
        'test_reduction_vectorize_along_output',
        'test_argminmax_large_axis',
        'test_argminmax_axis_with_dim_one',
        'test_median_real_values',
        'test_median_nan_values',
        'test_median_corner_cases',
        'test_quantile',
        'test_quantile_backward',
        'test_quantile_error',
        'test_std_mean',
        'test_std_mean_all_dims',
        'test_var_mean',
        'test_var_mean_all_dims',
        'test_std_mean_some_dims',
        'test_var_vs_numpy',
        'test_std_vs_numpy',
        'test_var_correction_vs_numpy',
        'test_std_correction_vs_numpy',
        'test_std_mean_correction',
        'test_var_mean_correction',
        'test_amin_amax_some_dims',
        'test_histc',
        'test_tensor_compare_ops_empty',
        'test_tensor_compare_ops_argmax_argmix_kthvalue_dim_empty',
        'test_reduction_empty_any_all',
        'test_reduce_dtype',
        'test_reference_masked',
        'test_reductions_large_half_tensors'
    },
    
    # test/test_sparse.py
    'TestSparseUnaryUfuncsDIPU': {
        
    },
    
    # test/test_sparse.py
    'TestSparseMaskedReductionsDIPU': {
        
    },
    
    # test/test_sparse.py
    'TestSparseDIPU': {
        
    },
    
    # test/test_sparse.py
    'TestSparseAnyDIPU': {
    },
    
    # test/test_sort_and_select.py
    'TestSortAndSelectDIPU': {
        'test_stable_sort',
        'test_sort_restride',
        'test_sort_discontiguous',
        'test_sort_discontiguous_slow',
        'test_sort_1d_output_discontiguous',
        'test_stable_sort_against_numpy',
        'test_msort',
        'test_sort_expanded_tensor',
        'test_kthvalue',
        'test_kthvalue_scalar',
        'test_isin',
        'test_isin_different_dtypes',
        'test_isin_different_devices',
    },
    
    # # test/test_torch.py
    # 'TestViewOpsDIPU': {
        
    # },
    
    # test/test_torch.py
    'TestVitalSignsCudaDIPU': {
        
    },
    
    # test/test_torch.py
    'TestTensorDeviceOpsDIPU': {
        
    },
    
    # test/test_torch.py
    'TestTorchDeviceTypeDIPU': {
        
    },
    
    # test/test_torch.py
    'TestDevicePrecisionDIPU': {
        
    }
}

DISABLED_TESTS = common_utils.prepare_match_set(DISABLED_TESTS_MLU)
