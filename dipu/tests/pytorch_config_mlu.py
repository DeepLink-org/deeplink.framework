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
    # test_linalg.py
    'TestLinalgDIPU': {
        'test_addmm_sizes', # diopiMulScalar: cnnlExpand] Check failed: output != NULL
        #'test_inner', # diopiMul :cnnlExpand] Check failed: output != NULL.
        'test_addr_bool',
        'test_addr_float_and_complex',
        'test_addr_integral',
        'test_addr_type_promotion',
        'test_blas_alpha_beta_empty',
        'test_blas_empty',
        'test_corner_cases_of_cublasltmatmul',
        'test_einsum_corner_cases',
        'test_householder_product_errors_and_warnings',
        'test_inverse',
        'test_ldl_factor',
        'test_linalg_lstsq',
        'test_linalg_lstsq_batch_broadcasting',
        'test_linalg_lstsq_input_checks',
        'test_linalg_lu_family',
        'test_linalg_solve_triangular_broadcasting',
        'test_matmul_small_brute_force_1d_Nd',
        'test_matmul_small_brute_force_2d_Nd',
        'test_matmul_small_brute_force_3d_Nd',
        'test_matrix_rank_atol',
        'test_norm_complex_old',
        'test_norm_fro_2_equivalence_old',
        'test_norm_matrix_degenerate_shapes',
        'test_norm_old',
        'test_norm_old_nan_propagation',
        'test_ormqr',
        'test_pca_lowrank',
        'test_pinv_errors_and_warnings',
        'test_renorm',
        'test_tensordot',
        'test_tensorsolve',
        'test_tensorsolve_empty',
    },
    # test_testing.py
    'TestTestParametrizationDeviceTypeDIPU': {
        # when change dipu device type to 'cuda', 'test_ops_composition_names' fail, because parameter
        # passed to testclass.device_type is 'dipu', different device seems have different case numbers.
        # to do: change test device_type='cuda'
        'test_ops_composition_names',
        'test_unparametrized_names',
        'test_make_tensor_dipu',
        'test_dtypes_composition_valid',
        'test_dtypes_composition_invalid',
        'test_multiple_handling_of_same_param_error',
    },

    'TestTestingDIPU': {
        'test_make_tensor',
        'test_assertEqual_numpy',
        'test_circular_dependencies',
        'test_multiple_handling_of_same_param_error',
    },

    'TestImports': {
        'test_circular_dependencies',
    },


    # test_utils.py
    'TestCheckpoint': {
        'test_checkpointing_without_reentrant_early_free',
        'test_checkpointing_without_reentrant_early_free',
        'test_checkpoint_rng_cuda',
    },
    'TestCheckpointDIPU': {
        'test_checkpointing_without_reentrant_early_free',
        'test_checkpointing_without_reentrant_early_free',
        'test_checkpoint_rng_cuda',
    },


    # test_unary_ufuncs.py
    'TestUnaryUfuncsDIPU': {
        'test_float_domains',
        'test_batch_vs_slicing',
        'test_contig_size1',
        'test_contig_size1_large_dim',
        'test_contig_vs_every_other',
        'test_cumulative_trapezoid',
        'test_non_contig',
        'test_non_contig_expand',
        'test_non_contig_index',
        'test_binary_ops_with_scalars',
        'test_contig_vs_transposed',
        'test_reference_numerics',
        'test_reference_numerics_large',
        'test_reference_numerics_small',
        'test_reference_numerics_extremal',
        'test_reference_numerics_normal',
        'test_trapezoid',
        'test_nan_to_num',
        'test_sinc',
        'test_nonzero_empty',
        'test_frexp_assert_raises',
        'test_special_ndtr_vs_scipy',
    },

    # test_binary_ufuncs.py
    'TestBinaryUfuncsDIPU': {
        # torch 2.1 new test and failed
        # todo: camb floor result incorrect: floor(-2.5) = -2.
        'test_div_and_floordiv_vs_python',
        # end

        # CRASHED
        'test_batch_vs_slicing',
        'test_contig_size1',
        'test_contig_size1_large_dim',
        'test_contig_vs_every_other',
        'test_cumulative_trapezoid',
        'test_non_contig',
        'test_non_contig_expand',
        'test_non_contig_index',
        'test_binary_ops_with_scalars',
        'test_contig_vs_transposed',
        'test_reference_numerics',
        'test_reference_numerics_extremal_values',
        'test_reference_numerics_large_values',
        'test_reference_numerics_small_values',
        'test_trapezoid',
         #  FAIL
        'test_add', #assertEqual(res, m1 + m2.contiguous()) failed
        'test_atan2',
        'test_broadcasting',
        'test_bitwise_ops',
        'test_comparison_ops_type_promotion_and_broadcasting',
        'test_copysign',
        'test_div_rounding_modes',
        'test_div_rounding_nonfinite',
        'test_div_rounding_numpy',
        'test_float_power',
        'test_float_power_exceptions',
        'test_floor_divide_scalar',
        'test_floor_divide_tensor',
        'test_fmod_remainder',
        'test_hypot',
        'test_int_and_float_pow',
        'test_int_tensor_pow_neg_ints',
        'test_long_tensor_pow_floats',
        'test_maximum_minimum_forward_ad_float32',
        'test_mul',
        'test_mul_intertype_scalar',
        'test_muldiv_scalar',
        'test_not_broadcastable_floor_divide',
        'test_pow',
        'test_pow_scalar_base',
        'test_pow_scalar_overloads_mem_overlap',
        'test_remainder_fmod_large_dividend',
        'test_remainder_overflow',
        'test_scalar_support',
        'test_type_promotion',
        'test_xlogy_xlog1py',
        'test_xlogy_xlog1py_bfloat16',
        'test_not_broadcastable',
        'test_maximum_and_minimum_subgradient',
        'test_copysign_subgradient',
        'test_fmod_remainder_by_zero_float', # skip for the remainder op
        'test_fmod_remainder_by_zero_integral',
        'test_lerp', # skip for the lerp op
    },

    # test_reductions.py
    'TestReductionsDIPU':{
        #"test_dim_arg_reduction_scalar",
        #"test_dim_default_argmax",
        # CASHED
        'test_dim_ndim_limit',
        'test_identity',
        'test_nan_policy_omit',
        'test_nan_policy_propagate',
        'test_noncontiguous_all',
        'test_noncontiguous_expanded',
        'test_noncontiguous_innermost',
        'test_noncontiguous_outermost',
        'test_noncontiguous_transposed',
        'test_reference_masked',
        'test_result_dtype',
        # ERROR
        'test_ref_duplicate_values',
        'test_ref_extremal_values',
        'test_ref_scalar_input',
        'test_ref_small_input',
        'test_tensor_reduce_ops_empty',
        'test_all_any_vs_numpy',
        'test_all_any',
        'test_all_any_empty',
        'test_count_nonzero',
        'test_dim_default_keepdim',
        'test_dim_single_keepdim',
        'test_dim_multi_unsorted_keepdim',
        'test_dim_multi_keepdim',
        'test_dim_multi',
        'test_dim_none_keepdim',
        'test_dim_none',
        'test_dim_default',
        'test_max',
        'test_min',
        'test_mode',
        'test_nansum',
        'test_dim_multi_unsorted',
        'test_dim_single',
        'test_histc',
        'test_nansum_out',
        'test_nansum_out_dtype',
        'test_quantile_backward',
        'test_logsumexp',
        'test_dim_offbounds', # skip for the std op
        'test_dim_reduction_less_than_64',
        'test_empty_tensor_empty_slice',
        'test_std_correction_vs_numpy',
        'test_std_mean_correction',
        'test_std_vs_numpy',
        'test_amax',
        'test_dim_empty', 
        'test_dim_empty_keepdim',
        # FAIL
        'test_argminmax_multiple',
        'test_dim_reduction',
        'test_empty_tensor_nonempty_slice',
        'test_sum_dim_reduction_uint8_overflow',
        'test_median_nan_values',
        'test_tensor_compare_ops_empty',
    },

    # test_torch.py
    'TestTorchDeviceTypeDIPU': {
        # torch 2.1 new test and failed
        # 1. torch 2.1 change overlap copy behavior,
        # 2. torch 2.1 has a new use_deterministic_algorithms() which dipu
        # not support. no-need support
        'test_deterministic_empty',
        'test_deterministic_resize',
        # torch2.1 add new op _unsafe_index.Tensor, dipu not support
        'test_deterministic_interpolate_bilinear',
        # dipu not support this api now, todo::
        'test_set_default_tensor_type_warnings',
        # end 2.1

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
        'test_scatter_reduce_non_unique_index',
        'test_scatter_reduce_operations_to_large_input',
        'test_scatter_reduce_scalar',
        'test_scatter_reduce_multiply_unsupported_dtypes',
        'test_cdist_empty',
        'test_cdist_norm',
        'test_cdist_norm_batch',
        'test_cdist_cuda_backward',
        'test_cdist_large',
        'test_cdist_large_batch',
        'test_cdist_non_contiguous',
        'test_cdist_non_contiguous_batch',
        'test_cdist_euclidean_large',
        'test_cdist_grad_p_lt_1_no_nan',
        'test_cdist_same_inputs',
        
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
        'test_logcumsumexp',
        'test_discontiguous_out_cumsum',
        'test_cdist_large',
        'test_cdist_same_inputs',
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
        'test_inplace',
        # ERROR
        'test_comparison_ops_with_type_promotion',
        'test_complex_promotion',
        'test_complex_scalar_mult_tensor_promotion',
        'test_create_bool_tensors',
        'test_lt_with_type_promotion',
        'test_many_promotions',
    },

    # test_tensor_creation_ops.py
    'TestTensorCreationDIPU': {
        # torch 2.1 new test and failed
        # cncl not support random of int/complex
        # todo: enhance ci to support disable specific dtype case.
        'test_cat_out_fast_path_dim0_dim1',
        # end 2.1

        'test_arange_device_vs_cpu',
        'test_arange_bfloat16',
        'test_arange', # camb impl have bug
        'test_linlogspace_mem_overlap',
        'test_cartesian_prod',
        'test_cat_all_dtypes_and_devices',
        'test_cat_empty_legacy',
        'test_cat_mem_overlap',
        'test_cat_out',
        'test_combinations',
        'test_linspace',
        'test_linspace_deduction',
        'test_linspace_vs_numpy_integral',
        'test_meshgrid_vs_numpy',
        'test_random',
        'test_random_bool',
        'test_random_default',
        'test_random_from_to',
        'test_random_from_to_bool',
        'test_random_full_range',
        'test_random_to',
        'test_repeat_interleave',
        'test_roll',
        'test_tensor_factories_empty',
        'test_zeros_dtype_layout_device_match',
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
        'test_clamp_propagates_nans', # [cnnlClip] Parameter min and max can not be null simultaneously.
    },
}

DISABLED_TESTS = common_utils.prepare_match_set(DISABLED_TESTS_MLU)
