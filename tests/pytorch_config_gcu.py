from torch_dipu.testing._internal import common_utils

TEST_PRECISIONS = {
    # test_name : floating_precision,
    'test_pow_dipu_float32': 0.0035,
    'test_pow_dipu_float64': 0.0045,
    'test_var_neg_dim_dipu_bfloat16': 0.01,
    'test_sum_dipu_bfloat16': 0.1,
}

DISABLED_TESTS_GCU = {
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
    }
}

DISABLED_TESTS = common_utils.prepare_match_set(DISABLED_TESTS_GCU)
