set -x

export PYTHONPATH=${PYTHONPATH}:${PWD}:${PYTORCH_TEST_DIR}/benchmarks/operator_benchmark/

options=' --num-runs 3 --iterations 10 --warmup-iterations 3 '

# dipu custom benchmark
python -m benchmark_all_dipu_test ${options}

# pytorch benchmark
python -m pt.add_test  ${options}
#python -m pt.diag_test  ${options}
python -m pt.layernorm_test  ${options}
python -m pt.qcat_test  ${options}
python -m pt.qpool_test  ${options}

python -m pt.ao_sparsifier_test  ${options}
python -m pt.embeddingbag_test  ${options}
#python -m pt.linear_prepack_fp16_test  ${options}
python -m pt.qcomparators_test  ${options}
#python -m pt.qrnn_test  ${options}

python -m pt.as_strided_test  ${options}
python -m pt.fill_test  ${options}
python -m pt.linear_test  ${options}
#python -m pt.qconv_test  ${options}
python -m pt.qtensor_method_test  ${options}

python -m pt.batchnorm_test  ${options}
python -m pt.gather_test  ${options}
#python -m pt.linear_unpack_fp16_test  ${options}
#python -m pt.qembedding_bag_lookups_test  ${options}
python -m pt.quantization_test  ${options}

python -m pt.binary_test  ${options}
python -m pt.gelu_test  ${options}
python -m pt.matmul_test  ${options}
python -m pt.qembeddingbag_test  ${options}
python -m pt.qunary_test  ${options}

python -m pt.bmm_test  ${options}
python -m pt.groupnorm_test  ${options}
python -m pt.matrix_mult_test  ${options}
python -m pt.qembedding_pack_test  ${options}
python -m pt.remainder_test  ${options}

python -m pt.cat_test  ${options}
python -m pt.hardsigmoid_test  ${options}
python -m pt.nan_to_num_test  ${options}
python -m pt.qgroupnorm_test  ${options}
python -m pt.softmax_test  ${options}

python -m pt.channel_shuffle_test  ${options}
python -m pt.hardswish_test  ${options}
python -m pt.pool_test  ${options}
python -m pt.qinstancenorm_test  ${options}
python -m pt.split_test  ${options}

python -m pt.chunk_test  ${options}
python -m pt.index_select_test  ${options}
python -m pt.qactivation_test  ${options}
python -m pt.qinterpolate_test  ${options}
python -m pt.stack_test  ${options}

#python -m pt.clip_ranges_test  ${options}
python -m pt.qarithmetic_test  ${options}
python -m pt.qlayernorm_test  ${options}
python -m pt.sum_test  ${options}

python -m pt.instancenorm_test  ${options}
python -m pt.qatembedding_ops_test  ${options}
#python -m pt.qlinear_test  ${options}
python -m pt.tensor_to_test  ${options}

python -m pt.conv_test  ${options}
python -m pt.interpolate_test  ${options}
python -m pt.qbatchnorm_test  ${options}
python -m pt.qobserver_test  ${options}
python -m pt.unary_test  ${options}

exit 0