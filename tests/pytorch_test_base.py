# Copyright (c) 2023, DeepLink.

import copy
import os
import re
import sys
import runpy
import torch
from torch_dipu import dipu
from torch_dipu.testing._internal import common_utils

_DEFAULT_FLOATING_PRECISION = 1e-3
_DISABLED_TESTS = dict()
_TEST_PRECISIONS = dict()
if dipu.vendor_type == "MLU":
    from tests.pytorch_config_mlu import DISABLED_TESTS, TEST_PRECISIONS
    _DISABLED_TESTS = DISABLED_TESTS
    _TEST_PRECISIONS = TEST_PRECISIONS
elif dipu.vendor_type == "GCU":
    from tests.pytorch_config_gcu import DISABLED_TESTS, TEST_PRECISIONS
    _DISABLED_TESTS = DISABLED_TESTS
    _TEST_PRECISIONS = TEST_PRECISIONS


class DIPUTestBase(DeviceTypeTestBase):
    device_type = 'dipu'
    unsupported_dtypes = {
        torch.complex32, torch.complex64, torch.complex128, torch.bfloat16
    }
    precision = _DEFAULT_FLOATING_PRECISION

    @staticmethod
    def _alt_lookup(d, keys, defval):
        for k in keys:
            value = d.get(k, None)
            if value is not None:
                return value
        return defval

    @classmethod
    def _get_dipu_types(cls, dtype_combination, test_name, test,
                        dipu_dtypes, disallowed_test):
        class_name = cls.__name__
        dtype_test_name = test_name
        skipped = False
        for dtype in dtype_combination:
            dtype_test_name += '_' + str(dtype).split('.')[1]
        for dtype in dtype_combination:
            if dtype in cls.unsupported_dtypes:
                reason = 'DIPU does not support dtype {0}'.format(str(dtype))
                @wraps(test)
                def skipped_test(self, *args, reason=reason, **kwargs):
                    raise unittest.SkipTest(reason)

                assert not hasattr(
                    cls, dtype_test_name), 'Redefinition of test {0}'.format(
                        dtype_test_name)
                setattr(cls, dtype_test_name, skipped_test)
                skipped = True
                break
            if dtype in [torch.float, torch.double, torch.bfloat16]:
                floating_precision = DIPUTestBase._alt_lookup(
                    _TEST_PRECISIONS,
                    [dtype_test_name, test_name, test.__name__],
                    _DEFAULT_FLOATING_PRECISION)
                if dtype not in test.precision_overrides or test.precision_overrides[
                        dtype] < floating_precision:
                    test.precision_overrides[dtype] = floating_precision
        if class_name in _DISABLED_TESTS and common_utils.match_name(
                dtype_test_name, _DISABLED_TESTS[class_name]):
            skipped = True
            setattr(cls, dtype_test_name, disallowed_test)
        if not skipped:
            dipu_dtypes.append(
                dtype_combination
                if len(dtype_combination) > 1 else dtype_combination[0])

    # Overrides to instantiate tests that are known to run quickly
    # and correctly on DIPU.
    @classmethod
    def instantiate_test(cls, name, test, *, generic_cls):
        test_name = name + '_' + cls.device_type
        class_name = cls.__name__

        @wraps(test)
        def disallowed_test(self, test=test):
            raise unittest.SkipTest('skipped on DIPU')

        if class_name in _DISABLED_TESTS and (
                common_utils.match_name(test_name, _DISABLED_TESTS[class_name]) or
                common_utils.match_name(name, _DISABLED_TESTS[class_name])):
            assert not hasattr(
                cls, test_name), 'Redefinition of test {0}'.format(test_name)
            setattr(cls, test_name, disallowed_test)
        else:  # Test is allowed
            dtype_combinations = cls._get_dtypes(test)
            if dtype_combinations is None:  # Tests without dtype variants are instantiated as usual
                super().instantiate_test(name, test, generic_cls=generic_cls)
            else:  # Tests with dtype variants have unsupported dtypes skipped
                # Sets default precision for floating types to bfloat16 precision
                if not hasattr(test, 'precision_overrides'):
                    test.precision_overrides = {}
                dipu_dtypes = []
                for dtype_combination in dtype_combinations:
                    if type(dtype_combination) == torch.dtype:
                        dtype_combination = (dtype_combination,)
                    DIPUTestBase._get_dipu_types(dtype_combination,
                                                 test_name,
                                                 test,
                                                 dipu_dtypes,
                                                 disallowed_test)
                if len(dipu_dtypes) != 0:
                    test.dtypes[cls.device_type] = dipu_dtypes
                    super().instantiate_test(name, test, generic_cls=generic_cls)

    @classmethod
    def get_primary_device(cls):
        return cls.primary_device

    @classmethod
    def setUpClass(cls):
        # Sets the primary test device to the dipu_device
        cls.primary_device = torch.device('dipu')

    def setUp(self):
        super().setUp()

    def prepare_for_compare(self, tx, ty):
        x, y = tx, ty
        if type(x) == torch.Tensor:
            x = tx.to(device='cpu')
        if type(y) == torch.Tensor:
            y = ty.to(device='cpu')
        return x, y

    def _override_prec(self, args, name):
        prec = args.get(name, None)
        if prec is None:
            args[name] = self.precision
        else:
            args[name] = max(self.precision, prec)
        return args

    def _rewrite_compare_args(self, kwargs):
        rwargs = copy.copy(kwargs)
        rwargs = self._override_prec(rwargs, 'rtol')
        rwargs = self._override_prec(rwargs, 'atol')
        return rwargs

    # Overrides assertEqual to popular custom precision
    def assertEqual(self, x, y, *args, **kwargs):
        if not args or isinstance(args[0], str):
            kwargs = self._rewrite_compare_args(kwargs)
        elif isinstance(args[0], (float, int)):
            args = [max(args[0], self.precision)] + list(args[1:])
        x, y = self.prepare_for_compare(x, y)
        return DeviceTypeTestBase.assertEqual(self, x, y, *args, **kwargs)


TEST_CLASS = DIPUTestBase
