import torch

# TODO(later): fix import *
from fp_formats import *

def _test(fp_ref, s_enc_ref, e_enc_ref, m_enc_ref, dtype):
    # test going from float to encoding
    x = torch.tensor(fp_ref, dtype=dtype)
    bitwidth = dtype_to_bitwidth[dtype]
    s_enc, e_enc, m_enc = get_sem_bits(x, bitwidth=bitwidth)
    assert s_enc_ref == s_enc
    assert e_enc_ref == e_enc
    assert m_enc_ref == m_enc

    # test going from encoding to float
    s_i, e_i, m_f, special_value = sem_bits_to_sem_vals(s_enc, e_enc, m_enc, dtype)
    fp = sem_vals_to_f32(s_i, e_i, m_f, special_value)
    assert_same(fp_ref, fp)

def test_fp32():
    # zero and neg zero
    _test(0.0, '0', '0' * 8, '0' * 23, torch.float)
    _test(-0.0, '1', '0' * 8, '0' * 23, torch.float)

    # special values
    _test(float('nan'), '0', '1' * 8, '1' + '0' * 22, torch.float)
    _test(float('inf'), '0', '1' * 8, '0' * 23, torch.float)
    _test(float('-inf'), '1', '1' * 8, '0' * 23, torch.float)

    # values below verified with from https://www.h-schmidt.net/FloatConverter/IEEE754.html

    # largest normal
    _test(3.402823466385288598117042e+38, '0', '1' * 7 + '0', '1' * 23, torch.float)
    _test(-3.402823466385288598117042e+38, '1', '1' * 7 + '0', '1' * 23, torch.float)

    # smallest normal
    _test(1.175494350822287507968737e-38, '0', '0' * 7 + '1', '0' * 23, torch.float)
    _test(-1.175494350822287507968737e-38, '1', '0' * 7 + '1', '0' * 23, torch.float)

    # largest denormal
    _test(1.175494210692441075487029e-38, '0', '0' * 8, '1' * 23, torch.float)
    _test(-1.175494210692441075487029e-38, '1', '0' * 8, '1' * 23, torch.float)

    # smallest denormal
    _test(1.401298464324817070923730e-45, '0', '0' * 8, '0' * 22 + '1', torch.float)
    _test(-1.401298464324817070923730e-45, '1', '0' * 8, '0' * 22 + '1', torch.float)

    # positive and negative value
    _test(30.0, '0', '10000011', '1' * 3 + '0' * 20, torch.float)
    _test(-24.0, '1', '10000011', '1' + '0' * 22, torch.float)

def test_bf16():
    # zero and neg zero
    _test(0.0, '0', '0' * 8, '0' * 7, torch.bfloat16)
    _test(-0.0, '1', '0' * 8, '0' * 7, torch.bfloat16)

    # special values
    _test(float('nan'), '0', '1' * 8, '1' + '0' * 6, torch.bfloat16)
    _test(float('inf'), '0', '1' * 8, '0' * 7, torch.bfloat16)
    _test(float('-inf'), '1', '1' * 8, '0' * 7, torch.bfloat16)

    # values below checked with TODO

    # largest normal
    _test(3.38953e+38, '0', '1' * 7 + '0', '1' * 7, torch.bfloat16)
    _test(-3.38953e+38, '1', '1' * 7 + '0', '1' * 7, torch.bfloat16)

    # smallest normal
    _test(1.17549e-38, '0', '0' * 7 + '1', '0' * 7, torch.bfloat16)
    _test(-1.17549e-38, '1', '0' * 7 + '1', '0' * 7, torch.bfloat16)

    # largest denormal
    _test(1.16631e-38, '0', '0' * 8, '1' * 7, torch.bfloat16)
    _test(-1.16631e-38, '1', '0' * 8, '1' * 7, torch.bfloat16)

    # smallest denormal
    _test(9.18355e-41, '0', '0' * 8, '0' * 6 + '1', torch.bfloat16)
    _test(-9.18355e-41, '1', '0' * 8, '0' * 6 + '1', torch.bfloat16)

    # positive and negative value
    _test(30.0, '0', '10000011', '1' * 3 + '0' * 4, torch.bfloat16)
    _test(-24.0, '1', '10000011', '1' + '0' * 6, torch.bfloat16)

def test_fp16():
    # zero and neg zero
    _test(0.0, '0', '0' * 5, '0' * 10, torch.float16)
    _test(-0.0, '1', '0' * 5, '0' * 10, torch.float16)

    # special values
    _test(float('nan'), '0', '1' * 5, '1' + '0' * 9, torch.float16)
    _test(float('inf'), '0', '1' * 5, '0' * 10, torch.float16)
    _test(float('-inf'), '1', '1' * 5, '0' * 10, torch.float16)

    # values below checked with https://en.wikipedia.org/wiki/Half-precision_floating-point_format

    # largest normal
    _test(65504, '0', '1' * 4 + '0', '1' * 10, torch.float16)
    _test(-65504, '1', '1' * 4 + '0', '1' * 10, torch.float16)

    # smallest normal
    _test(0.00006103515625, '0', '0' * 4 + '1', '0' * 10, torch.float16)
    _test(-0.00006103515625, '1', '0' * 4 + '1', '0' * 10, torch.float16)

    # largest denormal
    _test(0.000060975552, '0', '0' * 5, '1' * 10, torch.float16)
    _test(-0.000060975552, '1', '0' * 5, '1' * 10, torch.float16)

    # smallest denormal
    _test(0.000000059604645, '0', '0' * 5, '0' * 9 + '1', torch.float16)
    _test(-0.000000059604645, '1', '0' * 5, '0' * 9 + '1', torch.float16)

    # positive and negative value
    _test(30.0, '0', '10011', '1' * 3 + '0' * 7, torch.float16)
    _test(-24.0, '1', '10011', '1' + '0' * 9, torch.float16)
