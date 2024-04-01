"""
Solidifying knowledge of floating point formats

maybe later:
* look into rounding
"""

import math
import struct
# using PyTorch for bit manipulations to avoid dealing with Python's flexible 
# int/float types
import torch
import numpy as np
from typing import Tuple
import tabulate

dtype_to_bitwidth = {
    torch.float: 32,
    torch.bfloat16: 16,
    torch.float16: 16,
    torch.float8_e4m3fn: 8,
    torch.float8_e5m2: 8,
}
dtype_to_sem_len = {
    torch.float: (1, 8, 23),
    torch.bfloat16: (1, 8, 7),
    torch.float16: (1, 5, 10),
    torch.float8_e4m3fn: (1, 4, 3),
    torch.float8_e5m2: (1, 5, 2),
}
# bias = 2 ** (exp_bitwidth - 1) - 1
dtype_to_exp_bias = {
    torch.float: 127,
    torch.bfloat16: 127,
    torch.float16: 15,
    torch.float8_e4m3fn: 7,
    torch.float8_e5m2: 15,
}
dtype_to_int_dtype = {
    torch.float: torch.int32,
    torch.float16: torch.int16,
    torch.bfloat16: torch.int16,
    torch.float8_e4m3fn: torch.int8,
    torch.float8_e5m2: torch.int8,
}

def get_sem_bits(x: torch.Tensor, bitwidth: int) -> Tuple[str, str, str]:
    """
    Input: a tensor with a single floating point element
    Output: bit strings for sign, exponent, mantissa encodings of the input
    """
    assert x.numel() == 1
    s_len, e_len, m_len = dtype_to_sem_len[x.dtype]

    new_dtype = dtype_to_int_dtype[x.dtype]
    x = x.view(new_dtype)

    # Numpy has a nice function to get the string representation of binary. 
    # Since we are using ints as views of floats, need to specify the width 
    # to avoid numpy from using two's complement for negative numbers.
    # print('x_int8', x)
    np_res = np.binary_repr(x.numpy(), width=bitwidth)
    # print('bin', np_res)

    s, e, m = np_res[0], np_res[s_len:(s_len+e_len)], np_res[(s_len+e_len):]
    assert len(s) == s_len
    assert len(e) == e_len
    assert len(m) == m_len
    return s, e, m

def exp_encoding_to_exp(exp_bit_str: str, dtype):
    """
    Input: bit string of exponent for dtype
    Output: integer representation of exponent
    """
    exp_biased = int(exp_bit_str, 2)
    exp_bias = dtype_to_exp_bias[dtype]
    exp_unbiased = exp_biased - exp_bias

    # for denormalized values, increment exponent back
    # up by one
    if all([b == '0' for b in exp_bit_str]):
        exp_unbiased += 1

    return exp_unbiased

def sem_bits_to_sem_vals(s_enc, e_enc, m_enc, dtype):
    """
    Input: encodings of sign, exponent, mantissa for dtype
    Output: integer sign, integer exponent, float32 mantissa, special value

    If special value is filled out, sem are none
    If sem are filled out, special value is none
    """
    sign = 1 if s_enc == '0' else -1

    # handle special values
    if all([bit == '1' for bit in e_enc]):
        if all([bit == '0' for bit in m_enc]):
            if s_enc == '0':
                return None, None, None, float('inf')
            else:
                return None, None, None, float('-inf')
        else:
            return None, None, None, float('nan')

    exponent = exp_encoding_to_exp(e_enc, dtype)

    is_zero = all([b == '0' for b in e_enc + m_enc])
    is_denormal = (not is_zero) and all([b == '0' for b in e_enc])
    is_normal = not is_zero and not is_denormal

    if is_zero:
        return sign, exponent, 0.0, None
    
    mantissa = 1.0 if is_normal else 0.0
    cur_pow_2 = -1
    for m_bit in m_enc:
        mantissa += int(m_bit) * pow(2, cur_pow_2)
        cur_pow_2 -= 1
    return sign, exponent, mantissa, None

def sem_vals_to_f32(s_i, e_i, m_f, special_value):
    """
    Input: integer sign, integer exponent, float32 mantissa, special value
    Output: float32 value
    """
    if special_value is not None:
        return special_value
    f = s_i * pow(2, e_i) * m_f
    return f

def sem_vals_to_formula(s_i, e_i, m_f, special_value):
    """
    Input: integer sign, integer exponent, float32 mantissa, special value
    Output: formula to get the float32 value
    """
    if special_value is not None:
        return special_value
    return f'{s_i} * 2^{e_i} * {m_f}'

def assert_same(fp1, fp2):
    if math.isnan(fp1):
        assert math.isnan(fp2)
    elif math.isinf(fp1):
        if fp1 > 0:
            assert math.isinf(fp2) and fp2 > 0
        else:
            assert math.isinf(fp2) and fp2 < 0
    else:
        assert (abs(fp2 - fp1) / (fp1 + 1e-20)) - 1 < 1e-12, f'{fp2} != {fp1}'

def run(dtype):
    print('dtype', dtype)
    bitwidth = dtype_to_bitwidth[dtype]

    headers = ['orig_val', 'formula', 's_enc', 'e_enc', 'm_enc', 'note']
    results = []

    vals = [
        [0.0, None],
        [-0.0, None],
        # TODO(later): test that other bit formats corresponding to nan also work,
        # no way to get from them from a float nan
        [float('nan'), None],
        [float('inf'), None],
        [float('-inf'), None],
        [30.0, 'sample pos'],
        [-24.0, 'sample neg'],
    ]

    # extend the table programmaticaly for largest/smallest normals and 
    # subnormals
    s_len, e_len, m_len = dtype_to_sem_len[dtype]
    # s, e, m, notes
    bit_vals = [
        ('0', '1' * (e_len - 1) + '0', '1' * m_len, 'largest normal'),
        ('1', '1' * (e_len - 1) + '0', '1' * m_len, 'largest neg normal'),
        ('0', '0' * (e_len - 1) + '1', '0' * m_len, 'smallest normal'),
        ('1', '0' * (e_len - 1) + '1', '0' * m_len, 'smallest neg normal'),
        ('0', '0' * e_len, '1' * m_len, 'largest subnormal'),
        ('1', '0' * e_len, '1' * m_len, 'largest neg subnormal'),
        ('0', '0' * e_len, '0' * (m_len - 1) + '1', 'smallest subnormal'),
        ('1', '0' * e_len, '0' * (m_len - 1) + '1', 'smallest neg subnormal'),
    ]

    for s, e, m, notes in bit_vals:
        # https://stackoverflow.com/a/38283005/1058521 works for float, but will
        # not extend trivially to non-cpp formats
        # new_float = struct.unpack('!f',struct.pack('!I', int(s + e + m, 2)))[0]

        # for now, just use the utils we built in this file
        s, e, m, special_value = sem_bits_to_sem_vals(s, e, m, dtype)
        new_float = sem_vals_to_f32(s, e, m, special_value)
        vals.append([new_float, notes])

    for orig_val, note in vals:
        # print('orig_val', orig_val, 'note', note)
        x = torch.tensor(orig_val, dtype=dtype)
        # print('orig x', x)
        s_enc, e_enc, m_enc = get_sem_bits(x, bitwidth=bitwidth)
        # print('encodings', s_enc, e_enc, m_enc)
        s_i, e_i, m_f, special_value = sem_bits_to_sem_vals(s_enc, e_enc, m_enc, dtype)
        new_val = sem_vals_to_f32(s_i, e_i, m_f, special_value)
        formula = sem_vals_to_formula(s_i, e_i, m_f, special_value)
        # print('new_val', new_val)
        assert_same(orig_val, new_val)

        results.append([orig_val, formula, s_enc, e_enc, m_enc, note])

    print(tabulate.tabulate(results, headers=headers))
    print('\n')


if __name__ == '__main__':
    for dtype in (
        torch.float,
        torch.bfloat16,
        torch.float16,
        # torch.float8_e4m3fn,
    ):
        run(dtype)
