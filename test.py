import torch

from fp_formats import _assert_equals, dtype_to_interesting_values

def test_fp32():
    interesting_values = dtype_to_interesting_values[torch.float]
    for fp_ref, s_enc_ref, e_enc_ref, m_enc_ref, dtype, _notes in interesting_values:
        _assert_equals(fp_ref, s_enc_ref, e_enc_ref, m_enc_ref, dtype)

def test_bf16():
    interesting_values = dtype_to_interesting_values[torch.bfloat16]
    for fp_ref, s_enc_ref, e_enc_ref, m_enc_ref, dtype, _notes in interesting_values:
        _assert_equals(fp_ref, s_enc_ref, e_enc_ref, m_enc_ref, dtype)

def test_fp16():
    interesting_values = dtype_to_interesting_values[torch.float16]
    for fp_ref, s_enc_ref, e_enc_ref, m_enc_ref, dtype, _notes in interesting_values:
        _assert_equals(fp_ref, s_enc_ref, e_enc_ref, m_enc_ref, dtype)
