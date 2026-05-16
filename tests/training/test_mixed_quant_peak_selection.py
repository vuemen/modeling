"""Tests for peak TFLOPS routing by dtype."""
import pytest

from zrt.training.io.perf_tables import peak_tflops_for
from zrt.training.spec.dtype import Dtype
from zrt.training.spec.system import GPU


def _gpu(name="h100", *, bf16=989.0, fp8=3958.0, fp4=0.0):
    return GPU(name=name, flops_bf16=bf16, flops_fp8=fp8, flops_fp4=fp4,
               hbm_gb=80.0, hbm_bw_gbps=3350.0)


def test_bf16_returns_bf16_peak():
    gpu = _gpu()
    assert peak_tflops_for(gpu, Dtype.BF16) == pytest.approx(989.0e12)


def test_fp16_falls_back_to_bf16_peak():
    gpu = _gpu()
    assert peak_tflops_for(gpu, Dtype.FP16) == pytest.approx(989.0e12)


def test_fp8_e4m3_returns_fp8_peak():
    gpu = _gpu()
    assert peak_tflops_for(gpu, Dtype.FP8_E4M3) == pytest.approx(3958.0e12)


def test_fp8_e5m2_returns_fp8_peak():
    gpu = _gpu()
    assert peak_tflops_for(gpu, Dtype.FP8_E5M2) == pytest.approx(3958.0e12)


def test_fp4_returns_fp4_peak_when_supported():
    gpu = _gpu(fp4=30000.0)
    assert peak_tflops_for(gpu, Dtype.FP4) == pytest.approx(30000.0e12)


def test_fp4_falls_back_to_fp8_when_unsupported():
    # H100 has fp4=0 → fallback to fp8.
    gpu = _gpu(fp4=0.0)
    assert peak_tflops_for(gpu, Dtype.FP4) == pytest.approx(3958.0e12)


def test_fp8_falls_back_to_bf16_when_unsupported():
    # A100-like: no FP8 hardware
    gpu = _gpu(fp8=0.0, fp4=0.0)
    assert peak_tflops_for(gpu, Dtype.FP8_E4M3) == pytest.approx(989.0e12)


def test_b300_yaml_loads_with_fp4_tops():
    """B300 spec declares native FP4."""
    from zrt.hardware.registry import load
    hw = load("nvidia_b300")
    assert hw.compute.fp4_tops > 0
    assert hw.compute.fp8_tops > hw.compute.bf16_tflops  # FP8 >= 2x BF16


def test_h100_yaml_declares_fp4_tops_zero():
    """H100 lacks native FP4 hardware -> fp4_tops must be 0 in spec."""
    from zrt.hardware.registry import load
    hw = load("nvidia_h100_sxm")
    assert hw.compute.fp4_tops == 0.0
