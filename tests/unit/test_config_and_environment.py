import subprocess

import pytest

from sentinelai import environment
from sentinelai.config import Settings, check_model_url


@pytest.mark.parametrize("url", ["http://127.0.0.1:11434", "http://localhost:8080", "http://[::1]:11434"])
def test_loopback_model_urls_allowed(url):
    check_model_url(url, allow_remote=False)


@pytest.mark.parametrize("url", ["http://10.0.0.5:11434", "https://api.example.com", "ftp://127.0.0.1"])
def test_remote_model_urls_rejected(url):
    with pytest.raises(ValueError):
        check_model_url(url, allow_remote=False)


def test_remote_allowed_only_when_explicit():
    check_model_url("http://10.0.0.5:11434", allow_remote=True)
    with pytest.raises(ValueError):
        Settings(model_base_url="http://10.0.0.5:11434", _env_file=None)


def test_initial_policy_respects_classifier_flag():
    assert not Settings(classifier_enabled=False, _env_file=None).initial_policy().classifier_enabled


def test_gpu_detection_handles_missing_driver(monkeypatch):
    environment.detect_nvidia_gpus.cache_clear()
    monkeypatch.setattr(environment.shutil, "which", lambda name: None)
    assert environment.detect_nvidia_gpus() == []
    environment.detect_nvidia_gpus.cache_clear()


def test_gpu_detection_parses_nvidia_smi(monkeypatch):
    environment.detect_nvidia_gpus.cache_clear()
    monkeypatch.setattr(environment.shutil, "which", lambda name: "/usr/bin/nvidia-smi")

    class Done:
        stdout = "NVIDIA GeForce RTX 4060 Laptop GPU, 8188, 1200, 615.71, 37, [N/A]\n"

    monkeypatch.setattr(environment.subprocess, "run", lambda *a, **k: Done())
    (gpu,) = environment.detect_nvidia_gpus()
    assert gpu["memory_total_mib"] == 8188 and gpu["driver"] == "615.71"
    assert gpu["utilization_pct"] == 37 and gpu["temperature_c"] is None
    environment.detect_nvidia_gpus.cache_clear()


def test_gpu_detection_survives_failing_tool(monkeypatch):
    environment.detect_nvidia_gpus.cache_clear()
    monkeypatch.setattr(environment.shutil, "which", lambda name: "/usr/bin/nvidia-smi")

    def boom(*a, **k):
        raise subprocess.TimeoutExpired("nvidia-smi", 5)

    monkeypatch.setattr(environment.subprocess, "run", boom)
    assert environment.detect_nvidia_gpus() == []
    environment.detect_nvidia_gpus.cache_clear()


def test_environment_summary(tmp_path):
    info = environment.summarize_environment(tmp_path)
    assert {"python", "cpu_count", "disk_free_gib", "nvidia_gpus"} <= info.keys()
