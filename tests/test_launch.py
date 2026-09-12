"""Check paper launch geometry without allocating GPUs or downloading models."""

import os
import shlex
import subprocess
from pathlib import Path

import pytest

LAUNCHER = Path(__file__).resolve().parents[1] / "scripts" / "train_paper.sh"


def launch(tmp_path, axis="race5", method="maxk", **overrides):
    env = {
        "PATH": os.environ["PATH"],
        "DRY_RUN": "1",
        **{key: str(value) for key, value in overrides.items()},
    }
    return subprocess.run(
        ["bash", str(LAUNCHER), axis, method],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.parametrize("axis,k", [("race5", 5), ("gender2", 2), ("race5_gender2", 10)])
@pytest.mark.parametrize("method", ["maxk", "max1", "count"])
def test_paper_recipes(tmp_path, axis, k, method):
    result = launch(tmp_path, axis, method, TRAIN_SEED=0)
    assert result.returncode == 0, result.stderr
    expected_k = 1 if method == "max1" else k
    assert f"axis={axis} method={method} k={expected_k} seed=0" in result.stdout
    command = shlex.split(result.stdout.splitlines()[-1])
    assert command[:2] == ["accelerate", "launch"]
    assert command[command.index("--num_processes") + 1] == "32"


def test_multi_node_arguments(tmp_path):
    result = launch(
        tmp_path, NUM_MACHINES=4, MACHINE_RANK=2, MAIN_PROCESS_IP="10.0.0.1", NUM_PROCESSES=32
    )
    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout.splitlines()[-1])
    for key, value in {
        "--num_processes": "32",
        "--num_machines": "4",
        "--machine_rank": "2",
        "--main_process_ip": "10.0.0.1",
    }.items():
        assert command[command.index(key) + 1] == value


def test_single_gpu_uses_single_process_config(tmp_path):
    result = launch(tmp_path, NUM_PROCESSES=1, SAMPLE_BATCH_SIZE=16, TRAIN_BATCH_SIZE=16)
    assert result.returncode == 0, result.stderr
    assert "single_gpu.yaml" in result.stdout


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"NUM_PROCESSES": 1}, "global rollout batch"),
        ({"NUM_MACHINES": 4}, "MAIN_PROCESS_IP"),
        ({"NUM_MACHINES": 3}, "divisible by NUM_MACHINES"),
        ({"MACHINE_RANK": 1}, "MACHINE_RANK"),
        ({"NUM_PROCESSES": 0}, "positive integers"),
        ({"TRAIN_BATCH_SIZE": 4}, "equal SAMPLE_BATCH_SIZE"),
    ],
)
def test_invalid_launch_fails_before_accelerate(tmp_path, overrides, message):
    result = launch(tmp_path, **overrides)
    assert result.returncode == 2
    assert message in result.stderr


@pytest.mark.parametrize("method,k", [("maxk", 7), ("max1", 1)])
def test_color7_recipe(tmp_path, method, k):
    result = launch(tmp_path, "color7", method, TRAIN_SEED=0)
    assert result.returncode == 0, result.stderr
    assert f"axis=color7 method={method} k={k} seed=0" in result.stdout
    assert "single_gpu.yaml" in result.stdout
    command = shlex.split(result.stdout.splitlines()[-1])
    assert command[command.index("--num_processes") + 1] == "1"
    assert command[-1] == "--config=configs/sd35_paper.py:color7"
