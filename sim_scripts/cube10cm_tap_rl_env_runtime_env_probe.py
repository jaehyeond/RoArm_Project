"""Probe local CUDA visibility before cube10cm tap env runtime sanity.

This does not launch IsaacLab, run physics, build data, train, control a robot,
or use SSH. It only records local GPU/torch visibility for runtime gating.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480"
DEFAULT_OUT_JSON = LOG_DIR / "cube10cm_tap_rl_env_runtime_env_probe.json"
DEFAULT_OUT_SUMMARY = LOG_DIR / "cube10cm_tap_rl_env_runtime_env_probe_summary.out"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out_summary", type=Path, default=DEFAULT_OUT_SUMMARY)
    args = parser.parse_args()

    import torch

    nvidia = subprocess.run(["nvidia-smi"], check=False, capture_output=True, text=True)
    nvidia_text = (nvidia.stdout or "") + (nvidia.stderr or "")
    cuda_available = bool(torch.cuda.is_available())
    cuda_count = int(torch.cuda.device_count())
    device0 = torch.cuda.get_device_name(0) if cuda_available else "NA"
    nvidia_ok = nvidia.returncode == 0 and "NVIDIA-SMI" in nvidia_text

    result: dict[str, Any] = {
        "artifact_type": "cube10cm_tap_rl_env_runtime_env_probe_v1",
        "branch": "professor_cube10cm_tap_reaction_quality_tier",
        "local_env_probe_only": True,
        "no_isaaclab_runtime_dataset_training_robot_ssh": True,
        "nvidia_smi_ok": nvidia_ok,
        "nvidia_smi_returncode": nvidia.returncode,
        "nvidia_smi_first_lines": nvidia_text.splitlines()[:8],
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "torch_cuda_available": cuda_available,
        "torch_cuda_device_count": cuda_count,
        "torch_device0": device0,
        "verdict": "GPU_VISIBLE_TO_TORCH" if cuda_available and cuda_count > 0 else "GPU_NOT_VISIBLE_TO_TORCH",
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "line1 artifact=cube10cm_tap_rl_env_runtime_env_probe_v1 "
        "local_env_probe_only=YES isaaclab_runtime=NO dataset_generation=NO training=NO robot_control=NO ssh=NO",
        (
            "line2 nvidia_smi "
            f"ok={nvidia_ok} returncode={nvidia.returncode} "
            f"first_line={nvidia_text.splitlines()[0] if nvidia_text.splitlines() else 'NA'}"
        ),
        (
            "line3 torch_cuda "
            f"torch={torch.__version__} cuda_available={cuda_available} "
            f"device_count={cuda_count} torch_cuda_version={torch.version.cuda}"
        ),
        f"line4 device0 {device0}",
        (
            "line5 verdict "
            f"{result['verdict']} gpu_random_sanity_should_use_cuda=YES cpu_sanity_is_not_promotion_evidence=YES"
        ),
    ]
    args.out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        print(line)
    return 0 if cuda_available and cuda_count > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
