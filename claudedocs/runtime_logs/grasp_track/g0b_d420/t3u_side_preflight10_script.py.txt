#!/usr/bin/env python3
"""p16 / t3u — fixed-base RoArm side-midpoint parallel PhysX grasp.

Forward-only runner for the user-approved, sim-only D419 exception.  It consumes the
p15 Grasping-SDG candidate artifact, but never treats NVIDIA's flying-gripper root as a
RoArm TCP.  The asymmetric attempt3 64+64 collision hulls calibrate the physical pinch
centre before IK.  Physics uses an analytic D29 x H50, 24.83 g cylinder, no kinematic
attachment, parsed URDF joint limits, and explicit whole-moving-arm contact sensors.

This file deliberately imports the frozen p14 runner as a library for already-audited
version/USD/jaw extraction helpers.  It does not edit or execute p10--p14 as programs.

Canonical protocols:
  claudedocs/runtime_logs/grasp_track/g0b_d420/t3u_side_preflight10_prereg.md
  claudedocs/runtime_logs/grasp_track/g0b_d420/t3u_side_phys1_preflight10_prereg.md

``t3u_side_preflight1`` and ``t3u_side_preflight2`` are immutable failed evidence.
``t3u_side_preflight3`` and ``t3u_side_preflight4`` are immutable launch-infrastructure
evidence.  Preflight3 died with its sandbox namespace; preflight4 reached Supervisor V6
but failed before its contract or child fork when ``/proc/1/ns/pid`` denied ``stat``.
Neither tag contains Isaac/science evidence.  Preflight5 reached the first Dynamic
Control getter but returned no articulation candidate before the local task schedule.
This source admits only forward-only reactive ``side_preflight10`` and the still-blocked
canonical tag.  Preflight6 proved that one activation frame still left the deprecated
Dynamic Control articulation query empty and aborted before the task schedule.  The
reactive p7 replacement removed that deprecated query and uses exactly two same-process
PhysX behavioral controls (registered overlap, then clear HOME), followed by a strict
full-state re-baseline.  The scientific task remains 2,340 fresh steps; its subject,
controls, thresholds, trace, render, and lifecycle gates are otherwise unchanged.
Preflight7 then failed before either diagnostic because its one-filter identity gate
expected the configured regex spelling rather than PhysX's concrete env0 representative.
Preflight8 changed only that observed replicated-view representation contract.  Its
two-frame behavioral proof passed and its plan was generated, but its re-baseline tried
to clear ``DirectRLEnv.reward_buf`` before the first task ``step()``.  Isaac Lab 2.3
allocates that attribute inside ``step()``, not in ``DirectRLEnv.__init__``; p8 therefore
aborted with exactly zero task steps.  Preflight9 removes only that invalid clear,
explicitly gates the expected pre-first-step absence, and retains the same proof/task.
It completed all 2,340 task steps but rejected the valid observed PhysX clock because
it compared engine time to ideal decimal ``0.005`` products.  Preflight10 changes only
clock evidence/validation: the complete callback-dt sequence and ``math.fsum`` are
authoritative, manager/context counts and elapsed deltas must agree, and nominal
11.700/11.710 s remain informational trace labels.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
import signal
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _path in (REPO, REPO / "sim_scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

CASE_DIR = REPO / "claudedocs/runtime_logs/grasp_track/g0b_d420"
P14_PATH = REPO / "sim_scripts/p14_g0b_t3y_cyld29h50_workspace_parallel_physics_sweep.py"
P14_SHA256 = "fcaa7b1c6aeea65cd7fd335d9cd17ee5424a53d81764f67642d074a28e3e0133"
P10_PATH = REPO / "sim_scripts/p10_g0b_t3t_cyld29h50_tilted_close_sweep_grasp_probe.py"
P10_SHA256 = "63c6b2127d969e3291da6943eab6da1037034c154a8f21fe447519cbcb2f6cff"
P15_PATH = REPO / "sim_scripts/p15_g0b_t3s_cyld29h50_side_midpoint_sdg_candidates.py"
P15_SHA256 = "250a3f406f83d3b0cc95be7ccdc666d043e28eb5b5c0f9fb25e450e26ee17240"
SUPERVISOR_PATH = REPO / "sim_scripts/p16_g0b_t3u_detached_physics_render_supervisor_v12.py"
SUPERVISOR_SHA256 = "49c3ae4455e02706934ee5a8eb3e62d21d2231e724ecee9efb92e499fdb5565d"
P15_RUN_LABEL = "side_sdg2"
P15_PREFIX = "t3s_side_sdg2"
P15_PREREG_PATH = CASE_DIR / f"{P15_PREFIX}_prereg.md"
P15_PREREG_SHA256 = "23acb036cd1a26f577cff8145ef4031f1c4075af3e4e60f1df28a42d86da8330"
P15_CONFIG_PATH = CASE_DIR / f"{P15_PREFIX}_config.json"
P15_CONFIG_SHA256 = "dc93153cc2b8667b5156538b51140c3ea5eb1f1da19f507e5cd0f1227721638c"
P15_RERUN_VALIDATION_PATH = CASE_DIR / f"{P15_PREFIX}_rerun_validation.json"
P15_RERUN_VALIDATION_SHA256 = "18d98f66da9bb33da20a7965f9b04acf5bb0b9514c88911234f5f8a8959d8cc2"
P15_INSPECTION_PATH = CASE_DIR / f"{P15_PREFIX}_inspection.png"
P15_INSPECTION_SHA256 = "fb76856e17ba301ccd94e9388387af8b91eef6c004d91dd5df7c2462bb87cc8f"
P15_MANUAL_VISUAL_PATH = CASE_DIR / f"{P15_PREFIX}_manual_visual_inspection.json"
P15_MANUAL_VISUAL_SHA256 = "0c363c28cf71c4700496bb81cb118260a450505d7e664f3644873246223c95ce"
P15_EXIT_STATUS_PATH = CASE_DIR / f"{P15_PREFIX}_exit_status.txt"
P15_EXIT_STATUS_SHA256 = "9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa"
P15_STDOUT_PATH = CASE_DIR / f"{P15_PREFIX}_stdout.log"
P15_STDOUT_SHA256 = "9fd8b60355774d16d111cc89a4e45e17ebc09622659d44a6616883ddf646d3f8"
P15_PID_PATH = CASE_DIR / f"{P15_PREFIX}_pid.txt"
P15_PID_SHA256 = "074dd59abdb4ecfade74cfaf00e05f150534b479e265101d5f5f12958ff86353"
P15_FAILURE_PATH = CASE_DIR / f"{P15_PREFIX}_failure.json"
P15_CANDIDATES_SHA256 = "67eb07d68268be25c894c47fb6bee79347e2b201dea643c4823a0873bfcde384"
P15_BOUND_OUTPUT_PATHS = {
    "mesh_proxy.json": CASE_DIR / f"{P15_PREFIX}_mesh_proxy.json",
    "raw_candidates.json": CASE_DIR / f"{P15_PREFIX}_raw_candidates.json",
    "candidates.json": CASE_DIR / f"{P15_PREFIX}_candidates.json",
    "timeline.rrd": CASE_DIR / f"{P15_PREFIX}_timeline.rrd",
    "timeline.rbl": CASE_DIR / f"{P15_PREFIX}_timeline.rbl",
    "rerun_validation.json": P15_RERUN_VALIDATION_PATH,
    "inspection.png": P15_INSPECTION_PATH,
    "script.py.txt": CASE_DIR / f"{P15_PREFIX}_script.py.txt",
    "argv.txt": CASE_DIR / f"{P15_PREFIX}_argv.txt",
}
WITNESS_RESULTS_PATH = CASE_DIR / "t3y_workspace1_results.json"
WITNESS_RESULTS_SHA256 = "0f169bfababc458e98912c0aa3592def7935c791b30374235a0f1962f154fb26"
WITNESS_PLAN_PATH = CASE_DIR / "t3y_workspace1_plan.json"
WITNESS_PLAN_SHA256 = "2871c714a5b2f08519944a280d4a95b352566a0f5e0da53d41bf519282ade5bf"
JAW_PATH = REPO / "sim_scripts/g0b_t3_attempt3_jaw_throat_occlusion_readonly_vertex_audit.py"
JAW_SHA256 = "bca4f898023f63f21d540483a169499760038c582ce3a7919d7622e77946e1c3"
URDF_PATH = REPO / "local_assets/roarm_m3/urdf/roarm_m3.urdf"
URDF_SHA256 = "64dc8d082cbce9a13a0697a11a0eaeaebbc54bbcd97e7aafaa40da483969dae2"
ATTEMPT3_ROOT_PATH = (
    REPO
    / "claudedocs/runtime_logs/grasp_track/g0a_d344/collision_asset/attempt3"
    / "roarm_m3_fullmesh_fixed_point_parts/roarm_m3.usd"
)
ATTEMPT3_PHYSICS_PATH = (
    ATTEMPT3_ROOT_PATH.parent / "configuration/roarm_m3_physics.usd"
)
CANDIDATES_PATH = P15_BOUND_OUTPUT_PATHS["candidates.json"]
PREFLIGHT_PROFILE = "side_preflight10"
CANONICAL_PROFILE = "side_phys1"
EXECUTABLE_PROFILES = (PREFLIGHT_PROFILE, CANONICAL_PROFILE)
PREFLIGHT1_PREREG = CASE_DIR / "t3u_side_preflight1_prereg.md"
PREFLIGHT1_PREREG_SHA256 = "4f3ba53f3f350c11962120fbfc5d57818961f6d552db00997a682c074ad0faa4"
PREFLIGHT1_SOURCE_SHA256 = "0f253207dd5b77073cd504b8e344118575f2d6cc082eec6ae5c679ee180e31bf"
PREFLIGHT1_FAILURE_PATH = CASE_DIR / "t3u_side_preflight1_failure.json"
PREFLIGHT1_FAILURE_SHA256 = "ebbb6a6109a587d1547bdba02e0d1ed28eb2af16bbaf15239ee7275009e6179f"
PREFLIGHT1_OUTCOME_PATH = CASE_DIR / "t3u_side_preflight1_supervisor_outcome.json"
PREFLIGHT1_OUTCOME_SHA256 = "460ea9cff3a072b714c9a17d8d637ed53ff9f6f8b517613e99a046ffe7888c0c"
PREFLIGHT2_SOURCE_PATH = REPO / "sim_scripts/p16_g0b_t3u_cyld29h50_side_midpoint_parallel_physics.py"
PREFLIGHT2_SOURCE_SHA256 = "5c6132b68651549b2c54c9216a09ecfb4210e9b74ee1c3ba9ddf96f667dcf789"
PREFLIGHT2_SUPERVISOR_PATH = REPO / "sim_scripts/p16_g0b_t3u_detached_physics_render_supervisor.py"
PREFLIGHT2_SUPERVISOR_SHA256 = "527b06e5b9a090f4207c5f9ac5feb539c4b26f4c23f48ac59e4d802a153fa365"
PREFLIGHT2_PREREG = CASE_DIR / "t3u_side_preflight2_prereg.md"
PREFLIGHT2_PREREG_SHA256 = "e02b927edc493f4912ad9dbc5c9bd5713e4181c4e6512f0d61e50c62328bf329"
PREFLIGHT2_FAILURE_PATH = CASE_DIR / "t3u_side_preflight2_failure.json"
PREFLIGHT2_FAILURE_SHA256 = "f17e0c3a3f48c9a52ffea572b52957164b8e0adb54af1d2c9cbfe766ce88c4a3"
PREFLIGHT2_OUTCOME_PATH = CASE_DIR / "t3u_side_preflight2_supervisor_outcome.json"
PREFLIGHT2_OUTCOME_SHA256 = "443dd6a18ef7a0074a0ca04c64a3a6bcf55711991f403d4dea4ef9e733b56210"
PREFLIGHT2_PHASE_PATH = CASE_DIR / "t3u_side_preflight2_phase.jsonl"
PREFLIGHT2_PHASE_SHA256 = "010ae83487eb2cac6fc496ed9070cbe90242a3a8dd6f5079e90c93eb18e20ccb"
PREFLIGHT2_EXIT_STATUS_PATH = CASE_DIR / "t3u_side_preflight2_exit_status.txt"
PREFLIGHT2_EXIT_STATUS_SHA256 = "a5e45837a2959db847f7e67a915d0ecaddd47f943af2af5fa6453be497faabca"
PREFLIGHT2_TERMINAL_ATTESTATION_PATH = CASE_DIR / "t3u_side_preflight2_terminal_attestation.json"
PREFLIGHT2_TERMINAL_ATTESTATION_SHA256 = "6fbab4dc67a800d7a3d649fc4bf72fea2ad3dbffe5a57961a0284e96c923c58b"
RETIRED_CANONICAL_PREREG = CASE_DIR / "t3u_side_phys1_prereg.md"
RETIRED_CANONICAL_PREREG_SHA256 = "c52a31bddf6cfd64700074c66d0b6c1d43736379f37c581842334ce06819bbb2"
PREFLIGHT3_SOURCE_PATH = (
    REPO / "sim_scripts/p16_g0b_t3u_cyld29h50_side_midpoint_parallel_physics_v2.py"
)
PREFLIGHT3_SOURCE_SHA256 = "b9f987eef7f62527a64a80900a9811e73eea7a8d02885e2e820192af456f64ac"
PREFLIGHT3_SUPERVISOR_PATH = (
    REPO / "sim_scripts/p16_g0b_t3u_detached_physics_render_supervisor_v5.py"
)
PREFLIGHT3_SUPERVISOR_SHA256 = "998865694378509549841cac6fd1d486d49abf1ef8f53a5d74d423657213db5d"
PREFLIGHT3_PREREG = CASE_DIR / "t3u_side_preflight3_prereg.md"
PREFLIGHT3_PREREG_SHA256 = "4c5a068c28f54e5ba13313c55cac350f6aaff38fe10d52db7451dc962b5067a0"
PREFLIGHT3_CANONICAL_PREREG = CASE_DIR / "t3u_side_phys1_preflight3_prereg.md"
PREFLIGHT3_CANONICAL_PREREG_SHA256 = "b1b20f9e8eee24950f53c663f3712d787f77ac697cb66ada87b0502b17c51faf"
PREFLIGHT3_LAUNCHER_PATH = CASE_DIR / "t3u_side_preflight3_supervisor_launcher.log"
PREFLIGHT3_LAUNCHER_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
PREFLIGHT3_POSTHOC_AUDIT_FAILURE_PATH = (
    CASE_DIR / "t3u_side_preflight3_supervisor_failure.json"
)
PREFLIGHT3_POSTHOC_AUDIT_FAILURE_SHA256 = (
    "218ec29911134acaca1d472762fa27341f87fed136bd39849099c2eeca35ebcc"
)
PREFLIGHT3_POSTHOC_AUDIT_FAILURE_SIZE = 3074
PREFLIGHT3_POSTHOC_AUDIT_FAILURE_MTIME_NS = 1786507847537740266
PREFLIGHT4_SOURCE_PATH = (
    REPO / "sim_scripts/p16_g0b_t3u_cyld29h50_side_midpoint_parallel_physics_v3.py"
)
PREFLIGHT4_SOURCE_SHA256 = (
    "f03561858e12841d4b3eef3047083d69e96791136dbaa8e76bc0e9eb178e1d2a"
)
PREFLIGHT4_SUPERVISOR_PATH = (
    REPO / "sim_scripts/p16_g0b_t3u_detached_physics_render_supervisor_v6.py"
)
PREFLIGHT4_SUPERVISOR_SHA256 = (
    "40f46f3f94bf1926294831e4d41106b98fb9b69efd1cdb82d977e6be899f0f2f"
)
PREFLIGHT4_PREREG = CASE_DIR / "t3u_side_preflight4_prereg.md"
PREFLIGHT4_PREREG_SHA256 = (
    "6b413e343630cbac6dbec458769aac9310c9caea3cfedfb436d0f3582ac2ea13"
)
PREFLIGHT4_CANONICAL_PREREG = CASE_DIR / "t3u_side_phys1_preflight4_prereg.md"
PREFLIGHT4_CANONICAL_PREREG_SHA256 = (
    "6ccc5616d35abd8863c7bf48dc005cb7e058daf32414fd51df65d7f08a46466f"
)
PREFLIGHT4_FAILURE_PATH = CASE_DIR / "t3u_side_preflight4_supervisor_failure.json"
PREFLIGHT4_FAILURE_SHA256 = (
    "50cd5e0eec3444e44862dc0885137389c8073decbfdf7fbbe8d2a55b8bbf66b5"
)
PREFLIGHT4_FAILURE_SIZE = 1397
PREFLIGHT4_FAILURE_MTIME_NS = 1786510054805718060
PREFLIGHT4_LAUNCHER_PATH = CASE_DIR / "t3u_side_preflight4_supervisor_launcher.log"
PREFLIGHT4_LAUNCHER_SHA256 = (
    "3b37b2967c6dcb702f71dde28a8c3dd1d2069a7ec7f15a650f91667096bca2e9"
)
PREFLIGHT4_LAUNCHER_SIZE = 786
PREFLIGHT4_LAUNCHER_MTIME_NS = 1786510054822718284
PREFLIGHT5_SOURCE_PATH = (
    REPO / "sim_scripts/p16_g0b_t3u_cyld29h50_side_midpoint_parallel_physics_v4.py"
)
PREFLIGHT5_SOURCE_SHA256 = (
    "f019d55b437c93e53a2f6820af633821765c24a8741cd170fe3b4d189dc4a4ad"
)
PREFLIGHT5_SUPERVISOR_PATH = (
    REPO / "sim_scripts/p16_g0b_t3u_detached_physics_render_supervisor_v7.py"
)
PREFLIGHT5_SUPERVISOR_SHA256 = (
    "b344b49fb955a833ef4eee92c48f4ef7cf95ffdda4e4cef58cd806a681d15fcd"
)
PREFLIGHT5_CANONICAL_PREREG = CASE_DIR / "t3u_side_phys1_preflight5_prereg.md"
PREFLIGHT5_CANONICAL_PREREG_SHA256 = (
    "9415c0703897c1d3548c2db126c6a285e4c3418032fb71b6c973e5b9d4bb6e44"
)
PREFLIGHT5_EVIDENCE_SHA256 = {
    "argv.txt": "72075a703aa5a4ce4bff5fd80d3df6153761a985e06d4255527ccca4d66aeeaf",
    "exit_status.txt": "a5e45837a2959db847f7e67a915d0ecaddd47f943af2af5fa6453be497faabca",
    "failure.json": "0a051340dc4b448032fa4ebceee7927229497e4d5799502fbf90c71604746b5b",
    "nvidia_smi_after.csv": "4861bdfb03fa4e34a2d4dc6fa065e73e9a77e4f3d9535f9d54e86f8ddd8ff3a0",
    "nvidia_smi_before.csv": "4861bdfb03fa4e34a2d4dc6fa065e73e9a77e4f3d9535f9d54e86f8ddd8ff3a0",
    "nvidia_smi_supervisor_end.csv": "4861bdfb03fa4e34a2d4dc6fa065e73e9a77e4f3d9535f9d54e86f8ddd8ff3a0",
    "pgid.txt": "d12acd4cf7a466fa64f861c57f08aa78d6f312396889e5ac8ca3c16698df68d3",
    "phase.jsonl": "ea99ff199f00f3fe28fc0d0dfd28655c8163c0d410f178cd716ce43a89df8d76",
    "physics_python_pid.txt": "02f8a58b014ab34a712a6646a4c1023f8ed933d037c988fcbbb07ed4d3ebfd8b",
    "prereg.md": "319376d827f92355a51c71a0397f3aeace6f6a70c4ce4c3a41a8d8e7aa3c349b",
    "script.py.txt": "f019d55b437c93e53a2f6820af633821765c24a8741cd170fe3b4d189dc4a4ad",
    "stdout.log": "53115fbd845a855060af8f03e83cf1f049c131036b72427e10f5582f53b56cfe",
    "supervisor_contract.json": "c24535b82567070d9a1adff7a77ef398dd060ff5bbfbd393294f1b852dda6c31",
    "supervisor_launcher.log": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "supervisor_outcome.json": "cd44a132735d001b05f4c93bf9c9bdf05c76cded8a877b86d9f87570a24191d6",
    "supervisor_pid.txt": "d12acd4cf7a466fa64f861c57f08aa78d6f312396889e5ac8ca3c16698df68d3",
    "terminal_attestation.json": "d99a0f19d946d149d3307134cd79305b4ba5a1858662758d5a867717dbf9a84e",
}
PREFLIGHT5_DEPENDENCY_PINS = {
    "preflight5_source_sha256": PREFLIGHT5_SOURCE_SHA256,
    "preflight5_supervisor_sha256": PREFLIGHT5_SUPERVISOR_SHA256,
    "preflight5_canonical_prereg_sha256": PREFLIGHT5_CANONICAL_PREREG_SHA256,
    **{
        f"preflight5_{suffix.replace('.', '_')}_sha256": expected_sha
        for suffix, expected_sha in PREFLIGHT5_EVIDENCE_SHA256.items()
    },
}
PREFLIGHT6_SOURCE_PATH = (
    REPO / "sim_scripts/p16_g0b_t3u_cyld29h50_side_midpoint_parallel_physics_v5.py"
)
PREFLIGHT6_SOURCE_SHA256 = (
    "b6eb67cbec8e11752b926d8d04498c3d29fd993b8ac87b5aabbc207c92d06458"
)
PREFLIGHT6_SUPERVISOR_PATH = (
    REPO / "sim_scripts/p16_g0b_t3u_detached_physics_render_supervisor_v8.py"
)
PREFLIGHT6_SUPERVISOR_SHA256 = (
    "8cd7946b7dfb826a2fce8a9a9580603a945037aa48aa591d8979fc58ba03d9b2"
)
PREFLIGHT6_PREREG = CASE_DIR / "t3u_side_preflight6_prereg.md"
PREFLIGHT6_PREREG_SHA256 = (
    "198c81869ff8a547edb5bbc497e0c080864b39cf7ae47db676a03ba7d5028375"
)
PREFLIGHT6_CANONICAL_PREREG = CASE_DIR / "t3u_side_phys1_preflight6_prereg.md"
PREFLIGHT6_CANONICAL_PREREG_SHA256 = (
    "9dcaeee6840edeea81b0e7b7a1b92aa2415f57f03c1173be921692dda7556cc0"
)
PREFLIGHT6_EVIDENCE_SHA256 = {
    "argv.txt": "2b986a37216998a5c39614c2fd00c0f5956b89e75217103603ae30718f2a254d",
    "exit_status.txt": "a5e45837a2959db847f7e67a915d0ecaddd47f943af2af5fa6453be497faabca",
    "failure.json": "43e086551ee54063795fd915d5fa8c0dfd927855090928ca69f2518f609ab245",
    "nvidia_smi_after.csv": "4861bdfb03fa4e34a2d4dc6fa065e73e9a77e4f3d9535f9d54e86f8ddd8ff3a0",
    "nvidia_smi_before.csv": "4861bdfb03fa4e34a2d4dc6fa065e73e9a77e4f3d9535f9d54e86f8ddd8ff3a0",
    "nvidia_smi_supervisor_end.csv": "4861bdfb03fa4e34a2d4dc6fa065e73e9a77e4f3d9535f9d54e86f8ddd8ff3a0",
    "pgid.txt": "af9f0be71cb2098bec6697b688ff6586a1a2993b859d6024803b1ac3b2b3406a",
    "phase.jsonl": "ce6771980c37ab13dc4ed7f5ff52348be34d00ed750c14c2492ae73c274143d4",
    "physics_python_pid.txt": "b3f23e306dd4fe5f9f63716d899ead9cfb28ab8acec17ac91caff8f337bfdb70",
    "prereg.md": PREFLIGHT6_PREREG_SHA256,
    "script.py.txt": PREFLIGHT6_SOURCE_SHA256,
    "stdout.log": "50476db40a8c03594d09a2091ef8c01100afae196bc602a280146d933d94b1cf",
    "supervisor_contract.json": "b70810b9cebe8154490015c0b1fe4e3791d3a2b9034742342cf19b9a2b0f5554",
    "supervisor_launcher.log": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "supervisor_outcome.json": "da8161d632b6da1ba48e8a6c25a3cca240461bfd03b4f6fefc8c6561793adbe7",
    "supervisor_pid.txt": "af9f0be71cb2098bec6697b688ff6586a1a2993b859d6024803b1ac3b2b3406a",
    "terminal_attestation.json": "3d13364e2cb3113c69485e31aa12f30e1403ccef323852eb12bf58d464094d08",
}
PREFLIGHT6_DEPENDENCY_PINS = {
    "preflight6_source_sha256": PREFLIGHT6_SOURCE_SHA256,
    "preflight6_supervisor_sha256": PREFLIGHT6_SUPERVISOR_SHA256,
    "preflight6_canonical_prereg_sha256": PREFLIGHT6_CANONICAL_PREREG_SHA256,
    **{
        f"preflight6_{suffix.replace('.', '_')}_sha256": expected_sha
        for suffix, expected_sha in PREFLIGHT6_EVIDENCE_SHA256.items()
    },
}
PREFLIGHT7_SOURCE_PATH = (
    REPO / "sim_scripts/p16_g0b_t3u_cyld29h50_side_midpoint_parallel_physics_v6.py"
)
PREFLIGHT7_SOURCE_SHA256 = (
    "aabac6c76985682e32376195d187134da028bb6cc768148e883fbd56c18b3dbe"
)
PREFLIGHT7_SUPERVISOR_PATH = (
    REPO / "sim_scripts/p16_g0b_t3u_detached_physics_render_supervisor_v9.py"
)
PREFLIGHT7_SUPERVISOR_SHA256 = (
    "9f1cf1be075fe052f8d2db196be9a14207d80dc82f46a58a364a88513bacb716"
)
PREFLIGHT7_PREREG = CASE_DIR / "t3u_side_preflight7_prereg.md"
PREFLIGHT7_PREREG_SHA256 = (
    "c2b17be775ad4df465c967d3cbdf08c571eed3cfbd21325265798a906b0d6e96"
)
PREFLIGHT7_CANONICAL_PREREG = CASE_DIR / "t3u_side_phys1_preflight7_prereg.md"
PREFLIGHT7_CANONICAL_PREREG_SHA256 = (
    "1453c33642b5d32e2e24dba66da5732240afc6484af7142ddc7953c97b1efbbd"
)
PREFLIGHT7_EVIDENCE_SHA256 = {
    "argv.txt": "357f3e28c876adea544844850a947fd42e84b05a3a16d7632eaf222cbb0c13e4",
    "exit_status.txt": "a5e45837a2959db847f7e67a915d0ecaddd47f943af2af5fa6453be497faabca",
    "failure.json": "3b84af93ab399725a2fab220d9dd5883d6f5286bfa0241c63bec04b15d5bc01a",
    "nvidia_smi_after.csv": "4861bdfb03fa4e34a2d4dc6fa065e73e9a77e4f3d9535f9d54e86f8ddd8ff3a0",
    "nvidia_smi_before.csv": "4861bdfb03fa4e34a2d4dc6fa065e73e9a77e4f3d9535f9d54e86f8ddd8ff3a0",
    "nvidia_smi_supervisor_end.csv": "4861bdfb03fa4e34a2d4dc6fa065e73e9a77e4f3d9535f9d54e86f8ddd8ff3a0",
    "pgid.txt": "dd4e3756bcff21b948f61a8878c6ab30ed71609007888afa9ebbaee9167a4a0f",
    "phase.jsonl": "43788a37e63e000bf00d30346f6fdef9d152eef51041ca1ea604420cb028e2de",
    "physics_python_pid.txt": "1dbadd895274cd162f4b76bb1717e1fa6758e5674c98a8cb2184c391ec6e79ac",
    "prereg.md": PREFLIGHT7_PREREG_SHA256,
    "script.py.txt": PREFLIGHT7_SOURCE_SHA256,
    "stdout.log": "b03901b6ae7f2a6f3939b716f89165956a3144bf5869d1ce753e899dc6f389a3",
    "supervisor_contract.json": "98f4dd9b4007b770518abef87bbc1186554992b0c7c7c892164563b7e2e4ff0f",
    "supervisor_launcher.log": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "supervisor_outcome.json": "26e161620c725c45e24beaf163fe8221bff061ffbf667ea2fac811259bda64f0",
    "supervisor_pid.txt": "dd4e3756bcff21b948f61a8878c6ab30ed71609007888afa9ebbaee9167a4a0f",
    "terminal_attestation.json": "41a65b75453e4f97a326ec8ae4e966a09070812a1e8374696f3a46f421e4a8ea",
}
PREFLIGHT7_DEPENDENCY_PINS = {
    "preflight7_source_sha256": PREFLIGHT7_SOURCE_SHA256,
    "preflight7_supervisor_sha256": PREFLIGHT7_SUPERVISOR_SHA256,
    "preflight7_canonical_prereg_sha256": PREFLIGHT7_CANONICAL_PREREG_SHA256,
    **{
        f"preflight7_{suffix.replace('.', '_')}_sha256": expected_sha
        for suffix, expected_sha in PREFLIGHT7_EVIDENCE_SHA256.items()
    },
}
PREFLIGHT8_SOURCE_PATH = (
    REPO / "sim_scripts/p16_g0b_t3u_cyld29h50_side_midpoint_parallel_physics_v7.py"
)
PREFLIGHT8_SOURCE_SHA256 = (
    "e23606b9b51262bfd2c73bcd808ea4cf770e14f1d625ec6d1612f67e84d651ec"
)
PREFLIGHT8_SUPERVISOR_PATH = (
    REPO / "sim_scripts/p16_g0b_t3u_detached_physics_render_supervisor_v10.py"
)
PREFLIGHT8_SUPERVISOR_SHA256 = (
    "68d4cf3d6b1e81e7ff468b18148270242f2cbda3f39a607a1c46203c96813e32"
)
PREFLIGHT8_PREREG = CASE_DIR / "t3u_side_preflight8_prereg.md"
PREFLIGHT8_PREREG_SHA256 = (
    "2e733377e73940094513c158859d29d798a4ee92a0971930102eabfc68075689"
)
PREFLIGHT8_CANONICAL_PREREG = CASE_DIR / "t3u_side_phys1_preflight8_prereg.md"
PREFLIGHT8_CANONICAL_PREREG_SHA256 = (
    "e3df68274649e205afedea390e9dc09852b8fc19bd3a2052a00d29171c9cc240"
)
PREFLIGHT8_EVIDENCE_SHA256 = {
    "argv.txt": "2d4f68e6b106ce407de2c560b1865d7e36911e5d28da55dbd8ab93339f98e8d3",
    "exit_status.txt": "a5e45837a2959db847f7e67a915d0ecaddd47f943af2af5fa6453be497faabca",
    "failure.json": "dba5311f96202a3535f5b98007eea3ebffcd47c27e3b364787b3b887b2183f10",
    "nvidia_smi_after.csv": "4861bdfb03fa4e34a2d4dc6fa065e73e9a77e4f3d9535f9d54e86f8ddd8ff3a0",
    "nvidia_smi_before.csv": "4861bdfb03fa4e34a2d4dc6fa065e73e9a77e4f3d9535f9d54e86f8ddd8ff3a0",
    "nvidia_smi_supervisor_end.csv": "4861bdfb03fa4e34a2d4dc6fa065e73e9a77e4f3d9535f9d54e86f8ddd8ff3a0",
    "pgid.txt": "dce9307d70bf503725c075a083462d8a1dcc80d8a1156ee4306359e76b603091",
    "phase.jsonl": "a80d10afa9c99128296375cbbbddb58b5e534018234a23ca4eedc52bac268a6f",
    "physics_python_pid.txt": "d440472c17b45776cf2bafeef40274cf23bb79ec9adbcaaedf050a28c75836d1",
    "plan.json": "3af82f19b9a80769200b6099400721749a0df349b6ca1f5a52880dda2b8bda89",
    "prereg.md": PREFLIGHT8_PREREG_SHA256,
    "script.py.txt": PREFLIGHT8_SOURCE_SHA256,
    "stdout.log": "f80c3e190a73af4236335dc7fa8e65048dae0f3c3af3755790d6292c361383a1",
    "supervisor_contract.json": "fa3fb1f17b70d2a6c75ba2d347968e60a46a79dcaf70dde3b570f7dcdc348f7f",
    "supervisor_launcher.log": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "supervisor_outcome.json": "c9fc630b3842691a0af956f53b12bb72f07a5873560b9e9d99871a988264da68",
    "supervisor_pid.txt": "dce9307d70bf503725c075a083462d8a1dcc80d8a1156ee4306359e76b603091",
    "terminal_attestation.json": "ecdbb53867b3fadc049438d8f1e6cf8a78dd0ddc3e6ad2bae952ca40eaef6b36",
}
PREFLIGHT8_DEPENDENCY_PINS = {
    "preflight8_source_sha256": PREFLIGHT8_SOURCE_SHA256,
    "preflight8_supervisor_sha256": PREFLIGHT8_SUPERVISOR_SHA256,
    "preflight8_canonical_prereg_sha256": PREFLIGHT8_CANONICAL_PREREG_SHA256,
    **{
        f"preflight8_{suffix.replace('.', '_')}_sha256": expected_sha
        for suffix, expected_sha in PREFLIGHT8_EVIDENCE_SHA256.items()
    },
}
PREFLIGHT9_SOURCE_PATH = (
    REPO / "sim_scripts/p16_g0b_t3u_cyld29h50_side_midpoint_parallel_physics_v8.py"
)
PREFLIGHT9_SOURCE_SHA256 = (
    "56717ca98c4e1d1b19b026fb9d3b658b2a2dac6465f4c8318638251fb3eab2f1"
)
PREFLIGHT9_SUPERVISOR_PATH = (
    REPO / "sim_scripts/p16_g0b_t3u_detached_physics_render_supervisor_v11.py"
)
PREFLIGHT9_SUPERVISOR_SHA256 = (
    "bf36d2acbe9cb7fb6cb6721e8dca90ce4f7623fab5813449f459893131fae29b"
)
PREFLIGHT9_PREREG = CASE_DIR / "t3u_side_preflight9_prereg.md"
PREFLIGHT9_PREREG_SHA256 = (
    "c1020690231a343da95cc5b8b7e756e4651619eaf7b2338894e1157afc6e4987"
)
PREFLIGHT9_CANONICAL_PREREG = CASE_DIR / "t3u_side_phys1_preflight9_prereg.md"
PREFLIGHT9_CANONICAL_PREREG_SHA256 = (
    "8f16340c5edbbfe84730a7eb4bdbeebaf7015865154d9f890c3a517022d9cb25"
)
PREFLIGHT9_EVIDENCE_SHA256 = {
    "argv.txt": "90f467bf9de385b5abace70bd76bd45e5dcbbe1c24de92cce1a9c17e0a69dcb2",
    "exit_status.txt": "a5e45837a2959db847f7e67a915d0ecaddd47f943af2af5fa6453be497faabca",
    "failure.json": "6eccef7427294c413243f3b15025d24baae6c52bc0726269c14536e5656a4a4e",
    "nvidia_smi_after.csv": "4861bdfb03fa4e34a2d4dc6fa065e73e9a77e4f3d9535f9d54e86f8ddd8ff3a0",
    "nvidia_smi_before.csv": "4861bdfb03fa4e34a2d4dc6fa065e73e9a77e4f3d9535f9d54e86f8ddd8ff3a0",
    "nvidia_smi_supervisor_end.csv": "4861bdfb03fa4e34a2d4dc6fa065e73e9a77e4f3d9535f9d54e86f8ddd8ff3a0",
    "pgid.txt": "d1dc327519ebea4eb0bb3d11c8771051cde7db9788e76efa170fd55fe8941bff",
    "phase.jsonl": "9f3b4ec81e392b0e8ca13fa1bf846352bd543ef74e55e345de6be4af256cecf8",
    "physics_python_pid.txt": "fbc513f834845dfe78cacbd0548ab734d9763d9262dc7612f903ff1f1586cacf",
    "plan.json": "1cb380e86a7ce5685d8a14c576319bd95d34c82dede3c44030c541889971cf32",
    "prereg.md": PREFLIGHT9_PREREG_SHA256,
    "script.py.txt": PREFLIGHT9_SOURCE_SHA256,
    "stdout.log": "2bce27ac87e462666cf7994192ec6bb1b85976e33eba51b6654c4929cdc0605c",
    "supervisor_contract.json": "23b7ecc6cf0fb36cfdcfc5c7e227bb88e893210788ddaa68696da98162a32e7a",
    "supervisor_launcher.log": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "supervisor_outcome.json": "48031bdcf484a858f2bf5d9531a59569a78976fdd561c8de98e8092e89815b9f",
    "supervisor_pid.txt": "d1dc327519ebea4eb0bb3d11c8771051cde7db9788e76efa170fd55fe8941bff",
    "terminal_attestation.json": "055cd5bfb99927196f9bd015f9c0e1352883b6b1e2780bca352d29d61698d7e8",
}
PREFLIGHT9_DEPENDENCY_PINS = {
    "preflight9_source_sha256": PREFLIGHT9_SOURCE_SHA256,
    "preflight9_supervisor_sha256": PREFLIGHT9_SUPERVISOR_SHA256,
    "preflight9_canonical_prereg_sha256": PREFLIGHT9_CANONICAL_PREREG_SHA256,
    **{
        f"preflight9_{suffix.replace('.', '_')}_sha256": expected_sha
        for suffix, expected_sha in PREFLIGHT9_EVIDENCE_SHA256.items()
    },
}
PREFLIGHT_PREREG = CASE_DIR / "t3u_side_preflight10_prereg.md"
CANONICAL_PREREG = CASE_DIR / "t3u_side_phys1_preflight10_prereg.md"
PREFLIGHT_PREREG_SHA256 = "d75d790d7f5fff65af966fda928b0726907778a6361d71a06701973d8e3e26ee"
CANONICAL_PREREG_SHA256 = "2266679959cc670180ecf521ed0c04121b0906c893e936b6307d99f04fafeb9b"

LOG = "p16_t3u_side"
TAG = "t3u"
SCHEMA = "g0b.t3s.side_sdg_candidates.v1"
RERUN_VERSION = "0.34.1"
RERUN_CLI = "/home/cgxr/miniconda3/envs/isaaclab/bin/rerun"
ISAAC_PYTHON = "/home/cgxr/miniconda3/envs/isaaclab/bin/python"
VIDEO_FPS = 20
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
VIDEO_STEP_STRIDE = 10

OBJ_RADIUS_M = 0.0145
OBJ_DIAM_M = 0.029
OBJ_HEIGHT_M = 0.050
OBJ_MASS_KG = 0.02483
OBJECT_CENTER_M = np.asarray(
    [0.4235072423787768, 0.17237803311822986, 0.025], dtype=np.float64
)
SUPPORT_Z_M = 0.0
STATIC_FRICTION = 0.40
DYNAMIC_FRICTION = 0.30
RESTITUTION = 0.0
GRAVITY = 9.81

Q5_OPEN_DEG = 88.30998496351378
Q5_CLOSE_COMMAND_DEG = 22.0
HOME_DEG = np.asarray([0.0, 0.0, 90.0, 0.0, 0.0, Q5_OPEN_DEG])
SELF_COLLISION_CONTROL_PAIR = ("link2", "link4")
SELF_COLLISION_POSITIVE_EXPECTED_PAIRS = (
    ("link2", "link4"),
    ("link2", "link5"),
)
SELF_COLLISION_POSITIVE_Q_DEG = np.asarray(
    [0.0, 0.0, 165.0, 90.0, 0.0, 45.0], dtype=np.float64
)
SELF_COLLISION_NEGATIVE_Q_DEG = HOME_DEG.copy()
SELF_COLLISION_POSITIVE_INRADIUS_GATE_MM = 5.0
SELF_COLLISION_NEGATIVE_SEPARATION_GATE_MM = 60.0
SELF_COLLISION_NEGATIVE_FORCE_GATE_N = 1.0e-8
WITNESS_SOURCE_TRIAL_ID = "t3y_workspace1/trial_005948"
WITNESS_Q_APPROACH_DEG = np.asarray(
    [22.147551724447293, 54.009710735468865, 84.68324826073442,
     -26.586518474346644, 90.0, Q5_OPEN_DEG], dtype=np.float64
)
WITNESS_Q_DESCEND_DEG = np.asarray(
    [22.1475517244473, 64.01845721713758, 64.17330685057833,
     -17.790265742820885, 90.0, Q5_OPEN_DEG], dtype=np.float64
)
WITNESS_Q_CLOSE_DEG = WITNESS_Q_DESCEND_DEG.copy()
WITNESS_Q_CLOSE_DEG[5] = 66.4
WITNESS_Q_LIFT_DEG = np.asarray(
    [22.147551724447297, 59.77007943117819, 63.87783298418268,
     -12.942301413553086, 90.0, 66.4], dtype=np.float64
)
APPROACH_CLEARANCE_M = 0.040
ELEVATED_PREGRASP_Z_M = 0.040
NEAR_STAGE_BACKOFF_M = 0.005
NEAR_STAGE_Z_M = 0.010
LIFT_DELTA_M = 0.025
PINCH_Q_STEP_DEG = 0.1
PINCH_OFFSET_DELTAS_M = np.asarray([0.0, 0.00025, 0.00050, 0.00075, 0.00100])
PINCH_CONTACT_RESIDUAL_GATE_M = 0.0005
PLANNED_COLLISION_CLEARANCE_GATE_M = 0.001
FINAL_CLOSURE_FRAME_ERROR_GATE_DEG = 2.0
PREFLIGHT_CANDIDATE_INDEX = 5
PREFLIGHT_CANDIDATE_ID = "side_sdg_005_raw_025092"
P15_CANDIDATE_IDS = (
    "side_sdg_000_raw_050244",
    "side_sdg_001_raw_030852",
    "side_sdg_002_raw_041604",
    "side_sdg_003_raw_003796",
    "side_sdg_004_raw_009060",
    "side_sdg_005_raw_025092",
    "side_sdg_006_raw_008724",
    "side_sdg_007_raw_036164",
)
CANONICAL_STATIC_FEASIBLE_CANDIDATES = (
    (5, "side_sdg_005_raw_025092"),
    (7, "side_sdg_007_raw_036164"),
)

DT_S = 1.0 / 200.0
CLOCK_CALLBACK_NOMINAL_DT_ABS_TOL_S = 1.0e-9
CLOCK_ELAPSED_ULP_MULTIPLIER = 8
PHASE_STEPS = {
    "settle": 120,
    "approach": 400,
    "stage": 400,
    "descend": 400,
    "close": 400,
    "hold": 120,
    "lift": 500,
}
TOTAL_STEPS = sum(PHASE_STEPS.values())


def _clock_elapsed_abs_tolerance_s(*values: float) -> float:
    """Return the narrow, scale-aware tolerance for engine clock subtraction.

    Callback ``step_size`` values are the elapsed-time authority.  Manager and
    SimulationContext clocks may each incur one binary64 addition/subtraction
    rounding, so their independently observed deltas receive at most eight ULPs
    at the largest compared magnitude.  This is deliberately many orders of
    magnitude narrower than the p9 nominal-decimal discrepancy.
    """
    if not values or not all(
        type(value) is float and math.isfinite(value) for value in values
    ):
        raise ValueError("CLOCK_TOLERANCE_REQUIRES_FINITE_VALUES")
    comparison_magnitude = max(1.0, *(abs(value) for value in values))
    return float(
        CLOCK_ELAPSED_ULP_MULTIPLIER * math.ulp(comparison_magnitude)
    )


def _clock_elapsed_comparison(
    observed_delta_s: float,
    callback_fsum_s: float,
) -> dict[str, Any]:
    """Persist an independently reproducible observed-delta comparison."""
    if not (
        type(observed_delta_s) is float
        and type(callback_fsum_s) is float
        and math.isfinite(observed_delta_s)
        and math.isfinite(callback_fsum_s)
    ):
        raise ValueError("CLOCK_COMPARISON_REQUIRES_FINITE_FLOATS")
    ulp_budget_s = _clock_elapsed_abs_tolerance_s(
        observed_delta_s, callback_fsum_s
    )
    absolute_error_s = float(abs(observed_delta_s - callback_fsum_s))
    return {
        "observed_delta_s": observed_delta_s,
        "callback_fsum_s": callback_fsum_s,
        "absolute_error_s": absolute_error_s,
        "ulp_multiplier": CLOCK_ELAPSED_ULP_MULTIPLIER,
        "ulp_budget_s": ulp_budget_s,
        "pass": bool(absolute_error_s <= ulp_budget_s),
    }


def _clock_manager_context_equal(
    manager_steps: int,
    manager_time_s: float,
    context_steps: int,
    context_time_s: float,
) -> bool:
    """Strictly bind the two installed engine clock interfaces."""
    return bool(
        type(manager_steps) is int
        and type(context_steps) is int
        and type(manager_time_s) is float
        and type(context_time_s) is float
        and math.isfinite(manager_time_s)
        and math.isfinite(context_time_s)
        and manager_steps == context_steps
        and manager_time_s == context_time_s
    )


RESULT_SEMANTIC_CHECK_KEYS = frozenset(
    {
        "result_top_level_profile_and_mode_exact",
        "plan_exact_ordered_trial_set_recomputed",
        "composed_object_material_mass_units_all_clones_exact",
        "support_filter_actual_identity_exact",
        "self_filter_actual_identity_exact",
        "precontrol_and_postcontrol_self_filter_identity_reuse_exact",
        "cylinder_plus_base_and_six_reporters_every_clone_exact",
        "self_collision_stage_readback_exact",
        "diagnostic_one_frame_rebaseline_and_task_epoch_exact",
        "authored_rest_and_runtime_pose_exact_urdf_clearance_alignment",
        "fixed_base_metatype_joint_and_full_step_stability_exact",
        "composed_and_runtime_joint_limits_equal_frozen_urdf",
        "numeric_trace_metrics_quaternions_counts_recomputed",
        "authoritative_trace_step_time_phase_cadence_exact",
        "causal_classification_masks_counts_and_verdict_recomputed",
        "source_prereg_p15_and_dependency_pins_exact",
        "retired_preflight2_failure_provenance_exact_and_nonpromotable",
        "retired_preflight3_launch_provenance_exact_and_nonpromotable",
        "retired_preflight4_launch_provenance_exact_and_nonpromotable",
        "retired_preflight5_dynamic_control_abort_exact_and_nonpromotable",
        "retired_preflight6_post_activation_dynamic_control_abort_exact_and_nonpromotable",
        "retired_preflight7_filter_representation_abort_exact_and_nonpromotable",
        "retired_preflight8_reward_buffer_lifecycle_abort_exact_and_nonpromotable",
        "retired_preflight9_clock_accounting_abort_exact_and_nonpromotable",
        "runtime_instrumentation_recomputed_not_trusted",
    }
)

CONTACT_GATE_N = 0.02
JAW_LOAD_GATE_N = 0.01
LIFT_GATE_MM = 6.0
TIP_GATE_DEG = math.degrees(math.atan(OBJ_DIAM_M / OBJ_HEIGHT_M))
SETTLE_SUPPORT_N = OBJ_MASS_KG * GRAVITY
SETTLE_REL_TOL = 0.35

MOVING_BODIES = ("link1", "link2", "link3", "link4", "link5", "gripper_link")
FIXED_BASE_BODY = "world"  # attempt3 importer merges URDF world/base_link here
SELF_CONTACT_BODIES = (FIXED_BASE_BODY, *MOVING_BODIES)
CONTACT_REPORT_BODIES = SELF_CONTACT_BODIES
JAW_BODIES = ("link5", "gripper_link")
NONJAW_BODIES = ("link1", "link2", "link3", "link4")
ADJACENT_SELF_PAIR_EXCLUSIONS = tuple(
    zip(SELF_CONTACT_BODIES[:-1], SELF_CONTACT_BODIES[1:])
)
SELF_PAIRS = tuple(
    (a, b)
    for i, a in enumerate(SELF_CONTACT_BODIES)
    for j, b in enumerate(SELF_CONTACT_BODIES)
    if j > i + 1
)

RUN_OUTPUT_SUFFIXES = (
    "results.json",
    "plan.json",
    "trace.npz",
    "timeline.rrd",
    "timeline.rbl",
    "rerun_validation.json",
    "decision_snapshot.png",
    "inspection.png",
    "rgb_frames_manifest.json",
    "side_grasp.mp4",
    "script.py.txt",
    "argv.txt",
    "phase.jsonl",
    "render_phase.jsonl",
    "preclose_sentinel.json",
    "terminal_attestation.json",
    "manual_visual_inspection.json",
    "failure.json",
    "render_failure.json",
    "exit_status.txt",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _expected_object_physx_world_xyzw(object_world_pos: Any) -> Any:
    """Append an identity XYZW quaternion to authoritative world positions."""
    if len(object_world_pos.shape) != 2 or int(object_world_pos.shape[1]) != 3:
        raise RuntimeError(
            f"OBJECT_WORLD_POSITION_SHAPE_INVALID shape={object_world_pos.shape}"
        )
    expected = object_world_pos.new_zeros((int(object_world_pos.shape[0]), 7))
    expected[:, :3] = object_world_pos
    expected[:, 6] = 1.0
    return expected


def validate_object_physx_world_frame_regression() -> dict[str, Any]:
    """Pure regression: env-root offsets must not be dropped from PhysX transforms."""
    import torch

    env_origins = torch.as_tensor(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [2.0, 2.0, 0.0],
            [4.0, 2.0, 0.0],
            [6.0, 2.0, 0.0],
        ],
        dtype=torch.float64,
    )
    local_center = torch.as_tensor(OBJECT_CENTER_M, dtype=torch.float64).repeat(8, 1)
    world_positions = local_center + env_origins
    physx_world_fixture = torch.zeros((8, 7), dtype=torch.float64)
    physx_world_fixture[:, :3] = world_positions
    physx_world_fixture[:, 6] = 1.0
    world_expected = _expected_object_physx_world_xyzw(world_positions)
    local_candidate = _expected_object_physx_world_xyzw(local_center)
    local_matches = torch.all(local_candidate == physx_world_fixture, dim=1)
    checks = {
        "eight_environment_fixture": tuple(env_origins.shape) == (8, 3),
        "env0_origin_zero": bool(torch.count_nonzero(env_origins[0]).item() == 0),
        "env1_through_7_origins_nonzero": bool(
            torch.all(torch.linalg.vector_norm(env_origins[1:], dim=1) > 0.0).item()
        ),
        "world_expectation_matches_all_eight": bool(
            torch.equal(world_expected, physx_world_fixture)
        ),
        "local_expectation_matches_env0_only": bool(
            local_matches[0].item() and not local_matches[1:].any().item()
        ),
        "local_expectation_rejected_env1_through_7": bool(
            (~local_matches[1:]).all().item()
        ),
    }
    report = {
        "artifact": "T3U_OBJECT_PHYSX_WORLD_FRAME_REGRESSION_V1",
        "physx_view_subspace_root": "/",
        "expected_coordinate_frame": "world",
        "checks": checks,
        "pass": all(checks.values()),
    }
    if report["pass"] is not True:
        raise RuntimeError(f"OBJECT_PHYSX_WORLD_FRAME_REGRESSION_FAIL {report}")
    return report


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_json_x(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", closefd=False) as handle:
            json.dump(jsonable(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(fd)


def write_bytes_x(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(fd)


def append_phase(path: Path, phase: str, **fields: Any) -> None:
    row = {"time_unix": time.time(), "phase": phase, **fields}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(jsonable(row), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def run_paths(prefix: str) -> dict[str, Path]:
    return {suffix: CASE_DIR / f"{prefix}_{suffix}" for suffix in RUN_OUTPUT_SUFFIXES}


def validate_preflight2_retirement() -> dict[str, Any]:
    """Bind preflight2 only as immutable, non-promotable predecessor evidence."""
    document = json.loads(PREFLIGHT2_TERMINAL_ATTESTATION_PATH.read_text())
    if not isinstance(document, dict):
        raise RuntimeError("PREFLIGHT2_RETIRED_ATTESTATION_ROOT_NOT_OBJECT")
    expected = {
        "artifact": "T3U_EXTERNAL_TERMINAL_ABORT_ATTESTATION_V2",
        "profile": "side_preflight2",
        "attestation_valid": True,
        "pass": False,
        "promotion_allowed": False,
        "scientific_artifacts_complete": False,
        "physics_steps_claimed": None,
        "supervisor_combined_exit_status": 125,
        "verdict": "ATTESTED_UPSTREAM_ABORT_BEFORE_SCIENCE__NO_PROMOTION",
    }
    for key, value in expected.items():
        actual = document.get(key)
        if type(actual) is not type(value) or actual != value:
            raise RuntimeError(
                f"PREFLIGHT2_RETIRED_ATTESTATION_SEMANTIC_FAIL key={key} "
                f"expected={value!r} actual={actual!r}"
            )
    processes = document.get("processes")
    if not isinstance(processes, dict) or processes.get("render_pid") is not None:
        raise RuntimeError("PREFLIGHT2_RETIRED_RENDER_PID_NOT_NULL")
    return {
        "status": "retired_immutable_instrumentation_abort_only",
        "promotion_allowed": False,
        "scientific_grasp_verdict": None,
        "source_sha256": PREFLIGHT2_SOURCE_SHA256,
        "supervisor_sha256": PREFLIGHT2_SUPERVISOR_SHA256,
        "prereg_sha256": PREFLIGHT2_PREREG_SHA256,
        "failure_sha256": PREFLIGHT2_FAILURE_SHA256,
        "supervisor_outcome_sha256": PREFLIGHT2_OUTCOME_SHA256,
        "phase_sha256": PREFLIGHT2_PHASE_SHA256,
        "exit_status_sha256": PREFLIGHT2_EXIT_STATUS_SHA256,
        "terminal_attestation_sha256": PREFLIGHT2_TERMINAL_ATTESTATION_SHA256,
        "retired_canonical_prereg_sha256": RETIRED_CANONICAL_PREREG_SHA256,
        "terminal_verdict": expected["verdict"],
    }


def validate_preflight3_launch_retirement() -> dict[str, Any]:
    """Bind preflight3 only as an immutable sandbox-launch abort, never science.

    The launcher file was created by shell noclobber, but the sandbox owner used
    ``bwrap --die-with-parent`` and terminated the detached process tree when the tool
    call returned.  There is no supervisor contract/outcome, child PID, phase, frozen
    source or task artifact.  Exact prefix inventory and the zero-byte launcher are the
    durable facts.  The pinned V5 ordering fsyncs contract/PID/PGID before its physics
    fork, so their exact absence establishes that no Isaac child or task step began.
    """
    expected_prefix_files = {
        PREFLIGHT3_PREREG.resolve(),
        PREFLIGHT3_LAUNCHER_PATH.resolve(),
        PREFLIGHT3_POSTHOC_AUDIT_FAILURE_PATH.resolve(),
    }
    actual_prefix_entries = {
        path.resolve() for path in CASE_DIR.glob("t3u_side_preflight3_*")
    }
    if actual_prefix_entries != expected_prefix_files:
        raise RuntimeError(
            "PREFLIGHT3_RETIRED_PREFIX_INVENTORY_DRIFT "
            f"expected={sorted(map(str, expected_prefix_files))} "
            f"actual={sorted(map(str, actual_prefix_entries))}"
        )
    if not all(path.is_file() for path in actual_prefix_entries):
        raise RuntimeError("PREFLIGHT3_RETIRED_PREFIX_ENTRY_NOT_FILE")
    if not PREFLIGHT3_LAUNCHER_PATH.is_file():
        raise RuntimeError("PREFLIGHT3_RETIRED_LAUNCHER_MISSING")
    if PREFLIGHT3_LAUNCHER_PATH.stat().st_size != 0:
        raise RuntimeError("PREFLIGHT3_RETIRED_LAUNCHER_NOT_ZERO_BYTES")
    if sha256_file(PREFLIGHT3_LAUNCHER_PATH) != PREFLIGHT3_LAUNCHER_SHA256:
        raise RuntimeError("PREFLIGHT3_RETIRED_LAUNCHER_SHA256_MISMATCH")
    posthoc_stat = PREFLIGHT3_POSTHOC_AUDIT_FAILURE_PATH.stat()
    if (
        posthoc_stat.st_size != PREFLIGHT3_POSTHOC_AUDIT_FAILURE_SIZE
        or posthoc_stat.st_mtime_ns != PREFLIGHT3_POSTHOC_AUDIT_FAILURE_MTIME_NS
        or sha256_file(PREFLIGHT3_POSTHOC_AUDIT_FAILURE_PATH)
        != PREFLIGHT3_POSTHOC_AUDIT_FAILURE_SHA256
    ):
        raise RuntimeError("PREFLIGHT3_POSTHOC_AUDIT_FAILURE_BYTES_DRIFT")
    posthoc = json.loads(PREFLIGHT3_POSTHOC_AUDIT_FAILURE_PATH.read_text())
    posthoc_expected_keys = {
        "active_child_after_finally", "argv", "artifact", "cleanup_actions",
        "last_child_outcome", "message", "supervisor_signal", "time_unix",
        "traceback", "type",
    }
    posthoc_exact = bool(
        isinstance(posthoc, dict)
        and set(posthoc) == posthoc_expected_keys
        and posthoc.get("artifact") == "T3U_DETACHED_SUPERVISOR_FAILURE_V1"
        and posthoc.get("argv")
        == [
            "sim_scripts/p16_g0b_t3u_detached_physics_render_supervisor_v6.py",
            "--profile", "side_preflight3", "--candidates_sha256",
            P15_CANDIDATES_SHA256,
        ]
        and posthoc.get("type") == "SystemExit"
        and posthoc.get("message") == "2"
        and posthoc.get("last_child_outcome") is None
        and posthoc.get("active_child_after_finally") == {}
        and posthoc.get("cleanup_actions") == []
        and posthoc.get("supervisor_signal") is None
        and type(posthoc.get("time_unix")) is float
        and math.isfinite(posthoc["time_unix"])
        and abs(posthoc["time_unix"] - 1786507847.538827) < 1e-9
        and isinstance(posthoc.get("traceback"), str)
        and "argparse.ArgumentError" in posthoc["traceback"]
        and "invalid choice: 'side_preflight3'" in posthoc["traceback"]
        and "SystemExit: 2" in posthoc["traceback"]
    )
    if not posthoc_exact:
        raise RuntimeError("PREFLIGHT3_POSTHOC_AUDIT_FAILURE_SEMANTIC_DRIFT")
    return {
        "status": "retired_immutable_sandbox_launch_abort_only",
        "promotion_allowed": False,
        "scientific_grasp_verdict": None,
        "physics_steps_claimed": 0,
        "launch_failure_boundary": "sandbox_bwrap_die_with_parent_before_supervisor_evidence",
        "supervisor_start_evidence_present": False,
        "isaac_child_started": False,
        "zero_step_authority": (
            "pinned_v5_writes_fsyncs_contract_pid_pgid_before_physics_fork__"
            "exact_prefix_has_none"
        ),
        "science_artifacts_present": False,
        "original_launch_authority_files": sorted(
            str(path)
            for path in (PREFLIGHT3_PREREG.resolve(), PREFLIGHT3_LAUNCHER_PATH.resolve())
        ),
        "posthoc_static_audit_contamination": {
            "status": "non_science_non_launch_authority__audit_cli_side_effect",
            "creator": "preflight4_frozen_audit_child",
            "source": "supervisor_v6_argparse_rejection_test",
            "isaac_or_child_started": False,
            "sha256": PREFLIGHT3_POSTHOC_AUDIT_FAILURE_SHA256,
            "size_bytes": PREFLIGHT3_POSTHOC_AUDIT_FAILURE_SIZE,
            "mtime_ns": PREFLIGHT3_POSTHOC_AUDIT_FAILURE_MTIME_NS,
            "semantic_exact": True,
        },
        "prefix_file_inventory": sorted(str(path) for path in actual_prefix_entries),
        "source_sha256": PREFLIGHT3_SOURCE_SHA256,
        "supervisor_sha256": PREFLIGHT3_SUPERVISOR_SHA256,
        "prereg_sha256": PREFLIGHT3_PREREG_SHA256,
        "canonical_prereg_sha256": PREFLIGHT3_CANONICAL_PREREG_SHA256,
        "launcher_sha256": PREFLIGHT3_LAUNCHER_SHA256,
        "launcher_size_bytes": 0,
    }


def validate_preflight4_launch_retirement() -> dict[str, Any]:
    """Bind preflight4 only as an immutable pre-child host-readback abort.

    V6 reached ``main`` but raised while reading ``/proc/1/ns/pid``.  Its frozen
    ordering performs that read before contract/PID/PGID creation and before the
    physics fork.  The exact three-entry prefix therefore proves only that Supervisor
    V6 started and failed at its host-context readback; it is not Isaac or grasp data.
    """
    expected_prefix_files = {
        PREFLIGHT4_PREREG.resolve(),
        PREFLIGHT4_FAILURE_PATH.resolve(),
        PREFLIGHT4_LAUNCHER_PATH.resolve(),
    }
    actual_prefix_entries = {
        path.resolve() for path in CASE_DIR.glob("t3u_side_preflight4_*")
    }
    if actual_prefix_entries != expected_prefix_files:
        raise RuntimeError(
            "PREFLIGHT4_RETIRED_PREFIX_INVENTORY_DRIFT "
            f"expected={sorted(map(str, expected_prefix_files))} "
            f"actual={sorted(map(str, actual_prefix_entries))}"
        )
    if not all(path.is_file() for path in actual_prefix_entries):
        raise RuntimeError("PREFLIGHT4_RETIRED_PREFIX_ENTRY_NOT_FILE")
    expected_files = (
        (PREFLIGHT4_PREREG, PREFLIGHT4_PREREG_SHA256, 30676, None),
        (
            PREFLIGHT4_FAILURE_PATH,
            PREFLIGHT4_FAILURE_SHA256,
            PREFLIGHT4_FAILURE_SIZE,
            PREFLIGHT4_FAILURE_MTIME_NS,
        ),
        (
            PREFLIGHT4_LAUNCHER_PATH,
            PREFLIGHT4_LAUNCHER_SHA256,
            PREFLIGHT4_LAUNCHER_SIZE,
            PREFLIGHT4_LAUNCHER_MTIME_NS,
        ),
    )
    for path, expected_sha, expected_size, expected_mtime_ns in expected_files:
        stat = path.stat()
        if (
            stat.st_size != expected_size
            or sha256_file(path) != expected_sha
            or (
                expected_mtime_ns is not None
                and stat.st_mtime_ns != expected_mtime_ns
            )
        ):
            raise RuntimeError(f"PREFLIGHT4_RETIRED_FILE_DRIFT path={path}")

    failure = json.loads(PREFLIGHT4_FAILURE_PATH.read_text(encoding="utf-8"))
    expected_keys = {
        "active_child_after_finally", "argv", "artifact", "cleanup_actions",
        "last_child_outcome", "message", "supervisor_signal", "time_unix",
        "traceback", "type",
    }
    expected_argv = [
        str(PREFLIGHT4_SUPERVISOR_PATH),
        "--profile", "side_preflight4", "--candidates_sha256",
        P15_CANDIDATES_SHA256,
    ]
    failure_exact = bool(
        isinstance(failure, dict)
        and set(failure) == expected_keys
        and failure.get("artifact") == "T3U_DETACHED_SUPERVISOR_FAILURE_V1"
        and failure.get("argv") == expected_argv
        and failure.get("type") == "PermissionError"
        and failure.get("message")
        == "[Errno 13] Permission denied: '/proc/1/ns/pid'"
        and failure.get("last_child_outcome") is None
        and failure.get("active_child_after_finally") == {}
        and failure.get("cleanup_actions") == []
        and failure.get("supervisor_signal") is None
        and type(failure.get("time_unix")) is float
        and math.isfinite(failure["time_unix"])
        and abs(failure["time_unix"] - 1786510054.8072057) < 1e-9
        and isinstance(failure.get("traceback"), str)
        and PREFLIGHT4_LAUNCHER_PATH.read_text(encoding="utf-8")
        == failure["traceback"]
        and "host_launch_context = _host_launch_context()" in failure["traceback"]
        and 'os.stat("/proc/1/ns/pid")' in failure["traceback"]
        and "PermissionError: [Errno 13] Permission denied" in failure["traceback"]
    )
    if not failure_exact:
        raise RuntimeError("PREFLIGHT4_RETIRED_FAILURE_SEMANTIC_DRIFT")
    return {
        "status": "retired_immutable_host_context_readback_abort_only",
        "promotion_allowed": False,
        "scientific_grasp_verdict": None,
        "physics_steps_claimed": 0,
        "launch_failure_boundary": (
            "supervisor_v6_started__proc1_pid_namespace_stat_permission_error__"
            "before_contract_and_child_fork"
        ),
        "supervisor_started": True,
        "supervisor_contract_present": False,
        "isaac_child_started": False,
        "zero_step_authority": (
            "pinned_v6_host_context_precedes_contract_pid_pgid_and_physics_fork__"
            "exact_prefix_has_no_contract_or_child_evidence"
        ),
        "science_artifacts_present": False,
        "prefix_file_inventory": sorted(str(path) for path in actual_prefix_entries),
        "source_sha256": PREFLIGHT4_SOURCE_SHA256,
        "supervisor_sha256": PREFLIGHT4_SUPERVISOR_SHA256,
        "prereg_sha256": PREFLIGHT4_PREREG_SHA256,
        "canonical_prereg_sha256": PREFLIGHT4_CANONICAL_PREREG_SHA256,
        "failure_sha256": PREFLIGHT4_FAILURE_SHA256,
        "failure_size_bytes": PREFLIGHT4_FAILURE_SIZE,
        "failure_mtime_ns": PREFLIGHT4_FAILURE_MTIME_NS,
        "launcher_sha256": PREFLIGHT4_LAUNCHER_SHA256,
        "launcher_size_bytes": PREFLIGHT4_LAUNCHER_SIZE,
        "launcher_mtime_ns": PREFLIGHT4_LAUNCHER_MTIME_NS,
        "failure_semantic_exact": True,
    }


def validate_preflight5_retirement() -> dict[str, Any]:
    """Bind preflight5 as an immutable first-DC-read startup abort, never science."""
    expected_paths = {
        suffix: CASE_DIR / f"t3u_side_preflight5_{suffix}"
        for suffix in PREFLIGHT5_EVIDENCE_SHA256
    }
    actual_paths = {
        path.resolve() for path in CASE_DIR.glob("t3u_side_preflight5_*")
    }
    if actual_paths != {path.resolve() for path in expected_paths.values()}:
        raise RuntimeError(
            "PREFLIGHT5_RETIRED_PREFIX_INVENTORY_DRIFT "
            f"expected={sorted(str(path) for path in expected_paths.values())} "
            f"actual={sorted(map(str, actual_paths))}"
        )
    for suffix, path in expected_paths.items():
        if not path.is_file() or sha256_file(path) != PREFLIGHT5_EVIDENCE_SHA256[suffix]:
            raise RuntimeError(f"PREFLIGHT5_RETIRED_FILE_DRIFT path={path}")

    failure = json.loads(expected_paths["failure.json"].read_text(encoding="utf-8"))
    failure_prefix = "SELF_COLLISION_DYNAMIC_CONTROL_FAIL "
    if not (
        isinstance(failure, dict)
        and set(failure) == {"message", "profile", "source_sha256", "traceback", "type"}
        and failure.get("profile") == "side_preflight5"
        and failure.get("source_sha256") == PREFLIGHT5_SOURCE_SHA256
        and failure.get("type") == "RuntimeError"
        and isinstance(failure.get("message"), str)
        and failure["message"].startswith(failure_prefix)
        and isinstance(failure.get("traceback"), str)
        and failure_prefix.strip() in failure["traceback"]
    ):
        raise RuntimeError("PREFLIGHT5_RETIRED_FAILURE_SCHEMA_DRIFT")
    try:
        dc_report = ast.literal_eval(failure["message"][len(failure_prefix):])
    except Exception as exc:
        raise RuntimeError("PREFLIGHT5_RETIRED_DC_REPORT_PARSE_FAIL") from exc
    dc_rows = dc_report.get("rows") if isinstance(dc_report, dict) else None
    if not (
        isinstance(dc_rows, list)
        and len(dc_rows) == 8
        and dc_report.get("is_simulating") is True
        and dc_report.get("expected_clone_count") == 8
        and dc_report.get("actual_clone_count") == 8
        and dc_report.get("pass") is False
        and all(
            isinstance(row, dict)
            and row.get("articulation_object_candidates") == []
            and row.get("pass") is False
            for row in dc_rows
        )
    ):
        raise RuntimeError("PREFLIGHT5_RETIRED_DC_REPORT_SEMANTIC_DRIFT")

    outcome = json.loads(
        expected_paths["supervisor_outcome.json"].read_text(encoding="utf-8")
    )
    physics = outcome.get("physics") if isinstance(outcome, dict) else None
    gpu = outcome.get("gpu") if isinstance(outcome, dict) else None
    if not (
        outcome.get("artifact") == "T3U_DETACHED_SUPERVISOR_OUTCOME_V7"
        and outcome.get("profile") == "side_preflight5"
        and outcome.get("pass") is False
        and outcome.get("combined_exit_status") == 125
        and outcome.get("attempts")
        == {"automatic_retry_count": 0, "physics": 1, "render": 0}
        and outcome.get("render") is None
        and isinstance(physics, dict)
        and physics.get("exit_code") == 0
        and physics.get("raw_wait_status") == 0
        and physics.get("group_reaped") is True
        and physics.get("group_members_after_reap") == []
        and physics.get("timed_out") is False
        and physics.get("signal_actions") == []
        and isinstance(gpu, dict)
        and gpu.get("fresh_pid_delta") == []
        and gpu.get("no_fresh_pid_delta") is True
    ):
        raise RuntimeError("PREFLIGHT5_RETIRED_OUTCOME_SEMANTIC_DRIFT")

    attestation = json.loads(
        expected_paths["terminal_attestation.json"].read_text(encoding="utf-8")
    )
    evidence_checks = attestation.get("evidence_checks")
    processes = attestation.get("processes")
    if not (
        attestation.get("artifact") == "T3U_EXTERNAL_TERMINAL_ABORT_ATTESTATION_V2"
        and attestation.get("profile") == "side_preflight5"
        and attestation.get("attestation_valid") is True
        and attestation.get("pass") is False
        and attestation.get("promotion_allowed") is False
        and attestation.get("scientific_artifacts_complete") is False
        and attestation.get("physics_steps_claimed") is None
        and attestation.get("physics_step_count_authority")
        == "absent_on_aborted_run__no_step_count_claim"
        and attestation.get("supervisor_combined_exit_status") == 125
        and attestation.get("verdict")
        == "ATTESTED_UPSTREAM_ABORT_BEFORE_SCIENCE__NO_PROMOTION"
        and isinstance(evidence_checks, dict)
        and len(evidence_checks) == 13
        and all(value is True for value in evidence_checks.values())
        and isinstance(processes, dict)
        and processes.get("render_pid") is None
    ):
        raise RuntimeError("PREFLIGHT5_RETIRED_ATTESTATION_SEMANTIC_DRIFT")
    return {
        "status": "retired_immutable_dynamic_control_first_read_abort_only",
        "promotion_allowed": False,
        "scientific_grasp_verdict": None,
        "task_schedule_entered": False,
        "task_physics_steps_claimed": None,
        "task_step_count_authority": "control_flow_before_run_physics_only",
        "generic_physics_step_count_claimed": None,
        "dynamic_control_rows": 8,
        "dynamic_control_candidate_counts": [0] * 8,
        "physics_child_reaped": True,
        "supervisor_reaped": True,
        "render_started": False,
        "gpu_fresh_pid_delta": [],
        "source_sha256": PREFLIGHT5_SOURCE_SHA256,
        "supervisor_sha256": PREFLIGHT5_SUPERVISOR_SHA256,
        "canonical_prereg_sha256": PREFLIGHT5_CANONICAL_PREREG_SHA256,
        "evidence_sha256": dict(PREFLIGHT5_EVIDENCE_SHA256),
        "terminal_verdict": attestation["verdict"],
    }


def validate_preflight6_retirement() -> dict[str, Any]:
    """Bind p6 as the immutable post-activation DC absence abort, never science."""
    expected_paths = {
        suffix: CASE_DIR / f"t3u_side_preflight6_{suffix}"
        for suffix in PREFLIGHT6_EVIDENCE_SHA256
    }
    actual_paths = {
        path.resolve() for path in CASE_DIR.glob("t3u_side_preflight6_*")
    }
    if actual_paths != {path.resolve() for path in expected_paths.values()}:
        raise RuntimeError(
            "PREFLIGHT6_RETIRED_PREFIX_INVENTORY_DRIFT "
            f"expected={sorted(str(path) for path in expected_paths.values())} "
            f"actual={sorted(map(str, actual_paths))}"
        )
    for suffix, path in expected_paths.items():
        if not path.is_file() or sha256_file(path) != PREFLIGHT6_EVIDENCE_SHA256[suffix]:
            raise RuntimeError(f"PREFLIGHT6_RETIRED_FILE_DRIFT path={path}")

    failure = json.loads(expected_paths["failure.json"].read_text(encoding="utf-8"))
    failure_prefix = "SELF_COLLISION_DYNAMIC_CONTROL_FAIL "
    if not (
        isinstance(failure, dict)
        and set(failure) == {"message", "profile", "source_sha256", "traceback", "type"}
        and failure.get("profile") == "side_preflight6"
        and failure.get("source_sha256") == PREFLIGHT6_SOURCE_SHA256
        and failure.get("type") == "RuntimeError"
        and isinstance(failure.get("message"), str)
        and failure["message"].startswith(failure_prefix)
        and isinstance(failure.get("traceback"), str)
        and failure_prefix.strip() in failure["traceback"]
    ):
        raise RuntimeError("PREFLIGHT6_RETIRED_FAILURE_SCHEMA_DRIFT")
    try:
        report = ast.literal_eval(failure["message"][len(failure_prefix):])
    except Exception as exc:
        raise RuntimeError("PREFLIGHT6_RETIRED_DC_REPORT_PARSE_FAIL") from exc
    rows = report.get("rows") if isinstance(report, dict) else None
    activation = report.get("activation_sync") if isinstance(report, dict) else None
    if not (
        isinstance(rows, list)
        and len(rows) == 8
        and report.get("is_simulating") is True
        and report.get("expected_clone_count") == 8
        and report.get("actual_clone_count") == 8
        and report.get("pass") is False
        and all(
            isinstance(row, dict)
            and row.get("articulation_object_candidates") == []
            and row.get("pass") is False
            for row in rows
        )
        and isinstance(activation, dict)
        and activation.get("diagnostic_physics_steps") == 1
        and activation.get("task_physics_steps") == 0
        and activation.get("pass") is True
    ):
        raise RuntimeError("PREFLIGHT6_RETIRED_DC_REPORT_SEMANTIC_DRIFT")
    outcome = json.loads(
        expected_paths["supervisor_outcome.json"].read_text(encoding="utf-8")
    )
    terminal = json.loads(
        expected_paths["terminal_attestation.json"].read_text(encoding="utf-8")
    )
    physics = outcome.get("physics") if isinstance(outcome, dict) else None
    if not (
        outcome.get("artifact") == "T3U_DETACHED_SUPERVISOR_OUTCOME_V8"
        and outcome.get("profile") == "side_preflight6"
        and outcome.get("pass") is False
        and outcome.get("combined_exit_status") == 125
        and outcome.get("attempts")
        == {"automatic_retry_count": 0, "physics": 1, "render": 0}
        and outcome.get("render") is None
        and isinstance(physics, dict)
        and physics.get("exit_code") == 0
        and physics.get("raw_wait_status") == 0
        and physics.get("group_reaped") is True
        and terminal.get("profile") == "side_preflight6"
        and terminal.get("pass") is False
        and terminal.get("promotion_allowed") is False
    ):
        raise RuntimeError("PREFLIGHT6_RETIRED_LIFECYCLE_SEMANTIC_DRIFT")
    return {
        "status": "retired_immutable_one_activation_frame_dc_absence_abort_only",
        "promotion_allowed": False,
        "scientific_grasp_verdict": None,
        "task_physics_steps_claimed": 0,
        "diagnostic_physics_steps_observed": 1,
        "render_child_started": False,
        "dynamic_control_rows": 8,
        "dynamic_control_candidate_counts": [0] * 8,
        "source_sha256": PREFLIGHT6_SOURCE_SHA256,
        "supervisor_sha256": PREFLIGHT6_SUPERVISOR_SHA256,
        "prereg_sha256": PREFLIGHT6_PREREG_SHA256,
        "canonical_prereg_sha256": PREFLIGHT6_CANONICAL_PREREG_SHA256,
        "terminal_attestation_sha256": PREFLIGHT6_EVIDENCE_SHA256[
            "terminal_attestation.json"
        ],
        "failure_semantic_exact": True,
        "lifecycle_semantic_exact": True,
    }


def validate_preflight7_retirement() -> dict[str, Any]:
    """Bind p7 as immutable replicated-filter representation abort, never science."""
    expected_paths = {
        suffix: CASE_DIR / f"t3u_side_preflight7_{suffix}"
        for suffix in PREFLIGHT7_EVIDENCE_SHA256
    }
    actual_paths = {
        path.resolve() for path in CASE_DIR.glob("t3u_side_preflight7_*")
    }
    expected_inventory = {path.resolve() for path in expected_paths.values()}
    if actual_paths != expected_inventory:
        raise RuntimeError(
            "PREFLIGHT7_RETIRED_PREFIX_INVENTORY_DRIFT "
            f"expected={sorted(map(str, expected_inventory))} "
            f"actual={sorted(map(str, actual_paths))}"
        )
    for suffix, path in expected_paths.items():
        if not path.is_file() or sha256_file(path) != PREFLIGHT7_EVIDENCE_SHA256[suffix]:
            raise RuntimeError(f"PREFLIGHT7_RETIRED_FILE_DRIFT path={path}")
    if sha256_file(PREFLIGHT7_CANONICAL_PREREG) != PREFLIGHT7_CANONICAL_PREREG_SHA256:
        raise RuntimeError("PREFLIGHT7_RETIRED_CANONICAL_PREREG_DRIFT")

    failure = json.loads(expected_paths["failure.json"].read_text(encoding="utf-8"))
    message = failure.get("message") if isinstance(failure, dict) else None
    if not (
        set(failure) == {"message", "profile", "source_sha256", "traceback", "type"}
        and failure.get("profile") == "side_preflight7"
        and failure.get("source_sha256") == PREFLIGHT7_SOURCE_SHA256
        and failure.get("type") == "RuntimeError"
        and isinstance(message, str)
        and message.startswith("ONE_FILTER_SEMANTIC_IDENTITY_FAIL ")
        and "raw=['/World/envs/env_0/Robot/link2']" in message
        and "expected=/World/envs/env_.*/Robot/link2" in message
        and all(f"/World/envs/env_{i}/Robot/link2" in message for i in range(8))
    ):
        raise RuntimeError("PREFLIGHT7_RETIRED_FAILURE_SEMANTIC_DRIFT")

    outcome = json.loads(expected_paths["supervisor_outcome.json"].read_text())
    terminal = json.loads(expected_paths["terminal_attestation.json"].read_text())
    physics = outcome.get("physics") if isinstance(outcome, dict) else None
    gpu = outcome.get("gpu") if isinstance(outcome, dict) else None
    if not (
        outcome.get("artifact") == "T3U_DETACHED_SUPERVISOR_OUTCOME_V9"
        and outcome.get("profile") == "side_preflight7"
        and outcome.get("pass") is False
        and outcome.get("combined_exit_status") == 125
        and outcome.get("attempts") == {"automatic_retry_count": 0, "physics": 1, "render": 0}
        and outcome.get("render") is None
        and isinstance(physics, dict)
        and physics.get("exit_code") == 0
        and physics.get("raw_wait_status") == 0
        and physics.get("group_reaped") is True
        and physics.get("group_members_after_reap") == []
        and isinstance(gpu, dict)
        and gpu.get("fresh_pid_delta") == []
        and terminal.get("artifact") == "T3U_EXTERNAL_TERMINAL_ABORT_ATTESTATION_V2"
        and terminal.get("profile") == "side_preflight7"
        and terminal.get("attestation_valid") is True
        and terminal.get("pass") is False
        and terminal.get("promotion_allowed") is False
        and terminal.get("scientific_artifacts_complete") is False
        and terminal.get("physics_steps_claimed") is None
        and terminal.get("supervisor_combined_exit_status") == 125
        and terminal.get("verdict") == "ATTESTED_UPSTREAM_ABORT_BEFORE_SCIENCE__NO_PROMOTION"
    ):
        raise RuntimeError("PREFLIGHT7_RETIRED_LIFECYCLE_SEMANTIC_DRIFT")
    return {
        "status": "retired_immutable_replicated_filter_representation_abort_only",
        "promotion_allowed": False,
        "scientific_grasp_verdict": None,
        "task_physics_steps_claimed": None,
        "behavioral_diagnostic_entered": False,
        "observed_filter_paths": ["/World/envs/env_0/Robot/link2"],
        "configured_filter_expression": "/World/envs/env_.*/Robot/link2",
        "resolved_stage_target_count": 8,
        "render_child_started": False,
        "source_sha256": PREFLIGHT7_SOURCE_SHA256,
        "supervisor_sha256": PREFLIGHT7_SUPERVISOR_SHA256,
        "prereg_sha256": PREFLIGHT7_PREREG_SHA256,
        "canonical_prereg_sha256": PREFLIGHT7_CANONICAL_PREREG_SHA256,
        "terminal_attestation_sha256": PREFLIGHT7_EVIDENCE_SHA256[
            "terminal_attestation.json"
        ],
        "failure_semantic_exact": True,
        "lifecycle_semantic_exact": True,
    }


def validate_preflight8_retirement() -> dict[str, Any]:
    """Bind p8 as an immutable pre-task Isaac Lab buffer-lifecycle abort."""
    expected_paths = {
        suffix: CASE_DIR / f"t3u_side_preflight8_{suffix}"
        for suffix in PREFLIGHT8_EVIDENCE_SHA256
    }
    actual_paths = {path.resolve() for path in CASE_DIR.glob("t3u_side_preflight8_*")}
    expected_inventory = {path.resolve() for path in expected_paths.values()}
    if actual_paths != expected_inventory:
        raise RuntimeError(
            "PREFLIGHT8_RETIRED_PREFIX_INVENTORY_DRIFT "
            f"expected={sorted(map(str, expected_inventory))} "
            f"actual={sorted(map(str, actual_paths))}"
        )
    for suffix, path in expected_paths.items():
        if not path.is_file() or sha256_file(path) != PREFLIGHT8_EVIDENCE_SHA256[suffix]:
            raise RuntimeError(f"PREFLIGHT8_RETIRED_FILE_DRIFT path={path}")
    if sha256_file(PREFLIGHT8_CANONICAL_PREREG) != PREFLIGHT8_CANONICAL_PREREG_SHA256:
        raise RuntimeError("PREFLIGHT8_RETIRED_CANONICAL_PREREG_DRIFT")

    failure = json.loads(expected_paths["failure.json"].read_text(encoding="utf-8"))
    traceback_text = failure.get("traceback") if isinstance(failure, dict) else None
    if not (
        set(failure) == {"message", "profile", "source_sha256", "traceback", "type"}
        and failure.get("profile") == "side_preflight8"
        and failure.get("source_sha256") == PREFLIGHT8_SOURCE_SHA256
        and failure.get("type") == "AttributeError"
        and failure.get("message")
        == "'P16SideEnv' object has no attribute 'reward_buf'"
        and isinstance(traceback_text, str)
        and "rebaseline_after_behavioral_self_collision_diagnostic" in traceback_text
        and "env.reward_buf.zero_()" in traceback_text
        and "run_physics" in traceback_text
    ):
        raise RuntimeError("PREFLIGHT8_RETIRED_FAILURE_SEMANTIC_DRIFT")

    plan = json.loads(expected_paths["plan.json"].read_text(encoding="utf-8"))
    outcome = json.loads(expected_paths["supervisor_outcome.json"].read_text())
    terminal = json.loads(expected_paths["terminal_attestation.json"].read_text())
    physics = outcome.get("physics") if isinstance(outcome, dict) else None
    gpu = outcome.get("gpu") if isinstance(outcome, dict) else None
    if not (
        plan.get("profile") == "side_preflight8"
        and plan.get("n_planned") == 5
        and plan.get("n_feasible") == 5
        and plan.get("n_feasible_after_static_clearance") == 5
        and outcome.get("artifact") == "T3U_DETACHED_SUPERVISOR_OUTCOME_V10"
        and outcome.get("profile") == "side_preflight8"
        and outcome.get("pass") is False
        and outcome.get("combined_exit_status") == 125
        and outcome.get("attempts")
        == {"automatic_retry_count": 0, "physics": 1, "render": 0}
        and outcome.get("render") is None
        and isinstance(physics, dict)
        and physics.get("exit_code") == 0
        and physics.get("raw_wait_status") == 0
        and physics.get("group_reaped") is True
        and physics.get("group_members_after_reap") == []
        and isinstance(gpu, dict)
        and gpu.get("fresh_pid_delta") == []
        and terminal.get("artifact") == "T3U_EXTERNAL_TERMINAL_ABORT_ATTESTATION_V2"
        and terminal.get("profile") == "side_preflight8"
        and terminal.get("attestation_valid") is True
        and terminal.get("pass") is False
        and terminal.get("promotion_allowed") is False
        and terminal.get("scientific_artifacts_complete") is False
        and terminal.get("physics_steps_claimed") is None
        and terminal.get("supervisor_combined_exit_status") == 125
        and terminal.get("verdict")
        == "ATTESTED_UPSTREAM_ABORT_BEFORE_SCIENCE__NO_PROMOTION"
    ):
        raise RuntimeError("PREFLIGHT8_RETIRED_LIFECYCLE_SEMANTIC_DRIFT")
    return {
        "status": "retired_immutable_pre_task_reward_buffer_lifecycle_abort_only",
        "promotion_allowed": False,
        "scientific_grasp_verdict": None,
        "behavioral_diagnostic_physics_steps_observed": 2,
        "behavioral_diagnostic_completed_before_failure": True,
        "task_schedule_entered": False,
        "task_physics_steps_claimed": 0,
        "plan_generated": True,
        "render_child_started": False,
        "source_sha256": PREFLIGHT8_SOURCE_SHA256,
        "supervisor_sha256": PREFLIGHT8_SUPERVISOR_SHA256,
        "prereg_sha256": PREFLIGHT8_PREREG_SHA256,
        "canonical_prereg_sha256": PREFLIGHT8_CANONICAL_PREREG_SHA256,
        "plan_sha256": PREFLIGHT8_EVIDENCE_SHA256["plan.json"],
        "failure_sha256": PREFLIGHT8_EVIDENCE_SHA256["failure.json"],
        "terminal_attestation_sha256": PREFLIGHT8_EVIDENCE_SHA256[
            "terminal_attestation.json"
        ],
        "failure_semantic_exact": True,
        "lifecycle_semantic_exact": True,
    }


def validate_preflight9_retirement() -> dict[str, Any]:
    """Bind p9's observed valid callback clock as an immutable abort."""
    expected_paths = {
        suffix: CASE_DIR / f"t3u_side_preflight9_{suffix}"
        for suffix in PREFLIGHT9_EVIDENCE_SHA256
    }
    actual_paths = {path.resolve() for path in CASE_DIR.glob("t3u_side_preflight9_*")}
    expected_inventory = {path.resolve() for path in expected_paths.values()}
    if actual_paths != expected_inventory:
        raise RuntimeError(
            "PREFLIGHT9_RETIRED_PREFIX_INVENTORY_DRIFT "
            f"expected={sorted(map(str, expected_inventory))} "
            f"actual={sorted(map(str, actual_paths))}"
        )
    for suffix, path in expected_paths.items():
        if not path.is_file() or sha256_file(path) != PREFLIGHT9_EVIDENCE_SHA256[suffix]:
            raise RuntimeError(f"PREFLIGHT9_RETIRED_FILE_DRIFT path={path}")
    if (
        not PREFLIGHT9_CANONICAL_PREREG.is_file()
        or sha256_file(PREFLIGHT9_CANONICAL_PREREG)
        != PREFLIGHT9_CANONICAL_PREREG_SHA256
    ):
        raise RuntimeError("PREFLIGHT9_RETIRED_CANONICAL_PREREG_DRIFT")

    failure = json.loads(expected_paths["failure.json"].read_text(encoding="utf-8"))
    message = failure.get("message") if isinstance(failure, dict) else None
    traceback_text = failure.get("traceback") if isinstance(failure, dict) else None
    marker = "TASK_PHYSICS_CLOCK_ACCOUNTING_FAIL "
    try:
        clock = ast.literal_eval(message[len(marker):]) \
            if isinstance(message, str) and message.startswith(marker) else None
    except (SyntaxError, ValueError):
        clock = None
    if not (
        set(failure) == {"message", "profile", "source_sha256", "traceback", "type"}
        and failure.get("profile") == "side_preflight9"
        and failure.get("source_sha256") == PREFLIGHT9_SOURCE_SHA256
        and failure.get("type") == "RuntimeError"
        and isinstance(traceback_text, str)
        and "run_physics" in traceback_text
        and marker in traceback_text
        and isinstance(clock, dict)
    ):
        raise RuntimeError("PREFLIGHT9_RETIRED_FAILURE_SEMANTIC_DRIFT")

    observed_dt = 0.004999999888241291
    task_fsum = float(math.fsum([observed_dt] * TOTAL_STEPS))
    plan = json.loads(expected_paths["plan.json"].read_text(encoding="utf-8"))
    behavioral = plan.get("effective_self_collision_readback", {}).get(
        "behavioral_control", {}
    )
    diagnostic_dts = behavioral.get("callback_dts_s")
    diagnostic_fsum = (
        float(math.fsum(diagnostic_dts))
        if isinstance(diagnostic_dts, list) else math.nan
    )
    combined_fsum = (
        float(math.fsum([*diagnostic_dts, *([observed_dt] * TOTAL_STEPS)]))
        if isinstance(diagnostic_dts, list) else math.nan
    )
    baseline = clock.get("task_baseline", {})
    final = clock.get("task_final", {})
    behavioral_before = behavioral.get("before", {})
    behavioral_after = behavioral.get("after", {})
    checks = clock.get("checks", {})
    expected_old_checks = {
        "task_callback_count_2340": True,
        "all_task_callback_dt_005": True,
        "manager_task_step_delta_2340": True,
        "manager_task_time_delta_11_7": False,
        "simulation_context_task_step_delta_2340": True,
        "simulation_context_task_time_delta_11_7": False,
        "task_counters_2340": True,
        "combined_manager_steps_2342": True,
        "combined_manager_time_11_71": False,
        "combined_simulation_context_steps_2342": True,
        "combined_simulation_context_time_11_71": False,
    }
    if not (
        clock.get("artifact")
        == "T3U_DIAGNOSTIC_AND_TASK_PHYSICS_CLOCK_ACCOUNTING_V1"
        and clock.get("diagnostic_physics_steps") == 2
        and clock.get("task_physics_steps") == TOTAL_STEPS
        and clock.get("combined_physics_steps") == TOTAL_STEPS + 2
        and clock.get("task_local_step_range") == [1, TOTAL_STEPS]
        and clock.get("task_duration_s") == TOTAL_STEPS * DT_S
        and clock.get("combined_duration_s") == (TOTAL_STEPS + 2) * DT_S
        and clock.get("task_callback_count") == TOTAL_STEPS
        and clock.get("task_callback_dt_min_s") == observed_dt
        and clock.get("task_callback_dt_max_s") == observed_dt
        and checks == expected_old_checks
        and clock.get("pass") is False
        and diagnostic_dts == [observed_dt, observed_dt]
        and behavioral.get("pass") is True
        and baseline == {
            "simulation_manager_num_physics_steps": 4,
            "simulation_manager_time_s": 0.019999999552965164,
            "simulation_context_step_index": 4,
            "simulation_context_time_s": 0.019999999552965164,
            "robot_data_timestamp_s": 0.015,
            "object_data_timestamp_s": 0.015,
        }
        and final.get("simulation_manager_num_physics_steps") == 2344
        and final.get("simulation_context_step_index") == 2344
        and final.get("simulation_manager_time_s") == 11.719999738037586
        and final.get("simulation_context_time_s") == 11.719999738037586
        and final.get("env_sim_step_counter") == TOTAL_STEPS
        and final.get("common_step_counter") == TOTAL_STEPS
        and final.get("episode_length_buf") == [TOTAL_STEPS] * 8
        and behavioral_before.get("simulation_manager_time_s")
        == behavioral_before.get("simulation_context_time_s")
        == 0.009999999776482582
        and behavioral_after.get("simulation_manager_time_s")
        == behavioral_after.get("simulation_context_time_s")
        == baseline["simulation_manager_time_s"]
        and diagnostic_fsum == 0.009999999776482582
        and task_fsum == 11.699999738484621
        and combined_fsum == 11.709999738261104
        and final["simulation_manager_time_s"]
        - baseline["simulation_manager_time_s"] == task_fsum
        and final["simulation_context_time_s"]
        - baseline["simulation_context_time_s"] == task_fsum
        and final["simulation_manager_time_s"]
        - behavioral_before["simulation_manager_time_s"] == combined_fsum
        and final["simulation_context_time_s"]
        - behavioral_before["simulation_context_time_s"] == combined_fsum
        and plan.get("profile") == "side_preflight9"
        and plan.get("n_planned") == 5
        and plan.get("n_feasible") == 5
        and plan.get("n_feasible_after_static_clearance") == 5
    ):
        raise RuntimeError("PREFLIGHT9_RETIRED_CLOCK_EVIDENCE_DRIFT")

    outcome = json.loads(expected_paths["supervisor_outcome.json"].read_text())
    terminal = json.loads(expected_paths["terminal_attestation.json"].read_text())
    physics = outcome.get("physics", {})
    gate = outcome.get("physics_artifact_gate", {})
    if not (
        outcome.get("artifact") == "T3U_DETACHED_SUPERVISOR_OUTCOME_V11"
        and outcome.get("profile") == "side_preflight9"
        and outcome.get("pass") is False
        and outcome.get("combined_exit_status") == 125
        and outcome.get("attempts")
        == {"automatic_retry_count": 0, "physics": 1, "render": 0}
        and outcome.get("render") is None
        and physics.get("exit_code") == physics.get("raw_wait_status") == 0
        and physics.get("group_reaped") is True
        and physics.get("group_members_after_reap") == []
        and gate.get("pass") is False
        and gate.get("failure_marker", {}).get("sha256")
        == PREFLIGHT9_EVIDENCE_SHA256["failure.json"]
        and terminal.get("artifact") == "T3U_EXTERNAL_TERMINAL_ABORT_ATTESTATION_V2"
        and terminal.get("profile") == "side_preflight9"
        and terminal.get("attestation_valid") is True
        and terminal.get("pass") is False
        and terminal.get("promotion_allowed") is False
        and terminal.get("scientific_artifacts_complete") is False
        and terminal.get("physics_steps_claimed") is None
        and terminal.get("supervisor_combined_exit_status") == 125
        and terminal.get("verdict")
        == "ATTESTED_UPSTREAM_ABORT_BEFORE_SCIENCE__NO_PROMOTION"
    ):
        raise RuntimeError("PREFLIGHT9_RETIRED_LIFECYCLE_SEMANTIC_DRIFT")
    return {
        "status": "retired_immutable_valid_observed_callback_clock_abort_only",
        "promotion_allowed": False,
        "scientific_grasp_verdict": None,
        "diagnostic_physics_steps_observed": 2,
        "task_physics_steps_observed": TOTAL_STEPS,
        "combined_physics_steps_observed": TOTAL_STEPS + 2,
        "observed_callback_dt_s": observed_dt,
        "observed_task_callback_fsum_s": task_fsum,
        "observed_combined_callback_fsum_s": combined_fsum,
        "manager_context_elapsed_exact": True,
        "plan_generated": True,
        "results_generated": False,
        "render_child_started": False,
        "source_sha256": PREFLIGHT9_SOURCE_SHA256,
        "supervisor_sha256": PREFLIGHT9_SUPERVISOR_SHA256,
        "prereg_sha256": PREFLIGHT9_PREREG_SHA256,
        "canonical_prereg_sha256": PREFLIGHT9_CANONICAL_PREREG_SHA256,
        "failure_sha256": PREFLIGHT9_EVIDENCE_SHA256["failure.json"],
        "plan_sha256": PREFLIGHT9_EVIDENCE_SHA256["plan.json"],
        "terminal_attestation_sha256": PREFLIGHT9_EVIDENCE_SHA256[
            "terminal_attestation.json"
        ],
        "failure_semantic_exact": True,
        "lifecycle_semantic_exact": True,
    }


URDF_JOINT_MAP = {
    "base_link_to_link1": "base",
    "link1_to_link2": "shoulder",
    "link2_to_link3": "elbow",
    "link3_to_link4": "wrist_p",
    "link4_to_link5": "wrist_r",
    "link5_to_gripper_link": "gripper",
}
JOINT_ORDER = ("base", "shoulder", "elbow", "wrist_p", "wrist_r", "gripper")


def parse_urdf_limits() -> dict[str, tuple[float, float]]:
    if sha256_file(URDF_PATH) != URDF_SHA256:
        raise RuntimeError("URDF_SHA256_MISMATCH")
    root = ET.parse(URDF_PATH).getroot()
    out: dict[str, tuple[float, float]] = {}
    for joint in root.iter("joint"):
        label = URDF_JOINT_MAP.get(joint.get("name", ""))
        if label is None:
            continue
        limit = joint.find("limit")
        if limit is None:
            raise RuntimeError(f"URDF_LIMIT_MISSING {joint.get('name')}")
        out[label] = (
            math.degrees(float(limit.get("lower"))),
            math.degrees(float(limit.get("upper"))),
        )
    if tuple(out) != JOINT_ORDER:
        raise RuntimeError(f"URDF_JOINT_ORDER_MISMATCH expected={JOINT_ORDER} actual={tuple(out)}")
    return out


URDF_KINEMATIC_CHAIN_CONTRACT = (
    ("world_to_base_link", "world", "base_link", None),
    ("base_link_to_link1", "base_link", "link1", 0),
    ("link1_to_link2", "link1", "link2", 1),
    ("link2_to_link3", "link2", "link3", 2),
    ("link3_to_link4", "link3", "link4", 3),
    ("link4_to_link5", "link4", "link5", 4),
    ("link5_to_gripper_link", "link5", "gripper_link", 5),
)


def parse_urdf_kinematic_chain() -> list[dict[str, Any]]:
    """Parse the exact decimal URDF chain used by the USD importer.

    The legacy planner's compact ``_CHAIN`` intentionally rounds ``0.051959`` to
    ``0.05196`` and replaces authored ``1.5708`` with mathematical ``pi/2``.  That
    is acceptable for its millimetre IK, but it is not the frame authority for a
    transformed-collider clearance gate.  This parser preserves the frozen URDF
    decimal bytes and verifies every parent/child/axis before returning them.
    """
    if sha256_file(URDF_PATH) != URDF_SHA256:
        raise RuntimeError("URDF_SHA256_MISMATCH")
    root = ET.parse(URDF_PATH).getroot()
    by_name = {joint.get("name", ""): joint for joint in root.iter("joint")}
    rows: list[dict[str, Any]] = []
    for name, expected_parent, expected_child, q_index in URDF_KINEMATIC_CHAIN_CONTRACT:
        joint = by_name.get(name)
        if joint is None:
            raise RuntimeError(f"URDF_KINEMATIC_JOINT_MISSING {name}")
        parent_node = joint.find("parent")
        child_node = joint.find("child")
        origin_node = joint.find("origin")
        axis_node = joint.find("axis")
        if None in (parent_node, child_node, origin_node, axis_node):
            raise RuntimeError(f"URDF_KINEMATIC_SCHEMA_MISSING {name}")
        parent = str(parent_node.get("link"))
        child = str(child_node.get("link"))
        joint_type = str(joint.get("type"))
        expected_type = "fixed" if q_index is None else "revolute"
        xyz = [float(value) for value in str(origin_node.get("xyz")).split()]
        rpy = [float(value) for value in str(origin_node.get("rpy")).split()]
        axis = [float(value) for value in str(axis_node.get("xyz")).split()]
        if (
            parent != expected_parent
            or child != expected_child
            or joint_type != expected_type
            or len(xyz) != 3
            or len(rpy) != 3
            or len(axis) != 3
            or not np.isfinite(np.asarray([*xyz, *rpy, *axis])).all()
            or np.max(np.abs(np.asarray(axis) - [0.0, 0.0, 1.0])) > 1.0e-12
        ):
            raise RuntimeError(
                "URDF_KINEMATIC_CHAIN_DRIFT "
                f"name={name} parent={parent} child={child} type={joint_type} "
                f"xyz={xyz} rpy={rpy} axis={axis}"
            )
        rows.append(
            {
                "name": name,
                "parent": parent,
                "child": child,
                "type": joint_type,
                "q_index": q_index,
                "origin_xyz_m": xyz,
                "origin_rpy_rad": rpy,
                "axis": axis,
            }
        )
    return rows


def _rpy_transform_exact(xyz: Any, rpy: Any) -> np.ndarray:
    roll, pitch, yaw = (float(value) for value in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rotation = np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return transform


def _z_rotation_transform(theta_rad: float) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    cosine, sine = math.cos(float(theta_rad)), math.sin(float(theta_rad))
    transform[0, 0] = cosine
    transform[0, 1] = -sine
    transform[1, 0] = sine
    transform[1, 1] = cosine
    return transform


def _urdf_body_transforms_for_q(
    urdf_chain: list[dict[str, Any]], q_deg: np.ndarray
) -> dict[str, np.ndarray]:
    q_rad = np.radians(np.asarray(q_deg, dtype=np.float64))
    if q_rad.shape != (6,) or not np.isfinite(q_rad).all():
        raise RuntimeError(f"URDF_FK_Q_INVALID {q_deg}")
    transform = np.eye(4, dtype=np.float64)
    bodies: dict[str, np.ndarray] = {}
    for row in urdf_chain:
        transform = transform @ _rpy_transform_exact(
            row["origin_xyz_m"], row["origin_rpy_rad"]
        )
        if row["q_index"] is not None:
            transform = transform @ _z_rotation_transform(q_rad[int(row["q_index"])])
        if row["child"] in MOVING_BODIES:
            bodies[str(row["child"])] = transform.copy()
    if tuple(bodies) != MOVING_BODIES:
        raise RuntimeError(
            f"URDF_FK_BODY_MAP_INCOMPLETE expected={MOVING_BODIES} actual={tuple(bodies)}"
        )
    return bodies


def _vec(row: dict[str, Any], key: str) -> np.ndarray:
    value = np.asarray(row[key], dtype=np.float64)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise RuntimeError(f"candidate vector invalid key={key} value={row.get(key)}")
    return value


def _unit(vector: np.ndarray, label: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm < 1.0e-12:
        raise RuntimeError(f"zero/nonfinite axis {label}")
    return vector / norm


def load_candidates(path: Path, expected_sha256: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"P15_CANDIDATES_MISSING {path}")
    if expected_sha256 != P15_CANDIDATES_SHA256:
        raise RuntimeError(
            "P15_CANDIDATES_LAUNCH_PIN_MISMATCH "
            f"expected={P15_CANDIDATES_SHA256} actual={expected_sha256}"
        )
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha256:
        raise RuntimeError(
            f"P15_CANDIDATES_SHA256_MISMATCH expected={expected_sha256} actual={actual_sha}"
        )
    doc = json.loads(path.read_text())
    if not (
        doc.get("schema") == SCHEMA
        and doc.get("run_label") == P15_RUN_LABEL
        and doc.get("instrumentation_verdict")
        == "CANDIDATE_INSTRUMENTATION_PASS__NO_PHYSICS_OR_GRASP_VERDICT"
        and doc.get("scientific_physics_verdict") is None
    ):
        raise RuntimeError(
            f"P15_SCHEMA_OR_AUTHORITY_MISMATCH schema={doc.get('schema')!r} "
            f"run_label={doc.get('run_label')!r}"
        )
    rows = doc.get("candidates")
    if not isinstance(rows, list) or len(rows) != 8:
        raise RuntimeError(f"P15_CANDIDATE_COUNT_MISMATCH expected=8 actual={len(rows or [])}")
    actual_candidate_ids = tuple(str(row.get("candidate_id", "")) for row in rows)
    if actual_candidate_ids != P15_CANDIDATE_IDS:
        raise RuntimeError(
            "P15_ORDERED_CANDIDATE_IDS_DRIFT "
            f"expected={P15_CANDIDATE_IDS} actual={actual_candidate_ids}"
        )

    object_contract = doc.get("object_contract")
    expected_object_contract = {
        "physics_authority": "analytic_upright_cylinder_in_p16",
        "diameter_m": OBJ_DIAM_M,
        "radius_m": OBJ_RADIUS_M,
        "height_m": OBJ_HEIGHT_M,
        "mass_kg": OBJ_MASS_KG,
        "center_base_m": OBJECT_CENTER_M.tolist(),
        "pose_source": "p10.FOUR_SPONGE_SEED0_SOURCES.seed0_S4 + support_z + H/2",
        "yaw_deg": 0.0,
        "support_z_m": SUPPORT_Z_M,
        "grasp_point_case_exception": (
            "D419 top_center_to_side_midpoint__sim_only_user_approved"
        ),
        "material_friction": "not_sampled_not_measured_not_claimed_by_p15",
    }
    if object_contract != expected_object_contract:
        raise RuntimeError(
            "P15_OBJECT_CONTRACT_MISMATCH "
            f"expected={expected_object_contract} actual={object_contract}"
        )
    raw_contract = doc.get("frame_contract", {}).get("raw_sdg_root_calibration", {})
    if (
        raw_contract.get("gripper_frame_prim") is not None
        or raw_contract.get("T_sdg_gripper_link5") is not None
        or "PROVENANCE_ONLY" not in str(raw_contract.get("status"))
    ):
        raise RuntimeError(f"P15_RAW_ROOT_CONTRACT_INVALID {raw_contract}")

    ids: set[str] = set()
    radial = _unit(OBJECT_CENTER_M * np.asarray([1.0, 1.0, 0.0]), "radial")
    for index, row in enumerate(rows):
        cid = str(row.get("candidate_id", ""))
        if not cid or cid in ids:
            raise RuntimeError(f"P15_CANDIDATE_ID_INVALID index={index} id={cid!r}")
        ids.add(cid)
        if row.get("q5_control", {}).get("authority") != (
            "p16_physics_harness_unassigned_by_p15"
        ):
            raise RuntimeError(f"P15_Q5_AUTHORITY_INVALID candidate={cid}")
        if row["q5_control"].get("open_deg") is not None or row["q5_control"].get("close_deg") is not None:
            raise RuntimeError(f"P15_MUST_NOT_ASSIGN_Q5 candidate={cid}")
        axes = row.get("axes_base", {})
        closure = _unit(_vec(axes, "jaw_closure_x"), f"{cid}.jaw_closure_x")
        vertical = _unit(_vec(axes, "vertical_up_y"), f"{cid}.vertical_up_y")
        approach = _unit(_vec(axes, "tool_approach_z"), f"{cid}.tool_approach_z")
        R = np.column_stack((closure, vertical, approach))
        if np.max(np.abs(R.T @ R - np.eye(3))) > 1.0e-6 or np.linalg.det(R) < 0.999999:
            raise RuntimeError(f"P15_FRAME_NOT_RIGHT_HANDED candidate={cid}")
        proposed = np.asarray(row.get("R_base_link5_proposal"), dtype=np.float64)
        if proposed.shape != (3, 3) or np.max(np.abs(proposed - R)) > 1.0e-12:
            raise RuntimeError(f"P15_AXES_MATRIX_DRIFT candidate={cid}")
        # p15 admits measured SDG residuals; p16 records them rather than
        # silently replacing a proposal with the ideal radial frame.
        if float(approach @ radial) <= 0.0 or float(vertical[2]) <= 0.0:
            raise RuntimeError(f"P15_PROPOSAL_SIGN_INVALID candidate={cid}")
        midpoint = _vec(row, "antipodal_midpoint_base_m")
        midpoint_delta = midpoint - OBJECT_CENTER_M
        filter_contract = doc.get("filter_contract", {})
        if (
            np.linalg.norm(midpoint_delta[:2])
            > float(filter_contract.get("centerline_offset_max_m", -1.0)) + 1.0e-12
            or abs(float(midpoint_delta[2]))
            > float(filter_contract.get("midheight_abs_max_m", -1.0)) + 1.0e-12
            or row.get("filter_pass") is not None
            and not bool(row.get("filter_pass"))
            or not all(bool(value) for value in row.get("filter_checks", {}).values())
        ):
            raise RuntimeError(
                f"P15_ANTIPODAL_MIDPOINT_FILTER_DRIFT candidate={cid} "
                f"delta={midpoint_delta.tolist()}"
            )
        row_raw = row.get("raw_sdg_root_calibration", {})
        if (
            row_raw.get("gripper_frame_prim") is not None
            or row_raw.get("T_sdg_gripper_link5") is not None
            or "DO_NOT_USE_AS_ROARM_POSE" not in str(row_raw.get("status"))
        ):
            raise RuntimeError(f"P15_RAW_ROOT_MUST_NOT_CALIBRATE_ROARM candidate={cid}")
        mapped = row.get("geometry_mapped_roarm_targets", {})
        if mapped.get("status") != (
            "POSITION_MAPPING_DERIVED_FROM_PINNED_ATTEMPT3_GEOMETRY__IK_UNTESTED"
        ):
            raise RuntimeError(f"P15_GEOMETRY_MAPPING_STATUS_INVALID candidate={cid}")
        if np.max(np.abs(np.asarray(mapped.get("tcp_target_orientation_R_base_link5")) - R)) > 1.0e-12:
            raise RuntimeError(f"P15_GEOMETRY_MAPPING_ROTATION_DRIFT candidate={cid}")
    return {"path": str(path.relative_to(REPO)), "sha256": actual_sha, "doc": doc}


def validate_p15_observability(handoff: dict[str, Any]) -> dict[str, Any]:
    """Enforce p15's own no-consumption-before-D341-completion declaration."""
    required = {
        "config.json": P15_CONFIG_PATH,
        "rerun_validation.json": P15_RERUN_VALIDATION_PATH,
        "inspection.png": P15_INSPECTION_PATH,
        "manual_visual_inspection.json": P15_MANUAL_VISUAL_PATH,
        "exit_status.txt": P15_EXIT_STATUS_PATH,
        "stdout.log": P15_STDOUT_PATH,
        "pid.txt": P15_PID_PATH,
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"P15_OBSERVABILITY_ARTIFACT_MISSING {missing}")
    config = json.loads(P15_CONFIG_PATH.read_text())
    validation = json.loads(P15_RERUN_VALIDATION_PATH.read_text())
    manual = json.loads(P15_MANUAL_VISUAL_PATH.read_text())
    candidate_provenance = handoff["doc"].get("provenance", {})
    sampler = handoff["doc"].get("sampler", {})
    counts = handoff["doc"].get("counts", {})
    artifact_hashes = config.get("artifact_sha256", {})
    pinned_hashes = {
        P15_CONFIG_PATH: P15_CONFIG_SHA256,
        P15_RERUN_VALIDATION_PATH: P15_RERUN_VALIDATION_SHA256,
        P15_INSPECTION_PATH: P15_INSPECTION_SHA256,
        P15_MANUAL_VISUAL_PATH: P15_MANUAL_VISUAL_SHA256,
        P15_EXIT_STATUS_PATH: P15_EXIT_STATUS_SHA256,
        P15_STDOUT_PATH: P15_STDOUT_SHA256,
        P15_PID_PATH: P15_PID_SHA256,
    }
    pinned_failures = {
        str(path): {"expected": expected, "actual": sha256_file(path)}
        for path, expected in pinned_hashes.items()
        if sha256_file(path) != expected
    }
    if pinned_failures:
        raise RuntimeError(f"P15_PINNED_ARTIFACT_HASH_MISMATCH {pinned_failures}")
    absent_outputs = [
        str(path) for path in P15_BOUND_OUTPUT_PATHS.values() if not path.is_file()
    ]
    if absent_outputs:
        raise RuntimeError(f"P15_OUTPUT_SET_INCOMPLETE {absent_outputs}")
    hash_failures = {
        name: {"expected": artifact_hashes.get(name), "actual": sha256_file(path)}
        for name, path in P15_BOUND_OUTPUT_PATHS.items()
        if artifact_hashes.get(name) != sha256_file(path)
    }
    if hash_failures:
        raise RuntimeError(f"P15_CONFIG_ARTIFACT_HASH_MISMATCH {hash_failures}")
    if not (
        config.get("schema") == "g0b.t3s.side_sdg_run.v1"
        and config.get("run_label") == P15_RUN_LABEL
        and config.get("prefix") == P15_PREFIX
        and config.get("run_valid") is True
        and config.get("instrumentation_verdict")
        == "CANDIDATE_INSTRUMENTATION_PASS__NO_PHYSICS_OR_GRASP_VERDICT"
        and config.get("physics_steps") == 0
        and config.get("simulation_context_created") is False
        and config.get("grasping_manager_evaluate_calls") == 0
        and config.get("render_products") == 0
        and validation.get("pass") is True
        and candidate_provenance.get("prereg_sha256") == P15_PREREG_SHA256
        and candidate_provenance.get("executed_source_sha256") == P15_SHA256
        and candidate_provenance.get("executed_source_stable") is True
        and candidate_provenance.get("frozen_source_sha256") == P15_SHA256
        and artifact_hashes.get("script.py.txt") == P15_SHA256
        and artifact_hashes.get("candidates.json") == P15_CANDIDATES_SHA256
        and sampler.get("determinism_bit_identical") is True
        and counts.get("canonical_candidate_count") == 8
        and counts.get("configured_candidate_attempt_count") == 65536
        and counts.get("configured_surface_sample_count") == 4096
        and counts.get("raw_transform_count") == 51760
        and counts.get("filter_pass_count") == 20
        and counts.get("duplicate_raw_transform_count") == 0
        and P15_EXIT_STATUS_PATH.read_text().strip() == "0"
        and not P15_FAILURE_PATH.exists()
        and handoff["doc"].get(
            "p16_consumption_allowed_only_if_rerun_validation_pass_and_manual_inspection"
        )
        is True
    ):
        raise RuntimeError("P15_INSTRUMENTATION_OR_RERUN_CONTRACT_FAIL")
    expected_manual_checks = {
        "candidate_midpoints_at_side_midheight": True,
        "jaw_closure_x_tangential_horizontal": True,
        "vertical_up_y_world_up": True,
        "tool_approach_z_radial_outward": True,
    }
    observations = manual.get("observations")
    if not (
        manual.get("artifact") == "T3S_SIDE_SDG_MANUAL_VISUAL_INSPECTION_V1"
        and manual.get("pass") is True
        and manual.get("candidates_sha256") == handoff["sha256"]
        and manual.get("inspection_png_sha256") == sha256_file(P15_INSPECTION_PATH)
        and manual.get("rerun_validation_sha256")
        == sha256_file(P15_RERUN_VALIDATION_PATH)
        and manual.get("frame_checks") == expected_manual_checks
        and isinstance(observations, list)
        and len(observations) >= 1
        and all(isinstance(item, str) and item.strip() for item in observations)
    ):
        raise RuntimeError(f"P15_MANUAL_VISUAL_INSPECTION_CONTRACT_FAIL {manual}")
    return {
        "pass": True,
        "config_sha256": sha256_file(P15_CONFIG_PATH),
        "rerun_validation_sha256": sha256_file(P15_RERUN_VALIDATION_PATH),
        "inspection_png_sha256": sha256_file(P15_INSPECTION_PATH),
        "manual_visual_inspection_sha256": sha256_file(P15_MANUAL_VISUAL_PATH),
        "exit_status_sha256": sha256_file(P15_EXIT_STATUS_PATH),
        "stdout_sha256": sha256_file(P15_STDOUT_PATH),
        "pid_record_sha256": sha256_file(P15_PID_PATH),
        "failure_marker_absent": True,
        "frame_checks": expected_manual_checks,
    }


def validate_witness_source() -> dict[str, Any]:
    results = json.loads(WITNESS_RESULTS_PATH.read_text())
    plan = json.loads(WITNESS_PLAN_PATH.read_text())
    classification = next(
        (row for row in results.get("population_classifications", [])
         if row.get("trial_id") == "trial_005948"),
        None,
    )
    trial = next(
        (row for row in plan.get("trials", []) if row.get("trial_id") == "trial_005948"),
        None,
    )
    if classification != {
        "trial_id": "trial_005948",
        "mechanism": "JAW_SUPPORT_CONTACT_FAIL",
        "reason_flags": ["JAW_SUPPORT_CONTACT_OBSERVED_GT_0P02N"],
    }:
        raise RuntimeError(f"WITNESS_CLASSIFICATION_DRIFT {classification}")
    expected = {
        "pose_key": "seed0_S4",
        "theta_target_deg": 69.0,
        "q5_close_deg": 66.4,
    }
    if trial is None or any(trial.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"WITNESS_PLAN_IDENTITY_DRIFT {trial}")
    controls = (
        ("q_approach_deg", WITNESS_Q_APPROACH_DEG),
        ("q_descend_deg", WITNESS_Q_DESCEND_DEG),
        ("q_lift_deg", WITNESS_Q_LIFT_DEG),
    )
    for key, expected_control in controls:
        if np.max(np.abs(np.asarray(trial[key], dtype=np.float64) - expected_control)) > 1.0e-12:
            raise RuntimeError(f"WITNESS_CONTROL_DRIFT key={key}")
    return {
        "source_trial_id": WITNESS_SOURCE_TRIAL_ID,
        "results_sha256": WITNESS_RESULTS_SHA256,
        "plan_sha256": WITNESS_PLAN_SHA256,
        "source_mechanism": classification["mechanism"],
        "source_reason_flags": classification["reason_flags"],
        "replay_scope": "instrumentation_only_excluded_from_scientific_counts",
    }


def derive_pinch_calibration(jaw: Any, asset: dict[str, Any] | None = None) -> dict[str, Any]:
    """Derive the asymmetric pinch centre from actual 64+64 hull surfaces.

    The fixed RoArm frame is link5 +X=tangential closure, +Y=world up and
    +Z=radial approach.  Thus the upright finite cylinder axis is link5 +Y and
    its circular section lies in link5 XZ.  A corridor-width midpoint is wrong
    for this asymmetric fixed-plus-swinging-jaw mechanism: the fixed inward
    surface is made tangent first, then moving-jaw first contact is measured.
    """

    asset = jaw.extract_asset() if asset is None else asset
    for body in JAW_BODIES:
        parts = asset["bodies"][body]["parts"]
        if len(parts) != 64 or not all(bool(part["hull_ok"]) for part in parts):
            raise RuntimeError(f"PINCH_CALIBRATION_REQUIRES_64_HULLS body={body}")
    fixed, _ = jaw.concat_parts(asset["bodies"]["link5"]["parts"])
    moving_base, _ = jaw.concat_parts(asset["bodies"]["gripper_link"]["parts"])
    tcp_local = np.asarray(jaw.TCP_LOCAL, dtype=np.float64)
    if np.max(np.abs(tcp_local - [0.0, 0.0, 0.115428])) > 1.0e-12:
        raise RuntimeError(f"TCP_LOCAL_DRIFT {tcp_local}")

    def slab(points: np.ndarray) -> np.ndarray:
        return (
            (np.abs(points[:, 1] - tcp_local[1]) <= OBJ_HEIGHT_M / 2.0)
            & (points[:, 2] >= tcp_local[2] - OBJ_RADIUS_M)
            & (points[:, 2] <= tcp_local[2] + OBJ_RADIUS_M)
        )

    fixed_slab = fixed[slab(fixed)]
    if not len(fixed_slab):
        raise RuntimeError("PINCH_FIXED_SLAB_EMPTY")
    # Select the inward fixed surface at TCP depth.  The surface sampling pitch
    # supplies the explicit depth band and error bound; max X is the gap-facing
    # side because the moving jaw approaches from +X toward -X.
    at_tcp_depth = fixed_slab[
        np.abs(fixed_slab[:, 2] - tcp_local[2]) <= float(jaw.SAMPLE_SPACING_M)
    ]
    if not len(at_tcp_depth):
        raise RuntimeError("PINCH_FIXED_TCP_DEPTH_BAND_EMPTY")
    fixed_inner_x = float(at_tcp_depth[:, 0].max())
    nominal = fixed_inner_x + OBJ_RADIUS_M
    fixed_rho = np.hypot(
        fixed_slab[:, 0] - nominal, fixed_slab[:, 2] - tcp_local[2]
    )
    fixed_contact_index = int(np.argmin(fixed_rho))
    fixed_contact_residual = abs(float(fixed_rho[fixed_contact_index]) - OBJ_RADIUS_M)
    rows: list[dict[str, Any]] = []
    for q5 in np.arange(0.0, Q5_OPEN_DEG + 0.5 * PINCH_Q_STEP_DEG, PINCH_Q_STEP_DEG):
        moving = jaw.transform_pts(jaw.gripper_T_l5(asset["joint"], float(q5)), moving_base)
        moving_slab = moving[slab(moving)]
        if not len(moving_slab):
            continue
        moving_rho = np.hypot(
            moving_slab[:, 0] - nominal, moving_slab[:, 2] - tcp_local[2]
        )
        moving_contact_index = int(np.argmin(moving_rho))
        rho_min = float(moving_rho[moving_contact_index])
        rows.append(
            {
                "q5_deg": float(q5),
                "fixed_inner_x_m": fixed_inner_x,
                "pinch_center_link5_x_m": nominal,
                "moving_min_radius_m": rho_min,
                "moving_contact_residual_m": abs(rho_min - OBJ_RADIUS_M),
                "moving_contact_point_link5_m": moving_slab[moving_contact_index].tolist(),
                "n_fixed_samples": int(len(fixed_slab)),
                "n_moving_samples": int(len(moving_slab)),
            }
        )
    if not rows:
        raise RuntimeError("PINCH_Q5_SCAN_EMPTY")
    best = min(rows, key=lambda row: (row["moving_contact_residual_m"], row["q5_deg"]))
    if (
        fixed_contact_residual > PINCH_CONTACT_RESIDUAL_GATE_M
        or best["moving_contact_residual_m"] > PINCH_CONTACT_RESIDUAL_GATE_M
        or best["q5_deg"] <= 0.0
        or best["q5_deg"] >= Q5_OPEN_DEG
        or abs(fixed_inner_x - (-0.01002584956586361)) > 1.0e-12
        or abs(nominal - 0.00447415043413639) > 1.0e-12
        or abs(float(best["q5_deg"]) - 22.840) > 0.476
        or Q5_CLOSE_COMMAND_DEG >= float(best["q5_deg"])
    ):
        raise RuntimeError(
            f"PINCH_CALIBRATION_GATE_FAIL fixed_residual={fixed_contact_residual} best={best}"
        )
    offsets = nominal + PINCH_OFFSET_DELTAS_M
    return {
        "method": "attempt3_64_plus_64_fixed_tangent_then_moving_radial_first_contact",
        "frame": "link5; closure=+X_tangent; cylinder_axis=+Y_world_up; approach=+Z_radial",
        "q5_grid_step_deg": PINCH_Q_STEP_DEG,
        "hull_surface_sample_spacing_m": float(jaw.SAMPLE_SPACING_M),
        "contact_residual_gate_m": PINCH_CONTACT_RESIDUAL_GATE_M,
        "fixed_inner_x_m": fixed_inner_x,
        "fixed_contact_point_link5_m": fixed_slab[fixed_contact_index].tolist(),
        "fixed_contact_residual_m": fixed_contact_residual,
        "best": best,
        "nominal_offset_link5_x_m": nominal,
        "frozen_offset_deltas_m": PINCH_OFFSET_DELTAS_M.tolist(),
        "frozen_offsets_link5_x_m": offsets.tolist(),
        "n_q5_rows_with_both_slabs": len(rows),
        "drift_witness": {
            "expected_fixed_inner_x_mm": -10.025849566,
            "expected_q5_first_contact_deg": 22.840,
            "expected_q5_contact_uncertainty_deg": 0.476,
            "expected_pinch_center_mm": 4.474150434,
        },
        "asset_body_part_counts": {
            body: len(asset["bodies"][body]["parts"]) for body in JAW_BODIES
        },
    }


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.clip(_unit(a, "a") @ _unit(b, "b"), -1.0, 1.0))))


def _within_limits(q: np.ndarray, limits: dict[str, tuple[float, float]]) -> bool:
    return all(limits[name][0] - 1.0e-9 <= float(q[i]) <= limits[name][1] + 1.0e-9
               for i, name in enumerate(JOINT_ORDER))


def build_plan(
    p10: Any,
    handoff: dict[str, Any],
    calibration: dict[str, Any],
    limits: dict[str, tuple[float, float]],
    profile: str,
) -> dict[str, Any]:
    rows = handoff["doc"]["candidates"]
    selected = (
        [(PREFLIGHT_CANDIDATE_INDEX, rows[PREFLIGHT_CANDIDATE_INDEX])]
        if profile == PREFLIGHT_PROFILE
        else list(enumerate(rows))
    )
    p10.V6_LIMITS_DEG = {name: limits[name] for name in JOINT_ORDER[:4]}

    class PlanArgs:
        target_error_gate_m = 0.003
        # Solver-internal admission only.  The physical final frame uses the
        # stricter explicit 2-degree closure-frame and hull-clearance gates.
        plan_tilt_gate_deg = 5.0

    out: list[dict[str, Any]] = []
    for candidate_index, candidate in selected:
        if (
            profile == PREFLIGHT_PROFILE
            and candidate.get("candidate_id") != PREFLIGHT_CANDIDATE_ID
        ):
            raise RuntimeError(
                "PREFLIGHT_CANDIDATE_ID_DRIFT "
                f"index={candidate_index} actual={candidate.get('candidate_id')!r}"
            )
        axes = candidate["axes_base"]
        closure_x = _unit(_vec(axes, "jaw_closure_x"), "jaw_closure_x")
        vertical_y = _unit(_vec(axes, "vertical_up_y"), "vertical_up_y")
        approach_z = _unit(_vec(axes, "tool_approach_z"), "tool_approach_z")
        # Fixed, declared adapter after the independent 64+64 audit:
        # link5 +X=tangential closure, +Y=world up, +Z=radial approach.
        R_link5_target = np.column_stack((closure_x, vertical_y, approach_z))
        if np.max(np.abs(R_link5_target.T @ R_link5_target - np.eye(3))) > 1.0e-6:
            raise RuntimeError(f"LINK5_FRAME_MAP_INVALID candidate={candidate['candidate_id']}")
        if np.linalg.det(R_link5_target) < 0.999999:
            raise RuntimeError(f"LINK5_FRAME_LEFT_HANDED candidate={candidate['candidate_id']}")
        theta = _angle_deg(approach_z, np.asarray([0.0, 0.0, -1.0]))
        psi = math.degrees(math.atan2(float(approach_z[1]), float(approach_z[0]))) % 360.0
        world_down = np.asarray([0.0, 0.0, -1.0])
        phi = math.degrees(
            math.atan2(
                float(world_down @ R_link5_target[:, 1]),
                float(world_down @ R_link5_target[:, 0]),
            )
        ) % 360.0
        p10.set_target_axis(theta, psi)
        p10.PHI_STAR_DEG = phi
        midpoint = _vec(candidate, "antipodal_midpoint_base_m")
        mapped = candidate["geometry_mapped_roarm_targets"]
        mapped_offset = _vec(mapped, "midpoint_from_tcp_link5_m")
        nominal_offset = float(calibration["nominal_offset_link5_x_m"])
        if np.max(np.abs(mapped_offset - [nominal_offset, 0.0, 0.0])) > 1.0e-12:
            raise RuntimeError(
                f"P15_P16_PINCH_OFFSET_DRIFT candidate={candidate['candidate_id']} "
                f"p15={mapped_offset.tolist()} p16={nominal_offset}"
            )
        tcp_nominal_handoff = _vec(mapped, "tcp_target_base_m")
        tcp_nominal_rederived = midpoint - R_link5_target @ mapped_offset
        if np.max(np.abs(tcp_nominal_handoff - tcp_nominal_rederived)) > 1.0e-12:
            raise RuntimeError(
                f"P15_GEOMETRY_MAPPED_TCP_DRIFT candidate={candidate['candidate_id']}"
            )
        link5_tcp = mapped.get("T_link5_tcp", {})
        if (
            np.max(np.abs(np.asarray(link5_tcp.get("rotation")) - np.eye(3))) > 1.0e-12
            or np.max(
                np.abs(
                    np.asarray(link5_tcp.get("translation_link5_m"), dtype=np.float64)
                    - [0.0, 0.0, 0.115428]
                )
            )
            > 1.0e-12
        ):
            raise RuntimeError(f"P15_LINK5_TCP_CONTRACT_DRIFT candidate={candidate['candidate_id']}")
        for offset_index, offset_x in enumerate(calibration["frozen_offsets_link5_x_m"]):
            tcp_grasp = midpoint - R_link5_target @ np.asarray([offset_x, 0.0, 0.0])
            if offset_index == 0 and np.max(np.abs(tcp_grasp - tcp_nominal_handoff)) > 1.0e-12:
                raise RuntimeError(
                    f"P15_NOMINAL_TCP_NOT_CONSUMED candidate={candidate['candidate_id']}"
                )
            tcp_elevated = (
                tcp_grasp - APPROACH_CLEARANCE_M * approach_z
                + np.asarray([0.0, 0.0, ELEVATED_PREGRASP_Z_M])
            )
            tcp_stage = (
                tcp_grasp - NEAR_STAGE_BACKOFF_M * approach_z
                + np.asarray([0.0, 0.0, NEAR_STAGE_Z_M])
            )
            tcp_lift = tcp_grasp + np.asarray([0.0, 0.0, LIFT_DELTA_M])
            q_elevated, ok_elevated, err_elevated, tilt_elevated = p10._solve_q_vertical(
                tcp_elevated, HOME_DEG, Q5_OPEN_DEG, PlanArgs(),
                multi_seed_xy=(float(midpoint[0]), float(midpoint[1]))
            )
            q_stage, ok_stage, err_stage, tilt_stage = p10._solve_q_vertical(
                tcp_stage, q_elevated, Q5_OPEN_DEG, PlanArgs()
            )
            q_grasp, ok_grasp, err_grasp, tilt_grasp = p10._solve_q_vertical(
                tcp_grasp, q_stage, Q5_OPEN_DEG, PlanArgs()
            )
            q_lift, ok_lift, err_lift, tilt_lift = p10._solve_q_vertical(
                tcp_lift, q_grasp, Q5_CLOSE_COMMAND_DEG, PlanArgs()
            )
            q_close = q_grasp.copy()
            q_close[5] = Q5_CLOSE_COMMAND_DEG
            phase_q = {
                "home": HOME_DEG.copy(),
                "elevated_pregrasp": q_elevated,
                "near_side_stage": q_stage,
                "grasp_open": q_grasp,
                "grasp_closed_command": q_close,
                "lift": q_lift,
            }
            actual_frames: dict[str, Any] = {}
            for phase_name, q_now in phase_q.items():
                tcp_now, link5_now = p10.fk_full_5(q_now[:5])
                R_now = link5_now[:3, :3]
                actual_frames[phase_name] = {
                    "tcp_m": tcp_now[:3, 3].tolist(),
                    "axis_error_deg": {
                        axis: _angle_deg(R_now[:, i], R_link5_target[:, i])
                        for i, axis in enumerate(("closure_x", "vertical_up_y", "approach_z"))
                    },
                    "signed_adverse_pitch_deg": math.degrees(
                        math.asin(float(np.clip(R_now[2, 2], -1.0, 1.0)))
                    ),
                }
            final_frame_max_error = max(
                actual_frames["grasp_open"]["axis_error_deg"].values()
            )
            primary_limits_ok = all(_within_limits(q, limits) for q in phase_q.values())
            v6_limits = p10.JOINT_LIMITS_DEG
            v6_ok = all(
                all(v6_limits[name][0] - 1e-9 <= float(q[i]) <= v6_limits[name][1] + 1e-9
                    for i, name in enumerate(JOINT_ORDER))
                for q in phase_q.values()
            )
            feasible = bool(
                ok_elevated and ok_stage and ok_grasp and ok_lift
                and primary_limits_ok
                and final_frame_max_error <= FINAL_CLOSURE_FRAME_ERROR_GATE_DEG
            )
            out.append(
                {
                    "trial_id": f"c{candidate_index:02d}_o{offset_index:02d}",
                    "candidate_index": candidate_index,
                    "candidate_id": candidate["candidate_id"],
                    "source_raw_index": candidate["source_raw_index"],
                    "pinch_offset_index": offset_index,
                    "pinch_offset_delta_m": float(PINCH_OFFSET_DELTAS_M[offset_index]),
                    "pinch_center_link5_x_m": float(offset_x),
                    "center_m": OBJECT_CENTER_M.tolist(),
                    "side_surface_midpoint_base_m": candidate[
                        "d419_side_surface_midpoint_base_m"
                    ],
                    "antipodal_midpoint_base_m": midpoint.tolist(),
                    "candidate_axes_base": axes,
                    "p15_geometry_mapped_nominal_tcp_m": tcp_nominal_handoff.tolist(),
                    "p15_geometry_mapping_consumed": True,
                    "R_link5_target": R_link5_target.tolist(),
                    "theta_target_deg": theta,
                    "psi_axis_target_deg": psi,
                    "phi_target_deg": phi,
                    "tcp_elevated_pregrasp_m": tcp_elevated.tolist(),
                    "tcp_near_side_stage_m": tcp_stage.tolist(),
                    "tcp_grasp_m": tcp_grasp.tolist(),
                    "tcp_lift_m": tcp_lift.tolist(),
                    "q_home_deg": HOME_DEG.tolist(),
                    "q_elevated_pregrasp_deg": q_elevated.tolist(),
                    "q_near_side_stage_deg": q_stage.tolist(),
                    "q_grasp_open_deg": q_grasp.tolist(),
                    "q_grasp_closed_command_deg": q_close.tolist(),
                    "q_lift_deg": q_lift.tolist(),
                    "ik_ok": {"elevated_pregrasp": bool(ok_elevated),
                              "near_side_stage": bool(ok_stage),
                              "grasp": bool(ok_grasp), "lift": bool(ok_lift)},
                    "ik_position_error_mm": [float(err_elevated), float(err_stage),
                                             float(err_grasp), float(err_lift)],
                    "ik_axis_error_deg": [float(tilt_elevated), float(tilt_stage),
                                          float(tilt_grasp), float(tilt_lift)],
                    "actual_link5_frames": actual_frames,
                    "final_frame_max_axis_error_deg": final_frame_max_error,
                    "tcp_fk_residual_mm": float(
                        np.linalg.norm(
                            np.asarray(actual_frames["grasp_open"]["tcp_m"]) - tcp_grasp
                        ) * 1000.0
                    ),
                    "primary_urdf_limits_ok": primary_limits_ok,
                    "v6_distribution_compatible": v6_ok,
                    "feasible": feasible,
                    "reason": "" if feasible else "IK_LIMIT_OR_FINAL_FRAME_GATE_FAIL",
                }
            )
    plan = {
        "schema": "g0b.t3u.side_physics_plan.v1",
        "profile": profile,
        "candidate_count_input": len(rows),
        "candidate_count_selected": len(selected),
        "offsets_per_candidate": len(PINCH_OFFSET_DELTAS_M),
        "n_planned": len(out),
        "n_feasible": sum(bool(row["feasible"]) for row in out),
        "parsed_urdf_limits_deg": {name: list(limits[name]) for name in JOINT_ORDER},
        "pinch_calibration": calibration,
        "trials": out,
    }
    plan["trial_set_contracts"] = {
        "ik_frame": validate_trial_set_contract(plan, profile, "ik_frame", hard_fail=True)
    }
    return plan


def _trial_key(row: dict[str, Any]) -> list[Any]:
    return [
        int(row["candidate_index"]),
        str(row["candidate_id"]),
        int(row["pinch_offset_index"]),
    ]


def validate_trial_set_contract(
    plan: dict[str, Any],
    profile: str,
    stage: str,
    *,
    hard_fail: bool,
) -> dict[str, Any]:
    """Pin exact ordered planned/active rows before any physical outcome."""
    if profile == PREFLIGHT_PROFILE:
        expected_planned = [
            [PREFLIGHT_CANDIDATE_INDEX, PREFLIGHT_CANDIDATE_ID, offset]
            for offset in range(len(PINCH_OFFSET_DELTAS_M))
        ]
        expected_active = list(expected_planned)
    elif profile == CANONICAL_PROFILE:
        expected_planned = [
            [candidate_index, P15_CANDIDATE_IDS[candidate_index], offset]
            for candidate_index in range(len(P15_CANDIDATE_IDS))
            for offset in range(len(PINCH_OFFSET_DELTAS_M))
        ]
        expected_active = [
            [candidate_index, candidate_id, offset]
            for candidate_index, candidate_id in CANONICAL_STATIC_FEASIBLE_CANDIDATES
            for offset in range(len(PINCH_OFFSET_DELTAS_M))
        ]
    else:
        raise RuntimeError(f"TRIAL_SET_PROFILE_INVALID {profile}")
    actual_planned = [_trial_key(row) for row in plan["trials"]]
    actual_active = [_trial_key(row) for row in plan["trials"] if row["feasible"]]
    report = {
        "stage": stage,
        "profile": profile,
        "expected_planned_count": len(expected_planned),
        "actual_planned_count": len(actual_planned),
        "expected_active_count": len(expected_active),
        "actual_active_count": len(actual_active),
        "expected_planned_ordered": expected_planned,
        "actual_planned_ordered": actual_planned,
        "expected_active_ordered": expected_active,
        "actual_active_ordered": actual_active,
        "pass": actual_planned == expected_planned and actual_active == expected_active,
    }
    if hard_fail and not report["pass"]:
        raise RuntimeError(f"EXACT_TRIAL_SET_CONTRACT_FAIL {report}")
    return report


def _find_ground_collider(stage: Any) -> str:
    from pxr import Usd, UsdPhysics

    root = stage.GetPrimAtPath("/World/ground")
    if not root.IsValid():
        raise RuntimeError("GROUND_ROOT_MISSING")
    hits = [prim.GetPath().pathString for prim in Usd.PrimRange(root)
            if prim.HasAPI(UsdPhysics.CollisionAPI)]
    if not hits:
        raise RuntimeError("GROUND_COLLIDER_MISSING")
    return sorted(hits, key=lambda path: (-path.count("/"), path))[0]


def make_env(args: argparse.Namespace, p10: Any) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import ContactSensor, ContactSensorCfg
    from pxr import PhysxSchema
    from roarm_rl.roarm_stack_env import RoArmStackEnv, RoArmStackEnvCfg

    cfg = RoArmStackEnvCfg()
    if str(cfg.robot.spawn.usd_path) != str(p10.ATTEMPT3_USD):
        raise RuntimeError(f"ATTEMPT3_EFFECTIVE_PATH_DRIFT {cfg.robot.spawn.usd_path}")
    cfg.scene.num_envs = args.num_envs
    cfg.scene.replicate_physics = True
    cfg.scene.filter_collisions = True
    cfg.scene.clone_in_fabric = False
    cfg.decimation = 1
    cfg.sim.render_interval = 1
    cfg.episode_length_s = 120.0
    cfg.robot.spawn.articulation_props.enabled_self_collisions = True
    cfg.sim.physx = sim_utils.PhysxCfg(
        gpu_found_lost_pairs_capacity=2**23,
        gpu_total_aggregate_pairs_capacity=2**23,
        gpu_collision_stack_size=2**28,
        gpu_max_rigid_contact_count=2**23,
        solve_articulation_contact_last=False,
    )
    cfg.reward_phase = 6
    cfg.curriculum_pregrasp = False
    cfg.curriculum_pregrasp_hover = False
    cfg.curriculum_attached_transport_release = False
    cfg.curriculum_post_grasp_cap = False
    cfg.curriculum_disable_nearzone_cap = False
    cfg.curriculum_spawn_min_r = 0.0
    cfg.curriculum_spawn_max_r = 0.0
    cfg.sponge.spawn = sim_utils.CylinderCfg(
        radius=OBJ_RADIUS_M,
        height=OBJ_HEIGHT_M,
        axis="Z",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
            max_angular_velocity=10.0,
            max_linear_velocity=10.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=OBJ_MASS_KG),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=STATIC_FRICTION,
            dynamic_friction=DYNAMIC_FRICTION,
            restitution=RESTITUTION,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.80, 0.62, 0.38)),
    )
    cfg.sponge.init_state.pos = tuple(OBJECT_CENTER_M)
    cfg.sponge.init_state.rot = (1.0, 0.0, 0.0, 0.0)

    spawn_reports: dict[str, Any] = {"object": [], "robot": []}
    object_spawn = cfg.sponge.spawn.func

    def spawn_object(prim_path, spawn_cfg, translation=None, orientation=None, **kwargs):
        prim = object_spawn(prim_path, spawn_cfg, translation=translation,
                            orientation=orientation, **kwargs)
        sim_utils.activate_contact_sensors(prim.GetPath().pathString, threshold=0.0)
        api = PhysxSchema.PhysxContactReportAPI.Get(prim.GetStage(), prim.GetPath())
        threshold = api.GetThresholdAttr().Get()
        spawn_reports["object"].append({"path": prim.GetPath().pathString, "threshold": threshold})
        return prim

    cfg.sponge.spawn.func = spawn_object
    cfg.robot.spawn.activate_contact_sensors = False
    robot_spawn = cfg.robot.spawn.func

    def spawn_robot(prim_path, spawn_cfg, translation=None, orientation=None, **kwargs):
        prim = robot_spawn(prim_path, spawn_cfg, translation=translation,
                           orientation=orientation, **kwargs)
        stage = prim.GetStage()
        for body in CONTACT_REPORT_BODIES:
            path = f"{prim.GetPath().pathString}/{body}"
            body_prim = stage.GetPrimAtPath(path)
            if not body_prim.IsValid():
                raise RuntimeError(f"CONTACT_REPORT_BODY_MISSING {path}")
            sim_utils.activate_contact_sensors(path, threshold=0.0, stage=stage)
            api = PhysxSchema.PhysxContactReportAPI.Get(stage, body_prim.GetPath())
            threshold = api.GetThresholdAttr().Get()
            spawn_reports["robot"].append({"path": path, "body": body, "threshold": threshold})
        return prim

    cfg.robot.spawn.func = spawn_robot

    class P16SideEnv(RoArmStackEnv):
        def _setup_scene(self) -> None:
            super()._setup_scene()
            ground = _find_ground_collider(self.scene.stage)
            body_filters = [f"/World/envs/env_.*/Robot/{body}" for body in MOVING_BODIES]
            self._t3u_ground_path = ground
            self._t3u_object_sensor = ContactSensor(ContactSensorCfg(
                prim_path="/World/envs/env_.*/Sponge",
                filter_prim_paths_expr=[ground, *body_filters],
                update_period=0.0,
                history_length=1,
                track_pose=False,
                track_contact_points=True,
                max_contact_data_count_per_prim=args.contact_capacity,
                force_threshold=0.0,
                debug_vis=False,
            ))
            self.scene.sensors["t3u_object_contact"] = self._t3u_object_sensor
            self._t3u_support_sensors = {}
            for body in MOVING_BODIES:
                sensor = ContactSensor(ContactSensorCfg(
                    prim_path=f"/World/envs/env_.*/Robot/{body}",
                    filter_prim_paths_expr=[ground],
                    update_period=0.0,
                    history_length=1,
                    track_pose=False,
                    track_contact_points=False,
                    max_contact_data_count_per_prim=args.contact_capacity,
                    force_threshold=0.0,
                    debug_vis=False,
                ))
                self.scene.sensors[f"t3u_{body}_support"] = sensor
                self._t3u_support_sensors[body] = sensor
            self._t3u_self_sensors = {}
            for body_a, body_b in SELF_PAIRS:
                sensor = ContactSensor(ContactSensorCfg(
                    prim_path=f"/World/envs/env_.*/Robot/{body_a}",
                    filter_prim_paths_expr=[f"/World/envs/env_.*/Robot/{body_b}"],
                    update_period=0.0,
                    history_length=1,
                    track_pose=False,
                    track_contact_points=False,
                    max_contact_data_count_per_prim=args.contact_capacity,
                    force_threshold=0.0,
                    debug_vis=False,
                ))
                key = f"{body_a}__{body_b}"
                self.scene.sensors[f"t3u_self_{key}"] = sensor
                self._t3u_self_sensors[(body_a, body_b)] = sensor

        def _apply_action(self) -> None:
            self._robot.set_joint_position_target(self.robot_dof_targets)

        def _get_rewards(self):
            import torch
            return torch.zeros(self.num_envs, device=self.device)

        def _get_dones(self):
            import torch
            zeros = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            return zeros, zeros.clone()

    env = P16SideEnv(cfg=cfg)
    env._t3u_spawn_reports = spawn_reports
    return env


def audit_object_stage_readback(env: Any, num_envs: int) -> dict[str, Any]:
    """Read the composed USD object/material contract; never echo spawn cfg."""
    from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade

    stage = env.scene.stage
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    kilograms_per_unit = float(UsdPhysics.GetStageKilogramsPerUnit(stage))
    tolerances = {
        "stage_unit_abs": 1.0e-12,
        "linear_si_m_abs": 1.0e-9,
        "mass_si_kg_abs": 5.0e-9,
        "material_abs": 1.0e-6,
    }
    stage_units_pass = bool(
        math.isclose(meters_per_unit, 1.0, rel_tol=0.0,
                     abs_tol=tolerances["stage_unit_abs"])
        and math.isclose(kilograms_per_unit, 1.0, rel_tol=0.0,
                         abs_tol=tolerances["stage_unit_abs"])
    )
    if not stage_units_pass:
        raise RuntimeError(
            "OBJECT_STAGE_SI_UNIT_GATE_FAIL "
            f"meters_per_unit={meters_per_unit} kilograms_per_unit={kilograms_per_unit}"
        )
    rows: list[dict[str, Any]] = []
    for env_index in range(num_envs):
        root_path = f"/World/envs/env_{env_index}/Sponge"
        cylinder_path = f"{root_path}/geometry/mesh"
        expected_material_path = f"{root_path}/geometry/material"
        root = stage.GetPrimAtPath(root_path)
        cylinder_prim = stage.GetPrimAtPath(cylinder_path)
        if not root.IsValid() or not cylinder_prim.IsValid():
            raise RuntimeError(
                f"OBJECT_STAGE_PRIM_MISSING root={root_path} cylinder={cylinder_path}"
            )
        cylinder = UsdGeom.Cylinder(cylinder_prim)
        mass_api = UsdPhysics.MassAPI(root)
        rigid_api = UsdPhysics.RigidBodyAPI(root)
        physx_rigid_api = PhysxSchema.PhysxRigidBodyAPI(root)
        collision_api = UsdPhysics.CollisionAPI(cylinder_prim)
        mass_api_present = root.HasAPI(UsdPhysics.MassAPI)
        rigid_api_present = root.HasAPI(UsdPhysics.RigidBodyAPI)
        physx_rigid_api_present = root.HasAPI(PhysxSchema.PhysxRigidBodyAPI)
        collision_api_present = cylinder_prim.HasAPI(UsdPhysics.CollisionAPI)
        if not (
            cylinder
            and mass_api_present
            and rigid_api_present
            and physx_rigid_api_present
            and collision_api_present
        ):
            raise RuntimeError(
                "OBJECT_STAGE_SCHEMA_MISSING "
                f"root={root_path} cylinder={cylinder_path} "
                f"types=({root.GetTypeName()},{cylinder_prim.GetTypeName()})"
            )
        binding_api = UsdShade.MaterialBindingAPI(cylinder_prim)
        bound_material, bound_relationship = binding_api.ComputeBoundMaterial(
            materialPurpose="physics"
        )
        material_prim = bound_material.GetPrim() if bound_material else None
        material_path = (
            material_prim.GetPath().pathString
            if material_prim is not None and material_prim.IsValid()
            else ""
        )
        material_api = (
            UsdPhysics.MaterialAPI(material_prim)
            if material_prim is not None and material_prim.IsValid()
            else None
        )
        physx_material_api = (
            PhysxSchema.PhysxMaterialAPI(material_prim)
            if material_prim is not None and material_prim.IsValid()
            else None
        )
        material_api_present = bool(
            material_prim is not None
            and material_prim.IsValid()
            and material_prim.HasAPI(UsdPhysics.MaterialAPI)
        )
        physx_material_api_present = bool(
            material_prim is not None
            and material_prim.IsValid()
            and material_prim.HasAPI(PhysxSchema.PhysxMaterialAPI)
        )
        if (
            not material_path
            or material_prim is None
            or not material_prim.IsValid()
            or not material_api_present
            or not physx_material_api_present
            or bound_relationship is None
            or not bound_relationship.IsValid()
        ):
            raise RuntimeError(
                f"OBJECT_BOUND_PHYSICS_MATERIAL_INVALID cylinder={cylinder_path} "
                f"material={material_path!r}"
            )
        radius_stage_units = float(cylinder.GetRadiusAttr().Get())
        height_stage_units = float(cylinder.GetHeightAttr().Get())
        mass_stage_units = float(mass_api.GetMassAttr().Get())
        radius_si_m = radius_stage_units * meters_per_unit
        height_si_m = height_stage_units * meters_per_unit
        mass_si_kg = mass_stage_units * kilograms_per_unit
        row = {
            "env_index": env_index,
            "root_path": root_path,
            "root_type": root.GetTypeName(),
            "cylinder_path": cylinder_path,
            "cylinder_type": cylinder_prim.GetTypeName(),
            "axis": str(cylinder.GetAxisAttr().Get()),
            "radius_stage_units": radius_stage_units,
            "height_stage_units": height_stage_units,
            "radius_si_m": radius_si_m,
            "height_si_m": height_si_m,
            "collision_api_present": collision_api_present,
            "collision_enabled": bool(collision_api.GetCollisionEnabledAttr().Get()),
            "mass_api_path": root_path,
            "mass_api_present": mass_api_present,
            "mass_stage_units": mass_stage_units,
            "mass_si_kg": mass_si_kg,
            "rigid_body_api_present": rigid_api_present,
            "rigid_body_enabled": bool(rigid_api.GetRigidBodyEnabledAttr().Get()),
            "physx_rigid_body_api_present": physx_rigid_api_present,
            "disable_gravity": bool(physx_rigid_api.GetDisableGravityAttr().Get()),
            "material_binding_kind": "computed_bound_physics",
            "material_binding_relationship_path": (
                bound_relationship.GetPath().pathString
            ),
            "expected_material_path": expected_material_path,
            "material_path": material_path,
            "material_type": material_prim.GetTypeName(),
            "material_api_present": material_api_present,
            "physx_material_api_present": physx_material_api_present,
            "static_friction": float(material_api.GetStaticFrictionAttr().Get()),
            "dynamic_friction": float(material_api.GetDynamicFrictionAttr().Get()),
            "restitution": float(material_api.GetRestitutionAttr().Get()),
            "friction_combine_mode": str(
                physx_material_api.GetFrictionCombineModeAttr().Get()
            ),
            "restitution_combine_mode": str(
                physx_material_api.GetRestitutionCombineModeAttr().Get()
            ),
        }
        row["pass"] = bool(
            row["root_type"] == "Xform"
            and row["cylinder_type"] == "Cylinder"
            and row["axis"] == "Z"
            and math.isclose(row["radius_si_m"], OBJ_RADIUS_M, rel_tol=0.0,
                             abs_tol=tolerances["linear_si_m_abs"])
            and math.isclose(row["height_si_m"], OBJ_HEIGHT_M, rel_tol=0.0,
                             abs_tol=tolerances["linear_si_m_abs"])
            and row["collision_api_present"] is True
            and row["collision_enabled"] is True
            and row["mass_api_present"] is True
            and row["rigid_body_api_present"] is True
            and row["physx_rigid_body_api_present"] is True
            and row["rigid_body_enabled"] is True
            and math.isclose(row["mass_si_kg"], OBJ_MASS_KG, rel_tol=0.0,
                             abs_tol=tolerances["mass_si_kg_abs"])
            and row["disable_gravity"] is False
            and row["material_path"] == expected_material_path
            and row["material_api_present"] is True
            and row["physx_material_api_present"] is True
            and math.isclose(row["static_friction"], STATIC_FRICTION, rel_tol=0.0,
                             abs_tol=tolerances["material_abs"])
            and math.isclose(row["dynamic_friction"], DYNAMIC_FRICTION, rel_tol=0.0,
                             abs_tol=tolerances["material_abs"])
            and math.isclose(row["restitution"], RESTITUTION, rel_tol=0.0,
                             abs_tol=tolerances["material_abs"])
            and row["friction_combine_mode"] == "average"
            and row["restitution_combine_mode"] == "average"
        )
        rows.append(row)
    if len(rows) != num_envs or not all(row["pass"] for row in rows):
        raise RuntimeError(f"OBJECT_STAGE_EXACT_READBACK_FAIL {rows}")

    # Pair-effective friction is deliberately not inferred.  Record every
    # authored stage material so cylinder/jaw/support combine provenance is
    # inspectable without mistaking the cylinder's coefficients for a pair.
    material_rows: list[dict[str, Any]] = []
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.MaterialAPI):
            continue
        usd_material = UsdPhysics.MaterialAPI(prim)
        px_material = (
            PhysxSchema.PhysxMaterialAPI(prim)
            if prim.HasAPI(PhysxSchema.PhysxMaterialAPI)
            else None
        )
        material_rows.append(
            {
                "path": prim.GetPath().pathString,
                "static_friction": (
                    None if usd_material.GetStaticFrictionAttr().Get() is None
                    else float(usd_material.GetStaticFrictionAttr().Get())
                ),
                "dynamic_friction": (
                    None if usd_material.GetDynamicFrictionAttr().Get() is None
                    else float(usd_material.GetDynamicFrictionAttr().Get())
                ),
                "restitution": (
                    None if usd_material.GetRestitutionAttr().Get() is None
                    else float(usd_material.GetRestitutionAttr().Get())
                ),
                "friction_combine_mode": (
                    None if px_material is None
                    else str(px_material.GetFrictionCombineModeAttr().Get())
                ),
                "restitution_combine_mode": (
                    None if px_material is None
                    else str(px_material.GetRestitutionCombineModeAttr().Get())
                ),
            }
        )
    return {
        "pass": True,
        "authority": "composed_stage_schema_readback_before_task_step",
        "stage_units": {
            "meters_per_unit": meters_per_unit,
            "kilograms_per_unit": kilograms_per_unit,
            "expected_each": 1.0,
            "pass": stage_units_pass,
        },
        "tolerances": tolerances,
        "clone_count_expected": num_envs,
        "clone_count_actual": len(rows),
        "object_clones": rows,
        "all_stage_physics_materials": material_rows,
        "effective_pair_friction": "not_computed_not_measured_not_claimed",
    }


def _filter_map(sensor: Any, expected_envs: int) -> dict[str, int]:
    raw_outer = list(sensor.contact_physx_view.filter_paths)
    raw = [str(value) for value in (
        list(raw_outer[0]) if raw_outer and not isinstance(raw_outer[0], (str, bytes))
        else raw_outer
    )]
    shape = tuple(int(value) for value in sensor.data.force_matrix_w.shape)
    expected = (expected_envs, 1, 1 + len(MOVING_BODIES), 3)
    if shape != expected:
        raise RuntimeError(f"OBJECT_CONTACT_SHAPE_MISMATCH expected={expected} actual={shape} paths={raw}")
    mapping: dict[str, int] = {}
    for index, path in enumerate(raw[: expected[2]]):
        if "/ground" in path:
            mapping["support"] = index
        else:
            body = path.rsplit("/", 1)[-1]
            if body not in MOVING_BODIES:
                raise RuntimeError(f"OBJECT_FILTER_UNKNOWN {path}")
            mapping[body] = index
    if set(mapping) != {"support", *MOVING_BODIES}:
        raise RuntimeError(f"OBJECT_FILTER_MAP_INCOMPLETE {mapping}")
    return mapping


def _one_filter_gate(
    sensor: Any,
    label: str,
    expected_envs: int,
    expected_filter_expression: str,
    expected_stage_paths: list[str],
    stage: Any,
    *,
    replicated_concrete_representative: bool = False,
) -> dict[str, Any]:
    from isaaclab.sim.utils import find_matching_prim_paths

    shape = tuple(int(value) for value in sensor.data.force_matrix_w.shape)
    if sensor.num_bodies != 1 or shape != (expected_envs, 1, 1, 3):
        raise RuntimeError(f"ONE_FILTER_SHAPE_MISMATCH {label} shape={shape} bodies={sensor.num_bodies}")
    raw_outer = list(sensor.contact_physx_view.filter_paths)
    raw = [str(value) for value in (
        list(raw_outer[0]) if raw_outer and not isinstance(raw_outer[0], (str, bytes))
        else raw_outer
    )]
    expected_glob = expected_filter_expression.replace(".*", "*")
    expected_representative = expected_stage_paths[0] if expected_stage_paths else None
    # Cloned self-pair views expose one logical filter using the env0 concrete target
    # as representative. Other reporters retain the already-audited expression/glob.
    filter_identity_pass = (
        raw == [expected_representative]
        if replicated_concrete_representative
        else raw in ([expected_filter_expression], [expected_glob])
    )
    resolved_stage_paths = sorted(
        find_matching_prim_paths(expected_filter_expression, stage=stage)
    )
    expected_stage_paths_sorted = sorted(expected_stage_paths)
    stage_rows = [
        {
            "path": path,
            "valid": bool(stage.GetPrimAtPath(path).IsValid()),
            "type": (
                stage.GetPrimAtPath(path).GetTypeName()
                if stage.GetPrimAtPath(path).IsValid() else None
            ),
        }
        for path in expected_stage_paths
    ]
    if (
        len(raw) != 1
        or not filter_identity_pass
        or len(expected_stage_paths) == 0
        or resolved_stage_paths != expected_stage_paths_sorted
        or not all(row["valid"] for row in stage_rows)
        or int(sensor.contact_physx_view.filter_count) != 1
    ):
        raise RuntimeError(
            f"ONE_FILTER_SEMANTIC_IDENTITY_FAIL label={label} raw={raw} "
            f"expected={expected_filter_expression} resolved={resolved_stage_paths} "
            f"stage={stage_rows}"
        )
    return {
        "label": label,
        "force_matrix_shape": list(shape),
        "filter_count": int(sensor.contact_physx_view.filter_count),
        "actual_filter_paths": raw,
        "expected_concrete_env0_representative": expected_representative,
        "physx_replicated_filter_representation": (
            "single_logical_filter_as_env0_concrete_representative"
            if replicated_concrete_representative
            else "authored_expression_or_physx_glob"
        ),
        "expected_filter_expression": expected_filter_expression,
        "accepted_physx_glob": expected_glob,
        "resolved_stage_paths_from_expression": resolved_stage_paths,
        "expected_stage_paths": stage_rows,
        "pass": True,
    }


def audit_self_contact_filter_identity(
    env: Any,
    expected_envs: int,
    args: argparse.Namespace,
    audit_epoch: str,
) -> dict[str, Any]:
    """Fail closed on all 15 pair-view identities at a named runtime epoch."""
    if audit_epoch not in {"precontrol", "postcontrol_pre_task"}:
        raise RuntimeError(f"SELF_FILTER_AUDIT_EPOCH_INVALID {audit_epoch}")
    stage = env.scene.stage
    clock = _diagnostic_clock_snapshot(env)
    cfg_values = {
        "replicate_physics": env.scene.cfg.replicate_physics,
        "filter_collisions": env.scene.cfg.filter_collisions,
        "clone_in_fabric": env.scene.cfg.clone_in_fabric,
    }
    cfg_expected = {
        "replicate_physics": True,
        "filter_collisions": True,
        "clone_in_fabric": False,
    }
    cfg_pass = bool(
        all(type(value) is bool for value in cfg_values.values())
        and cfg_values == cfg_expected
    )
    if not cfg_pass:
        raise RuntimeError(
            "PRECONTROL_SCENE_CLONE_CONTRACT_FAIL "
            f"actual={cfg_values} expected={cfg_expected}"
        )

    rows: dict[str, Any] = {}
    for body_a, body_b in SELF_PAIRS:
        key = f"{body_a}__{body_b}"
        sensor = env._t3u_self_sensors[(body_a, body_b)]
        view = sensor.contact_physx_view
        subject_expr = f"/World/envs/env_.*/Robot/{body_a}"
        filter_expr = f"/World/envs/env_.*/Robot/{body_b}"
        expected_sensor_paths = [
            f"/World/envs/env_{index}/Robot/{body_a}"
            for index in range(expected_envs)
        ]
        expected_filter_paths = [
            f"/World/envs/env_{index}/Robot/{body_b}"
            for index in range(expected_envs)
        ]
        base_filter = _one_filter_gate(
            sensor,
            f"self:{key}",
            expected_envs,
            filter_expr,
            expected_filter_paths,
            stage,
            replicated_concrete_representative=True,
        )
        sensor_paths = [str(path) for path in list(view.sensor_paths)]
        _forces, _points, _normals, _distances, raw_count, _starts = (
            view.get_contact_data(dt=sensor._sim_physics_dt)
        )
        raw_count_shape = [int(value) for value in raw_count.shape]
        sensor_stage_rows = [
            {
                "path": path,
                "valid": bool(stage.GetPrimAtPath(path).IsValid()),
                "type": (
                    stage.GetPrimAtPath(path).GetTypeName()
                    if stage.GetPrimAtPath(path).IsValid() else None
                ),
            }
            for path in expected_sensor_paths
        ]
        actual_capacity = int(view.max_contact_data_count)
        expected_capacity = int(args.contact_capacity) * expected_envs
        checks = {
            "base_filter_identity_pass": base_filter["pass"] is True,
            "sensor_count_exact": type(view.sensor_count) is int
            and int(view.sensor_count) == expected_envs,
            "filter_count_exact": type(view.filter_count) is int
            and int(view.filter_count) == 1,
            "sensor_paths_ordered_exact": sensor_paths == expected_sensor_paths,
            "sensor_stage_paths_valid": all(row["valid"] for row in sensor_stage_rows),
            "raw_contact_count_shape_exact": raw_count_shape == [expected_envs, 1],
            "max_contact_data_count_exact": actual_capacity == expected_capacity,
            "configured_per_prim_capacity_exact": (
                type(args.contact_capacity) is int and args.contact_capacity == 256
            ),
        }
        row = {
            "pair": [body_a, body_b],
            "subject_expression": subject_expr,
            "expected_sensor_paths": expected_sensor_paths,
            "actual_sensor_paths": sensor_paths,
            "sensor_stage_paths": sensor_stage_rows,
            "sensor_count": int(view.sensor_count),
            "filter_count": int(view.filter_count),
            "raw_contact_count_shape": raw_count_shape,
            "configured_max_contact_data_count_per_prim": int(args.contact_capacity),
            "expected_max_contact_data_count": expected_capacity,
            "actual_max_contact_data_count": actual_capacity,
            "filter_identity": base_filter,
            "checks": checks,
            "pass": all(checks.values()),
        }
        if row["pass"] is not True:
            raise RuntimeError(f"PRECONTROL_SELF_FILTER_IDENTITY_FAIL {key} {row}")
        rows[key] = row

    report_checks = {
        "scene_clone_contract_exact": cfg_pass,
        "pair_inventory_exact": list(rows) == [f"{a}__{b}" for a, b in SELF_PAIRS],
        "pair_count_exactly_15": len(rows) == 15,
        "all_pair_views_pass": all(row["pass"] for row in rows.values()),
        "task_counters_zero_before_control": (
            clock["env_sim_step_counter"] == 0
            and clock["common_step_counter"] == 0
            and clock["episode_length_buf"] == [0] * expected_envs
        ),
    }
    report = {
        "artifact": "T3U_SELF_CONTACT_FILTER_IDENTITY_V1",
        "authority": "actual_rigid_contact_view_runtime_identity",
        "audit_epoch": audit_epoch,
        "scene_clone_configuration": {
            "actual": cfg_values,
            "expected": cfg_expected,
            "pass": cfg_pass,
        },
        "expected_env_count": expected_envs,
        "expected_pair_count": len(SELF_PAIRS),
        "clock_before_control": clock,
        "pair_rows": rows,
        "checks": report_checks,
        "pass": all(report_checks.values()),
    }
    if report["pass"] is not True:
        raise RuntimeError(f"PRECONTROL_SELF_FILTER_CONTRACT_FAIL {report}")
    return report


def audit_cloned_reporters(stage: Any, num_envs: int) -> dict[str, Any]:
    from pxr import PhysxSchema

    failures: list[dict[str, Any]] = []
    checked = 0
    for env_index in range(num_envs):
        subjects = [
            ("object", f"/World/envs/env_{env_index}/Sponge"),
            *[
                (body, f"/World/envs/env_{env_index}/Robot/{body}")
                for body in CONTACT_REPORT_BODIES
            ],
        ]
        for subject, path in subjects:
            prim = stage.GetPrimAtPath(path)
            api = None if not prim.IsValid() else PhysxSchema.PhysxContactReportAPI.Get(
                stage, prim.GetPath()
            )
            threshold = None if not api else api.GetThresholdAttr().Get()
            if threshold is None or abs(float(threshold)) > 1.0e-12:
                failures.append(
                    {"subject": subject, "path": path, "threshold": threshold}
                )
            else:
                checked += 1
    expected = num_envs * (1 + len(CONTACT_REPORT_BODIES))
    if failures or checked != expected:
        raise RuntimeError(
            f"CLONED_REPORTER_AUDIT_FAIL checked={checked}/{expected} failures={failures[:20]}"
        )
    return {
        "pass": True,
        "checked": checked,
        "expected": expected,
        "subjects_per_clone": ["object", *CONTACT_REPORT_BODIES],
        "threshold": 0.0,
    }


SELF_COLLISION_ATTR = "physxArticulation:enabledSelfCollisions"
ARTICULATION_ROOT_SUFFIX = "/root_joint"
DYNAMIC_CONTROL_VERSION = "2.0.7"
DYNAMIC_CONTROL_DEPRECATED_SINCE = "Isaac Sim 4.5.0"
DYNAMIC_CONTROL_PINNED_PRODUCT = "Isaac Sim 5.1"


def _property_stack_rows(attribute: Any) -> list[dict[str, Any]]:
    """Serialize authored defaults in strength order without bool coercion."""
    from pxr import Sdf

    rows: list[dict[str, Any]] = []
    for index, spec in enumerate(attribute.GetPropertyStack()):
        authored = bool(spec.HasInfo(Sdf.AttributeSpec.DefaultValueKey))
        value = spec.GetInfo(Sdf.AttributeSpec.DefaultValueKey) if authored else None
        real_path = str(spec.layer.realPath or "")
        rows.append(
            {
                "strength_index": index,
                "layer_identifier": str(spec.layer.identifier),
                "layer_real_path": (
                    str(Path(real_path).resolve()) if real_path else None
                ),
                "spec_path": str(spec.path),
                "type_name": str(spec.typeName),
                "default_authored": authored,
                "default_value": value if type(value) is bool else None,
                "default_python_type": type(value).__name__ if authored else None,
            }
        )
    return rows


def _articulation_root_paths(stage: Any, container_path: str) -> list[str]:
    from pxr import Usd, UsdPhysics

    container = stage.GetPrimAtPath(container_path)
    if not container.IsValid():
        return []
    return sorted(
        prim.GetPath().pathString
        for prim in Usd.PrimRange(container, Usd.TraverseInstanceProxies())
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    )


def _audit_attempt3_source_self_collision(p10: Any) -> dict[str, Any]:
    """Prove the pinned asset itself explicitly authors False at root_joint."""
    from pxr import PhysxSchema, Sdf, Usd, UsdPhysics

    source_path = Path(p10.ATTEMPT3_USD).resolve()
    source_physics_path = Path(p10.ATTEMPT3_PHYSICS_LAYER).resolve()
    source_stage = Usd.Stage.Open(str(source_path))
    if source_stage is None:
        raise RuntimeError(f"SELF_COLLISION_SOURCE_STAGE_OPEN_FAIL {source_path}")
    roots = sorted(
        prim.GetPath().pathString
        for prim in Usd.PrimRange(
            source_stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()
        )
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    )
    if len(roots) != 1 or not roots[0].endswith(ARTICULATION_ROOT_SUFFIX):
        raise RuntimeError(f"SELF_COLLISION_SOURCE_ROOT_IDENTITY_FAIL {roots}")
    root = source_stage.GetPrimAtPath(roots[0])
    attr = root.GetAttribute(SELF_COLLISION_ATTR)
    stack = _property_stack_rows(attr) if attr.IsValid() else []
    source_rows = [
        row for row in stack
        if row["layer_real_path"] == str(source_physics_path)
    ]
    checks = {
        "root_exactly_one": len(roots) == 1,
        "root_suffix_exact": roots == [f"/roarm_m3{ARTICULATION_ROOT_SUFFIX}"],
        "usd_articulation_root_api": root.HasAPI(UsdPhysics.ArticulationRootAPI),
        "physx_articulation_api": root.HasAPI(PhysxSchema.PhysxArticulationAPI),
        "attribute_valid": attr.IsValid(),
        "attribute_type_bool": (
            attr.IsValid() and attr.GetTypeName() == Sdf.ValueTypeNames.Bool
        ),
        "authored_value_opinion": (
            attr.IsValid() and attr.HasAuthoredValueOpinion()
        ),
        "resolved_value_strict_false": (
            attr.IsValid() and type(attr.Get()) is bool and attr.Get() is False
        ),
        "pinned_physics_spec_exactly_one": len(source_rows) == 1,
        "pinned_physics_spec_typed_explicit_false": (
            len(source_rows) == 1
            and source_rows[0]["type_name"] == "bool"
            and source_rows[0]["default_authored"] is True
            and source_rows[0]["default_python_type"] == "bool"
            and source_rows[0]["default_value"] is False
        ),
    }
    report = {
        "source_usd_path": str(source_path),
        "source_physics_layer_path": str(source_physics_path),
        "root_candidates": roots,
        "root_path": roots[0] if len(roots) == 1 else None,
        "property_stack_strong_to_weak": stack,
        "checks": checks,
        "pass": all(checks.values()),
    }
    if report["pass"] is not True:
        raise RuntimeError(f"SELF_COLLISION_SOURCE_AUTHORITY_FAIL {report}")
    return report


def _audit_composed_usd_self_collision(
    stage: Any,
    num_envs: int,
    p10: Any,
) -> tuple[dict[str, Any], list[str]]:
    """Gate clone-authored True above the pinned source's explicit False."""
    from pxr import PhysxSchema, Sdf, UsdPhysics

    source_physics_path = str(Path(p10.ATTEMPT3_PHYSICS_LAYER).resolve())
    rows: list[dict[str, Any]] = []
    discovered_root_paths: list[str] = []
    for env_index in range(num_envs):
        container_path = f"/World/envs/env_{env_index}/Robot"
        roots = _articulation_root_paths(stage, container_path)
        root_path = roots[0] if len(roots) == 1 else None
        root = stage.GetPrimAtPath(root_path) if root_path is not None else None
        attr = (
            root.GetAttribute(SELF_COLLISION_ATTR)
            if root is not None and root.IsValid()
            else None
        )
        stack = _property_stack_rows(attr) if attr is not None and attr.IsValid() else []
        authored_rows = [row for row in stack if row["default_authored"] is True]
        source_false_indices = [
            row["strength_index"]
            for row in authored_rows
            if row["layer_real_path"] == source_physics_path
            and row["type_name"] == "bool"
            and row["default_python_type"] == "bool"
            and row["default_value"] is False
        ]
        strongest = authored_rows[0] if authored_rows else None
        expected_root = f"{container_path}{ARTICULATION_ROOT_SUFFIX}"
        checks = {
            "container_valid": stage.GetPrimAtPath(container_path).IsValid(),
            "articulation_root_exactly_one": len(roots) == 1,
            "root_suffix_exact": root_path == expected_root,
            "usd_articulation_root_api": bool(
                root is not None
                and root.IsValid()
                and root.HasAPI(UsdPhysics.ArticulationRootAPI)
            ),
            "physx_articulation_api": bool(
                root is not None
                and root.IsValid()
                and root.HasAPI(PhysxSchema.PhysxArticulationAPI)
            ),
            "attribute_valid": bool(attr is not None and attr.IsValid()),
            "attribute_type_bool": bool(
                attr is not None
                and attr.IsValid()
                and attr.GetTypeName() == Sdf.ValueTypeNames.Bool
            ),
            "authored_value_opinion": bool(
                attr is not None
                and attr.IsValid()
                and attr.HasAuthoredValueOpinion()
            ),
            "resolved_value_strict_true": bool(
                attr is not None
                and attr.IsValid()
                and type(attr.Get()) is bool
                and attr.Get() is True
            ),
            "strongest_explicit_value_strict_true": bool(
                strongest is not None
                and strongest["type_name"] == "bool"
                and strongest["default_python_type"] == "bool"
                and strongest["default_value"] is True
            ),
            "pinned_source_explicit_false_exactly_one": (
                len(source_false_indices) == 1
            ),
            "strong_true_precedes_pinned_false": bool(
                strongest is not None
                and len(source_false_indices) == 1
                and strongest["strength_index"] < source_false_indices[0]
            ),
        }
        row = {
            "env_index": env_index,
            "container_path": container_path,
            "articulation_root_candidates": roots,
            "root_path": root_path,
            "expected_root_path": expected_root,
            "attribute_name": SELF_COLLISION_ATTR,
            "attribute_type_name": (
                str(attr.GetTypeName()) if attr is not None and attr.IsValid() else None
            ),
            "resolved_value": (
                attr.Get() if attr is not None and attr.IsValid()
                and type(attr.Get()) is bool else None
            ),
            "property_stack_strong_to_weak": stack,
            "strongest_authored_strength_index": (
                strongest["strength_index"] if strongest is not None else None
            ),
            "pinned_source_false_strength_index": (
                source_false_indices[0] if len(source_false_indices) == 1 else None
            ),
            "checks": checks,
            "pass": all(checks.values()),
        }
        rows.append(row)
        if row["pass"] is True and root_path is not None:
            discovered_root_paths.append(root_path)
    report = {
        "authority": "composed_usd_authorship_and_property_stack",
        "attribute": SELF_COLLISION_ATTR,
        "expected_clone_count": num_envs,
        "actual_clone_count": len(rows),
        "root_suffix": ARTICULATION_ROOT_SUFFIX,
        "pinned_source_layer_path": source_physics_path,
        "rows": rows,
        "pass": (
            len(rows) == num_envs
            and len(discovered_root_paths) == num_envs
            and len(set(discovered_root_paths)) == num_envs
            and all(row["pass"] is True for row in rows)
        ),
    }
    if report["pass"] is not True:
        raise RuntimeError(f"SELF_COLLISION_COMPOSED_USD_FAIL {report}")
    return report, discovered_root_paths


def _audit_root_physx_view(
    env: Any,
    num_envs: int,
    discovered_root_paths: list[str],
) -> dict[str, Any]:
    """Bind Isaac Lab's effective PhysX view to the discovered root prims."""
    view = env._robot.root_physx_view
    backend = getattr(view, "_backend", None)
    raw_count = view.count
    raw_paths = list(view.prim_paths)
    prim_paths = [str(path) for path in raw_paths]
    check_value = view.check()
    checks = {
        "backend_present": backend is not None,
        "view_check_strict_true": type(check_value) is bool and check_value is True,
        "count_strict_int": type(raw_count) is int,
        "count_exact": type(raw_count) is int and raw_count == num_envs,
        "prim_paths_are_strings": all(type(path) is str for path in raw_paths),
        "prim_paths_exact_ordered_identity": prim_paths == discovered_root_paths,
    }
    report = {
        "authority": "isaaclab_root_physx_view_runtime_identity",
        "backend_python_type": (
            None if backend is None
            else f"{type(backend).__module__}.{type(backend).__name__}"
        ),
        "frontend_python_type": (
            f"{type(getattr(view, '_frontend', None)).__module__}."
            f"{type(getattr(view, '_frontend', None)).__name__}"
        ),
        "check": check_value if type(check_value) is bool else None,
        "expected_count": num_envs,
        "actual_count": raw_count if type(raw_count) is int else None,
        "expected_prim_paths": discovered_root_paths,
        "actual_prim_paths": prim_paths,
        "checks": checks,
        "pass": all(checks.values()),
    }
    if report["pass"] is not True:
        raise RuntimeError(f"SELF_COLLISION_ROOT_PHYSX_VIEW_FAIL {report}")
    return report


def _diagnostic_clock_snapshot(env: Any) -> dict[str, Any]:
    from isaacsim.core.simulation_manager import SimulationManager

    return {
        "simulation_manager_num_physics_steps": int(
            SimulationManager.get_num_physics_steps()
        ),
        "simulation_manager_time_s": float(SimulationManager.get_simulation_time()),
        "simulation_context_step_index": int(env.sim.current_time_step_index),
        "simulation_context_time_s": float(env.sim.current_time),
        "env_sim_step_counter": int(env._sim_step_counter),
        "common_step_counter": int(env.common_step_counter),
        "episode_length_buf": env.episode_length_buf.detach().cpu().tolist(),
    }


def _build_self_collision_geometry_model(
    stage: Any,
    collision_provenance: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Recover each enabled composed collider as a body-local convex hull."""
    from pxr import Gf, Usd, UsdGeom
    from scipy.spatial import ConvexHull

    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    model: dict[str, list[dict[str, Any]]] = {}
    for body in SELF_CONTACT_BODIES:
        body_path = f"/World/envs/env_0/Robot/{body}"
        body_prim = stage.GetPrimAtPath(body_path)
        if not body_prim.IsValid():
            raise RuntimeError(f"SELF_CONTROL_BODY_MISSING {body_path}")
        body_w2l = cache.GetLocalToWorldTransform(body_prim).GetInverse()
        parts: list[dict[str, Any]] = []
        for mesh_row in collision_provenance[body]["meshes"]:
            path = str(mesh_row["path"])
            mesh_prim = stage.GetPrimAtPath(path)
            raw = np.asarray(UsdGeom.Mesh(mesh_prim).GetPointsAttr().Get(), dtype=np.float64)
            mesh_l2w = cache.GetLocalToWorldTransform(mesh_prim)
            local = np.asarray(
                [
                    [
                        float(value)
                        for value in body_w2l.Transform(
                            mesh_l2w.Transform(
                                Gf.Vec3d(*[float(component) for component in point])
                            )
                        )
                    ]
                    for point in raw
                ],
                dtype=np.float64,
            )
            hull = ConvexHull(local)
            parts.append(
                {
                    "path": path,
                    "vertices": local[hull.vertices],
                    "normals": hull.equations[:, :3].astype(np.float64),
                    "offsets": hull.equations[:, 3].astype(np.float64),
                }
            )
        if len(parts) != int(collision_provenance[body]["unique_collision_mesh_count"]):
            raise RuntimeError(f"SELF_CONTROL_PART_COUNT_DRIFT body={body}")
        model[body] = parts
    return model


def _map_convex_part(part: dict[str, Any], transform: np.ndarray) -> dict[str, Any]:
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    vertices = part["vertices"] @ rotation.T + translation
    normals = part["normals"] @ rotation.T
    return {
        "path": part["path"],
        "vertices": vertices,
        "normals": normals,
        "offsets": part["offsets"] - normals @ translation,
    }


def _convex_intersection_inradius_m(
    first: dict[str, Any], second: dict[str, Any]
) -> float | None:
    from scipy.optimize import linprog

    normals = np.vstack([first["normals"], second["normals"]])
    offsets = np.concatenate([first["offsets"], second["offsets"]])
    result = linprog(
        np.asarray([0.0, 0.0, 0.0, -1.0], dtype=np.float64),
        A_ub=np.column_stack([normals, np.linalg.norm(normals, axis=1)]),
        b_ub=-offsets,
        bounds=[(None, None), (None, None), (None, None), (0.0, None)],
        method="highs",
    )
    return float(result.x[3]) if result.success else None


def _convex_face_separation_m(
    first: dict[str, Any], second: dict[str, Any]
) -> float:
    """Largest certified separating gap along either hull's face normals."""
    largest = -math.inf
    for raw_normal in np.vstack([first["normals"], second["normals"]]):
        unit = raw_normal / np.linalg.norm(raw_normal)
        largest = max(
            largest,
            float(
                np.min(second["vertices"] @ unit)
                - np.max(first["vertices"] @ unit)
            ),
            float(
                np.min(first["vertices"] @ unit)
                - np.max(second["vertices"] @ unit)
            ),
        )
    return largest


def _geometry_pose_report(
    model: dict[str, list[dict[str, Any]]],
    transforms: dict[str, np.ndarray],
) -> dict[str, Any]:
    mapped = {
        body: [_map_convex_part(part, transforms[body]) for part in model[body]]
        for body in SELF_CONTACT_BODIES
    }
    pair_rows: dict[str, Any] = {}
    positive_pairs: list[str] = []
    for body_a, body_b in SELF_PAIRS:
        key = f"{body_a}__{body_b}"
        intersections: list[dict[str, Any]] = []
        minimum_separation = math.inf
        for part_a in mapped[body_a]:
            lo_a, hi_a = part_a["vertices"].min(axis=0), part_a["vertices"].max(axis=0)
            for part_b in mapped[body_b]:
                lo_b, hi_b = part_b["vertices"].min(axis=0), part_b["vertices"].max(axis=0)
                if np.all(np.minimum(hi_a, hi_b) - np.maximum(lo_a, lo_b) > 0.0):
                    radius = _convex_intersection_inradius_m(part_a, part_b)
                    if radius is not None and radius > 1.0e-12:
                        intersections.append(
                            {
                                "part_a": part_a["path"],
                                "part_b": part_b["path"],
                                "inradius_mm": radius * 1000.0,
                            }
                        )
                        continue
                minimum_separation = min(
                    minimum_separation,
                    _convex_face_separation_m(part_a, part_b),
                )
        if intersections:
            positive_pairs.append(key)
        pair_rows[key] = {
            "positive_intersection_count": len(intersections),
            "max_intersection_inradius_mm": (
                max(row["inradius_mm"] for row in intersections)
                if intersections else None
            ),
            "minimum_separating_face_margin_mm": (
                minimum_separation * 1000.0 if math.isfinite(minimum_separation) else None
            ),
            "intersections": intersections,
        }
    moving_min_z = min(
        float(part["vertices"][:, 2].min())
        for body in MOVING_BODIES for part in mapped[body]
    )
    # This is explicitly a frozen-vertex-to-analytic-cylinder diagnostic, not
    # an exact convex-to-cylinder distance proof.  It is sufficient here only
    # to rule out the far-away task object as the cause of the positive control.
    object_vertex_proxy_gap = math.inf
    for body in MOVING_BODIES:
        for part in mapped[body]:
            vertices = part["vertices"]
            radial_gap = np.maximum(
                np.linalg.norm(vertices[:, :2] - OBJECT_CENTER_M[:2], axis=1)
                - OBJ_RADIUS_M,
                0.0,
            )
            vertical_gap = np.maximum(
                np.abs(vertices[:, 2] - OBJECT_CENTER_M[2])
                - OBJ_HEIGHT_M / 2.0,
                0.0,
            )
            object_vertex_proxy_gap = min(
                object_vertex_proxy_gap,
                float(np.sqrt(radial_gap**2 + vertical_gap**2).min()),
            )
    return {
        "positive_pair_set": positive_pairs,
        "pair_rows": pair_rows,
        "moving_min_z_mm": moving_min_z * 1000.0,
        "object_cylinder_vertex_proxy_separation_mm": (
            object_vertex_proxy_gap * 1000.0
        ),
    }


def _self_collision_geometry_certificate(
    stage: Any,
    collision_provenance: dict[str, Any],
    urdf_chain: list[dict[str, Any]],
    limits: dict[str, tuple[float, float]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    import scipy

    model = _build_self_collision_geometry_model(stage, collision_provenance)
    positive_tf = {
        "world": np.eye(4),
        **_urdf_body_transforms_for_q(urdf_chain, SELF_COLLISION_POSITIVE_Q_DEG),
    }
    negative_tf = {
        "world": np.eye(4),
        **_urdf_body_transforms_for_q(urdf_chain, SELF_COLLISION_NEGATIVE_Q_DEG),
    }
    positive = _geometry_pose_report(model, positive_tf)
    negative = _geometry_pose_report(model, negative_tf)
    expected = [f"{a}__{b}" for a, b in SELF_COLLISION_POSITIVE_EXPECTED_PAIRS]
    limit_rows = {
        name: {
            "value_deg": float(SELF_COLLISION_POSITIVE_Q_DEG[index]),
            "lower_deg": float(limits[name][0]),
            "upper_deg": float(limits[name][1]),
            "pass": bool(
                limits[name][0] <= SELF_COLLISION_POSITIVE_Q_DEG[index]
                <= limits[name][1]
            ),
        }
        for index, name in enumerate(JOINT_ORDER)
    }
    checks = {
        "model_body_inventory_exact": set(model) == set(SELF_CONTACT_BODIES),
        "positive_q_inside_exact_urdf_limits": all(row["pass"] for row in limit_rows.values()),
        "positive_overlap_pair_set_exact": positive["positive_pair_set"] == expected,
        "positive_link2_link4_inradius_gte_5mm": (
            positive["pair_rows"]["link2__link4"]["max_intersection_inradius_mm"]
            >= SELF_COLLISION_POSITIVE_INRADIUS_GATE_MM
        ),
        "positive_link2_link5_inradius_gte_5mm": (
            positive["pair_rows"]["link2__link5"]["max_intersection_inradius_mm"]
            >= SELF_COLLISION_POSITIVE_INRADIUS_GATE_MM
        ),
        "positive_all_moving_floor_clearance_gte_71mm": positive["moving_min_z_mm"] >= 71.0,
        "positive_object_cylinder_vertex_proxy_separation_gte_395mm": (
            positive["object_cylinder_vertex_proxy_separation_mm"] >= 395.0
        ),
        "negative_home_has_zero_positive_intersections": negative["positive_pair_set"] == [],
        "negative_link2_link4_signed_margin_lte_minus_60mm": (
            -negative["pair_rows"]["link2__link4"]["minimum_separating_face_margin_mm"]
            <= -SELF_COLLISION_NEGATIVE_SEPARATION_GATE_MM
        ),
    }
    report = {
        "artifact": "T3U_FROZEN_CONVEX_SELF_COLLISION_GEOMETRY_CERTIFICATE_V1",
        "authority": "composed_attempt3_enabled_convex_hulls_plus_exact_decimal_urdf_fk",
        "scipy_version": scipy.__version__,
        "positive_control_pair": list(SELF_COLLISION_CONTROL_PAIR),
        "positive_expected_overlap_pairs": [
            list(pair) for pair in SELF_COLLISION_POSITIVE_EXPECTED_PAIRS
        ],
        "positive_q_deg": SELF_COLLISION_POSITIVE_Q_DEG.tolist(),
        "negative_q_deg": SELF_COLLISION_NEGATIVE_Q_DEG.tolist(),
        "positive": positive,
        "negative": negative,
        "joint_limit_rows": limit_rows,
        "checks": checks,
        "pass": all(checks.values()),
    }
    if report["pass"] is not True:
        raise RuntimeError(f"SELF_COLLISION_GEOMETRY_CERTIFICATE_FAIL {report}")
    return model, report


def _runtime_pair_inradius_mm(
    model: dict[str, list[dict[str, Any]]],
    transform_a: np.ndarray,
    transform_b: np.ndarray,
    pair: tuple[str, str],
) -> float | None:
    radii: list[float] = []
    for raw_a in model[pair[0]]:
        part_a = _map_convex_part(raw_a, transform_a)
        lo_a, hi_a = part_a["vertices"].min(axis=0), part_a["vertices"].max(axis=0)
        for raw_b in model[pair[1]]:
            part_b = _map_convex_part(raw_b, transform_b)
            lo_b, hi_b = part_b["vertices"].min(axis=0), part_b["vertices"].max(axis=0)
            if not np.all(np.minimum(hi_a, hi_b) - np.maximum(lo_a, lo_b) > 0.0):
                continue
            radius = _convex_intersection_inradius_m(part_a, part_b)
            if radius is not None and radius > 0.0:
                radii.append(radius * 1000.0)
    return max(radii) if radii else None


def _run_self_collision_behavioral_control(
    env: Any,
    num_envs: int,
    args: argparse.Namespace,
    model: dict[str, list[dict[str, Any]]],
    precontrol_filter_identity: dict[str, Any],
) -> dict[str, Any]:
    """Run exactly two raw PhysX frames: colliding pose, then HOME."""
    import omni.physx
    import torch

    n, device = int(num_envs), env.device
    expected_positive = {
        f"{a}__{b}" for a, b in SELF_COLLISION_POSITIVE_EXPECTED_PAIRS
    }
    callback_dts: list[float] = []

    def on_step(dt: float) -> None:
        callback_dts.append(float(dt))

    subscription = omni.physx.get_physx_interface().subscribe_physics_step_events(on_step)

    def write_state(q_deg: np.ndarray) -> tuple[Any, Any]:
        q = torch.as_tensor(
            np.tile(np.radians(q_deg), (n, 1)), dtype=torch.float32, device=device
        )
        zeros = torch.zeros_like(q)
        origins = env.scene.env_origins
        object_pos = torch.as_tensor(
            np.tile(OBJECT_CENTER_M, (n, 1)), dtype=torch.float32, device=device
        ) + origins
        object_quat = torch.zeros((n, 4), dtype=torch.float32, device=device)
        object_quat[:, 0] = 1.0
        env._sponge.write_root_pose_to_sim(torch.cat([object_pos, object_quat], dim=-1))
        env._sponge.write_root_velocity_to_sim(torch.zeros((n, 6), device=device))
        env._robot.write_joint_state_to_sim(q, zeros)
        env.robot_dof_targets[:] = q
        env._robot.set_joint_position_target(q)
        env.scene.write_data_to_sim()
        env.sim.forward()
        for sensor in env.scene.sensors.values():
            sensor.reset()
        return q, origins

    def direct_runtime_transforms(
        origins: Any, bodies: tuple[str, ...] | set[str]
    ) -> tuple[Any, list[dict[str, np.ndarray]]]:
        q_actual = env._robot.root_physx_view.get_dof_positions().clone()
        # Direct PhysX-view readback avoids accepting an Isaac Lab cached body tensor
        # from the pose that preceded the diagnostic write.  PhysX returns xyzw.
        link_pose_xyzw = (
            env._robot.root_physx_view.get_link_transforms()
            .clone().detach().cpu().numpy()
        )
        origins_np = origins.detach().cpu().numpy()
        rows: list[dict[str, np.ndarray]] = []
        for env_index in range(n):
            transforms: dict[str, np.ndarray] = {}
            for body in bodies:
                pose = link_pose_xyzw[env_index, env._robot.body_names.index(body)]
                tf = np.eye(4)
                tf[:3, :3] = _quat_to_rot(
                    [pose[6], pose[3], pose[4], pose[5]]
                )
                tf[:3, 3] = pose[:3] - origins_np[env_index]
                transforms[body] = tf
            rows.append(transforms)
        return q_actual, rows

    def runtime_overlap(q: Any, origins: Any) -> dict[str, Any]:
        positive_bodies = {
            part for pair in SELF_COLLISION_POSITIVE_EXPECTED_PAIRS for part in pair
        }
        q_actual, transforms_by_env = direct_runtime_transforms(
            origins, positive_bodies
        )
        rows: list[dict[str, Any]] = []
        for env_index in range(n):
            transforms = transforms_by_env[env_index]
            pair_values = {
                f"{pair[0]}__{pair[1]}": _runtime_pair_inradius_mm(
                    model, transforms[pair[0]], transforms[pair[1]], pair
                )
                for pair in SELF_COLLISION_POSITIVE_EXPECTED_PAIRS
            }
            rows.append(
                {
                    "env_index": env_index,
                    "actual_q_max_abs_error_rad": float(
                        torch.max(torch.abs(q_actual[env_index] - q[env_index])).item()
                    ),
                    "pair_inradius_mm": pair_values,
                    "pass": bool(
                        torch.max(torch.abs(q_actual[env_index] - q[env_index])).item()
                        <= 1.0e-7
                        and all(
                            value is not None
                            and value >= SELF_COLLISION_POSITIVE_INRADIUS_GATE_MM
                            for value in pair_values.values()
                        )
                    ),
                }
            )
        return {"rows": rows, "pass": all(row["pass"] for row in rows)}

    def runtime_negative_clearance(q: Any, origins: Any) -> dict[str, Any]:
        q_actual, transforms_by_env = direct_runtime_transforms(
            origins, SELF_CONTACT_BODIES
        )
        rows: list[dict[str, Any]] = []
        for env_index in range(n):
            pose_geometry = _geometry_pose_report(
                model, transforms_by_env[env_index]
            )
            pair_separation = {
                key: value["minimum_separating_face_margin_mm"]
                for key, value in pose_geometry["pair_rows"].items()
            }
            q_error = float(
                torch.max(torch.abs(q_actual[env_index] - q[env_index])).item()
            )
            row_checks = {
                "actual_q_equals_written_HOME": q_error <= 1.0e-7,
                "all_15_pair_runtime_overlap_set_empty": (
                    pose_geometry["positive_pair_set"] == []
                ),
                "all_15_pair_separations_finite_nonnegative": (
                    set(pair_separation) == {f"{a}__{b}" for a, b in SELF_PAIRS}
                    and all(
                        type(value) is float
                        and math.isfinite(value)
                        and value >= 0.0
                        for value in pair_separation.values()
                    )
                ),
                "link2_link4_runtime_separation_gte_60mm": (
                    pair_separation.get("link2__link4", -math.inf)
                    >= SELF_COLLISION_NEGATIVE_SEPARATION_GATE_MM
                ),
            }
            rows.append(
                {
                    "env_index": env_index,
                    "actual_q_max_abs_error_rad": q_error,
                    "positive_pair_set": pose_geometry["positive_pair_set"],
                    "pair_minimum_separating_face_margin_mm": pair_separation,
                    "checks": row_checks,
                    "pass": all(row_checks.values()),
                }
            )
        return {
            "authority": (
                "direct_root_physx_view_dof_positions_and_link_transforms_"
                "after_HOME_write_before_negative_step"
            ),
            "rows": rows,
            "pass": all(row["pass"] for row in rows),
        }

    def read_after_one_step(label: str) -> dict[str, Any]:
        before = _diagnostic_clock_snapshot(env)
        callback_count_before = len(callback_dts)
        env.sim.step(render=False)
        env.scene.update(DT_S)
        after = _diagnostic_clock_snapshot(env)
        observed_callback_dts = callback_dts[callback_count_before:]
        observed_callback_dt = (
            observed_callback_dts[0]
            if len(observed_callback_dts) == 1 else math.nan
        )
        manager_elapsed = (
            after["simulation_manager_time_s"]
            - before["simulation_manager_time_s"]
        )
        context_elapsed = (
            after["simulation_context_time_s"]
            - before["simulation_context_time_s"]
        )
        callback_elapsed_tolerance = (
            _clock_elapsed_abs_tolerance_s(
                observed_callback_dt, manager_elapsed, context_elapsed
            )
            if len(observed_callback_dts) == 1
            else 0.0
        )
        pair_rows: dict[str, Any] = {}
        for pair, sensor in env._t3u_self_sensors.items():
            key = f"{pair[0]}__{pair[1]}"
            view = sensor.contact_physx_view
            force = sensor.data.force_matrix_w[:, 0, 0].norm(dim=-1)
            _f, _p, _n, _d, raw, _s = view.get_contact_data(
                dt=sensor._sim_physics_dt
            )
            raw_shape = [int(value) for value in raw.shape]
            counts = raw[:, 0] if raw_shape == [n, 1] else raw.reshape(n, -1)[:, 0]
            raw_total = int(counts.to(torch.int64).sum().item())
            actual_capacity = int(view.max_contact_data_count)
            contact_expected = label == "positive" and key in expected_positive
            count_valid = bool(
                raw_shape == [n, 1]
                and
                torch.isfinite(counts.to(torch.float64)).all().item()
                and (counts >= 0).all().item()
                and (counts < args.contact_capacity).all().item()
                and torch.equal(counts.to(torch.float64), torch.round(counts.to(torch.float64)))
                and raw_total < actual_capacity
            )
            semantic = bool(
                ((counts >= 1) & (counts < args.contact_capacity)).all().item()
                and (force > CONTACT_GATE_N).all().item()
                if contact_expected
                else (counts == 0).all().item()
                and (force <= SELF_COLLISION_NEGATIVE_FORCE_GATE_N).all().item()
            )
            pair_rows[key] = {
                "expected_contact": contact_expected,
                "raw_count_shape": raw_shape,
                "raw_count_per_env": counts.detach().cpu().tolist(),
                "raw_count_total": raw_total,
                "actual_max_contact_data_count": actual_capacity,
                "raw_count_total_strictly_below_actual_capacity": (
                    raw_total < actual_capacity
                ),
                "force_norm_n_per_env": force.detach().cpu().tolist(),
                "pass": bool(count_valid and torch.isfinite(force).all().item() and semantic),
            }

        def raw_all_zero(sensor: Any) -> bool:
            _f, _p, _n, _d, raw, _s = sensor.contact_physx_view.get_contact_data(
                dt=sensor._sim_physics_dt
            )
            return bool((raw == 0).all().item())

        object_map = _filter_map(env._t3u_object_sensor, n)
        object_robot_filter_indices = [object_map[body] for body in MOVING_BODIES]
        object_force = env._t3u_object_sensor.data.force_matrix_w[
            :, 0, object_robot_filter_indices
        ].norm(dim=-1).max(dim=1).values
        support_force = torch.stack(
            [
                env._t3u_support_sensors[body].data.force_matrix_w[:, 0, 0].norm(dim=-1)
                for body in MOVING_BODIES
            ], dim=1,
        ).max(dim=1).values
        checks = {
            "pair_inventory_exact": set(pair_rows) == {f"{a}__{b}" for a, b in SELF_PAIRS},
            "all_pair_rows_pass": all(row["pass"] for row in pair_rows.values()),
            "robot_object_raw_zero": bool(
                (
                    env._t3u_object_sensor.contact_physx_view.get_contact_data(
                        dt=env._t3u_object_sensor._sim_physics_dt
                    )[4].reshape(n, -1)[:, object_robot_filter_indices]
                    == 0
                ).all().item()
            ),
            "support_raw_zero": all(raw_all_zero(sensor) for sensor in env._t3u_support_sensors.values()),
            "robot_object_force_lte_1e_minus_8": bool((object_force <= SELF_COLLISION_NEGATIVE_FORCE_GATE_N).all().item()),
            "support_force_lte_1e_minus_8": bool((support_force <= SELF_COLLISION_NEGATIVE_FORCE_GATE_N).all().item()),
            "manager_step_delta_one": after["simulation_manager_num_physics_steps"] - before["simulation_manager_num_physics_steps"] == 1,
            "manager_time_delta_callback_dt": math.isclose(
                manager_elapsed, observed_callback_dt,
                rel_tol=0.0, abs_tol=callback_elapsed_tolerance,
            ),
            "simulation_context_step_delta_one": after["simulation_context_step_index"] - before["simulation_context_step_index"] == 1,
            "simulation_context_time_delta_callback_dt": math.isclose(
                context_elapsed, observed_callback_dt,
                rel_tol=0.0, abs_tol=callback_elapsed_tolerance,
            ),
            "task_counters_remain_zero": (
                before["env_sim_step_counter"] == after["env_sim_step_counter"] == 0
                and before["common_step_counter"] == after["common_step_counter"] == 0
                and before["episode_length_buf"] == after["episode_length_buf"] == [0] * n
            ),
        }
        return {
            "label": label,
            "before": before,
            "after": after,
            "pair_rows": pair_rows,
            "object_force_max_n_per_env": object_force.detach().cpu().tolist(),
            "support_force_max_n_per_env": support_force.detach().cpu().tolist(),
            "checks": checks,
            "pass": all(checks.values()),
        }

    before = _diagnostic_clock_snapshot(env)
    positive_q, origins = write_state(SELF_COLLISION_POSITIVE_Q_DEG)
    runtime_geometry = runtime_overlap(positive_q, origins)
    positive = read_after_one_step("positive")
    negative_q, negative_origins = write_state(SELF_COLLISION_NEGATIVE_Q_DEG)
    negative_runtime_geometry = runtime_negative_clearance(
        negative_q, negative_origins
    )
    negative = read_after_one_step("negative")
    subscription = None
    after = _diagnostic_clock_snapshot(env)
    checks = {
        "runtime_geometry_all8_pass": runtime_geometry["pass"] is True,
        "negative_runtime_geometry_all8_pass": (
            negative_runtime_geometry["pass"] is True
        ),
        "precontrol_filter_identity_pass": (
            precontrol_filter_identity["pass"] is True
        ),
        "precontrol_clock_equals_behavior_start": _json_type_value_exact(
            precontrol_filter_identity["clock_before_control"], before
        ),
        "positive_pass": positive["pass"] is True,
        "negative_pass": negative["pass"] is True,
        "callback_count_exactly_two": len(callback_dts) == 2,
        "callback_dt_both_finite_nominal": bool(
            len(callback_dts) == 2
            and all(
                math.isfinite(dt)
                and math.isclose(
                    dt, DT_S, rel_tol=0.0,
                    abs_tol=CLOCK_CALLBACK_NOMINAL_DT_ABS_TOL_S,
                )
                for dt in callback_dts
            )
        ),
        "total_manager_step_delta_two": after["simulation_manager_num_physics_steps"] - before["simulation_manager_num_physics_steps"] == 2,
        "total_manager_time_delta_callback_fsum": math.isclose(
            after["simulation_manager_time_s"] - before["simulation_manager_time_s"],
            math.fsum(callback_dts), rel_tol=0.0,
            abs_tol=_clock_elapsed_abs_tolerance_s(
                after["simulation_manager_time_s"] - before["simulation_manager_time_s"],
                math.fsum(callback_dts),
            ),
        ),
        "total_sim_context_step_delta_two": after["simulation_context_step_index"] - before["simulation_context_step_index"] == 2,
        "total_sim_context_time_delta_callback_fsum": math.isclose(
            after["simulation_context_time_s"] - before["simulation_context_time_s"],
            math.fsum(callback_dts), rel_tol=0.0,
            abs_tol=_clock_elapsed_abs_tolerance_s(
                after["simulation_context_time_s"] - before["simulation_context_time_s"],
                math.fsum(callback_dts),
            ),
        ),
        "task_counters_zero_after_both": (
            after["env_sim_step_counter"] == 0
            and after["common_step_counter"] == 0
            and after["episode_length_buf"] == [0] * n
        ),
    }
    report = {
        "artifact": "T3U_SAME_PROCESS_SELF_COLLISION_BEHAVIORAL_CONTROL_V1",
        "authority": "actual_contact_sensor_force_matrix_and_raw_physx_contact_data",
        "behavioral_proof_scope": (
            "two_preregistered_overlap_pairs_detected_and_HOME_all_15_pairs_clear; "
            "not a proof of every pose or manifold"
        ),
        "deprecated_dynamic_control_queries": 0,
        "diagnostic_physics_steps": 2,
        "task_physics_steps": 0,
        "configured_contact_capacity_per_prim": int(args.contact_capacity),
        "positive_raw_count_saturation_forbidden": True,
        "callback_dts_s": callback_dts,
        "before": before,
        "after": after,
        "precontrol_self_contact_filter_identity": precontrol_filter_identity,
        "runtime_geometry": runtime_geometry,
        "negative_runtime_geometry_before_step": negative_runtime_geometry,
        "positive": positive,
        "negative": negative,
        "checks": checks,
        "pass": all(checks.values()),
    }
    if report["pass"] is not True:
        raise RuntimeError(f"SELF_COLLISION_BEHAVIORAL_CONTROL_FAIL {report}")
    return report


def audit_self_collision_readback(
    env: Any,
    num_envs: int,
    p10: Any,
    collision_provenance: dict[str, Any],
    urdf_chain: list[dict[str, Any]],
    limits: dict[str, tuple[float, float]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Gate authored True, runtime view identity, and observed pair behavior."""
    source = _audit_attempt3_source_self_collision(p10)
    composed, root_paths = _audit_composed_usd_self_collision(
        env.scene.stage, num_envs, p10
    )
    root_view = _audit_root_physx_view(env, num_envs, root_paths)
    geometry_model, geometry = _self_collision_geometry_certificate(
        env.scene.stage, collision_provenance, urdf_chain, limits
    )
    precontrol_filter_identity = audit_self_contact_filter_identity(
        env, num_envs, args, "precontrol"
    )
    behavioral = _run_self_collision_behavioral_control(
        env, num_envs, args, geometry_model, precontrol_filter_identity
    )
    report = {
        "authority": "usd_authorship_root_view_plus_same_process_behavioral_control",
        "attribute": SELF_COLLISION_ATTR,
        "source_asset": source,
        "composed_usd": composed,
        "root_physx_view": root_view,
        "geometry_certificate": geometry,
        "precontrol_self_contact_filter_identity": precontrol_filter_identity,
        "behavioral_control": behavioral,
        "deprecated_dynamic_control_removed": True,
        "behavioral_proof_scope_limited_to_preregistered_poses": True,
        "pairwise_contact_evidence_authority": (
            "full_step_trace.self_contact_force_w_n_and_self_raw_contact_count"
        ),
        "task_physics_steps_before_gate": 0,
        "diagnostic_physics_steps_before_gate": 2,
        "pass": bool(
            source["pass"] is True
            and composed["pass"] is True
            and root_view["pass"] is True
            and geometry["pass"] is True
            and precontrol_filter_identity["pass"] is True
            and behavioral["pass"] is True
        ),
    }
    if report["pass"] is not True:
        raise RuntimeError(f"SELF_COLLISION_READBACK_FAIL {report}")
    return report


def audit_effective_joint_limits(
    env: Any,
    parsed_limits_deg: dict[str, tuple[float, float]],
    num_envs: int,
) -> dict[str, Any]:
    """Prove that composed/runtime limits equal the frozen URDF authority."""
    from pxr import UsdPhysics

    expected_joint_names = list(URDF_JOINT_MAP)
    if list(env._robot.joint_names) != expected_joint_names:
        raise RuntimeError(
            "JOINT_LIMIT_READBACK_ORDER_FAIL "
            f"expected={expected_joint_names} actual={env._robot.joint_names}"
        )
    expected_deg = np.asarray(
        [parsed_limits_deg[label] for label in JOINT_ORDER], dtype=np.float64
    )
    expected_rad = np.radians(expected_deg)
    soft = (
        env._robot.data.soft_joint_pos_limits.detach().cpu().numpy().astype(np.float64)
    )
    cached = np.stack(
        [
            env.robot_dof_lower_limits.detach().cpu().numpy(),
            env.robot_dof_upper_limits.detach().cpu().numpy(),
        ],
        axis=-1,
    ).astype(np.float64)
    soft_tolerance_rad = 5.0e-7
    usd_tolerance_deg = 2.0e-5
    soft_error = np.abs(soft - expected_rad[None, :, :])
    cached_error = np.abs(cached - expected_rad)
    soft_pass = bool(
        soft.shape == (num_envs, len(JOINT_ORDER), 2)
        and np.isfinite(soft).all()
        and float(soft_error.max()) <= soft_tolerance_rad
    )
    cached_pass = bool(
        cached.shape == (len(JOINT_ORDER), 2)
        and np.isfinite(cached).all()
        and float(cached_error.max()) <= soft_tolerance_rad
    )
    stage = env.scene.stage
    usd_rows: list[dict[str, Any]] = []
    for env_index in range(num_envs):
        for joint_index, joint_name in enumerate(expected_joint_names):
            path = f"/World/envs/env_{env_index}/Robot/joints/{joint_name}"
            prim = stage.GetPrimAtPath(path)
            joint = UsdPhysics.RevoluteJoint(prim) if prim.IsValid() else None
            lower = None if not joint else joint.GetLowerLimitAttr().Get()
            upper = None if not joint else joint.GetUpperLimitAttr().Get()
            lower_error = (
                math.inf if lower is None
                else abs(float(lower) - float(expected_deg[joint_index, 0]))
            )
            upper_error = (
                math.inf if upper is None
                else abs(float(upper) - float(expected_deg[joint_index, 1]))
            )
            row_pass = bool(
                prim.IsValid()
                and bool(joint)
                and lower is not None
                and upper is not None
                and math.isfinite(float(lower))
                and math.isfinite(float(upper))
                and lower_error <= usd_tolerance_deg
                and upper_error <= usd_tolerance_deg
            )
            usd_rows.append(
                {
                    "env_index": env_index,
                    "joint_index": joint_index,
                    "joint_name": joint_name,
                    "joint_label": JOINT_ORDER[joint_index],
                    "path": path,
                    "lower_deg": None if lower is None else float(lower),
                    "upper_deg": None if upper is None else float(upper),
                    "lower_error_deg": lower_error,
                    "upper_error_deg": upper_error,
                    "pass": row_pass,
                }
            )
    usd_pass = bool(
        len(usd_rows) == num_envs * len(JOINT_ORDER)
        and all(row["pass"] for row in usd_rows)
    )
    report = {
        "authority": "parsed_frozen_urdf_cross_checked_against_composed_usd_and_runtime_soft_limits",
        "urdf_sha256": URDF_SHA256,
        "joint_names": expected_joint_names,
        "joint_labels": list(JOINT_ORDER),
        "expected_urdf_limits_deg": expected_deg.tolist(),
        "expected_urdf_limits_rad": expected_rad.tolist(),
        "soft_limit_abs_tolerance_rad": soft_tolerance_rad,
        "usd_limit_abs_tolerance_deg": usd_tolerance_deg,
        "soft_joint_pos_limits_shape": list(soft.shape),
        "soft_joint_pos_limits_rad": soft.tolist(),
        "soft_joint_pos_limits_max_abs_error_rad": float(soft_error.max()),
        "cached_lower_upper_limits_rad": cached.tolist(),
        "cached_limits_max_abs_error_rad": float(cached_error.max()),
        "composed_revolute_joint_rows": usd_rows,
        "soft_limits_pass": soft_pass,
        "cached_limits_pass": cached_pass,
        "composed_usd_limits_pass": usd_pass,
        "pass": bool(soft_pass and cached_pass and usd_pass),
    }
    if not report["pass"]:
        raise RuntimeError(
            "EFFECTIVE_JOINT_LIMIT_READBACK_FAIL "
            f"soft={soft_pass} cached={cached_pass} usd={usd_pass}"
        )
    return report


def audit_fixed_base_contract(env: Any, num_envs: int) -> dict[str, Any]:
    """Read the actual PhysX fixed-base metatype and composed root joint."""
    from pxr import UsdPhysics

    is_fixed_base = bool(env._robot.is_fixed_base)
    body_names = list(env._robot.body_names)
    rows: list[dict[str, Any]] = []
    for env_index in range(num_envs):
        root = f"/World/envs/env_{env_index}/Robot"
        joint_path = f"{root}/root_joint"
        fixed_body_path = f"{root}/{FIXED_BASE_BODY}"
        prim = env.scene.stage.GetPrimAtPath(joint_path)
        fixed_body_prim = env.scene.stage.GetPrimAtPath(fixed_body_path)
        joint = UsdPhysics.FixedJoint(prim) if prim.IsValid() else None
        enabled = None if not joint else joint.GetJointEnabledAttr().Get()
        body0 = [] if not joint else [str(path) for path in joint.GetBody0Rel().GetTargets()]
        body1 = [] if not joint else [str(path) for path in joint.GetBody1Rel().GetTargets()]
        row_pass = bool(
            prim.IsValid()
            and prim.GetTypeName() == "PhysicsFixedJoint"
            and bool(joint)
            and enabled is True
            and body0 == []
            and body1 == [fixed_body_path]
            and fixed_body_prim.IsValid()
            and fixed_body_prim.HasAPI(UsdPhysics.RigidBodyAPI)
        )
        rows.append(
            {
                "env_index": env_index,
                "joint_path": joint_path,
                "joint_type": prim.GetTypeName() if prim.IsValid() else None,
                "joint_enabled": enabled,
                "body0_targets": body0,
                "body1_targets": body1,
                "fixed_body_path": fixed_body_path,
                "fixed_body_rigid_api_present": bool(
                    fixed_body_prim.IsValid()
                    and fixed_body_prim.HasAPI(UsdPhysics.RigidBodyAPI)
                ),
                "pass": row_pass,
            }
        )
    report = {
        "authority": "isaaclab_physx_metatype_plus_composed_enabled_root_fixed_joint",
        "urdf_conceptual_fixed_body": "base_link",
        "composed_fixed_body": FIXED_BASE_BODY,
        "urdf_to_composed_mapping": "world+base_link_merged_to_Robot/world",
        "isaaclab_is_fixed_base": is_fixed_base,
        "body_names": body_names,
        "expected_clone_count": num_envs,
        "actual_clone_count": len(rows),
        "root_fixed_joint_rows": rows,
        "pass": bool(
            is_fixed_base
            and body_names == list(SELF_CONTACT_BODIES)
            and len(rows) == num_envs
            and all(row["pass"] for row in rows)
        ),
    }
    if not report["pass"]:
        raise RuntimeError(f"FIXED_BASE_RUNTIME_READBACK_FAIL {report}")
    return report


def extract_enabled_collision_vertices(stage: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Extract the composed env-0 collision vertices in each rigid-body frame.

    Raw vertices are sufficient for the support-plane minimum of a convex hull:
    every linear directional extremum is attained at a hull vertex.  The two jaw
    bodies are the frozen authored convex-hull pieces themselves.  No visual mesh
    is admitted by this traversal.
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    all_prims = list(Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()))
    env0_prim = stage.GetPrimAtPath("/World/envs/env_0")
    if not env0_prim.IsValid():
        raise RuntimeError("STATIC_CLEARANCE_ENV0_ROOT_MISSING")
    env0_w2l = UsdGeom.Xformable(env0_prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    ).GetInverse()
    out: dict[str, np.ndarray] = {}
    provenance: dict[str, Any] = {}
    for body in SELF_CONTACT_BODIES:
        body_path = f"/World/envs/env_0/Robot/{body}"
        body_prim = stage.GetPrimAtPath(body_path)
        if not body_prim.IsValid():
            raise RuntimeError(f"STATIC_CLEARANCE_BODY_MISSING {body_path}")
        body_l2w = UsdGeom.Xformable(body_prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        body_w2l = body_l2w.GetInverse()
        body_origin_env = np.asarray(
            [
                float(value)
                for value in env0_w2l.Transform(
                    body_l2w.Transform(Gf.Vec3d(0.0, 0.0, 0.0))
                )
            ],
            dtype=np.float64,
        )
        body_R_env = np.column_stack(
            [
                np.asarray(
                    [
                        float(value)
                        for value in env0_w2l.Transform(
                            body_l2w.Transform(Gf.Vec3d(*axis))
                        )
                    ],
                    dtype=np.float64,
                )
                - body_origin_env
                for axis in np.eye(3, dtype=np.float64)
            ]
        )
        body_T_env = np.eye(4, dtype=np.float64)
        body_T_env[:3, :3] = body_R_env
        body_T_env[:3, 3] = body_origin_env
        collision_roots = []
        for prim in all_prims:
            path = prim.GetPath().pathString
            if path != body_path and not path.startswith(body_path + "/"):
                continue
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
            if enabled is False:
                continue
            collision_roots.append(prim)
        mesh_prims: dict[str, Any] = {}
        root_rows: list[dict[str, Any]] = []
        for root in collision_roots:
            root_path = root.GetPath().pathString
            found: list[str] = []
            for prim in all_prims:
                path = prim.GetPath().pathString
                if path != root_path and not path.startswith(root_path + "/"):
                    continue
                if prim.IsA(UsdGeom.Mesh):
                    mesh_prims[path] = prim
                    found.append(path)
            root_rows.append(
                {
                    "collision_root": root_path,
                    "enabled": True,
                    "mesh_paths": sorted(found),
                    "legacy_visual_proxy_used": False,
                }
            )
        if not collision_roots or not mesh_prims:
            raise RuntimeError(
                f"STATIC_CLEARANCE_ENABLED_COLLISION_MESH_MISSING body={body} "
                f"roots={[p.GetPath().pathString for p in collision_roots]}"
            )
        body_vertices: list[np.ndarray] = []
        mesh_rows: list[dict[str, Any]] = []
        for path, prim in sorted(mesh_prims.items()):
            raw = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=np.float64)
            if raw.ndim != 2 or raw.shape[1] != 3 or not len(raw):
                raise RuntimeError(f"STATIC_CLEARANCE_MESH_POINTS_INVALID path={path}")
            mesh_l2w = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()
            )
            mapped = np.asarray(
                [
                    [
                        float(value)
                        for value in body_w2l.Transform(
                            mesh_l2w.Transform(Gf.Vec3d(*[float(value) for value in point]))
                        )
                    ]
                    for point in raw
                ],
                dtype=np.float64,
            )
            body_vertices.append(mapped)
            mesh_rows.append(
                {
                    "path": path,
                    "raw_vertex_count": int(len(raw)),
                    "body_local_bounds_m": np.vstack(
                        [mapped.min(axis=0), mapped.max(axis=0)]
                    ).tolist(),
                }
            )
        vertices = np.vstack(body_vertices)
        out[body] = vertices
        provenance[body] = {
            "enabled_collision_root_count": len(collision_roots),
            "unique_collision_mesh_count": len(mesh_prims),
            "raw_vertex_count": int(len(vertices)),
            "body_local_bounds_m": np.vstack(
                [vertices.min(axis=0), vertices.max(axis=0)]
            ).tolist(),
            "roots": root_rows,
            "meshes": mesh_rows,
            "directional_minimum_authority": (
                "composed_enabled_collision_raw_vertices; convex directional extrema"
            ),
            # This is the composed USD authored/rest xform at default time.  It
            # is deliberately not called the current PhysX/HOME pose; those are
            # read independently from the articulation tensors below.
            "authored_rest_body_T_env0_body": body_T_env.tolist(),
        }
    for body in JAW_BODIES:
        if provenance[body]["enabled_collision_root_count"] != 64:
            raise RuntimeError(
                f"STATIC_CLEARANCE_JAW_NOT_64 body={body} "
                f"roots={provenance[body]['enabled_collision_root_count']}"
            )
    return out, provenance


def _gf_pose_transform(position: Any, quaternion: Any) -> np.ndarray:
    w = float(quaternion.GetReal())
    x, y, z = (float(value) for value in quaternion.GetImaginary())
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _quat_to_rot([w, x, y, z])
    transform[:3, 3] = np.asarray(position, dtype=np.float64)
    return transform


def _stage_prim_transform_in_env(stage: Any, path: str) -> np.ndarray:
    from pxr import Gf, Usd, UsdGeom

    prim = stage.GetPrimAtPath(path)
    env0 = stage.GetPrimAtPath("/World/envs/env_0")
    if not prim.IsValid() or not env0.IsValid():
        raise RuntimeError(f"AUTHORED_REST_PRIM_MISSING path={path}")
    env_w2l = UsdGeom.Xformable(env0).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    ).GetInverse()
    prim_l2w = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    origin = np.asarray(
        [float(value) for value in env_w2l.Transform(prim_l2w.Transform(Gf.Vec3d()))],
        dtype=np.float64,
    )
    rotation = np.column_stack(
        [
            np.asarray(
                [
                    float(value)
                    for value in env_w2l.Transform(
                        prim_l2w.Transform(Gf.Vec3d(*axis))
                    )
                ],
                dtype=np.float64,
            )
            - origin
            for axis in np.eye(3, dtype=np.float64)
        ]
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = origin
    return transform


def _audit_stage_authored_rest_joint_coordinates(stage: Any) -> dict[str, Any]:
    """Recover q from authored body/joint frames; this is not runtime state."""
    from pxr import UsdPhysics

    rows: list[dict[str, Any]] = []
    for joint_index, joint_name in enumerate(URDF_JOINT_MAP):
        path = f"/World/envs/env_0/Robot/joints/{joint_name}"
        prim = stage.GetPrimAtPath(path)
        joint = UsdPhysics.RevoluteJoint(prim) if prim.IsValid() else None
        if not joint:
            raise RuntimeError(f"AUTHORED_REST_JOINT_MISSING {path}")
        body0 = [str(value) for value in joint.GetBody0Rel().GetTargets()]
        body1 = [str(value) for value in joint.GetBody1Rel().GetTargets()]
        expected_body0_name = FIXED_BASE_BODY if joint_index == 0 else MOVING_BODIES[joint_index - 1]
        expected_body1_name = MOVING_BODIES[joint_index]
        expected_body0 = f"/World/envs/env_0/Robot/{expected_body0_name}"
        expected_body1 = f"/World/envs/env_0/Robot/{expected_body1_name}"
        if body0 != [expected_body0] or body1 != [expected_body1]:
            raise RuntimeError(
                f"AUTHORED_REST_JOINT_BODY_DRIFT {joint_name} body0={body0} body1={body1}"
            )
        parent = _stage_prim_transform_in_env(stage, expected_body0)
        child = _stage_prim_transform_in_env(stage, expected_body1)
        local0 = _gf_pose_transform(
            joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get()
        )
        local1 = _gf_pose_transform(
            joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get()
        )
        axis_transform = np.linalg.inv(local0) @ np.linalg.inv(parent) @ child @ local1
        q_deg = math.degrees(
            math.atan2(float(axis_transform[1, 0]), float(axis_transform[0, 0]))
        )
        ideal = _z_rotation_transform(math.radians(q_deg))
        residual = float(np.max(np.abs(axis_transform - ideal)))
        row_pass = bool(
            str(joint.GetAxisAttr().Get()) == "Z"
            and abs(q_deg) <= 2.0e-5
            and residual <= 5.0e-7
        )
        rows.append(
            {
                "joint_index": joint_index,
                "joint_name": joint_name,
                "path": path,
                "body0": body0,
                "body1": body1,
                "axis": str(joint.GetAxisAttr().Get()),
                "derived_authored_rest_q_deg": q_deg,
                "pure_axis_transform_max_abs_residual": residual,
                "pass": row_pass,
            }
        )
    report = {
        "authority": "composed_usd_default_time_body_and_joint_local_frames",
        "semantic_scope": "authored_rest_only__not_current_physx_or_HOME_pose",
        "derived_q_abs_gate_deg": 2.0e-5,
        "axis_transform_residual_gate": 5.0e-7,
        "rows": rows,
        "pass": bool(len(rows) == 6 and all(row["pass"] for row in rows)),
    }
    if not report["pass"]:
        raise RuntimeError(f"AUTHORED_REST_JOINT_COORDINATE_FAIL {report}")
    return report


def _legacy_p10_body_transforms_for_q(p10: Any, q_deg: np.ndarray) -> dict[str, np.ndarray]:
    q_rad = np.radians(np.asarray(q_deg, dtype=np.float64))
    transform = np.eye(4, dtype=np.float64)
    bodies: dict[str, np.ndarray] = {}
    chain_to_body = {
        "base_to_link1": "link1",
        "link1_to_link2": "link2",
        "link2_to_link3": "link3",
        "link3_to_link4": "link4",
        "link4_to_link5": "link5",
    }
    for name, xyz, rpy, qi in p10._CHAIN:
        transform = transform @ p10.Tmat(xyz, rpy)
        if qi is not None:
            transform = transform @ p10.Trot_z(q_rad[qi])
        body = chain_to_body.get(name)
        if body is not None:
            bodies[body] = transform.copy()
    return bodies


def audit_static_fk_alignment(
    stage: Any,
    env: Any,
    collision_provenance: dict[str, Any],
    p10: Any,
    urdf_chain: list[dict[str, Any]],
) -> dict[str, Any]:
    """Separate authored-rest alignment from current PhysX articulation state."""
    expected_joint_names = list(URDF_JOINT_MAP)
    if list(env._robot.joint_names) != expected_joint_names:
        raise RuntimeError("STATIC_ALIGNMENT_JOINT_ORDER_DRIFT")

    authored_rest_joint_report = _audit_stage_authored_rest_joint_coordinates(stage)
    authored_q = np.zeros(6, dtype=np.float64)
    exact_authored = _urdf_body_transforms_for_q(urdf_chain, authored_q)
    authored_rows: dict[str, Any] = {}
    authored_pass = True
    for body in MOVING_BODIES:
        observed = np.asarray(
            collision_provenance[body]["authored_rest_body_T_env0_body"],
            dtype=np.float64,
        )
        translation_error = float(
            np.linalg.norm(observed[:3, 3] - exact_authored[body][:3, 3])
        )
        rotation_error = float(
            np.max(np.abs(observed[:3, :3] - exact_authored[body][:3, :3]))
        )
        row_pass = translation_error <= 1.0e-6 and rotation_error <= 1.0e-6
        authored_rows[body] = {
            "translation_error_m": translation_error,
            "rotation_matrix_max_abs": rotation_error,
            "pass": row_pass,
        }
        authored_pass = authored_pass and row_pass

    runtime_epoch_before_raw = getattr(env._robot.data, "_sim_timestamp", None)
    runtime_epoch_before = (
        None if runtime_epoch_before_raw is None else float(runtime_epoch_before_raw)
    )
    runtime_q_rad = (
        env._robot.data.joint_pos.detach().cpu().numpy().astype(np.float64)
    )
    runtime_body_pos_w = (
        env._robot.data.body_pos_w.detach().cpu().numpy().astype(np.float64)
    )
    runtime_body_quat_w = (
        env._robot.data.body_quat_w.detach().cpu().numpy().astype(np.float64)
    )
    env_origins = env.scene.env_origins.detach().cpu().numpy().astype(np.float64)
    runtime_epoch_after_raw = getattr(env._robot.data, "_sim_timestamp", None)
    runtime_epoch_after = (
        None if runtime_epoch_after_raw is None else float(runtime_epoch_after_raw)
    )
    body_names = list(env._robot.body_names)
    num_envs = int(env.num_envs)
    tensor_schema_pass = bool(
        runtime_q_rad.shape == (num_envs, 6)
        and runtime_body_pos_w.shape == (num_envs, len(body_names), 3)
        and runtime_body_quat_w.shape == (num_envs, len(body_names), 4)
        and env_origins.shape == (num_envs, 3)
        and body_names == list(SELF_CONTACT_BODIES)
        and np.isfinite(runtime_q_rad).all()
        and np.isfinite(runtime_body_pos_w).all()
        and np.isfinite(runtime_body_quat_w).all()
        and np.isfinite(env_origins).all()
        and runtime_epoch_before is not None
        and runtime_epoch_after == runtime_epoch_before
    )
    runtime_rows: list[dict[str, Any]] = []
    runtime_pass = tensor_schema_pass
    if tensor_schema_pass:
        for env_index in range(num_envs):
            expected = _urdf_body_transforms_for_q(
                urdf_chain, np.degrees(runtime_q_rad[env_index])
            )
            for body in MOVING_BODIES:
                body_index = body_names.index(body)
                actual_position = runtime_body_pos_w[env_index, body_index] - env_origins[env_index]
                actual_quaternion = runtime_body_quat_w[env_index, body_index]
                actual_rotation = _quat_to_rot(actual_quaternion)
                translation_error = float(
                    np.linalg.norm(actual_position - expected[body][:3, 3])
                )
                rotation_error = float(
                    np.max(np.abs(actual_rotation - expected[body][:3, :3]))
                )
                quaternion_norm_error = abs(float(np.linalg.norm(actual_quaternion)) - 1.0)
                row_pass = bool(
                    translation_error <= 5.0e-6
                    and rotation_error <= 1.0e-5
                    and quaternion_norm_error <= 1.0e-6
                )
                runtime_rows.append(
                    {
                        "env_index": env_index,
                        "body": body,
                        "joint_state_deg": np.degrees(runtime_q_rad[env_index]).tolist(),
                        "translation_error_m": translation_error,
                        "rotation_matrix_max_abs": rotation_error,
                        "quaternion_norm_abs_error": quaternion_norm_error,
                        "pass": row_pass,
                    }
                )
                runtime_pass = runtime_pass and row_pass

    legacy_q0 = _legacy_p10_body_transforms_for_q(p10, authored_q)
    legacy_rows = {
        body: {
            "translation_delta_vs_exact_urdf_m": float(
                np.linalg.norm(legacy_q0[body][:3, 3] - exact_authored[body][:3, 3])
            ),
            "rotation_matrix_max_abs_delta_vs_exact_urdf": float(
                np.max(np.abs(legacy_q0[body][:3, :3] - exact_authored[body][:3, :3]))
            ),
        }
        for body in MOVING_BODIES[:-1]
    }
    report = {
        "pass": bool(authored_pass and runtime_pass),
        "clearance_fk_authority": "exact_decimal_parsed_frozen_urdf",
        "p10_role": "IK_only__not_clearance_frame_authority",
        "authored_rest_alignment": {
            "semantic_scope": "Usd default-time authored/rest q=0",
            "joint_coordinate_derivation": authored_rest_joint_report,
            "expected_joint_state_deg": authored_q.tolist(),
            "translation_gate_m": 1.0e-6,
            "rotation_matrix_max_abs_gate": 1.0e-6,
            "bodies": authored_rows,
            "pass": authored_pass,
        },
        "same_epoch_runtime_articulation_alignment": {
            "authority": "env._robot.data joint_pos/body_pos_w/body_quat_w single readback epoch",
            "joint_names": expected_joint_names,
            "body_names": body_names,
            "num_envs": num_envs,
            "articulation_data_sim_timestamp_before": runtime_epoch_before,
            "articulation_data_sim_timestamp_after": runtime_epoch_after,
            "single_epoch_pass": runtime_epoch_after == runtime_epoch_before,
            "tensor_schema_pass": tensor_schema_pass,
            "translation_gate_m": 5.0e-6,
            "rotation_matrix_max_abs_gate": 1.0e-5,
            "quaternion_norm_abs_gate": 1.0e-6,
            "rows": runtime_rows,
            "pass": runtime_pass,
        },
        "legacy_p10_rounded_chain_diagnostic_at_q0": legacy_rows,
        "preflight1_failure_reproduced_as_state_plus_rounding_mismatch": True,
    }
    if not report["pass"]:
        raise RuntimeError(f"STATIC_CLEARANCE_EXACT_URDF_ALIGNMENT_FAIL {report}")
    return report


def _trial_schedule_deg(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    controls = {
        "home": np.asarray(row["q_home_deg"], dtype=np.float64),
        "elevated": np.asarray(row["q_elevated_pregrasp_deg"], dtype=np.float64),
        "stage": np.asarray(row["q_near_side_stage_deg"], dtype=np.float64),
        "grasp": np.asarray(row["q_grasp_open_deg"], dtype=np.float64),
        "close": np.asarray(row["q_grasp_closed_command_deg"], dtype=np.float64),
        "lift": np.asarray(row["q_lift_deg"], dtype=np.float64),
    }
    specs = (
        ("settle", "home", "home"),
        ("approach", "home", "elevated"),
        ("stage", "elevated", "stage"),
        ("descend", "stage", "grasp"),
        ("close", "grasp", "close"),
        ("hold", "close", "close"),
        ("lift", "close", "lift"),
    )
    q_rows: list[np.ndarray] = []
    phase_ids: list[int] = []
    phase_steps: list[int] = []
    for phase_id, (phase, start, end) in enumerate(specs):
        count = PHASE_STEPS[phase]
        fractions = np.arange(1, count + 1, dtype=np.float64) / float(count)
        q_rows.append(
            controls[start][None, :]
            + fractions[:, None] * (controls[end] - controls[start])[None, :]
        )
        phase_ids.extend([phase_id] * count)
        phase_steps.extend(range(count))
    result = np.vstack(q_rows)
    if result.shape != (TOTAL_STEPS, 6):
        raise RuntimeError(f"STATIC_SCHEDULE_SHAPE_DRIFT {result.shape}")
    return result, np.asarray(phase_ids, dtype=np.int16), np.asarray(phase_steps, dtype=np.int32)


def apply_planned_clearance_gate(
    plan: dict[str, Any],
    collision_vertices: dict[str, np.ndarray],
    collision_provenance: dict[str, Any],
    urdf_chain: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate all 2,340 frozen command samples before any PhysX step."""
    phase_names = tuple(PHASE_STEPS)
    precontact_phase_ids = {phase_names.index(name) for name in (
        "settle", "approach", "stage", "descend"
    )}
    evaluated = 0
    for row in plan["trials"]:
        if not row["feasible"]:
            row["planned_clearance"] = {
                "evaluated": False,
                "gate_pass": False,
                "reason": "IK_OR_LIMIT_OR_FRAME_GATE_FAILED_BEFORE_COLLISION_EVALUATION",
            }
            continue
        q_schedule, phase_ids, phase_steps = _trial_schedule_deg(row)
        transforms = {body: [] for body in MOVING_BODIES}
        for q_now in q_schedule:
            body_tf = _urdf_body_transforms_for_q(urdf_chain, q_now)
            for body in MOVING_BODIES:
                transforms[body].append(body_tf[body])
        min_z = np.empty((TOTAL_STEPS, len(MOVING_BODIES)), dtype=np.float64)
        for body_index, body in enumerate(MOVING_BODIES):
            tf = np.asarray(transforms[body], dtype=np.float64)
            vertices = collision_vertices[body]
            for start in range(0, TOTAL_STEPS, 128):
                stop = min(start + 128, TOTAL_STEPS)
                world_z = tf[start:stop, 2, :3] @ vertices.T + tf[start:stop, 2, 3, None]
                min_z[start:stop, body_index] = world_z.min(axis=1)
        pre_mask = np.isin(phase_ids, sorted(precontact_phase_ids))
        pre_values = min_z[pre_mask]
        pre_flat_index = int(np.argmin(pre_values))
        masked_steps = np.flatnonzero(pre_mask)
        pre_row_index, pre_worst_body_index = np.unravel_index(
            pre_flat_index, pre_values.shape
        )
        pre_worst_step = int(masked_steps[pre_row_index])
        all_flat_index = int(np.argmin(min_z))
        worst_step, worst_body_index = np.unravel_index(all_flat_index, min_z.shape)
        worst_step = int(worst_step)
        worst_body_index = int(worst_body_index)
        phase_body_min: dict[str, dict[str, float]] = {}
        for phase_id, phase in enumerate(phase_names):
            mask = phase_ids == phase_id
            phase_body_min[phase] = {
                body: float(min_z[mask, body_index].min())
                for body_index, body in enumerate(MOVING_BODIES)
            }
        grasp_step = int(np.flatnonzero(phase_ids == phase_names.index("descend"))[-1])
        precontact_min = float(min_z[pre_worst_step, pre_worst_body_index])
        all_phase_min = float(min_z[worst_step, worst_body_index])
        # Object contact is intentional after descend, support contact is not.
        # Therefore the enabled-collider/support clearance applies to all 2,340
        # command samples, including close, hold, and lift.
        clearance_pass = all_phase_min >= PLANNED_COLLISION_CLEARANCE_GATE_M
        final_frame = row["actual_link5_frames"]["grasp_open"]
        final_adverse_abs = abs(float(final_frame["signed_adverse_pitch_deg"]))
        final_jaw_min = float(
            min(min_z[grasp_step, MOVING_BODIES.index(body)] for body in JAW_BODIES)
        )
        # The p15 one-degree proposal filter and this delivered-pose gate are
        # separate authorities.  The latter is a hard gate on the IK/FK pose
        # actually commanded by p16; planned clearance cannot waive it.
        delivered_pose_pitch_pass = bool(final_adverse_abs <= 1.0)
        row["planned_clearance"] = {
            "evaluated": True,
            "all_scheduled_command_samples": TOTAL_STEPS,
            "precontact_phase_names": [
                phase_names[index] for index in sorted(precontact_phase_ids)
            ],
            "gate_m_inclusive": PLANNED_COLLISION_CLEARANCE_GATE_M,
            "precontact_min_z_m": precontact_min,
            "precontact_worst_body": MOVING_BODIES[pre_worst_body_index],
            "precontact_worst_physics_step": pre_worst_step,
            "all_phase_min_z_m": all_phase_min,
            "worst_body": MOVING_BODIES[worst_body_index],
            "worst_physics_step": worst_step,
            "worst_phase": phase_names[int(phase_ids[worst_step])],
            "worst_phase_step": int(phase_steps[worst_step]),
            "q_command_at_worst_deg": q_schedule[worst_step].tolist(),
            "final_grasp_jaw_min_z_m": final_jaw_min,
            "phase_body_min_z_m": phase_body_min,
            "gate_scope": "all_2340_command_samples__support_contact_never_intentional",
            "gate_pass": clearance_pass,
        }
        row["final_orientation_acceptance"] = {
            "signed_adverse_pitch_deg": float(final_frame["signed_adverse_pitch_deg"]),
            "hard_gate_abs_lte_deg": 1.0,
            "max_axis_error_deg": float(row["final_frame_max_axis_error_deg"]),
            "clearance_is_not_a_pitch_waiver": True,
            "constrained_residual_used": False,
            "pass": delivered_pose_pitch_pass,
            "wording": "delivered IK/FK pose is within hard adverse-pitch gate",
        }
        row["feasible"] = bool(
            row["feasible"] and clearance_pass and delivered_pose_pitch_pass
        )
        if not row["feasible"]:
            row["reason"] = "STATIC_CLEARANCE_OR_FINAL_ORIENTATION_GATE_FAIL"
        evaluated += 1
    plan["static_collision_geometry"] = {
        "source": "composed attempt3 env_0 enabled collision prims only",
        "clearance_fk_authority": (
            "exact frozen URDF decimal origins/rpy/axes parsed at runtime; p10 excluded"
        ),
        "urdf_kinematic_chain": urdf_chain,
        "support_plane_z_m": SUPPORT_Z_M,
        "gate_m_inclusive": PLANNED_COLLISION_CLEARANCE_GATE_M,
        "gate_scope": "all_2340_command_samples_for_every_moving_body",
        "evaluated_trial_count": evaluated,
        "body_provenance": collision_provenance,
    }
    plan["n_feasible_after_static_clearance"] = sum(
        bool(row["feasible"]) for row in plan["trials"]
    )
    return plan


def rebaseline_after_behavioral_self_collision_diagnostic(
    env: Any,
    q_home: Any,
    behavioral_control: dict[str, Any],
) -> dict[str, Any]:
    """Restore and gate the complete task epoch without advancing physics."""
    import torch
    from isaacsim.core.simulation_manager import SimulationManager

    n = int(env.num_envs)
    device = env.device
    env.reset()
    origins = env.scene.env_origins
    object_pos = torch.as_tensor(
        np.tile(OBJECT_CENTER_M, (n, 1)), dtype=torch.float32, device=device
    ) + origins
    object_quat_wxyz = torch.zeros((n, 4), dtype=torch.float32, device=device)
    object_quat_wxyz[:, 0] = 1.0
    zero_object_velocity = torch.zeros((n, 6), dtype=torch.float32, device=device)
    zero_joint_velocity = torch.zeros_like(q_home)

    env._sponge.write_root_pose_to_sim(
        torch.cat([object_pos, object_quat_wxyz], dim=-1)
    )
    env._sponge.write_root_velocity_to_sim(zero_object_velocity)
    env._robot.write_joint_state_to_sim(q_home, zero_joint_velocity)
    env.robot_dof_targets[:] = q_home
    env._robot.set_joint_position_target(q_home)
    env._robot.write_joint_stiffness_to_sim(
        torch.full((n, len(JOINT_ORDER)), 100.0, device=device)
    )
    env._robot.write_joint_damping_to_sim(
        torch.full((n, len(JOINT_ORDER)), 5.0, device=device)
    )

    env._sim_step_counter = 0
    env.common_step_counter = 0
    env.episode_length_buf.zero_()
    env.reset_terminated.zero_()
    env.reset_time_outs.zero_()
    env.reset_buf.zero_()
    env.actions = torch.zeros_like(q_home)
    env.extras = {}
    for latch_name in (
        "_grasped", "_was_grasped", "_lift_counter", "_lift_success_flag",
        "_lift_bonus_paid", "_place_counter", "_place_success_flag",
        "_place_bonus_paid", "_stage3_fired",
    ):
        getattr(env, latch_name).zero_()

    env.scene.write_data_to_sim()
    env.sim.forward()
    env._compute_intermediate_values()
    for sensor in env.scene.sensors.values():
        sensor.reset()

    robot_view = env._robot.root_physx_view
    object_view = env._sponge.root_physx_view
    actual_q = robot_view.get_dof_positions().clone()
    actual_qd = robot_view.get_dof_velocities().clone()
    actual_targets = robot_view.get_dof_position_targets().clone()
    actual_stiffness = robot_view.get_dof_stiffnesses().clone()
    actual_damping = robot_view.get_dof_dampings().clone()
    object_transforms_xyzw = object_view.get_transforms().clone()
    object_velocities = object_view.get_velocities().clone()
    expected_object_xyzw = _expected_object_physx_world_xyzw(object_pos)

    sensor_rows: dict[str, Any] = {}
    for name, sensor in sorted(env.scene.sensors.items()):
        data = sensor._data
        managed_fields: dict[str, Any] = {}
        managed_pass = True
        for field_name in (
            "net_forces_w", "net_forces_w_history", "force_matrix_w",
            "force_matrix_w_history",
        ):
            value = getattr(data, field_name)
            if value is None:
                managed_fields[field_name] = None
                continue
            is_zero = bool(torch.count_nonzero(value).item() == 0)
            managed_fields[field_name] = {
                "shape": list(value.shape),
                "finite": bool(torch.isfinite(value).all().item()),
                "all_zero": is_zero,
            }
            managed_pass = managed_pass and is_zero
        contact_pos = data.contact_pos_w
        contact_position_state = None
        if contact_pos is not None:
            contact_position_state = {
                "shape": list(contact_pos.shape),
                "all_nan": bool(torch.isnan(contact_pos).all().item()),
                "no_inf": bool((~torch.isinf(contact_pos)).all().item()),
            }
            managed_pass = managed_pass and contact_position_state["all_nan"]
        aggregate = getattr(sensor, "_contact_position_aggregate_buffer", None)
        aggregate_state = None
        if aggregate is not None:
            aggregate_state = {
                "shape": list(aggregate.shape),
                "all_nan": bool(torch.isnan(aggregate).all().item()),
                "no_inf": bool((~torch.isinf(aggregate)).all().item()),
            }
            managed_pass = managed_pass and aggregate_state["all_nan"]
        timestamps_zero = bool(
            torch.count_nonzero(sensor._timestamp).item() == 0
            and torch.count_nonzero(sensor._timestamp_last_update).item() == 0
        )
        outdated_true = bool(sensor._is_outdated.all().item())
        sensor_rows[name] = {
            "timestamp_all_zero": timestamps_zero,
            "outdated_all_true": outdated_true,
            "managed_fields": managed_fields,
            "contact_position": contact_position_state,
            "contact_position_aggregate": aggregate_state,
            "raw_contact_counts_pre_task_not_asserted": True,
            "pass": bool(timestamps_zero and outdated_true and managed_pass),
        }

    expected_sensor_names = {
        "t3u_object_contact",
        *{f"t3u_{body}_support" for body in MOVING_BODIES},
        *{f"t3u_self_{a}__{b}" for a, b in SELF_PAIRS},
    }
    manager_steps = int(SimulationManager.get_num_physics_steps())
    manager_time = float(SimulationManager.get_simulation_time())
    sim_step_index = int(env.sim.current_time_step_index)
    sim_time = float(env.sim.current_time)
    q_error = float(torch.max(torch.abs(actual_q - q_home)).item())
    qd_error = float(torch.max(torch.abs(actual_qd)).item())
    target_error = float(torch.max(torch.abs(actual_targets - q_home)).item())
    stiffness_error = float(torch.max(torch.abs(actual_stiffness - 100.0)).item())
    damping_error = float(torch.max(torch.abs(actual_damping - 5.0)).item())
    object_transform_error = float(
        torch.max(torch.abs(object_transforms_xyzw - expected_object_xyzw)).item()
    )
    object_velocity_error = float(torch.max(torch.abs(object_velocities)).item())
    task_latches_zero = bool(
        env._sim_step_counter == 0
        and env.common_step_counter == 0
        and torch.count_nonzero(env.episode_length_buf).item() == 0
        and torch.count_nonzero(env.reset_terminated).item() == 0
        and torch.count_nonzero(env.reset_time_outs).item() == 0
        and torch.count_nonzero(env.reset_buf).item() == 0
        and torch.count_nonzero(env.actions).item() == 0
        and all(
            torch.count_nonzero(getattr(env, name)).item() == 0
            for name in (
                "_grasped", "_was_grasped", "_lift_counter",
                "_lift_success_flag", "_lift_bonus_paid", "_place_counter",
                "_place_success_flag", "_place_bonus_paid", "_stage3_fired",
            )
        )
    )
    authoritative_shapes = {
        "robot_q": list(actual_q.shape),
        "robot_qd": list(actual_qd.shape),
        "robot_position_targets": list(actual_targets.shape),
        "robot_stiffness": list(actual_stiffness.shape),
        "robot_damping": list(actual_damping.shape),
        "object_transforms_xyzw": list(object_transforms_xyzw.shape),
        "object_velocities": list(object_velocities.shape),
    }
    expected_authoritative_shapes = {
        "robot_q": [n, len(JOINT_ORDER)],
        "robot_qd": [n, len(JOINT_ORDER)],
        "robot_position_targets": [n, len(JOINT_ORDER)],
        "robot_stiffness": [n, len(JOINT_ORDER)],
        "robot_damping": [n, len(JOINT_ORDER)],
        "object_transforms_xyzw": [n, 7],
        "object_velocities": [n, 6],
    }
    diagnostic_after = behavioral_control["after"]
    checks = {
        "authoritative_physx_view_shapes_exact": (
            authoritative_shapes == expected_authoritative_shapes
        ),
        "authoritative_robot_positions_home": q_error <= 1.0e-7,
        "authoritative_robot_velocities_zero": qd_error <= 1.0e-7,
        "authoritative_robot_position_targets_home": target_error <= 1.0e-7,
        "authoritative_robot_stiffness_100": stiffness_error <= 1.0e-6,
        "authoritative_robot_damping_5": damping_error <= 1.0e-6,
        "authoritative_object_transform_exact": object_transform_error <= 1.0e-7,
        "authoritative_object_velocity_zero": object_velocity_error <= 1.0e-7,
        "task_counters_and_latches_zero": task_latches_zero,
        "reward_buf_absent_before_first_task_step": not hasattr(env, "reward_buf"),
        "sensor_inventory_exact": set(sensor_rows) == expected_sensor_names,
        "sensor_managed_buffers_reset": bool(
            sensor_rows and all(row["pass"] for row in sensor_rows.values())
        ),
        "manager_clock_preserved_not_rewritten": bool(
            manager_steps == diagnostic_after["simulation_manager_num_physics_steps"]
            and math.isclose(
                manager_time, diagnostic_after["simulation_manager_time_s"],
                rel_tol=0.0, abs_tol=1.0e-12,
            )
        ),
        "simulation_context_clock_preserved_not_rewritten": bool(
            sim_step_index == diagnostic_after["simulation_context_step_index"]
            and math.isclose(
                sim_time, diagnostic_after["simulation_context_time_s"],
                rel_tol=0.0, abs_tol=1.0e-12,
            )
        ),
        "asset_data_epoch_baselines_recorded_finite": bool(
            math.isfinite(float(env._robot.data._sim_timestamp))
            and float(env._robot.data._sim_timestamp) >= 0.0
            and math.isfinite(float(env._sponge.data._sim_timestamp))
            and float(env._sponge.data._sim_timestamp) >= 0.0
        ),
    }
    report = {
        "artifact": "T3U_POST_DIAGNOSTIC_FULL_TASK_REBASELINE_V1",
        "operations": [
            "env.reset",
            "write_object_pose_velocity",
            "write_robot_joint_state_and_position_target",
            "write_robot_stiffness_damping",
            "clear_task_counters_latches_actions",
            "scene.write_data_to_sim",
            "sim.forward_zero_physics",
            "compute_intermediate_values",
            "reset_all_scene_sensors",
        ],
        "physics_steps_added": 0,
        "authoritative_physx_view_shapes": authoritative_shapes,
        "authoritative_physx_view": {
            "robot_q_max_abs_error_rad": q_error,
            "robot_qd_max_abs_rad_s": qd_error,
            "robot_position_target_max_abs_error_rad": target_error,
            "robot_stiffness_max_abs_error": stiffness_error,
            "robot_damping_max_abs_error": damping_error,
            "object_transform_xyzw_max_abs_error": object_transform_error,
            "object_velocity_max_abs": object_velocity_error,
        },
        "task_baseline": {
            "simulation_manager_num_physics_steps": manager_steps,
            "simulation_manager_time_s": manager_time,
            "simulation_context_step_index": sim_step_index,
            "simulation_context_time_s": sim_time,
            "robot_data_timestamp_s": float(env._robot.data._sim_timestamp),
            "object_data_timestamp_s": float(env._sponge.data._sim_timestamp),
        },
        "sensor_rows": sensor_rows,
        "checks": checks,
        "pass": all(checks.values()),
    }
    if report["pass"] is not True:
        raise RuntimeError(f"POST_DIAGNOSTIC_TASK_REBASELINE_FAIL {report}")
    return report


def run_physics(
    args: argparse.Namespace,
    env: Any,
    feasible: list[dict[str, Any]],
    joint_limits_readback: dict[str, Any],
    fixed_base_readback: dict[str, Any],
    behavioral_control: dict[str, Any],
    precontrol_filter_identity: dict[str, Any],
) -> dict[str, Any]:
    import omni.physx
    import torch
    from isaacsim.core.simulation_manager import SimulationManager
    from roarm_rl.roarm_stack_env import _quat_rotate

    active_count = len(feasible)
    if not 0 < active_count < args.num_envs:
        raise RuntimeError(f"ACTIVE_COUNT_INVALID active={active_count} envs={args.num_envs}")
    witness_slot = active_count
    samples = feasible + [feasible[-1]] * (args.num_envs - active_count)
    n = args.num_envs
    device = env.device
    object_sensor = env._t3u_object_sensor
    postcontrol_filter_identity = audit_self_contact_filter_identity(
        env, n, args, "postcontrol_pre_task"
    )
    filter_identity_reuse_checks = {
        "precontrol_pass": precontrol_filter_identity.get("pass") is True,
        "postcontrol_pass": postcontrol_filter_identity.get("pass") is True,
        "scene_clone_configuration_equal": _json_type_value_exact(
            precontrol_filter_identity.get("scene_clone_configuration"),
            postcontrol_filter_identity.get("scene_clone_configuration"),
        ),
        "pair_rows_exactly_equal": _json_type_value_exact(
            precontrol_filter_identity.get("pair_rows"),
            postcontrol_filter_identity.get("pair_rows"),
        ),
        "expected_counts_equal": (
            precontrol_filter_identity.get("expected_env_count")
            == postcontrol_filter_identity.get("expected_env_count") == n
            and precontrol_filter_identity.get("expected_pair_count")
            == postcontrol_filter_identity.get("expected_pair_count")
            == len(SELF_PAIRS)
        ),
        "postcontrol_clock_equals_rebaseline_input": _json_type_value_exact(
            postcontrol_filter_identity.get("clock_before_control"),
            behavioral_control.get("after"),
        ),
    }
    filter_identity_reuse = {
        "artifact": "T3U_SELF_CONTACT_FILTER_IDENTITY_REUSE_V1",
        "precontrol": precontrol_filter_identity,
        "postcontrol_pre_task": postcontrol_filter_identity,
        "checks": filter_identity_reuse_checks,
        "pass": all(filter_identity_reuse_checks.values()),
    }
    if filter_identity_reuse["pass"] is not True:
        raise RuntimeError(f"SELF_FILTER_IDENTITY_REUSE_FAIL {filter_identity_reuse}")
    fmap = _filter_map(object_sensor, n)
    support_filter_audit: dict[str, Any] = {}
    for body, sensor in env._t3u_support_sensors.items():
        support_filter_audit[body] = _one_filter_gate(
            sensor,
            f"support:{body}",
            n,
            env._t3u_ground_path,
            [env._t3u_ground_path],
            env.scene.stage,
        )
    self_filter_audit: dict[str, Any] = {}
    for pair, sensor in env._t3u_self_sensors.items():
        body_a, body_b = pair
        expression = f"/World/envs/env_.*/Robot/{body_b}"
        key = f"{body_a}__{body_b}"
        self_filter_audit[key] = _one_filter_gate(
            sensor,
            f"self:{key}",
            n,
            expression,
            [f"/World/envs/env_{index}/Robot/{body_b}" for index in range(n)],
            env.scene.stage,
            replicated_concrete_representative=True,
        )

    expected_joints = [
        "base_link_to_link1", "link1_to_link2", "link2_to_link3",
        "link3_to_link4", "link4_to_link5", "link5_to_gripper_link",
    ]
    if list(env._robot.joint_names) != expected_joints:
        raise RuntimeError(f"JOINT_IDENTITY_MISMATCH {env._robot.joint_names}")

    attach_calls = {"actual_attach_or_follow_pose_writes": 0, "disabled_hook_invocations": 0}

    def no_attach() -> None:
        attach_calls["disabled_hook_invocations"] += 1

    env._update_grasp_attach = no_attach

    def tensor(values: Any) -> Any:
        return torch.as_tensor(np.asarray(values), dtype=torch.float32, device=device)

    expected_limits_rad = tensor(
        joint_limits_readback["expected_urdf_limits_rad"]
    )
    planned_applied_target_tolerance_rad = 1.0e-7
    actual_joint_limit_tolerance_rad = 1.0e-5
    fixed_base_position_tolerance_m = 1.0e-7
    fixed_base_quaternion_component_tolerance = 1.0e-7
    fixed_base_velocity_tolerance = 1.0e-7

    q_home = tensor([sample["q_home_deg"] for sample in samples]) * math.pi / 180.0
    q_elevated = tensor(
        [sample["q_elevated_pregrasp_deg"] for sample in samples]
    ) * math.pi / 180.0
    q_stage = tensor(
        [sample["q_near_side_stage_deg"] for sample in samples]
    ) * math.pi / 180.0
    q_grasp = tensor([sample["q_grasp_open_deg"] for sample in samples]) * math.pi / 180.0
    q_close = tensor([sample["q_grasp_closed_command_deg"] for sample in samples]) * math.pi / 180.0
    q_lift = tensor([sample["q_lift_deg"] for sample in samples]) * math.pi / 180.0
    q_home[witness_slot] = tensor(HOME_DEG) * math.pi / 180.0
    q_elevated[witness_slot] = tensor(WITNESS_Q_APPROACH_DEG) * math.pi / 180.0
    q_stage[witness_slot] = tensor(WITNESS_Q_APPROACH_DEG) * math.pi / 180.0
    q_grasp[witness_slot] = tensor(WITNESS_Q_DESCEND_DEG) * math.pi / 180.0
    q_close[witness_slot] = tensor(WITNESS_Q_CLOSE_DEG) * math.pi / 180.0
    q_lift[witness_slot] = tensor(WITNESS_Q_LIFT_DEG) * math.pi / 180.0
    tcp_grasp_ref = tensor([sample["tcp_grasp_m"] for sample in samples])

    origins = env.scene.env_origins
    rebaseline = rebaseline_after_behavioral_self_collision_diagnostic(
        env, q_home, behavioral_control
    )
    fixed_base_index = env._robot.body_names.index(FIXED_BASE_BODY)
    fixed_base_initial_pos = (
        env._robot.data.body_pos_w[:, fixed_base_index] - origins
    ).clone()
    fixed_base_initial_quat = env._robot.data.body_quat_w[:, fixed_base_index].clone()
    schedule = [
        ("settle", PHASE_STEPS["settle"], q_home, q_home),
        ("approach", PHASE_STEPS["approach"], q_home, q_elevated),
        ("stage", PHASE_STEPS["stage"], q_elevated, q_stage),
        ("descend", PHASE_STEPS["descend"], q_stage, q_grasp),
        ("close", PHASE_STEPS["close"], q_grasp, q_close),
        ("hold", PHASE_STEPS["hold"], q_close, q_close),
        ("lift", PHASE_STEPS["lift"], q_close, q_lift),
    ]
    zeros = lambda: torch.zeros(n, device=device)
    acc: dict[str, Any] = {
        "settle_support_fz": [],
        "preclose_jaw_max": zeros(),
        "preclose_nonjaw_max": zeros(),
        "close_fixed_max": zeros(),
        "close_moving_max": zeros(),
        "lift_fixed_max": zeros(),
        "lift_moving_max": zeros(),
        "close_bilateral": zeros(),
        "lift_bilateral": zeros(),
        "moving_link_support_max": zeros(),
        "nonjaw_object_max": zeros(),
        "self_contact_max": zeros(),
        "max_tilt_deg": zeros(),
    }
    all_sensors: dict[str, Any] = {"object": object_sensor}
    all_sensors.update({f"support:{body}": sensor
                        for body, sensor in env._t3u_support_sensors.items()})
    all_sensors.update({f"self:{a}__{b}": sensor
                        for (a, b), sensor in env._t3u_self_sensors.items()})
    task_callback_dts: list[float] = []

    def on_task_physics_step(dt: float) -> None:
        task_callback_dts.append(float(dt))

    task_physics_subscription = (
        omni.physx.get_physx_interface().subscribe_physics_step_events(
            on_task_physics_step
        )
    )
    first_task_step_freshness: dict[str, Any] | None = None
    raw_contact_total_peak = {name: 0 for name in all_sensors}
    trace: dict[str, list[np.ndarray]] = {
        "physics_step": [], "sim_time_s": [], "phase_id": [], "phase_step": [],
        "joint_pos_deg": [], "joint_planned_target_deg": [],
        "joint_target_deg": [], "joint_vel_rad_s": [],
        "object_pos_m": [], "object_quat_wxyz": [],
        "object_lin_vel_m_s": [], "object_ang_vel_rad_s": [],
        "tcp_pos_m": [], "moving_body_pos_m": [], "moving_body_quat_wxyz": [],
        "moving_body_lin_vel_m_s": [], "moving_body_ang_vel_rad_s": [],
        "object_force_w_n": [], "object_contact_pos_m": [],
        "moving_link_support_force_w_n": [], "self_contact_force_w_n": [],
        "self_contact_body_pos_m": [],
        "fixed_base_pos_m": [], "fixed_base_quat_wxyz": [],
        "fixed_base_lin_vel_m_s": [], "fixed_base_ang_vel_rad_s": [],
        "object_raw_contact_count": [], "support_raw_contact_count": [],
        "self_raw_contact_count": [], "object_tilt_deg": [],
        "witness_moving_support_force_w_n": [],
        "witness_joint_pos_deg": [], "witness_joint_target_deg": [],
    }
    null_action = torch.zeros((n, 6), device=device)
    tcp_local = env._tcp_local
    body_indices = [env._robot.body_names.index(body) for body in MOVING_BODIES]
    self_contact_body_indices = [
        env._robot.body_names.index(body) for body in SELF_CONTACT_BODIES
    ]
    obj_rest_z = obj_rest_tilt = tcp_at_grasp_z = None
    witness_moving_support_max = torch.zeros((), device=device)
    numeric_integrity = torch.ones(n, dtype=torch.bool, device=device)
    numeric_failure_counts: dict[str, int] = {}
    quaternion_norm_abs_tolerance = 1.0e-3

    def accumulate_numeric_check(name: str, check: Any) -> None:
        nonlocal numeric_integrity
        if tuple(check.shape) != (n,):
            raise RuntimeError(f"NUMERIC_CHECK_SHAPE_INVALID name={name} shape={check.shape}")
        check_bool = check.to(dtype=torch.bool)
        numeric_integrity &= check_bool
        numeric_failure_counts[name] = numeric_failure_counts.get(name, 0) + int(
            (~check_bool).sum().item()
        )

    def finite_by_env(value: Any) -> Any:
        return torch.isfinite(value.reshape(n, -1)).all(dim=1)

    global_step = 0
    for phase_id, (phase, steps, q_from, q_to) in enumerate(schedule):
        for phase_step in range(steps):
            fraction = (phase_step + 1) / float(steps)
            target = q_from + fraction * (q_to - q_from)
            env.robot_dof_targets[:] = target
            _obs, step_reward, _terminated, _time_outs, _extras = env.step(null_action)
            applied_target = env.robot_dof_targets.clone()

            obj_pos_local = env._sponge.data.root_pos_w - origins
            obj_quat_now = env._sponge.data.root_quat_w
            obj_lin_vel = env._sponge.data.root_lin_vel_w
            obj_ang_vel = env._sponge.data.root_ang_vel_w
            w, x, y, z = obj_quat_now.unbind(-1)
            tilt = torch.rad2deg(torch.acos(torch.clamp(1.0 - 2.0 * (x*x + y*y), -1.0, 1.0)))
            link5_pos_w = env._robot.data.body_pos_w[:, env.link5_idx]
            link5_quat_w = env._robot.data.body_quat_w[:, env.link5_idx]
            tcp = link5_pos_w + _quat_rotate(link5_quat_w, tcp_local.expand(n, 3)) - origins

            moving_body_pos = (
                env._robot.data.body_pos_w[:, body_indices] - origins[:, None, :]
            )
            moving_body_quat = env._robot.data.body_quat_w[:, body_indices]
            moving_body_lin_vel = env._robot.data.body_lin_vel_w[:, body_indices]
            moving_body_ang_vel = env._robot.data.body_ang_vel_w[:, body_indices]
            self_contact_body_pos = (
                env._robot.data.body_pos_w[:, self_contact_body_indices]
                - origins[:, None, :]
            )
            fixed_base_pos = (
                env._robot.data.body_pos_w[:, fixed_base_index] - origins
            )
            fixed_base_quat = env._robot.data.body_quat_w[:, fixed_base_index]
            fixed_base_lin_vel = env._robot.data.body_lin_vel_w[:, fixed_base_index]
            fixed_base_ang_vel = env._robot.data.body_ang_vel_w[:, fixed_base_index]

            fm = object_sensor.data.force_matrix_w[:, 0]
            object_forces = torch.stack([fm[:, fmap[name]] for name in ("support", *MOVING_BODIES)], dim=1)
            cp = object_sensor.data.contact_pos_w
            if cp is None:
                raise RuntimeError("OBJECT_CONTACT_POINTS_UNALLOCATED")
            cp_local_raw = cp[:, 0] - origins[:, None, :]
            object_contact_pos = torch.stack(
                [cp_local_raw[:, fmap[name]] for name in ("support", *MOVING_BODIES)],
                dim=1,
            )
            support_v = object_forces[:, 0]
            fixed_force = fm[:, fmap["link5"]].norm(dim=-1)
            moving_force = fm[:, fmap["gripper_link"]].norm(dim=-1)
            nonjaw_force = torch.stack(
                [fm[:, fmap[body]].norm(dim=-1) for body in NONJAW_BODIES], dim=1
            ).max(dim=1).values
            support_vectors = torch.stack(
                [env._t3u_support_sensors[body].data.force_matrix_w[:, 0, 0]
                 for body in MOVING_BODIES], dim=1
            )
            support_by_link = support_vectors.norm(dim=-1)
            witness_moving_support_max = torch.maximum(
                witness_moving_support_max,
                support_by_link[witness_slot, MOVING_BODIES.index("gripper_link")],
            )
            self_vectors = torch.stack(
                [env._t3u_self_sensors[pair].data.force_matrix_w[:, 0, 0]
                 for pair in SELF_PAIRS], dim=1
            )
            self_by_pair = self_vectors.norm(dim=-1)
            acc["moving_link_support_max"] = torch.maximum(
                acc["moving_link_support_max"], support_by_link.max(dim=1).values
            )
            acc["self_contact_max"] = torch.maximum(
                acc["self_contact_max"], self_by_pair.max(dim=1).values
            )
            acc["nonjaw_object_max"] = torch.maximum(acc["nonjaw_object_max"], nonjaw_force)
            acc["max_tilt_deg"] = torch.maximum(acc["max_tilt_deg"], tilt)
            raw_counts: dict[str, Any] = {}
            for sensor_name, sensor_now in all_sensors.items():
                _forces, _points, _normals, _distances, raw_count, _starts = (
                    sensor_now.contact_physx_view.get_contact_data(
                        dt=sensor_now._sim_physics_dt
                    )
                )
                total_count = int(raw_count.sum().item())
                raw_contact_total_peak[sensor_name] = max(
                    raw_contact_total_peak[sensor_name], total_count
                )
                raw_counts[sensor_name] = raw_count.reshape(n, -1)
            object_counts_ordered = torch.stack(
                [raw_counts["object"][:, fmap[name]] for name in ("support", *MOVING_BODIES)],
                dim=1,
            )
            support_counts_ordered = torch.stack(
                [raw_counts[f"support:{body}"][:, 0] for body in MOVING_BODIES],
                dim=1,
            )
            self_counts_ordered = torch.stack(
                [raw_counts[f"self:{a}__{b}"][:, 0] for a, b in SELF_PAIRS],
                dim=1,
            )
            if global_step == 0:
                baseline = rebaseline["task_baseline"]
                sensor_freshness: dict[str, Any] = {}
                for sensor_name, sensor_now in sorted(all_sensors.items()):
                    managed_force_values = [
                        value
                        for value in (
                            sensor_now._data.net_forces_w,
                            sensor_now._data.net_forces_w_history,
                            sensor_now._data.force_matrix_w,
                            sensor_now._data.force_matrix_w_history,
                        )
                        if value is not None
                    ]
                    sensor_freshness[sensor_name] = {
                        "timestamp_all_dt": bool(
                            torch.allclose(
                                sensor_now._timestamp,
                                torch.full_like(sensor_now._timestamp, DT_S),
                                rtol=0.0, atol=1.0e-7,
                            )
                        ),
                        "last_update_all_dt": bool(
                            torch.allclose(
                                sensor_now._timestamp_last_update,
                                torch.full_like(
                                    sensor_now._timestamp_last_update, DT_S
                                ),
                                rtol=0.0, atol=1.0e-7,
                            )
                        ),
                        "outdated_all_false": bool(
                            (~sensor_now._is_outdated).all().item()
                        ),
                        "managed_public_force_buffers_finite": bool(
                            managed_force_values
                            and all(
                                torch.isfinite(value).all().item()
                                for value in managed_force_values
                            )
                        ),
                        "raw_counts_finite_nonnegative_integer": bool(
                            torch.isfinite(
                                raw_counts[sensor_name].to(dtype=torch.float64)
                            ).all().item()
                            and (raw_counts[sensor_name] >= 0).all().item()
                            and torch.equal(
                                raw_counts[sensor_name].to(dtype=torch.float64),
                                torch.round(
                                    raw_counts[sensor_name].to(dtype=torch.float64)
                                ),
                            )
                        ),
                        "raw_counts_within_capacity": bool(
                            (raw_counts[sensor_name] <= args.contact_capacity).all().item()
                            and int(raw_counts[sensor_name].sum().item())
                            <= int(args.contact_capacity)
                            * n * int(sensor_now.num_bodies)
                        ),
                    }
                    sensor_freshness[sensor_name]["pass"] = all(
                        sensor_freshness[sensor_name].values()
                    )
                first_callback_dt = (
                    task_callback_dts[0]
                    if len(task_callback_dts) == 1 else math.nan
                )
                first_manager_elapsed = (
                    float(SimulationManager.get_simulation_time())
                    - baseline["simulation_manager_time_s"]
                )
                first_context_elapsed = (
                    float(env.sim.current_time)
                    - baseline["simulation_context_time_s"]
                )
                first_elapsed_tolerance = (
                    _clock_elapsed_abs_tolerance_s(
                        first_callback_dt,
                        first_manager_elapsed,
                        first_context_elapsed,
                    )
                    if len(task_callback_dts) == 1
                    else 0.0
                )
                first_clock_before = {
                    "simulation_manager_num_physics_steps": baseline[
                        "simulation_manager_num_physics_steps"
                    ],
                    "simulation_manager_time_s": baseline[
                        "simulation_manager_time_s"
                    ],
                    "simulation_context_step_index": baseline[
                        "simulation_context_step_index"
                    ],
                    "simulation_context_time_s": baseline[
                        "simulation_context_time_s"
                    ],
                }
                first_clock_after = {
                    "simulation_manager_num_physics_steps": int(
                        SimulationManager.get_num_physics_steps()
                    ),
                    "simulation_manager_time_s": float(
                        SimulationManager.get_simulation_time()
                    ),
                    "simulation_context_step_index": int(
                        env.sim.current_time_step_index
                    ),
                    "simulation_context_time_s": float(env.sim.current_time),
                }
                first_elapsed_comparisons = {
                    "manager": _clock_elapsed_comparison(
                        float(first_manager_elapsed), float(first_callback_dt)
                    ),
                    "context": _clock_elapsed_comparison(
                        float(first_context_elapsed), float(first_callback_dt)
                    ),
                }
                first_checks = {
                    "task_callback_exactly_one": len(task_callback_dts) == 1,
                    "task_callback_dt_finite_nominal": bool(
                        len(task_callback_dts) == 1
                        and math.isfinite(task_callback_dts[0])
                        and math.isclose(
                            task_callback_dts[0], DT_S,
                            rel_tol=0.0,
                            abs_tol=CLOCK_CALLBACK_NOMINAL_DT_ABS_TOL_S,
                        )
                    ),
                    "manager_clock_new_epoch_plus_one": bool(
                        int(SimulationManager.get_num_physics_steps())
                        - baseline["simulation_manager_num_physics_steps"] == 1
                        and math.isclose(
                            first_manager_elapsed,
                            first_callback_dt,
                            rel_tol=0.0, abs_tol=first_elapsed_tolerance,
                        )
                    ),
                    "simulation_context_new_epoch_plus_one": bool(
                        int(env.sim.current_time_step_index)
                        - baseline["simulation_context_step_index"] == 1
                        and math.isclose(
                            first_context_elapsed,
                            first_callback_dt,
                            rel_tol=0.0, abs_tol=first_elapsed_tolerance,
                        )
                    ),
                    "task_counters_exactly_one": bool(
                        env._sim_step_counter == 1
                        and env.common_step_counter == 1
                        and torch.equal(
                            env.episode_length_buf,
                            torch.ones_like(env.episode_length_buf),
                        )
                    ),
                    "task_reset_flags_false": bool(
                        not env.reset_terminated.any().item()
                        and not env.reset_time_outs.any().item()
                        and not env.reset_buf.any().item()
                    ),
                    "asset_data_timestamps_fresh_dt": bool(
                        math.isclose(
                            float(env._robot.data._sim_timestamp)
                            - baseline["robot_data_timestamp_s"],
                            DT_S,
                            rel_tol=0.0, abs_tol=1.0e-9,
                        )
                        and math.isclose(
                            float(env._sponge.data._sim_timestamp)
                            - baseline["object_data_timestamp_s"],
                            DT_S,
                            rel_tol=0.0, abs_tol=1.0e-9,
                        )
                    ),
                    "all_sensor_epochs_and_public_data_fresh": bool(
                        sensor_freshness
                        and all(row["pass"] for row in sensor_freshness.values())
                    ),
                    "reward_buf_created_from_return_exact_zero_finite": bool(
                        hasattr(env, "reward_buf")
                        and torch.is_tensor(step_reward)
                        and torch.is_tensor(env.reward_buf)
                        and tuple(step_reward.shape) == (n,)
                        and tuple(env.reward_buf.shape) == (n,)
                        and torch.isfinite(step_reward).all().item()
                        and torch.count_nonzero(step_reward).item() == 0
                        and torch.equal(step_reward, env.reward_buf)
                    ),
                }
                first_task_step_freshness = {
                    "artifact": "T3U_FIRST_TASK_STEP_FRESH_EPOCH_GATE_V1",
                    "local_task_step": 1,
                    "nominal_dt_s_informational": float(DT_S),
                    "physics_callback_dts_s": list(task_callback_dts),
                    "elapsed_abs_tolerance_s": float(first_elapsed_tolerance),
                    "clock_before": first_clock_before,
                    "clock_after": first_clock_after,
                    "elapsed_comparisons": first_elapsed_comparisons,
                    "sensor_rows": sensor_freshness,
                    "support_positive_control_authority": (
                        "unchanged_median_of_final_60_settle_samples_not_first_frame"
                    ),
                    "reward_lifecycle": {
                        "returned_shape": list(step_reward.shape),
                        "env_reward_buf_shape": list(env.reward_buf.shape),
                        "returned_finite": bool(torch.isfinite(step_reward).all().item()),
                        "returned_all_zero": bool(torch.count_nonzero(step_reward).item() == 0),
                        "returned_equals_env_reward_buf": bool(
                            torch.equal(step_reward, env.reward_buf)
                        ),
                    },
                    "checks": first_checks,
                    "pass": all(first_checks.values()),
                }
                if first_task_step_freshness["pass"] is not True:
                    raise RuntimeError(
                        "FIRST_TASK_STEP_FRESH_EPOCH_FAIL "
                        f"{first_task_step_freshness}"
                    )
            count_tensors = {
                "object_contact_counts": object_counts_ordered,
                "support_contact_counts": support_counts_ordered,
                "self_contact_counts": self_counts_ordered,
            }
            for count_name, count_tensor in count_tensors.items():
                count_float = count_tensor.to(dtype=torch.float64)
                per_prim_capacity_ok = (
                    count_float.sum(dim=1) <= float(args.contact_capacity)
                    if count_name == "object_contact_counts"
                    else (count_float <= float(args.contact_capacity)).all(dim=1)
                )
                accumulate_numeric_check(
                    f"{count_name}_finite_integer_nonnegative_within_capacity",
                    torch.isfinite(count_float).all(dim=1)
                    & (count_float >= 0.0).all(dim=1)
                    & (count_float == torch.round(count_float)).all(dim=1)
                    & (count_float <= float(args.contact_capacity)).all(dim=1)
                    & per_prim_capacity_ok,
                )
            contact_position_finite = torch.isfinite(object_contact_pos).all(dim=-1)
            accumulate_numeric_check(
                "contact_position_no_inf_and_finite_when_raw_count_positive",
                (
                    (~torch.isinf(object_contact_pos)).all(dim=-1)
                    & ((object_counts_ordered <= 0) | contact_position_finite)
                ).all(dim=1),
            )
            finite_tensors = {
                "joint_planned_target": target,
                "joint_applied_target": applied_target,
                "joint_position": env._robot.data.joint_pos,
                "joint_velocity": env._robot.data.joint_vel,
                "object_position": obj_pos_local,
                "object_quaternion": obj_quat_now,
                "object_linear_velocity": obj_lin_vel,
                "object_angular_velocity": obj_ang_vel,
                "tcp_position": tcp,
                "moving_body_position": moving_body_pos,
                "moving_body_quaternion": moving_body_quat,
                "moving_body_linear_velocity": moving_body_lin_vel,
                "moving_body_angular_velocity": moving_body_ang_vel,
                "object_contact_forces": object_forces,
                "moving_link_support_forces": support_vectors,
                "self_contact_forces": self_vectors,
                "self_contact_body_position": self_contact_body_pos,
                "fixed_base_position": fixed_base_pos,
                "fixed_base_quaternion": fixed_base_quat,
                "fixed_base_linear_velocity": fixed_base_lin_vel,
                "fixed_base_angular_velocity": fixed_base_ang_vel,
                "object_tilt": tilt,
            }
            for tensor_name, tensor_value in finite_tensors.items():
                accumulate_numeric_check(
                    f"{tensor_name}_finite", finite_by_env(tensor_value)
                )
            accumulate_numeric_check(
                "planned_equals_post_clamp_applied_target",
                (
                    (applied_target - target).abs()
                    <= planned_applied_target_tolerance_rad
                ).all(dim=1),
            )
            accumulate_numeric_check(
                "applied_target_inside_parsed_urdf_limits",
                (
                    (applied_target >= expected_limits_rad[:, 0])
                    & (applied_target <= expected_limits_rad[:, 1])
                ).all(dim=1),
            )
            accumulate_numeric_check(
                "actual_joint_position_inside_parsed_urdf_limits",
                (
                    (
                        env._robot.data.joint_pos
                        >= expected_limits_rad[:, 0] - actual_joint_limit_tolerance_rad
                    )
                    & (
                        env._robot.data.joint_pos
                        <= expected_limits_rad[:, 1] + actual_joint_limit_tolerance_rad
                    )
                ).all(dim=1),
            )
            fixed_base_quat_direct = (
                fixed_base_quat - fixed_base_initial_quat
            ).abs().max(dim=1).values
            fixed_base_quat_negated = (
                fixed_base_quat + fixed_base_initial_quat
            ).abs().max(dim=1).values
            accumulate_numeric_check(
                "fixed_base_position_no_drift",
                (fixed_base_pos - fixed_base_initial_pos).abs().max(dim=1).values
                <= fixed_base_position_tolerance_m,
            )
            accumulate_numeric_check(
                "fixed_base_orientation_no_drift_sign_invariant",
                torch.minimum(fixed_base_quat_direct, fixed_base_quat_negated)
                <= fixed_base_quaternion_component_tolerance,
            )
            accumulate_numeric_check(
                "fixed_base_linear_velocity_zero",
                fixed_base_lin_vel.abs().max(dim=1).values
                <= fixed_base_velocity_tolerance,
            )
            accumulate_numeric_check(
                "fixed_base_angular_velocity_zero",
                fixed_base_ang_vel.abs().max(dim=1).values
                <= fixed_base_velocity_tolerance,
            )
            accumulate_numeric_check(
                "object_quaternion_unit_norm",
                (
                    torch.linalg.vector_norm(obj_quat_now, dim=1) - 1.0
                ).abs() <= quaternion_norm_abs_tolerance,
            )
            accumulate_numeric_check(
                "moving_body_quaternion_unit_norm",
                (
                    (torch.linalg.vector_norm(moving_body_quat, dim=-1) - 1.0).abs()
                    <= quaternion_norm_abs_tolerance
                ).all(dim=1),
            )
            accumulate_numeric_check(
                "fixed_base_quaternion_unit_norm",
                (
                    torch.linalg.vector_norm(fixed_base_quat, dim=1) - 1.0
                ).abs() <= quaternion_norm_abs_tolerance,
            )
            if phase == "settle":
                acc["settle_support_fz"].append(support_v[:, 2].clone())
            if phase in {"settle", "approach", "stage", "descend"}:
                acc["preclose_jaw_max"] = torch.maximum(
                    acc["preclose_jaw_max"], torch.maximum(fixed_force, moving_force)
                )
                acc["preclose_nonjaw_max"] = torch.maximum(acc["preclose_nonjaw_max"], nonjaw_force)
            if phase == "close":
                acc["close_fixed_max"] = torch.maximum(acc["close_fixed_max"], fixed_force)
                acc["close_moving_max"] = torch.maximum(acc["close_moving_max"], moving_force)
                acc["close_bilateral"] = torch.maximum(
                    acc["close_bilateral"], torch.minimum(fixed_force, moving_force)
                )
            if phase == "lift":
                acc["lift_fixed_max"] = torch.maximum(acc["lift_fixed_max"], fixed_force)
                acc["lift_moving_max"] = torch.maximum(acc["lift_moving_max"], moving_force)
                acc["lift_bilateral"] = torch.maximum(
                    acc["lift_bilateral"], torch.minimum(fixed_force, moving_force)
                )

            active = slice(0, active_count)
            trace["physics_step"].append(np.asarray(global_step + 1, dtype=np.int64))
            trace["sim_time_s"].append(np.asarray((global_step + 1) * DT_S, dtype=np.float64))
            trace["phase_id"].append(np.asarray(phase_id, dtype=np.int16))
            trace["phase_step"].append(np.asarray(phase_step, dtype=np.int32))
            trace["joint_pos_deg"].append(torch.rad2deg(env._robot.data.joint_pos[active]).cpu().numpy())
            trace["joint_planned_target_deg"].append(
                torch.rad2deg(target[active]).cpu().numpy()
            )
            trace["joint_target_deg"].append(
                torch.rad2deg(applied_target[active]).cpu().numpy()
            )
            trace["joint_vel_rad_s"].append(env._robot.data.joint_vel[active].cpu().numpy())
            trace["object_pos_m"].append(obj_pos_local[active].cpu().numpy())
            trace["object_quat_wxyz"].append(obj_quat_now[active].cpu().numpy())
            trace["object_lin_vel_m_s"].append(obj_lin_vel[active].cpu().numpy())
            trace["object_ang_vel_rad_s"].append(obj_ang_vel[active].cpu().numpy())
            trace["tcp_pos_m"].append(tcp[active].cpu().numpy())
            trace["moving_body_pos_m"].append(moving_body_pos[active].cpu().numpy())
            trace["moving_body_quat_wxyz"].append(moving_body_quat[active].cpu().numpy())
            trace["moving_body_lin_vel_m_s"].append(
                moving_body_lin_vel[active].cpu().numpy()
            )
            trace["moving_body_ang_vel_rad_s"].append(
                moving_body_ang_vel[active].cpu().numpy()
            )
            trace["object_force_w_n"].append(object_forces[active].cpu().numpy())
            trace["object_contact_pos_m"].append(object_contact_pos[active].cpu().numpy())
            trace["moving_link_support_force_w_n"].append(support_vectors[active].cpu().numpy())
            trace["self_contact_force_w_n"].append(self_vectors[active].cpu().numpy())
            trace["self_contact_body_pos_m"].append(
                self_contact_body_pos[active].cpu().numpy()
            )
            trace["fixed_base_pos_m"].append(fixed_base_pos[active].cpu().numpy())
            trace["fixed_base_quat_wxyz"].append(
                fixed_base_quat[active].cpu().numpy()
            )
            trace["fixed_base_lin_vel_m_s"].append(
                fixed_base_lin_vel[active].cpu().numpy()
            )
            trace["fixed_base_ang_vel_rad_s"].append(
                fixed_base_ang_vel[active].cpu().numpy()
            )
            trace["object_raw_contact_count"].append(
                object_counts_ordered[active].cpu().numpy()
            )
            trace["support_raw_contact_count"].append(
                support_counts_ordered[active].cpu().numpy()
            )
            trace["self_raw_contact_count"].append(
                self_counts_ordered[active].cpu().numpy()
            )
            trace["object_tilt_deg"].append(tilt[active].cpu().numpy())
            trace["witness_moving_support_force_w_n"].append(
                support_vectors[witness_slot, MOVING_BODIES.index("gripper_link")]
                .cpu().numpy()
            )
            trace["witness_joint_pos_deg"].append(
                torch.rad2deg(env._robot.data.joint_pos[witness_slot]).cpu().numpy()
            )
            trace["witness_joint_target_deg"].append(
                torch.rad2deg(target[witness_slot]).cpu().numpy()
            )

            if phase_step == steps - 1:
                if phase == "settle":
                    obj_rest_z = obj_pos_local[:, 2].clone()
                    obj_rest_tilt = tilt.clone()
                elif phase == "descend":
                    acc["grasp_arrival_mm"] = (tcp - tcp_grasp_ref).norm(dim=-1) * 1000.0
                    acc["grasp_arm_q_error_deg"] = torch.rad2deg(
                        (env._robot.data.joint_pos[:, :5] - q_grasp[:, :5]).abs().max(dim=1).values
                    )
                    tcp_at_grasp_z = tcp[:, 2].clone()
                elif phase == "lift":
                    acc["lift_arm_q_error_deg"] = torch.rad2deg(
                        (env._robot.data.joint_pos[:, :5] - q_lift[:, :5]).abs().max(dim=1).values
                    )
                    acc["lift_tcp_rise_mm"] = (tcp[:, 2] - tcp_at_grasp_z) * 1000.0
            global_step += 1

    task_physics_subscription = None
    if global_step != TOTAL_STEPS or obj_rest_z is None or obj_rest_tilt is None:
        raise RuntimeError(f"PHASE_COMPLETION_FAIL steps={global_step}")
    if first_task_step_freshness is None:
        raise RuntimeError("FIRST_TASK_STEP_FRESH_EPOCH_GATE_MISSING")
    task_baseline = rebaseline["task_baseline"]
    manager_final_steps = int(SimulationManager.get_num_physics_steps())
    manager_final_time = float(SimulationManager.get_simulation_time())
    sim_final_step = int(env.sim.current_time_step_index)
    sim_final_time = float(env.sim.current_time)
    diagnostic_before = behavioral_control["before"]
    diagnostic_after = behavioral_control["after"]
    diagnostic_callback_dts = list(behavioral_control["callback_dts_s"])
    task_callback_dts_observed = list(task_callback_dts)
    diagnostic_callback_fsum = math.fsum(diagnostic_callback_dts)
    task_callback_fsum = math.fsum(task_callback_dts_observed)
    combined_callback_fsum = math.fsum(
        [*diagnostic_callback_dts, *task_callback_dts_observed]
    )
    elapsed_deltas = {
        "diagnostic_manager_s": float(
            diagnostic_after["simulation_manager_time_s"]
            - diagnostic_before["simulation_manager_time_s"]
        ),
        "diagnostic_context_s": float(
            diagnostic_after["simulation_context_time_s"]
            - diagnostic_before["simulation_context_time_s"]
        ),
        "task_manager_s": float(
            manager_final_time - task_baseline["simulation_manager_time_s"]
        ),
        "task_context_s": float(
            sim_final_time - task_baseline["simulation_context_time_s"]
        ),
        "combined_manager_s": float(
            manager_final_time - diagnostic_before["simulation_manager_time_s"]
        ),
        "combined_context_s": float(
            sim_final_time - diagnostic_before["simulation_context_time_s"]
        ),
    }
    elapsed_tolerances = {
        "diagnostic_s": _clock_elapsed_abs_tolerance_s(
            diagnostic_callback_fsum,
            elapsed_deltas["diagnostic_manager_s"],
            elapsed_deltas["diagnostic_context_s"],
        ),
        "task_s": _clock_elapsed_abs_tolerance_s(
            task_callback_fsum,
            elapsed_deltas["task_manager_s"],
            elapsed_deltas["task_context_s"],
        ),
        "combined_s": _clock_elapsed_abs_tolerance_s(
            combined_callback_fsum,
            elapsed_deltas["combined_manager_s"],
            elapsed_deltas["combined_context_s"],
        ),
    }
    elapsed_comparisons = {
        "diagnostic_manager": _clock_elapsed_comparison(
            elapsed_deltas["diagnostic_manager_s"],
            float(diagnostic_callback_fsum),
        ),
        "diagnostic_context": _clock_elapsed_comparison(
            elapsed_deltas["diagnostic_context_s"],
            float(diagnostic_callback_fsum),
        ),
        "task_manager": _clock_elapsed_comparison(
            elapsed_deltas["task_manager_s"], float(task_callback_fsum)
        ),
        "task_context": _clock_elapsed_comparison(
            elapsed_deltas["task_context_s"], float(task_callback_fsum)
        ),
        "combined_manager": _clock_elapsed_comparison(
            elapsed_deltas["combined_manager_s"],
            float(combined_callback_fsum),
        ),
        "combined_context": _clock_elapsed_comparison(
            elapsed_deltas["combined_context_s"],
            float(combined_callback_fsum),
        ),
    }
    task_clock_checks = {
        "diagnostic_callback_count_2": len(diagnostic_callback_dts) == 2,
        "task_callback_count_2340": len(task_callback_dts) == TOTAL_STEPS,
        "combined_callback_count_2342": bool(
            len(diagnostic_callback_dts) + len(task_callback_dts)
            == TOTAL_STEPS + 2
        ),
        "all_callback_dts_finite_nominal": bool(
            len(diagnostic_callback_dts) == 2
            and len(task_callback_dts) == TOTAL_STEPS
            and all(
                math.isfinite(dt)
                and math.isclose(
                    dt, DT_S, rel_tol=0.0,
                    abs_tol=CLOCK_CALLBACK_NOMINAL_DT_ABS_TOL_S,
                )
                for dt in [*diagnostic_callback_dts, *task_callback_dts]
            )
        ),
        "manager_diagnostic_step_delta_2": (
            diagnostic_after["simulation_manager_num_physics_steps"]
            - diagnostic_before["simulation_manager_num_physics_steps"] == 2
        ),
        "manager_diagnostic_time_delta_callback_fsum": (
            elapsed_comparisons["diagnostic_manager"]["pass"] is True
        ),
        "simulation_context_diagnostic_step_delta_2": (
            diagnostic_after["simulation_context_step_index"]
            - diagnostic_before["simulation_context_step_index"] == 2
        ),
        "simulation_context_diagnostic_time_delta_callback_fsum": (
            elapsed_comparisons["diagnostic_context"]["pass"] is True
        ),
        "manager_task_step_delta_2340": (
            manager_final_steps
            - task_baseline["simulation_manager_num_physics_steps"] == TOTAL_STEPS
        ),
        "manager_task_time_delta_callback_fsum": (
            elapsed_comparisons["task_manager"]["pass"] is True
        ),
        "simulation_context_task_step_delta_2340": (
            sim_final_step - task_baseline["simulation_context_step_index"]
            == TOTAL_STEPS
        ),
        "simulation_context_task_time_delta_callback_fsum": (
            elapsed_comparisons["task_context"]["pass"] is True
        ),
        "task_counters_2340": bool(
            env._sim_step_counter == TOTAL_STEPS
            and env.common_step_counter == TOTAL_STEPS
            and torch.equal(
                env.episode_length_buf,
                torch.full_like(env.episode_length_buf, TOTAL_STEPS),
            )
        ),
        "combined_manager_steps_2342": (
            manager_final_steps
            - diagnostic_before["simulation_manager_num_physics_steps"]
            == TOTAL_STEPS + 2
        ),
        "combined_manager_time_delta_callback_fsum": (
            elapsed_comparisons["combined_manager"]["pass"] is True
        ),
        "combined_simulation_context_steps_2342": (
            sim_final_step - diagnostic_before["simulation_context_step_index"]
            == TOTAL_STEPS + 2
        ),
        "combined_simulation_context_time_delta_callback_fsum": (
            elapsed_comparisons["combined_context"]["pass"] is True
        ),
        "manager_context_task_baselines_exact": bool(
            _clock_manager_context_equal(
                task_baseline["simulation_manager_num_physics_steps"],
                task_baseline["simulation_manager_time_s"],
                task_baseline["simulation_context_step_index"],
                task_baseline["simulation_context_time_s"],
            )
        ),
        "manager_context_diagnostic_before_after_exact": bool(
            _clock_manager_context_equal(
                diagnostic_before["simulation_manager_num_physics_steps"],
                diagnostic_before["simulation_manager_time_s"],
                diagnostic_before["simulation_context_step_index"],
                diagnostic_before["simulation_context_time_s"],
            )
            and _clock_manager_context_equal(
                diagnostic_after["simulation_manager_num_physics_steps"],
                diagnostic_after["simulation_manager_time_s"],
                diagnostic_after["simulation_context_step_index"],
                diagnostic_after["simulation_context_time_s"],
            )
        ),
        "manager_context_finals_exact": bool(
            _clock_manager_context_equal(
                manager_final_steps, manager_final_time,
                sim_final_step, sim_final_time,
            )
        ),
    }
    physics_clock_accounting = {
        "artifact": "T3U_DIAGNOSTIC_AND_TASK_PHYSICS_CLOCK_ACCOUNTING_V2",
        "elapsed_time_authority": "math.fsum_of_durable_physics_callback_step_size_vectors",
        "diagnostic_physics_steps": 2,
        "task_physics_steps": TOTAL_STEPS,
        "combined_physics_steps": TOTAL_STEPS + 2,
        "task_local_step_range": [1, TOTAL_STEPS],
        "nominal_dt_s_informational": float(DT_S),
        "nominal_task_duration_s_informational": float(TOTAL_STEPS * DT_S),
        "nominal_combined_duration_s_informational": float(
            (TOTAL_STEPS + 2) * DT_S
        ),
        "callback_nominal_dt_abs_tolerance_s": (
            CLOCK_CALLBACK_NOMINAL_DT_ABS_TOL_S
        ),
        "elapsed_ulp_multiplier": CLOCK_ELAPSED_ULP_MULTIPLIER,
        "diagnostic_callback_dts_s": diagnostic_callback_dts,
        "task_callback_dts_s": task_callback_dts_observed,
        "task_callback_count": len(task_callback_dts),
        "task_callback_dt_min_s": float(min(task_callback_dts)),
        "task_callback_dt_max_s": float(max(task_callback_dts)),
        "callback_fsum_s": {
            "diagnostic": float(diagnostic_callback_fsum),
            "task": float(task_callback_fsum),
            "combined": float(combined_callback_fsum),
        },
        "observed_elapsed_deltas_s": elapsed_deltas,
        "elapsed_abs_tolerance_s": elapsed_tolerances,
        "elapsed_comparisons": elapsed_comparisons,
        "task_baseline": task_baseline,
        "task_final": {
            "simulation_manager_num_physics_steps": manager_final_steps,
            "simulation_manager_time_s": manager_final_time,
            "simulation_context_step_index": sim_final_step,
            "simulation_context_time_s": sim_final_time,
            "env_sim_step_counter": int(env._sim_step_counter),
            "common_step_counter": int(env.common_step_counter),
            "episode_length_buf": env.episode_length_buf.detach().cpu().tolist(),
        },
        "checks": task_clock_checks,
        "pass": all(task_clock_checks.values()),
    }
    if physics_clock_accounting["pass"] is not True:
        raise RuntimeError(f"TASK_PHYSICS_CLOCK_ACCOUNTING_FAIL {physics_clock_accounting}")
    final_pos = env._sponge.data.root_pos_w - origins
    final_quat = env._sponge.data.root_quat_w
    fw, fx, fy, fz = final_quat.unbind(-1)
    final_tilt = torch.rad2deg(torch.acos(torch.clamp(1.0 - 2.0 * (fx*fx + fy*fy), -1.0, 1.0)))

    def lowest(center_z: Any, tilt_deg: Any) -> Any:
        tilt_rad = torch.deg2rad(tilt_deg)
        return center_z - (
            (OBJ_HEIGHT_M / 2.0) * torch.cos(tilt_rad)
            + OBJ_RADIUS_M * torch.sin(tilt_rad)
        )

    corrected_lift = (lowest(final_pos[:, 2], final_tilt) - lowest(obj_rest_z, obj_rest_tilt)) * 1000.0
    settle_tail = torch.stack(acc["settle_support_fz"][-60:], dim=0).median(dim=0).values
    positive_control = (
        (settle_tail >= SETTLE_SUPPORT_N * (1.0 - SETTLE_REL_TOL))
        & (settle_tail <= SETTLE_SUPPORT_N * (1.0 + SETTLE_REL_TOL))
    )
    raw_contact_capacity = {
        name: int(args.contact_capacity) * n * int(sensor.num_bodies)
        for name, sensor in all_sensors.items()
    }
    saturated_sensors = [
        name for name in all_sensors
        if raw_contact_total_peak[name] >= raw_contact_capacity[name]
    ]
    buffers_ok = not saturated_sensors
    arrival = (
        (acc["grasp_arrival_mm"] <= 3.0)
        & (acc["grasp_arm_q_error_deg"] <= 3.0)
        & (acc["lift_arm_q_error_deg"] <= 3.0)
        & (acc["lift_tcp_rise_mm"] >= 15.0)
    )
    premature_jaw_contact = acc["preclose_jaw_max"] > CONTACT_GATE_N
    collision_block = (
        (acc["preclose_nonjaw_max"] > CONTACT_GATE_N)
        | (acc["moving_link_support_max"] > CONTACT_GATE_N)
        | (acc["nonjaw_object_max"] > CONTACT_GATE_N)
        | (acc["self_contact_max"] > CONTACT_GATE_N)
    )
    task_clear = ~(premature_jaw_contact | collision_block)
    both_close = acc["close_bilateral"] > JAW_LOAD_GATE_N
    both_lift = acc["lift_bilateral"] > JAW_LOAD_GATE_N
    derived_float_tensors = {
        "final_object_position": final_pos,
        "final_object_quaternion": final_quat,
        "final_object_tilt": final_tilt,
        "corrected_lift": corrected_lift,
        "settle_support_tail": settle_tail,
        "grasp_arrival": acc["grasp_arrival_mm"],
        "grasp_arm_q_error": acc["grasp_arm_q_error_deg"],
        "lift_arm_q_error": acc["lift_arm_q_error_deg"],
        "lift_tcp_rise": acc["lift_tcp_rise_mm"],
        "preclose_jaw_max": acc["preclose_jaw_max"],
        "preclose_nonjaw_max": acc["preclose_nonjaw_max"],
        "close_fixed_max": acc["close_fixed_max"],
        "close_moving_max": acc["close_moving_max"],
        "lift_fixed_max": acc["lift_fixed_max"],
        "lift_moving_max": acc["lift_moving_max"],
        "close_bilateral": acc["close_bilateral"],
        "lift_bilateral": acc["lift_bilateral"],
        "moving_link_support_max": acc["moving_link_support_max"],
        "nonjaw_object_max": acc["nonjaw_object_max"],
        "self_contact_max": acc["self_contact_max"],
    }
    for tensor_name, tensor_value in derived_float_tensors.items():
        accumulate_numeric_check(
            f"derived_{tensor_name}_finite", finite_by_env(tensor_value)
        )
    witness_numeric_integrity = bool(
        numeric_integrity[witness_slot].item()
        and math.isfinite(float(witness_moving_support_max.item()))
    )
    witness_pass = bool(
        witness_numeric_integrity
        and float(witness_moving_support_max.item()) > CONTACT_GATE_N
    )
    measurement_valid = (
        positive_control & buffers_ok & witness_pass & numeric_integrity
    )
    success = (
        measurement_valid & arrival & task_clear & both_close & both_lift
        & (corrected_lift > LIFT_GATE_MM) & (final_tilt < TIP_GATE_DEG)
    )
    metric_tensors = {
        **acc,
        "settle_support_fz_n": settle_tail,
        "positive_control": positive_control,
        "numeric_integrity": numeric_integrity,
        "measurement_valid": measurement_valid,
        "arrival_pass": arrival,
        "premature_jaw_contact": premature_jaw_contact,
        "collision_block": collision_block,
        "task_clear": task_clear,
        "both_jaws_close": both_close,
        "both_jaws_lift": both_lift,
        "lift_corrected_mm": corrected_lift,
        "final_tilt_deg": final_tilt,
        "success": success,
    }
    metrics = {
        key: value[:active_count].detach().cpu().numpy()
        for key, value in metric_tensors.items()
        if not isinstance(value, list)
    }
    trace_np = {key: np.stack(value, axis=0) for key, value in trace.items()}
    trace_finite_failures = {
        name: int((~np.isfinite(value)).sum())
        for name, value in trace_np.items()
        if name != "object_contact_pos_m"
        and np.issubdtype(value.dtype, np.number)
        and not np.isfinite(value).all()
    }
    count_integrity: dict[str, Any] = {}
    for name in (
        "object_raw_contact_count",
        "support_raw_contact_count",
        "self_raw_contact_count",
    ):
        value = trace_np[name]
        finite = bool(np.isfinite(value).all())
        nonnegative = bool((value >= 0).all())
        integer = bool(np.equal(value, np.round(value)).all())
        within_capacity = bool((value <= args.contact_capacity).all())
        per_sample_sum_within_capacity = bool(
            (value.sum(axis=-1) <= args.contact_capacity).all()
            if name == "object_raw_contact_count"
            else True
        )
        count_integrity[name] = {
            "dtype": str(value.dtype),
            "finite": finite,
            "nonnegative": nonnegative,
            "integer_valued": integer,
            "max_value": float(np.max(value)),
            "per_sample_sum_max": (
                float(np.max(value.sum(axis=-1)))
                if name == "object_raw_contact_count" else None
            ),
            "capacity_per_prim": int(args.contact_capacity),
            "within_capacity": within_capacity,
            "per_sample_sum_within_capacity": per_sample_sum_within_capacity,
            "pass": bool(
                finite and nonnegative and integer and within_capacity
                and per_sample_sum_within_capacity
            ),
        }
    contact_count = trace_np["object_raw_contact_count"]
    contact_position = trace_np["object_contact_pos_m"]
    conditional_contact_position_pass = bool(
        np.logical_and(
            ~np.isinf(contact_position).any(axis=-1),
            np.logical_or(
                contact_count <= 0,
                np.isfinite(contact_position).all(axis=-1),
            ),
        ).all()
    )
    trace_quaternion_norm = {
        "object": np.linalg.norm(trace_np["object_quat_wxyz"], axis=-1),
        "moving_body": np.linalg.norm(
            trace_np["moving_body_quat_wxyz"], axis=-1
        ),
        "fixed_base": np.linalg.norm(
            trace_np["fixed_base_quat_wxyz"], axis=-1
        ),
    }
    quaternion_norm_report = {
        name: {
            "min": float(np.min(value)),
            "max": float(np.max(value)),
            "max_abs_error_from_one": float(np.max(np.abs(value - 1.0))),
            "abs_tolerance": quaternion_norm_abs_tolerance,
            "pass": bool(
                np.isfinite(value).all()
                and np.max(np.abs(value - 1.0)) <= quaternion_norm_abs_tolerance
            ),
        }
        for name, value in trace_quaternion_norm.items()
    }
    metrics_finite = bool(
        all(
            np.isfinite(value).all()
            for value in metrics.values()
            if np.issubdtype(value.dtype, np.number)
        )
    )
    numeric_integrity_report = {
        "authority": "all_task_steps_all_envs_plus_derived_metrics_and_npz_trace",
        "per_env": numeric_integrity.detach().cpu().numpy().astype(bool).tolist(),
        "active_per_env": (
            numeric_integrity[:active_count].detach().cpu().numpy().astype(bool).tolist()
        ),
        "witness_environment_slot": witness_slot,
        "witness_numeric_integrity": witness_numeric_integrity,
        "failure_counts_by_check": numeric_failure_counts,
        "trace_nonfinite_failures_excluding_conditionally_allowed_contact_positions": (
            trace_finite_failures
        ),
        "contact_position_rule": (
            "Inf_forbidden; NaN_allowed_only_when_corresponding_raw_count_zero; "
            "finite_required_when_raw_count_positive"
        ),
        "conditional_contact_position_pass": conditional_contact_position_pass,
        "count_integrity": count_integrity,
        "quaternion_norm": quaternion_norm_report,
        "joint_limit_application": {
            "readback_pass": joint_limits_readback["pass"],
            "planned_applied_target_abs_tolerance_rad": (
                planned_applied_target_tolerance_rad
            ),
            "actual_position_limit_abs_tolerance_rad": (
                actual_joint_limit_tolerance_rad
            ),
            "planned_equals_applied_failure_count": numeric_failure_counts.get(
                "planned_equals_post_clamp_applied_target", -1
            ),
            "applied_inside_limit_failure_count": numeric_failure_counts.get(
                "applied_target_inside_parsed_urdf_limits", -1
            ),
            "actual_inside_limit_failure_count": numeric_failure_counts.get(
                "actual_joint_position_inside_parsed_urdf_limits", -1
            ),
        },
        "fixed_base_stability": {
            "readback_pass": fixed_base_readback["pass"],
            "position_abs_tolerance_m": fixed_base_position_tolerance_m,
            "quaternion_component_abs_tolerance_sign_invariant": (
                fixed_base_quaternion_component_tolerance
            ),
            "linear_angular_velocity_abs_tolerance": (
                fixed_base_velocity_tolerance
            ),
            "position_drift_failure_count": numeric_failure_counts.get(
                "fixed_base_position_no_drift", -1
            ),
            "orientation_drift_failure_count": numeric_failure_counts.get(
                "fixed_base_orientation_no_drift_sign_invariant", -1
            ),
            "linear_velocity_failure_count": numeric_failure_counts.get(
                "fixed_base_linear_velocity_zero", -1
            ),
            "angular_velocity_failure_count": numeric_failure_counts.get(
                "fixed_base_angular_velocity_zero", -1
            ),
        },
        "metrics_finite": metrics_finite,
        "pass_all_active": bool(numeric_integrity[:active_count].all().item()),
        "pass_all_envs": bool(numeric_integrity.all().item()),
    }
    numeric_integrity_report["pass"] = bool(
        numeric_integrity_report["pass_all_envs"]
        and not trace_finite_failures
        and conditional_contact_position_pass
        and all(row["pass"] for row in count_integrity.values())
        and all(row["pass"] for row in quaternion_norm_report.values())
        and metrics_finite
        and joint_limits_readback["pass"] is True
        and all(
            numeric_integrity_report["joint_limit_application"][key] == 0
            for key in (
                "planned_equals_applied_failure_count",
                "applied_inside_limit_failure_count",
                "actual_inside_limit_failure_count",
            )
        )
        and fixed_base_readback["pass"] is True
        and all(
            numeric_integrity_report["fixed_base_stability"][key] == 0
            for key in (
                "position_drift_failure_count",
                "orientation_drift_failure_count",
                "linear_velocity_failure_count",
                "angular_velocity_failure_count",
            )
        )
    )
    return {
        "active_count": active_count,
        "total_steps": global_step,
        "post_diagnostic_task_rebaseline": rebaseline,
        "first_task_step_freshness": first_task_step_freshness,
        "physics_clock_accounting": physics_clock_accounting,
        "metrics": metrics,
        "trace": trace_np,
        "kinematic_attach_calls": attach_calls["actual_attach_or_follow_pose_writes"],
        "disabled_attach_hook_invocations": attach_calls["disabled_hook_invocations"],
        "object_filter_map": fmap,
        "self_contact_filter_identity_reuse": filter_identity_reuse,
        "support_filter_audit": support_filter_audit,
        "self_filter_audit": self_filter_audit,
        "joint_limits_readback": joint_limits_readback,
        "fixed_base_readback": fixed_base_readback,
        "moving_bodies": list(MOVING_BODIES),
        "self_contact_bodies": list(SELF_CONTACT_BODIES),
        "nonadjacent_self_pairs": [list(pair) for pair in SELF_PAIRS],
        "adjacent_self_pair_exclusions": [
            list(pair) for pair in ADJACENT_SELF_PAIR_EXCLUSIONS
        ],
        "fixed_base_support_contact_excluded_from_task_gate": True,
        "spawn_reports": env._t3u_spawn_reports,
        "raw_contact_total_peak": raw_contact_total_peak,
        "raw_contact_capacity": raw_contact_capacity,
        "saturated_sensors": saturated_sensors,
        "contact_buffers_ok": buffers_ok,
        "numeric_integrity": numeric_integrity_report,
        "instrumentation_witness": {
            "source_trial_id": WITNESS_SOURCE_TRIAL_ID,
            "environment_slot": witness_slot,
            "excluded_from_scientific_counts": True,
            "moving_jaw_support_force_max_n": float(witness_moving_support_max.item()),
            "gate_n_strict_gt": CONTACT_GATE_N,
            "pass": witness_pass,
            "controls_deg": {
                "approach": WITNESS_Q_APPROACH_DEG.tolist(),
                "descend": WITNESS_Q_DESCEND_DEG.tolist(),
                "close": WITNESS_Q_CLOSE_DEG.tolist(),
                "lift": WITNESS_Q_LIFT_DEG.tolist(),
            },
        },
    }


def _quat_to_rot(q_wxyz: Any) -> np.ndarray:
    w, x, y, z = np.asarray(q_wxyz, dtype=np.float64)
    norm = max(float(np.linalg.norm([w, x, y, z])), 1.0e-12)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _rr_quaternion(rr: Any, q_wxyz: Any) -> Any:
    q = np.asarray(q_wxyz, dtype=np.float64)
    return rr.Quaternion(xyzw=[float(q[1]), float(q[2]), float(q[3]), float(q[0])])


def _cylinder_mesh(segments: int = 64) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    half = OBJ_HEIGHT_M / 2.0
    for z in (-half, half):
        for index in range(segments):
            angle = 2.0 * math.pi * index / segments
            vertices.append(
                [OBJ_RADIUS_M * math.cos(angle), OBJ_RADIUS_M * math.sin(angle), z]
            )
    vertices.extend([[0.0, 0.0, -half], [0.0, 0.0, half]])
    triangles: list[list[int]] = []
    for index in range(segments):
        nxt = (index + 1) % segments
        triangles.extend(
            [
                [index, nxt, segments + nxt],
                [index, segments + nxt, segments + index],
                [2 * segments, nxt, index],
                [2 * segments + 1, segments + index, segments + nxt],
            ]
        )
    return np.asarray(vertices, dtype=np.float32), np.asarray(triangles, dtype=np.uint32)


def emit_decision_snapshot(
    path: Path,
    representative: dict[str, Any],
    representative_slot: int,
    trace: dict[str, np.ndarray],
    profile: str,
) -> dict[str, Any]:
    from roarm_rl import viz_debug

    target_R = np.asarray(representative["R_link5_target"], dtype=np.float64)
    target_tcp = np.asarray(representative["tcp_grasp_m"], dtype=np.float64)
    final_body_q = trace["moving_body_quat_wxyz"][-1, representative_slot]
    final_body_p = trace["moving_body_pos_m"][-1, representative_slot]
    link5_index = MOVING_BODIES.index("link5")
    actual_q = final_body_q[link5_index]
    frames = [
        viz_debug.frame_from_axes(
            "target_link5",
            target_tcp,
            x_axis=target_R[:, 0],
            z_axis=target_R[:, 2],
            role="target",
            label="target link5: +X closure, +Y up, +Z radial",
        ),
        {
            "name": "actual_link5_final",
            "label": "actual link5 at final physics sample",
            "position": final_body_p[link5_index].tolist(),
            "quat_wxyz": actual_q.tolist(),
            "role": "actual",
            "metadata": {},
        },
        {
            "name": "actual_object_final",
            "label": "actual D29xH50 object at final physics sample",
            "position": trace["object_pos_m"][-1, representative_slot].tolist(),
            "quat_wxyz": trace["object_quat_wxyz"][-1, representative_slot].tolist(),
            "role": "object",
            "metadata": {},
        },
    ]
    status = viz_debug.snapshot(
        path,
        pairs=frames,
        prefer_viewport=False,
        title=f"p16 t3u {profile}: side-midpoint target vs actual",
        annotations=[
            "D419 sim-only side-midpoint exception",
            f"trial={representative['trial_id']} slot={representative_slot}",
            "fixed base; analytic D29xH50; no kinematic attach",
            "actual final sample is after the vertical-lift phase",
        ],
    )
    if not status.get("ok") or not path.is_file():
        raise RuntimeError(f"DECISION_SNAPSHOT_FAIL {status}")
    return status


def emit_rerun(
    paths: dict[str, Path],
    representative: dict[str, Any],
    representative_slot: int,
    trace: dict[str, np.ndarray],
    collision_vertices: dict[str, np.ndarray],
    profile: str,
) -> dict[str, Any]:
    """Emit one preregistered representative with every authoritative time step."""
    import rerun as rr
    import rerun.blueprint as rrb
    from roarm_rl.rerun_contract import validate_rerun_artifact

    if rr.__version__ != RERUN_VERSION:
        raise RuntimeError(f"RERUN_VERSION_DRIFT expected={RERUN_VERSION} actual={rr.__version__}")
    if trace["physics_step"].shape != (TOTAL_STEPS,):
        raise RuntimeError(f"RERUN_TRACE_STEP_SHAPE_DRIFT {trace['physics_step'].shape}")
    vertices, triangles = _cylinder_mesh()
    phase_names = tuple(PHASE_STEPS)
    expected: set[str] = set()

    def remember(path: str) -> str:
        expected.add(path.strip("/"))
        return path

    trial_id = representative["trial_id"]
    base = f"replay/{trial_id}"
    authoritative = profile == CANONICAL_PROFILE
    summary = (
        f"# p16 / t3u {profile}\n\n"
        f"scientific_authoritative: `{str(authoritative).lower()}`  \n"
        f"representative: `{trial_id}`, environment slot `{representative_slot}`  \n"
        "D419 sim-only side-midpoint; fixed base; analytic D29 x H50, 24.83 g; "
        "attempt3 64+64 collision jaws; no kinematic attach. Native JSON/NPZ is "
        "the numerical authority. Rerun is a Float32 inspection copy. Full moving-body "
        "transforms are logged; each collision cloud is the composed enabled collider's "
        "raw vertices in that body frame (not a visual proxy).\n"
    )
    blueprint = rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.TextDocumentView(origin="/metadata", contents="/metadata/run", name="1 | run"),
                rrb.TextLogView(origin="/events", contents="/events/**", name="2 | phases"),
                column_shares=[0.62, 0.38],
            ),
            rrb.Horizontal(
                rrb.Spatial3DView(origin="/", contents="/replay/**", name="3 | side grasp"),
                rrb.TimeSeriesView(origin="/metrics", contents="/metrics/**", name="4 | forces and pose"),
                column_shares=[0.58, 0.42],
            ),
            row_shares=[0.28, 0.72],
        ),
        auto_layout=False,
        auto_views=False,
        collapse_panels=True,
    )
    app_id = f"roarm_g0b_t3u_{profile}"
    recording_id = f"g0b_d420_t3u_{profile}"
    target_tcp = np.asarray(representative["tcp_grasp_m"], dtype=np.float32)
    target_midpoint = np.asarray(representative["antipodal_midpoint_base_m"], dtype=np.float32)
    target_R = np.asarray(representative["R_link5_target"], dtype=np.float32)
    support_lines = np.asarray(
        [[[-0.05, -0.60, SUPPORT_Z_M], [0.65, -0.60, SUPPORT_Z_M],
          [0.65, 0.60, SUPPORT_Z_M], [-0.05, 0.60, SUPPORT_Z_M],
          [-0.05, -0.60, SUPPORT_Z_M]]],
        dtype=np.float32,
    )
    with rr.RecordingStream(
        app_id, recording_id=recording_id, make_default=False, send_properties=True
    ) as recording:
        recording.save(str(paths["timeline.rrd"]), write_footer=True)
        recording.send_blueprint(blueprint, make_active=True, make_default=True)
        recording.log(
            remember("metadata/run"),
            rr.TextDocument(summary, media_type=rr.MediaType.MARKDOWN),
            static=True,
        )
        recording.log(
            remember("metadata/representative"),
            rr.TextDocument(
                json.dumps(
                    {"trial_id": trial_id, "slot": representative_slot}, sort_keys=True
                ),
                media_type=rr.MediaType.TEXT,
            ),
            static=True,
        )
        recording.log(
            remember(f"{base}/support_plane"), rr.LineStrips3D(support_lines), static=True
        )
        recording.log(
            remember(f"{base}/geometry/cylinder"),
            rr.Mesh3D(
                vertex_positions=vertices,
                triangle_indices=triangles,
                albedo_factor=[220, 155, 70, 210],
            ),
            rr.CoordinateFrame(f"{trial_id}/object"),
            static=True,
        )
        for body_index, body in enumerate(MOVING_BODIES):
            colors = [60, 155, 245] if body == "link5" else (
                [235, 70, 105] if body == "gripper_link" else [125, 135, 150]
            )
            recording.log(
                remember(f"{base}/geometry/{body}_collision_vertices"),
                rr.Points3D(
                    np.asarray(collision_vertices[body], dtype=np.float32),
                    colors=[colors],
                    radii=[0.00035 if body in JAW_BODIES else 0.0005],
                ),
                rr.CoordinateFrame(f"{trial_id}/{body}"),
                static=True,
            )
        recording.log(
            remember(f"{base}/target/tcp"),
            rr.Points3D([target_tcp], colors=[[255, 45, 35]], radii=[0.0028], labels=["target TCP"]),
            rr.CoordinateFrame("world"),
            static=True,
        )
        recording.log(
            remember(f"{base}/target/antipodal_midpoint"),
            rr.Points3D([target_midpoint], colors=[[255, 225, 45]], radii=[0.0028], labels=["side midpoint"]),
            rr.CoordinateFrame("world"),
            static=True,
        )
        for axis_index, (axis_name, color) in enumerate(
            (("closure_x", [45, 225, 85]), ("vertical_up_y", [50, 130, 250]),
             ("approach_z", [250, 70, 45]))
        ):
            recording.log(
                remember(f"{base}/target/{axis_name}"),
                rr.Arrows3D(
                    origins=[target_midpoint],
                    vectors=[target_R[:, axis_index] * 0.045],
                    colors=[color],
                    radii=[0.00075],
                ),
                rr.CoordinateFrame("world"),
                static=True,
            )

        previous_phase = None
        for step_index in range(TOTAL_STEPS):
            physics_step = int(trace["physics_step"][step_index])
            sim_time = float(trace["sim_time_s"][step_index])
            phase_id = int(trace["phase_id"][step_index])
            recording.reset_time()
            recording.set_time("physics_step", sequence=physics_step)
            recording.set_time("sim_time_s", duration=sim_time)
            if phase_id != previous_phase:
                recording.log(
                    remember("events/phase"),
                    rr.TextLog(
                        f"phase={phase_names[phase_id]} starts at step={physics_step}",
                        level=rr.TextLogLevel.INFO,
                    ),
                )
                previous_phase = phase_id
            object_pos = trace["object_pos_m"][step_index, representative_slot]
            object_quat = trace["object_quat_wxyz"][step_index, representative_slot]
            recording.log(
                remember(f"{base}/transforms/object"),
                rr.Transform3D(
                    translation=object_pos,
                    rotation=_rr_quaternion(rr, object_quat),
                    parent_frame="world",
                    child_frame=f"{trial_id}/object",
                ),
            )
            for body_index, body in enumerate(MOVING_BODIES):
                body_pos = trace["moving_body_pos_m"][step_index, representative_slot, body_index]
                body_quat = trace["moving_body_quat_wxyz"][step_index, representative_slot, body_index]
                recording.log(
                    remember(f"{base}/transforms/{body}"),
                    rr.Transform3D(
                        translation=body_pos,
                        rotation=_rr_quaternion(rr, body_quat),
                        parent_frame="world",
                        child_frame=f"{trial_id}/{body}",
                    ),
                )
                support_force = trace["moving_link_support_force_w_n"][
                    step_index, representative_slot, body_index
                ]
                entity = remember(f"{base}/contacts/{body}_support_force")
                if float(np.linalg.norm(support_force)) > 0.0:
                    recording.log(
                        entity,
                        rr.Arrows3D(
                            origins=[body_pos], vectors=[support_force * 0.005],
                            radii=[0.00055], labels=[f"{body}/support x0.005 m/N"],
                        ),
                        rr.CoordinateFrame("world"),
                    )
                else:
                    recording.log(
                        entity,
                        rr.Arrows3D(
                            origins=np.empty((0, 3), dtype=np.float32),
                            vectors=np.empty((0, 3), dtype=np.float32),
                        ),
                        rr.CoordinateFrame("world"),
                    )
            for pair_index, (body_a, body_b) in enumerate(SELF_PAIRS):
                self_force = trace["self_contact_force_w_n"][
                    step_index, representative_slot, pair_index
                ]
                origin = trace["self_contact_body_pos_m"][
                    step_index,
                    representative_slot,
                    SELF_CONTACT_BODIES.index(body_a),
                ]
                entity = remember(f"{base}/contacts/self_{body_a}__{body_b}_force")
                if float(np.linalg.norm(self_force)) > 0.0:
                    recording.log(
                        entity,
                        rr.Arrows3D(
                            origins=[origin], vectors=[self_force * 0.005],
                            radii=[0.0005],
                            labels=[f"{body_a}/{body_b} force at {body_a} origin"],
                        ),
                        rr.CoordinateFrame("world"),
                    )
                else:
                    recording.log(
                        entity,
                        rr.Arrows3D(
                            origins=np.empty((0, 3), dtype=np.float32),
                            vectors=np.empty((0, 3), dtype=np.float32),
                        ),
                        rr.CoordinateFrame("world"),
                    )
            tcp = trace["tcp_pos_m"][step_index, representative_slot]
            link5_q = trace["moving_body_quat_wxyz"][
                step_index, representative_slot, MOVING_BODIES.index("link5")
            ]
            actual_R = _quat_to_rot(link5_q)
            recording.log(
                remember(f"{base}/actual/tcp"),
                rr.Points3D([tcp], colors=[[35, 190, 245]], radii=[0.0023]),
                rr.CoordinateFrame("world"),
            )
            for axis_index, axis_name in enumerate(("closure_x", "vertical_up_y", "approach_z")):
                recording.log(
                    remember(f"{base}/actual/{axis_name}"),
                    rr.Arrows3D(
                        origins=[tcp], vectors=[actual_R[:, axis_index] * 0.04], radii=[0.0006]
                    ),
                    rr.CoordinateFrame("world"),
                )
            for contact_index, label in enumerate(("support", *MOVING_BODIES)):
                force = trace["object_force_w_n"][step_index, representative_slot, contact_index]
                point = trace["object_contact_pos_m"][step_index, representative_slot, contact_index]
                point_entity = remember(f"{base}/contacts/object_{label}_point")
                force_entity = remember(f"{base}/contacts/object_{label}_force")
                if np.isfinite(point).all() and float(np.linalg.norm(force)) > 0.0:
                    recording.log(
                        point_entity,
                        rr.Points3D([point], radii=[0.0017], labels=[f"object/{label}"]),
                        rr.CoordinateFrame("world"),
                    )
                    recording.log(
                        force_entity,
                        rr.Arrows3D(
                            origins=[point], vectors=[force * 0.005], radii=[0.00055]
                        ),
                        rr.CoordinateFrame("world"),
                    )
                else:
                    recording.log(
                        point_entity, rr.Points3D(np.empty((0, 3), dtype=np.float32)),
                        rr.CoordinateFrame("world"),
                    )
                    recording.log(
                        force_entity,
                        rr.Arrows3D(
                            origins=np.empty((0, 3), dtype=np.float32),
                            vectors=np.empty((0, 3), dtype=np.float32),
                        ),
                        rr.CoordinateFrame("world"),
                    )
            metric_values = {
                "phase_id": float(phase_id),
                "object_z_mm": float(object_pos[2] * 1000.0),
                "object_tilt_deg": float(trace["object_tilt_deg"][step_index, representative_slot]),
                "q5_actual_deg": float(trace["joint_pos_deg"][step_index, representative_slot, 5]),
                "q5_target_deg": float(trace["joint_target_deg"][step_index, representative_slot, 5]),
                "fixed_jaw_force_n": float(np.linalg.norm(trace["object_force_w_n"][step_index, representative_slot, 1 + MOVING_BODIES.index("link5")])),
                "moving_jaw_force_n": float(np.linalg.norm(trace["object_force_w_n"][step_index, representative_slot, 1 + MOVING_BODIES.index("gripper_link")])),
            }
            for name, value in metric_values.items():
                recording.log(remember(f"metrics/{trial_id}/{name}"), rr.Scalars([value]))
            recording.log(
                remember("metrics/instrumentation_witness/moving_jaw_support_force_n"),
                rr.Scalars(
                    [
                        float(
                            np.linalg.norm(
                                trace["witness_moving_support_force_w_n"][step_index]
                            )
                        )
                    ]
                ),
            )
        recording.flush(timeout_sec=180.0)
    blueprint.save(app_id, str(paths["timeline.rbl"]))

    components: dict[str, list[str]] = {
        "metadata/run": ["TextDocument:text"],
        "metadata/representative": ["TextDocument:text"],
        "events/phase": ["TextLog:level", "TextLog:text"],
        f"{base}/support_plane": ["LineStrips3D:strips"],
        f"{base}/geometry/cylinder": ["Mesh3D:vertex_positions", "Mesh3D:triangle_indices"],
        f"{base}/target/tcp": ["Points3D:positions"],
        f"{base}/target/antipodal_midpoint": ["Points3D:positions"],
        f"{base}/actual/tcp": ["Points3D:positions"],
    }
    transform_components = [
        "Transform3D:translation", "Transform3D:quaternion",
        "Transform3D:parent_frame", "Transform3D:child_frame",
    ]
    components[f"{base}/transforms/object"] = transform_components
    for body in MOVING_BODIES:
        components[f"{base}/geometry/{body}_collision_vertices"] = [
            "Points3D:positions", "CoordinateFrame:frame"
        ]
        components[f"{base}/transforms/{body}"] = transform_components
        components[f"{base}/contacts/{body}_support_force"] = [
            "Arrows3D:origins", "Arrows3D:vectors", "CoordinateFrame:frame"
        ]
    for body_a, body_b in SELF_PAIRS:
        components[f"{base}/contacts/self_{body_a}__{body_b}_force"] = [
            "Arrows3D:origins", "Arrows3D:vectors", "CoordinateFrame:frame"
        ]
    for axis_name in ("closure_x", "vertical_up_y", "approach_z"):
        components[f"{base}/target/{axis_name}"] = ["Arrows3D:origins", "Arrows3D:vectors"]
        components[f"{base}/actual/{axis_name}"] = ["Arrows3D:origins", "Arrows3D:vectors"]
    for label in ("support", *MOVING_BODIES):
        components[f"{base}/contacts/object_{label}_point"] = ["Points3D:positions"]
        components[f"{base}/contacts/object_{label}_force"] = ["Arrows3D:origins", "Arrows3D:vectors"]
    for name in (
        "phase_id", "object_z_mm", "object_tilt_deg", "q5_actual_deg",
        "q5_target_deg", "fixed_jaw_force_n", "moving_jaw_force_n",
    ):
        components[f"metrics/{trial_id}/{name}"] = ["Scalars:scalars"]
    components["metrics/instrumentation_witness/moving_jaw_support_force_n"] = [
        "Scalars:scalars"
    ]
    validation = validate_rerun_artifact(
        paths["timeline.rrd"],
        expected_entity_paths=sorted(expected),
        exact_entity_paths=sorted(expected),
        expected_timeline_names=["physics_step", "sim_time_s"],
        exact_timeline_names=["blueprint", "log_time", "physics_step", "sim_time_s"],
        expected_entity_components=components,
        blueprint_path=paths["timeline.rbl"],
        screenshot_path=paths["inspection.png"],
        screenshot_window_size="2400x1400",
        expected_version=RERUN_VERSION,
        cli_path=RERUN_CLI,
        timeout_s=600.0,
    )
    validation["representative_binding"] = {
        "trial_id": trial_id,
        "environment_slot": representative_slot,
        "physics_steps": TOTAL_STEPS,
        "candidate_index": int(representative["candidate_index"]),
        "pinch_offset_index": int(representative["pinch_offset_index"]),
        "nominal_offset_required": True,
        "pass": bool(
            representative["candidate_index"] == PREFLIGHT_CANDIDATE_INDEX
            and representative["pinch_offset_index"] == 0
            and TOTAL_STEPS == len(trace["physics_step"])
        ),
    }
    if not validation["representative_binding"]["pass"]:
        validation["pass"] = False
        validation.setdefault("errors", []).append("representative binding drift")
    write_json_x(paths["rerun_validation.json"], validation)
    return {
        "technical_pass": bool(validation.get("pass")),
        "errors": validation.get("errors", []),
        "representative_binding": validation["representative_binding"],
        "manual_visual_inspection": "PENDING_USER_OR_ROOT_REVIEW_DO_NOT_CLAIM_COMPLETE",
    }


def classify(
    metrics: dict[str, np.ndarray],
) -> tuple[str, list[str], dict[str, Any]]:
    """Apply one exact causal ladder to rows and the population verdict."""
    arrays = {
        name: np.asarray(metrics[name])
        for name in (
            "measurement_valid", "success", "arrival_pass",
            "premature_jaw_contact", "collision_block", "both_jaws_close",
            "both_jaws_lift", "lift_corrected_mm", "final_tilt_deg",
        )
    }
    n = len(arrays["success"])
    bool_names = (
        "measurement_valid", "success", "arrival_pass",
        "premature_jaw_contact", "collision_block", "both_jaws_close",
        "both_jaws_lift",
    )
    for name in bool_names:
        arrays[name] = arrays[name].astype(bool)
        if arrays[name].shape != (n,):
            raise RuntimeError(f"CLASSIFICATION_METRIC_SHAPE_FAIL {name} {arrays[name].shape}")
    for name in ("lift_corrected_mm", "final_tilt_deg"):
        arrays[name] = arrays[name].astype(np.float64)
        if arrays[name].shape != (n,) or not np.isfinite(arrays[name]).all():
            raise RuntimeError(f"CLASSIFICATION_FLOAT_METRIC_INVALID {name}")

    unassigned = np.ones(n, dtype=bool)
    branch_specs: list[tuple[str, str, np.ndarray]] = [
        ("measurement_invalid", "MEASUREMENT_INVALID", ~arrays["measurement_valid"]),
        ("success", "SIDE_MIDPOINT_GRASP_PASS_IN_FIXED_SIM_CONTROLS", arrays["success"]),
        ("tracking_or_arrival_fail", "TRACKING_OR_ARRIVAL_GATE_FAIL", ~arrays["arrival_pass"]),
        (
            "premature_jaw_contact",
            "PREMATURE_JAW_CONTACT_BLOCKS_SIDE_GRASP",
            arrays["premature_jaw_contact"],
        ),
        (
            "nonjaw_support_or_self_collision",
            "NONJAW_OR_SUPPORT_COLLISION_BLOCKS_SIDE_GRASP",
            arrays["collision_block"],
        ),
        ("no_bilateral_close", "NO_BILATERAL_SIDE_CONTACT", ~arrays["both_jaws_close"]),
        (
            "bilateral_lost_before_lift",
            "BILATERAL_CONTACT_LOST_BEFORE_LIFT",
            ~arrays["both_jaws_lift"],
        ),
        (
            "corrected_lift_gate_fail",
            "BILATERAL_CONTACT_BUT_NO_CORRECTED_LIFT",
            arrays["lift_corrected_mm"] <= LIFT_GATE_MM,
        ),
        (
            "lifted_but_tipped",
            "OBJECT_LIFTED_BUT_TIPPED",
            arrays["final_tilt_deg"] >= TIP_GATE_DEG,
        ),
        ("other_exact_gate_fail", "OTHER_EXACT_GATE_FAIL", np.ones(n, dtype=bool)),
    ]
    labels = [""] * n
    masks: dict[str, list[bool]] = {}
    counts: dict[str, int] = {}
    verdict_by_label: dict[str, str] = {}
    for label, branch_verdict, raw_mask in branch_specs:
        assigned = unassigned & np.asarray(raw_mask, dtype=bool)
        masks[label] = assigned.tolist()
        counts[label] = int(assigned.sum())
        verdict_by_label[label] = branch_verdict
        for index in np.flatnonzero(assigned):
            labels[int(index)] = label
        unassigned &= ~assigned
    if unassigned.any() or any(not label for label in labels):
        raise RuntimeError("CLASSIFICATION_PARTITION_INCOMPLETE")

    # Invalid authority blocks every scientific claim; otherwise one success
    # establishes local existence.  With no success, report the deepest
    # progress reached by any row, not the earliest failure seen in some other
    # row.  This preserves the per-row causal partition while preventing one
    # weak row from hiding nine rows that reached bilateral lift contact.
    if counts["measurement_invalid"]:
        selected_label = "measurement_invalid"
    elif counts["success"]:
        selected_label = "success"
    else:
        selected_label = next(
            label
            for label, _verdict, _mask in reversed(branch_specs[2:])
            if counts[label]
        )
    verdict = verdict_by_label[selected_label]
    summary = {
        "precedence": [label for label, _verdict, _mask in branch_specs],
        "verdict_by_branch": verdict_by_label,
        "branch_masks": masks,
        "branch_counts": counts,
        "population_selection_rule": (
            "measurement_invalid_any__else_success_any__else_deepest_populated_failure"
        ),
        "deepest_populated_failure_branch": next(
            (
                label
                for label, _verdict, _mask in reversed(branch_specs[2:])
                if counts[label]
            ),
            None,
        ),
        "selected_branch": selected_label,
        "selected_verdict": verdict,
        "row_count": n,
        "partition_exactly_once": all(
            sum(bool(masks[label][index]) for label in masks) == 1
            for index in range(n)
        ),
    }
    return verdict, labels, summary


def _matrix_from_pose(pos: Any, quat_wxyz: Any) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = _quat_to_rot(quat_wxyz)
    matrix[:3, 3] = np.asarray(pos, dtype=np.float64)
    return matrix


def _passing_preflight_dependency_paths() -> dict[str, Path]:
    """Return the exact preflight evidence set that admits the canonical run."""
    preflight_prefix = f"{TAG}_{PREFLIGHT_PROFILE}"
    preflight_paths = run_paths(preflight_prefix)
    core = {
        name: preflight_paths[name]
        for name in (
            "results.json", "plan.json", "trace.npz", "timeline.rrd",
            "timeline.rbl", "rerun_validation.json", "decision_snapshot.png",
            "inspection.png", "rgb_frames_manifest.json", "side_grasp.mp4",
            "script.py.txt", "argv.txt", "phase.jsonl", "render_phase.jsonl",
            "preclose_sentinel.json", "exit_status.txt",
            "terminal_attestation.json", "manual_visual_inspection.json",
        )
    }
    external = {
        "stdout": CASE_DIR / f"{preflight_prefix}_stdout.log",
        "supervisor_launcher": CASE_DIR / f"{preflight_prefix}_supervisor_launcher.log",
        "supervisor_pid": CASE_DIR / f"{preflight_prefix}_supervisor_pid.txt",
        "physics_python_pid": CASE_DIR / f"{preflight_prefix}_physics_python_pid.txt",
        "render_python_pid": CASE_DIR / f"{preflight_prefix}_render_python_pid.txt",
        "pgid": CASE_DIR / f"{preflight_prefix}_pgid.txt",
        "supervisor_contract": CASE_DIR / f"{preflight_prefix}_supervisor_contract.json",
        "supervisor_outcome": CASE_DIR / f"{preflight_prefix}_supervisor_outcome.json",
        "gpu_before": CASE_DIR / f"{preflight_prefix}_nvidia_smi_before.csv",
        "gpu_supervisor_end": CASE_DIR / f"{preflight_prefix}_nvidia_smi_supervisor_end.csv",
        "gpu_after": CASE_DIR / f"{preflight_prefix}_nvidia_smi_after.csv",
    }
    return {**core, **external}


def profile_decision_dependency_paths(
    profile: str,
    pinned_local_sources: dict[str, Any],
    asset: dict[str, Any],
) -> dict[str, Path]:
    """Single source for every byte that can affect a physics decision."""
    if profile not in EXECUTABLE_PROFILES:
        raise RuntimeError(f"DEPENDENCY_PROFILE_INVALID {profile}")
    prereg = PREFLIGHT_PREREG if profile == PREFLIGHT_PROFILE else CANONICAL_PREREG
    dependency_paths: dict[str, Path] = {
        "p16_source": Path(__file__).resolve(),
        "p14": P14_PATH,
        "p10": P10_PATH,
        "p15": P15_PATH,
        "p16_supervisor": SUPERVISOR_PATH,
        "p15_prereg": P15_PREREG_PATH,
        "jaw": JAW_PATH,
        "urdf": URDF_PATH,
        "p16_selected_prereg": prereg,
        "p16_preflight1_historical_prereg": PREFLIGHT1_PREREG,
        "p16_preflight1_historical_failure": PREFLIGHT1_FAILURE_PATH,
        "p16_preflight1_historical_supervisor_outcome": PREFLIGHT1_OUTCOME_PATH,
        "p16_preflight2_historical_source": PREFLIGHT2_SOURCE_PATH,
        "p16_preflight2_historical_supervisor": PREFLIGHT2_SUPERVISOR_PATH,
        "p16_preflight2_historical_prereg": PREFLIGHT2_PREREG,
        "p16_preflight2_historical_failure": PREFLIGHT2_FAILURE_PATH,
        "p16_preflight2_historical_supervisor_outcome": PREFLIGHT2_OUTCOME_PATH,
        "p16_preflight2_historical_phase": PREFLIGHT2_PHASE_PATH,
        "p16_preflight2_historical_exit_status": PREFLIGHT2_EXIT_STATUS_PATH,
        "p16_preflight2_historical_terminal_attestation": (
            PREFLIGHT2_TERMINAL_ATTESTATION_PATH
        ),
        "p16_retired_canonical_prereg": RETIRED_CANONICAL_PREREG,
        "p16_preflight3_historical_source": PREFLIGHT3_SOURCE_PATH,
        "p16_preflight3_historical_supervisor": PREFLIGHT3_SUPERVISOR_PATH,
        "p16_preflight3_historical_prereg": PREFLIGHT3_PREREG,
        "p16_preflight3_historical_canonical_prereg": (
            PREFLIGHT3_CANONICAL_PREREG
        ),
        "p16_preflight3_historical_zero_launcher": PREFLIGHT3_LAUNCHER_PATH,
        "p16_preflight3_posthoc_audit_failure": (
            PREFLIGHT3_POSTHOC_AUDIT_FAILURE_PATH
        ),
        "p16_preflight4_historical_source": PREFLIGHT4_SOURCE_PATH,
        "p16_preflight4_historical_supervisor": PREFLIGHT4_SUPERVISOR_PATH,
        "p16_preflight4_historical_prereg": PREFLIGHT4_PREREG,
        "p16_preflight4_historical_canonical_prereg": (
            PREFLIGHT4_CANONICAL_PREREG
        ),
        "p16_preflight4_historical_failure": PREFLIGHT4_FAILURE_PATH,
        "p16_preflight4_historical_launcher": PREFLIGHT4_LAUNCHER_PATH,
        "p16_preflight5_historical_source": PREFLIGHT5_SOURCE_PATH,
        "p16_preflight5_historical_supervisor": PREFLIGHT5_SUPERVISOR_PATH,
        "p16_preflight5_historical_canonical_prereg": (
            PREFLIGHT5_CANONICAL_PREREG
        ),
        **{
            f"p16_preflight5_historical_{suffix}": (
                CASE_DIR / f"t3u_side_preflight5_{suffix}"
            )
            for suffix in PREFLIGHT5_EVIDENCE_SHA256
        },
        "p16_preflight6_historical_source": PREFLIGHT6_SOURCE_PATH,
        "p16_preflight6_historical_supervisor": PREFLIGHT6_SUPERVISOR_PATH,
        "p16_preflight6_historical_prereg": PREFLIGHT6_PREREG,
        "p16_preflight6_historical_canonical_prereg": (
            PREFLIGHT6_CANONICAL_PREREG
        ),
        **{
            f"p16_preflight6_historical_{suffix}": (
                CASE_DIR / f"t3u_side_preflight6_{suffix}"
            )
            for suffix in PREFLIGHT6_EVIDENCE_SHA256
        },
        "p16_preflight7_historical_source": PREFLIGHT7_SOURCE_PATH,
        "p16_preflight7_historical_supervisor": PREFLIGHT7_SUPERVISOR_PATH,
        "p16_preflight7_historical_prereg": PREFLIGHT7_PREREG,
        "p16_preflight7_historical_canonical_prereg": PREFLIGHT7_CANONICAL_PREREG,
        **{
            f"p16_preflight7_historical_{suffix}": (
                CASE_DIR / f"t3u_side_preflight7_{suffix}"
            )
            for suffix in PREFLIGHT7_EVIDENCE_SHA256
        },
        "p16_preflight8_historical_source": PREFLIGHT8_SOURCE_PATH,
        "p16_preflight8_historical_supervisor": PREFLIGHT8_SUPERVISOR_PATH,
        "p16_preflight8_historical_prereg": PREFLIGHT8_PREREG,
        "p16_preflight8_historical_canonical_prereg": PREFLIGHT8_CANONICAL_PREREG,
        **{
            f"p16_preflight8_historical_{suffix}": (
                CASE_DIR / f"t3u_side_preflight8_{suffix}"
            )
            for suffix in PREFLIGHT8_EVIDENCE_SHA256
        },
        "p16_preflight9_historical_source": PREFLIGHT9_SOURCE_PATH,
        "p16_preflight9_historical_supervisor": PREFLIGHT9_SUPERVISOR_PATH,
        "p16_preflight9_historical_prereg": PREFLIGHT9_PREREG,
        "p16_preflight9_historical_canonical_prereg": PREFLIGHT9_CANONICAL_PREREG,
        **{
            f"p16_preflight9_historical_{suffix}": (
                CASE_DIR / f"t3u_side_preflight9_{suffix}"
            )
            for suffix in PREFLIGHT9_EVIDENCE_SHA256
        },
        "p16_preflight_prereg": PREFLIGHT_PREREG,
        "p16_canonical_prereg": CANONICAL_PREREG,
        "p15_candidates": CANDIDATES_PATH.resolve(),
        "p15_config": P15_CONFIG_PATH,
        "p15_rerun_validation": P15_RERUN_VALIDATION_PATH,
        "p15_inspection": P15_INSPECTION_PATH,
        "p15_manual_visual": P15_MANUAL_VISUAL_PATH,
        "p15_exit_status": P15_EXIT_STATUS_PATH,
        "p15_stdout": P15_STDOUT_PATH,
        "p15_pid_record": P15_PID_PATH,
        "witness_results": WITNESS_RESULTS_PATH,
        "witness_plan": WITNESS_PLAN_PATH,
        **{
            f"p15_output_{name}": path
            for name, path in P15_BOUND_OUTPUT_PATHS.items()
        },
        **{
            f"p14_local_{name}": Path(row["path"])
            for name, row in pinned_local_sources.items()
        },
        **{
            f"attempt3_usd:{row['relative_path']}": Path(row["path"])
            for row in asset["expected_recursive_composition_layers"]
        },
    }
    if profile == CANONICAL_PROFILE:
        dependency_paths.update(
            {
                f"passing_preflight:{name}": path
                for name, path in _passing_preflight_dependency_paths().items()
            }
        )
    return dependency_paths


def render_dependency_snapshot(
    profile: str,
) -> tuple[dict[str, Path], dict[str, str]]:
    """Rehash the full profile-specific physics decision set for rendering."""
    if P15_FAILURE_PATH.exists():
        raise RuntimeError("RENDER_P15_FAILURE_MARKER_APPEARED")
    if profile == CANONICAL_PROFILE:
        preflight_paths = run_paths(f"{TAG}_{PREFLIGHT_PROFILE}")
        preflight_supervisor_failure = (
            CASE_DIR / f"{TAG}_{PREFLIGHT_PROFILE}_supervisor_failure.json"
        )
        if (
            preflight_paths["failure.json"].exists()
            or preflight_paths["render_failure.json"].exists()
            or preflight_supervisor_failure.exists()
        ):
            raise RuntimeError("RENDER_PASSING_PREFLIGHT_FAILURE_MARKER_APPEARED")
    p14 = load_module("p16_render_dependency_p14", P14_PATH)
    if sha256_file(P14_PATH) != P14_SHA256:
        raise RuntimeError("RENDER_P14_PIN_DRIFT")
    pinned_local = p14._verify_pinned_local_sources()
    p10 = p14._import_p10()
    asset = p14._asset_gate(p10)
    dependency_paths = profile_decision_dependency_paths(profile, pinned_local, asset)
    hashes = {name: sha256_file(path) for name, path in dependency_paths.items()}
    exact_expected = {
        "p16_supervisor": SUPERVISOR_SHA256,
        "p16_preflight_prereg": PREFLIGHT_PREREG_SHA256,
        "p16_canonical_prereg": CANONICAL_PREREG_SHA256,
        "p16_preflight2_historical_source": PREFLIGHT2_SOURCE_SHA256,
        "p16_preflight2_historical_supervisor": PREFLIGHT2_SUPERVISOR_SHA256,
        "p16_preflight2_historical_prereg": PREFLIGHT2_PREREG_SHA256,
        "p16_preflight2_historical_failure": PREFLIGHT2_FAILURE_SHA256,
        "p16_preflight2_historical_supervisor_outcome": PREFLIGHT2_OUTCOME_SHA256,
        "p16_preflight2_historical_phase": PREFLIGHT2_PHASE_SHA256,
        "p16_preflight2_historical_exit_status": PREFLIGHT2_EXIT_STATUS_SHA256,
        "p16_preflight2_historical_terminal_attestation": (
            PREFLIGHT2_TERMINAL_ATTESTATION_SHA256
        ),
        "p16_retired_canonical_prereg": RETIRED_CANONICAL_PREREG_SHA256,
        "p16_preflight3_historical_source": PREFLIGHT3_SOURCE_SHA256,
        "p16_preflight3_historical_supervisor": PREFLIGHT3_SUPERVISOR_SHA256,
        "p16_preflight3_historical_prereg": PREFLIGHT3_PREREG_SHA256,
        "p16_preflight3_historical_canonical_prereg": (
            PREFLIGHT3_CANONICAL_PREREG_SHA256
        ),
        "p16_preflight3_historical_zero_launcher": PREFLIGHT3_LAUNCHER_SHA256,
        "p16_preflight3_posthoc_audit_failure": (
            PREFLIGHT3_POSTHOC_AUDIT_FAILURE_SHA256
        ),
        "p16_preflight4_historical_source": PREFLIGHT4_SOURCE_SHA256,
        "p16_preflight4_historical_supervisor": PREFLIGHT4_SUPERVISOR_SHA256,
        "p16_preflight4_historical_prereg": PREFLIGHT4_PREREG_SHA256,
        "p16_preflight4_historical_canonical_prereg": (
            PREFLIGHT4_CANONICAL_PREREG_SHA256
        ),
        "p16_preflight4_historical_failure": PREFLIGHT4_FAILURE_SHA256,
        "p16_preflight4_historical_launcher": PREFLIGHT4_LAUNCHER_SHA256,
        "p16_preflight5_historical_source": PREFLIGHT5_SOURCE_SHA256,
        "p16_preflight5_historical_supervisor": PREFLIGHT5_SUPERVISOR_SHA256,
        "p16_preflight5_historical_canonical_prereg": (
            PREFLIGHT5_CANONICAL_PREREG_SHA256
        ),
        **{
            f"p16_preflight5_historical_{suffix}": expected_sha
            for suffix, expected_sha in PREFLIGHT5_EVIDENCE_SHA256.items()
        },
        "p16_preflight6_historical_source": PREFLIGHT6_SOURCE_SHA256,
        "p16_preflight6_historical_supervisor": PREFLIGHT6_SUPERVISOR_SHA256,
        "p16_preflight6_historical_prereg": PREFLIGHT6_PREREG_SHA256,
        "p16_preflight6_historical_canonical_prereg": (
            PREFLIGHT6_CANONICAL_PREREG_SHA256
        ),
        **{
            f"p16_preflight6_historical_{suffix}": expected_sha
            for suffix, expected_sha in PREFLIGHT6_EVIDENCE_SHA256.items()
        },
        "p16_preflight7_historical_source": PREFLIGHT7_SOURCE_SHA256,
        "p16_preflight7_historical_supervisor": PREFLIGHT7_SUPERVISOR_SHA256,
        "p16_preflight7_historical_prereg": PREFLIGHT7_PREREG_SHA256,
        "p16_preflight7_historical_canonical_prereg": (
            PREFLIGHT7_CANONICAL_PREREG_SHA256
        ),
        **{
            f"p16_preflight7_historical_{suffix}": expected_sha
            for suffix, expected_sha in PREFLIGHT7_EVIDENCE_SHA256.items()
        },
        "p16_preflight8_historical_source": PREFLIGHT8_SOURCE_SHA256,
        "p16_preflight8_historical_supervisor": PREFLIGHT8_SUPERVISOR_SHA256,
        "p16_preflight8_historical_prereg": PREFLIGHT8_PREREG_SHA256,
        "p16_preflight8_historical_canonical_prereg": (
            PREFLIGHT8_CANONICAL_PREREG_SHA256
        ),
        **{
            f"p16_preflight8_historical_{suffix}": expected_sha
            for suffix, expected_sha in PREFLIGHT8_EVIDENCE_SHA256.items()
        },
        "p16_preflight9_historical_source": PREFLIGHT9_SOURCE_SHA256,
        "p16_preflight9_historical_supervisor": PREFLIGHT9_SUPERVISOR_SHA256,
        "p16_preflight9_historical_prereg": PREFLIGHT9_PREREG_SHA256,
        "p16_preflight9_historical_canonical_prereg": (
            PREFLIGHT9_CANONICAL_PREREG_SHA256
        ),
        **{
            f"p16_preflight9_historical_{suffix}": expected_sha
            for suffix, expected_sha in PREFLIGHT9_EVIDENCE_SHA256.items()
        },
        "p14": P14_SHA256,
        "p10": P10_SHA256,
        "p15": P15_SHA256,
        "p15_prereg": P15_PREREG_SHA256,
        "p15_candidates": P15_CANDIDATES_SHA256,
        "jaw": JAW_SHA256,
        "urdf": URDF_SHA256,
        **{
            f"p14_local_{name}": row["sha256"]
            for name, row in pinned_local.items()
        },
        **{
            f"attempt3_usd:{row['relative_path']}": row["sha256"]
            for row in asset["expected_recursive_composition_layers"]
        },
    }
    bad = {
        name: {"expected": expected, "actual": hashes.get(name)}
        for name, expected in exact_expected.items()
        if hashes.get(name) != expected
    }
    if bad:
        raise RuntimeError(f"RENDER_DEPENDENCY_START_PIN_FAIL {bad}")
    return dependency_paths, hashes


def render_physics_finalize_cross_bind(
    results: dict[str, Any], render_start_hashes: dict[str, str]
) -> dict[str, Any]:
    """Project physics-final evidence onto every render decision dependency."""
    provenance = results.get("provenance", {})
    finalize = provenance.get("dependency_hashes_at_finalize", {})
    projection: dict[str, Any] = {}
    for render_name, render_sha in render_start_hashes.items():
        evidence_key = f"dependency_hashes_at_finalize.{render_name}"
        physics_sha = finalize.get(render_name)
        projection[render_name] = {
            "physics_evidence_key": evidence_key,
            "physics_finalize_sha256": physics_sha,
            "render_start_sha256": render_sha,
            "pass": isinstance(physics_sha, str) and physics_sha == render_sha,
        }
    return {
        "projection": projection,
        "exact_render_dependency_key_set": bool(
            set(projection) == set(render_start_hashes)
        ),
        "pass": bool(
            projection
            and set(projection) == set(render_start_hashes)
            and all(row["pass"] for row in projection.values())
            and set(finalize) == set(render_start_hashes)
            and finalize == render_start_hashes
            and provenance.get("dependency_hashes_equal") is True
            and provenance.get("dependency_hashes_at_start") == finalize
            and provenance.get("source_stable") is True
        ),
    }


def record_render_failure_evidence(
    profile: str, paths: dict[str, Path], exc: BaseException
) -> None:
    """Write one forward-only render failure record before Kit close can exit 0."""
    if paths["render_failure.json"].exists():
        return
    failure = {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
        "phase": "render_trace",
        "profile": profile,
        "source_sha256": sha256_file(Path(__file__)),
    }
    write_json_x(paths["render_failure.json"], failure)
    append_phase(
        paths["render_phase.jsonl"],
        "render_failure",
        type=type(exc).__name__,
        message=str(exc),
        failure_sha256=sha256_file(paths["render_failure.json"]),
    )


def render_trace_mode(profile: str) -> int:
    """Replay the frozen NPZ in RTX with no physics scene or physics-step call."""
    prefix = f"{TAG}_{profile}"
    paths = run_paths(prefix)
    frame_dir = CASE_DIR / f"{prefix}_rgb_frames"
    if (
        paths["rgb_frames_manifest.json"].exists()
        or paths["side_grasp.mp4"].exists()
        or paths["render_phase.jsonl"].exists()
        or paths["render_failure.json"].exists()
        or frame_dir.exists()
    ):
        raise RuntimeError(
            "RENDER_G0_OUTPUT_EXISTS "
            f"manifest={paths['rgb_frames_manifest.json'].exists()} "
            f"mp4={paths['side_grasp.mp4'].exists()} "
            f"render_phase={paths['render_phase.jsonl'].exists()} "
            f"render_failure={paths['render_failure.json'].exists()} "
            f"frame_dir={frame_dir.exists()}"
        )
    required = {
        name: paths[name]
        for name in (
            "results.json", "plan.json", "trace.npz", "script.py.txt",
            "preclose_sentinel.json", "timeline.rrd", "rerun_validation.json",
        )
    }
    missing = [f"{name}:{path}" for name, path in required.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"RENDER_SOURCE_ARTIFACT_MISSING {missing}")
    source_sha = sha256_file(Path(__file__))
    if sha256_file(paths["script.py.txt"]) != source_sha:
        raise RuntimeError("RENDER_FROZEN_SOURCE_MISMATCH")
    results = json.loads(paths["results.json"].read_text())
    if results.get("provenance", {}).get("source_sha256") != source_sha:
        raise RuntimeError("RENDER_RESULTS_SOURCE_BINDING_MISMATCH")
    if results.get("artifact_hashes_preclose", {}).get("trace.npz") != sha256_file(paths["trace.npz"]):
        raise RuntimeError("RENDER_TRACE_HASH_BINDING_MISMATCH")
    plan = json.loads(paths["plan.json"].read_text())
    binding = results.get("representative_binding", {})
    if binding != plan.get("representative_binding"):
        raise RuntimeError("RENDER_REPRESENTATIVE_PLAN_RESULTS_DRIFT")
    if (
        binding.get("candidate_index") != PREFLIGHT_CANDIDATE_INDEX
        or binding.get("pinch_offset_index") != 0
    ):
        raise RuntimeError(f"RENDER_REPRESENTATIVE_NOT_NOMINAL {binding}")
    representative_slot = int(binding["environment_slot"])
    with np.load(paths["trace.npz"], allow_pickle=False) as archive:
        trace = {name: archive[name] for name in archive.files}
    if (
        trace["physics_step"].shape != (TOTAL_STEPS,)
        or int(trace["representative_environment_slot"]) != representative_slot
        or str(trace["trial_id"][representative_slot]) != binding["trial_id"]
    ):
        raise RuntimeError("RENDER_TRACE_REPRESENTATIVE_BINDING_DRIFT")
    frame_indices = np.arange(
        VIDEO_STEP_STRIDE - 1, TOTAL_STEPS, VIDEO_STEP_STRIDE, dtype=np.int64
    )
    if (
        len(frame_indices) != TOTAL_STEPS // VIDEO_STEP_STRIDE
        or int(trace["physics_step"][frame_indices[0]]) != VIDEO_STEP_STRIDE
        or int(trace["physics_step"][frame_indices[-1]]) != TOTAL_STEPS
    ):
        raise RuntimeError("RENDER_FRAME_CADENCE_DRIFT")
    render_dependency_paths, render_dependency_hashes_start = render_dependency_snapshot(
        profile
    )
    physics_finalize_cross_bind = render_physics_finalize_cross_bind(
        results, render_dependency_hashes_start
    )
    if not physics_finalize_cross_bind["pass"]:
        raise RuntimeError(
            "RENDER_PHYSICS_FINALIZE_TO_RENDER_START_DEPENDENCY_DRIFT "
            f"{physics_finalize_cross_bind}"
        )

    # No output directory is created until every source/binding gate passes.
    frame_dir.mkdir(parents=False, exist_ok=False)
    simulation_app = None
    try:
        from isaaclab.app import AppLauncher

        launcher = AppLauncher(headless=True, enable_cameras=True)
        simulation_app = launcher.app
        import omni.replicator.core as rep
        import omni.physx
        import omni.timeline
        import omni.usd
        from isaacsim.core.simulation_manager import SimulationManager
        from PIL import Image, ImageDraw
        from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade, Vt

        timeline = omni.timeline.get_timeline_interface()
        timeline.stop()
        timeline.commit()
        if timeline.is_playing():
            raise RuntimeError("RENDER_TIMELINE_REFUSED_TO_STOP_BEFORE_STAGE_SETUP")
        physics_step_observation = {"event_count": 0, "dt_sum_s": 0.0}

        def on_render_physics_step(dt: float) -> None:
            physics_step_observation["event_count"] += 1
            physics_step_observation["dt_sum_s"] += float(dt)

        physics_step_subscription = (
            omni.physx.get_physx_interface().subscribe_physics_step_events(
                on_render_physics_step
            )
        )

        def clock_snapshot() -> dict[str, Any]:
            return {
                "timeline_is_playing": bool(timeline.is_playing()),
                "timeline_time_s": float(timeline.get_current_time()),
                "simulation_manager_num_physics_steps": int(
                    SimulationManager.get_num_physics_steps()
                ),
                "simulation_manager_time_s": float(
                    SimulationManager.get_simulation_time()
                ),
                "physics_step_event_count": int(
                    physics_step_observation["event_count"]
                ),
                "physics_step_event_dt_sum_s": float(
                    physics_step_observation["dt_sum_s"]
                ),
            }

        clock_baseline = clock_snapshot()
        if clock_baseline["timeline_is_playing"]:
            raise RuntimeError(f"RENDER_CLOCK_BASELINE_PLAYING {clock_baseline}")
        clock_audits: list[dict[str, Any]] = []

        def gate_clock_unchanged(label: str, before: dict[str, Any]) -> dict[str, Any]:
            after = clock_snapshot()
            passed = bool(
                not after["timeline_is_playing"]
                and after["timeline_time_s"] == clock_baseline["timeline_time_s"]
                and after["simulation_manager_num_physics_steps"]
                == clock_baseline["simulation_manager_num_physics_steps"]
                and after["simulation_manager_time_s"]
                == clock_baseline["simulation_manager_time_s"]
                and after["physics_step_event_count"]
                == clock_baseline["physics_step_event_count"]
                and after["physics_step_event_dt_sum_s"]
                == clock_baseline["physics_step_event_dt_sum_s"]
                and before == clock_baseline
            )
            row = {"label": label, "before": before, "after": after, "pass": passed}
            clock_audits.append(row)
            if not passed:
                raise RuntimeError(f"RENDER_CLOCK_OR_PHYSICS_STEP_DRIFT {row}")
            return after

        def audited_app_update(label: str) -> None:
            before = clock_snapshot()
            simulation_app.update()
            gate_clock_unchanged(f"app.update:{label}", before)

        def audited_replicator_step(label: str) -> None:
            before = clock_snapshot()
            rep.orchestrator.step(rt_subframes=1)
            gate_clock_unchanged(f"rep.orchestrator.step:{label}", before)

        context = omni.usd.get_context()
        context.new_stage()
        gate_clock_unchanged("context.new_stage", clock_baseline)
        for _ in range(3):
            audited_app_update("new_stage_settle")
        stage = context.get_stage()
        UsdGeom.Xform.Define(stage, "/World")
        robot_root = UsdGeom.Xform.Define(stage, "/World/Robot").GetPrim()
        robot_root.GetReferences().AddReference(
            str(
                REPO
                / "claudedocs/runtime_logs/grasp_track/g0a_d344/collision_asset/attempt3"
                / "roarm_m3_fullmesh_fixed_point_parts/roarm_m3.usd"
            )
        )
        for _ in range(6):
            audited_app_update("robot_reference_settle")

        body_ops: dict[str, Any] = {}
        for body in MOVING_BODIES:
            prim = stage.GetPrimAtPath(f"/World/Robot/{body}")
            if not prim.IsValid():
                raise RuntimeError(f"RENDER_ROBOT_BODY_MISSING body={body}")
            xform = UsdGeom.Xformable(prim)
            xform.ClearXformOpOrder()
            body_ops[body] = xform.AddTransformOp()

        def material(path: str, color: tuple[float, float, float], roughness: float) -> Any:
            mat = UsdShade.Material.Define(stage, path)
            shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
            mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            return mat

        support_mat = material("/World/Looks/Support", (0.34, 0.38, 0.42), 0.88)
        object_mat = material("/World/Looks/Object", (0.86, 0.50, 0.12), 0.55)
        support = UsdGeom.Cube.Define(stage, "/World/Support")
        support.CreateSizeAttr(1.0)
        support_xf = UsdGeom.Xformable(support.GetPrim())
        support_xf.ClearXformOpOrder()
        support_xf.AddTranslateOp().Set(Gf.Vec3d(0.30, 0.0, -0.005))
        support_xf.AddScaleOp().Set(Gf.Vec3f(0.70, 0.55, 0.01))
        UsdShade.MaterialBindingAPI(support.GetPrim()).Bind(support_mat)

        cylinder = UsdGeom.Cylinder.Define(stage, "/World/Object")
        cylinder.CreateAxisAttr("Z")
        cylinder.CreateRadiusAttr(OBJ_RADIUS_M)
        cylinder.CreateHeightAttr(OBJ_HEIGHT_M)
        cylinder_xf = UsdGeom.Xformable(cylinder.GetPrim())
        cylinder_xf.ClearXformOpOrder()
        cylinder_op = cylinder_xf.AddTransformOp()
        UsdShade.MaterialBindingAPI(cylinder.GetPrim()).Bind(object_mat)

        dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
        dome.CreateIntensityAttr(2200.0)
        dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))

        def look_at_matrix(eye: np.ndarray, target: np.ndarray) -> Any:
            forward = target - eye
            forward /= np.linalg.norm(forward)
            right = np.cross(forward, np.asarray([0.0, 0.0, 1.0]))
            right /= np.linalg.norm(right)
            up = np.cross(right, forward)
            matrix = np.eye(4, dtype=np.float64)
            matrix[:3, 0] = right
            matrix[:3, 1] = up
            matrix[:3, 2] = -forward
            matrix[:3, 3] = eye
            return Gf.Matrix4d(*matrix.T.flatten().tolist())

        camera = UsdGeom.Camera.Define(stage, "/World/RenderCam")
        camera.CreateFocalLengthAttr(22.0)
        camera.CreateHorizontalApertureAttr(24.0)
        camera.CreateVerticalApertureAttr(24.0 * VIDEO_HEIGHT / VIDEO_WIDTH)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.03, 5.0))
        camera_xf = UsdGeom.Xformable(camera.GetPrim())
        camera_xf.ClearXformOpOrder()
        camera_xf.AddTransformOp().Set(
            look_at_matrix(
                np.asarray([0.72, -0.48, 0.36], dtype=np.float64),
                np.asarray([0.28, 0.08, 0.11], dtype=np.float64),
            )
        )
        render_product = rep.create.render_product(
            "/World/RenderCam", (VIDEO_WIDTH, VIDEO_HEIGHT)
        )
        rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb_annotator.attach([render_product])
        def physics_scene_paths() -> list[str]:
            return sorted(
                prim.GetPath().pathString
                for prim in stage.Traverse()
                if prim.GetTypeName() == "PhysicsScene"
            )

        physics_scenes_before_first_render = physics_scene_paths()
        if physics_scenes_before_first_render:
            raise RuntimeError(
                "RENDER_PHYSICS_SCENE_PRESENT_BEFORE_FIRST_RENDER "
                f"{physics_scenes_before_first_render}"
            )
        # Explicit counterexample: an app.update while stopped must leave every
        # independently observed simulation clock/callback unchanged.
        audited_app_update("stopped_clock_counterexample")

        def set_trace_frame(step_index: int) -> None:
            for body_index, body in enumerate(MOVING_BODIES):
                matrix = _matrix_from_pose(
                    trace["moving_body_pos_m"][step_index, representative_slot, body_index],
                    trace["moving_body_quat_wxyz"][step_index, representative_slot, body_index],
                )
                body_ops[body].Set(Gf.Matrix4d(*matrix.T.flatten().tolist()))
            object_matrix = _matrix_from_pose(
                trace["object_pos_m"][step_index, representative_slot],
                trace["object_quat_wxyz"][step_index, representative_slot],
            )
            cylinder_op.Set(Gf.Matrix4d(*object_matrix.T.flatten().tolist()))

        def op_matrix(op: Any) -> np.ndarray:
            value = op.Get()
            if value is None:
                raise RuntimeError("RENDER_XFORM_OP_VALUE_MISSING")
            return np.asarray(value, dtype=np.float64).T

        def frame_fidelity(step_index: int) -> dict[str, Any]:
            body_errors: dict[str, float] = {}
            for body_index, body in enumerate(MOVING_BODIES):
                expected = _matrix_from_pose(
                    trace["moving_body_pos_m"][step_index, representative_slot, body_index],
                    trace["moving_body_quat_wxyz"][step_index, representative_slot, body_index],
                )
                body_errors[body] = float(np.max(np.abs(op_matrix(body_ops[body]) - expected)))
            expected_object = _matrix_from_pose(
                trace["object_pos_m"][step_index, representative_slot],
                trace["object_quat_wxyz"][step_index, representative_slot],
            )
            object_error = float(np.max(np.abs(op_matrix(cylinder_op) - expected_object)))
            joint_now = np.asarray(
                trace["joint_pos_deg"][step_index, representative_slot], dtype=np.float64
            )
            joint_target_now = np.asarray(
                trace["joint_target_deg"][step_index, representative_slot], dtype=np.float64
            )
            finite = bool(
                np.isfinite(joint_now).all()
                and np.isfinite(joint_target_now).all()
                and np.isfinite(list(body_errors.values())).all()
                and math.isfinite(object_error)
            )
            passed = bool(
                finite
                and max(body_errors.values()) <= 1.0e-12
                and object_error <= 1.0e-12
            )
            row = {
                "moving_body_transform_max_abs": body_errors,
                "object_transform_max_abs": object_error,
                "joint_pos_deg": joint_now.tolist(),
                "joint_target_deg": joint_target_now.tolist(),
                "joint_source_finite": finite,
                "gate_max_abs": 1.0e-12,
                "pass": passed,
            }
            if not passed:
                raise RuntimeError(f"RENDER_TRACE_STATE_FIDELITY_FAIL {row}")
            return row

        set_trace_frame(int(frame_indices[0]))
        warmup_render_updates = 6
        for warmup_index in range(warmup_render_updates):
            audited_replicator_step(f"warmup:{warmup_index}")
            audited_app_update(f"warmup:{warmup_index}")
            frame_fidelity(int(frame_indices[0]))
        frame_rows: list[dict[str, Any]] = []
        phase_names = tuple(PHASE_STEPS)
        for output_index, step_index_raw in enumerate(frame_indices):
            step_index = int(step_index_raw)
            set_trace_frame(step_index)
            fidelity_pre = frame_fidelity(step_index)
            frame_clock_before = clock_snapshot()
            audited_replicator_step(f"frame:{output_index}")
            clock_after_replicator = clock_snapshot()
            audited_app_update(f"frame:{output_index}")
            frame_clock_after = clock_snapshot()
            fidelity_post = frame_fidelity(step_index)
            rgba = rgb_annotator.get_data()
            if rgba is None or getattr(rgba, "ndim", 0) != 3:
                raise RuntimeError(f"RENDER_RGB_DATA_INVALID frame={output_index}")
            image = Image.fromarray(np.asarray(rgba[:, :, :3], dtype=np.uint8))
            if image.size != (VIDEO_WIDTH, VIDEO_HEIGHT):
                raise RuntimeError(f"RENDER_FRAME_SIZE_DRIFT frame={output_index} size={image.size}")
            draw = ImageDraw.Draw(image)
            draw.rectangle((12, 10, 775, 76), fill=(0, 0, 0))
            phase_id = int(trace["phase_id"][step_index])
            physics_step = int(trace["physics_step"][step_index])
            draw.text(
                (24, 18),
                "p16 fixed-base side-midpoint PhysX trace replay\n"
                f"trial={binding['trial_id']} | phase={phase_names[phase_id]} | "
                f"physics_step={physics_step}/{TOTAL_STEPS} | replay physics steps=0",
                fill=(255, 255, 255),
            )
            frame_path = frame_dir / f"frame_{output_index:04d}.png"
            image.save(frame_path)
            frame_rows.append(
                {
                    "frame_index": output_index,
                    "source_trace_index": step_index,
                    "physics_step": physics_step,
                    "sim_time_s": float(trace["sim_time_s"][step_index]),
                    "phase_id": phase_id,
                    "phase": phase_names[phase_id],
                    "path": str(frame_path.relative_to(REPO)),
                    "sha256": sha256_file(frame_path),
                    "bytes": frame_path.stat().st_size,
                    "clock": {
                        "before": frame_clock_before,
                        "after_replicator": clock_after_replicator,
                        "after_app_update": frame_clock_after,
                        "pass": bool(
                            frame_clock_before == clock_baseline
                            and clock_after_replicator == clock_baseline
                            and frame_clock_after == clock_baseline
                        ),
                    },
                    "state_fidelity_pre": fidelity_pre,
                    "state_fidelity_post": fidelity_post,
                }
            )
        rgb_annotator.detach([render_product])

        import imageio_ffmpeg

        ffmpeg = Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve()
        command = [
            str(ffmpeg), "-hide_banner", "-loglevel", "error", "-n",
            "-framerate", str(VIDEO_FPS), "-start_number", "0",
            "-i", str(frame_dir / "frame_%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(VIDEO_FPS),
            str(paths["side_grasp.mp4"]),
        ]
        encoded = subprocess.run(command, capture_output=True, text=True, check=False)
        if encoded.returncode != 0 or not paths["side_grasp.mp4"].is_file():
            raise RuntimeError(
                f"RENDER_FFMPEG_ENCODE_FAIL rc={encoded.returncode} stderr={encoded.stderr}"
            )
        reader = imageio_ffmpeg.read_frames(
            str(paths["side_grasp.mp4"]), pix_fmt="rgb24"
        )
        decode_metadata = next(reader)
        decoded_frames = 0
        try:
            for _decoded in reader:
                decoded_frames += 1
        finally:
            reader.close()
        decode_pass = bool(
            decoded_frames == len(frame_rows)
            and tuple(decode_metadata.get("size", ())) == (VIDEO_WIDTH, VIDEO_HEIGHT)
            and abs(float(decode_metadata.get("fps", 0.0)) - VIDEO_FPS) < 1.0e-9
        )
        if not decode_pass:
            raise RuntimeError(
                f"RENDER_FULL_DECODE_GATE_FAIL frames={decoded_frames} meta={decode_metadata}"
            )
        physics_scenes_end = physics_scene_paths()
        render_dependency_hashes_end = {
            name: sha256_file(path) for name, path in render_dependency_paths.items()
        }
        dependency_hashes_equal = (
            render_dependency_hashes_end == render_dependency_hashes_start
        )
        if not dependency_hashes_equal:
            raise RuntimeError(
                "RENDER_DECISION_DEPENDENCY_CHANGED "
                f"start={render_dependency_hashes_start} "
                f"end={render_dependency_hashes_end}"
            )
        clock_final = clock_snapshot()
        clock_contract_pass = bool(
            clock_final == clock_baseline
            and not physics_scenes_before_first_render
            and not physics_scenes_end
            and clock_audits
            and all(row["pass"] for row in clock_audits)
            and all(row["clock"]["pass"] for row in frame_rows)
            and all(
                row["state_fidelity_pre"]["pass"]
                and row["state_fidelity_post"]["pass"]
                for row in frame_rows
            )
        )
        if not clock_contract_pass:
            raise RuntimeError(
                "RENDER_ZERO_PHYSICS_OR_FIDELITY_CONTRACT_FAIL "
                f"baseline={clock_baseline} final={clock_final} "
                f"before_scenes={physics_scenes_before_first_render} "
                f"end_scenes={physics_scenes_end}"
            )
        manifest = {
            "artifact": "T3U_ISOLATED_POSTHOC_RGB_TRACE_REPLAY_V1",
            "profile": profile,
            "argv": list(sys.argv),
            "scientific_authoritative": False,
            "render_is_posthoc_observability_only": True,
            "source_trace_path": str(paths["trace.npz"].relative_to(REPO)),
            "source_trace_sha256": sha256_file(paths["trace.npz"]),
            "source_results_sha256": sha256_file(paths["results.json"]),
            "source_plan_sha256": sha256_file(paths["plan.json"]),
            "executed_source_sha256": source_sha,
            "representative_binding": binding,
            "cadence": {
                "physics_hz": int(round(1.0 / DT_S)),
                "video_fps": VIDEO_FPS,
                "physics_step_stride": VIDEO_STEP_STRIDE,
                "mapping": "frame k -> trace index (k+1)*10-1; physics steps 10..2340",
            },
            "resolution": [VIDEO_WIDTH, VIDEO_HEIGHT],
            "frame_count": len(frame_rows),
            "frames": frame_rows,
            "first_frame_sha256": frame_rows[0]["sha256"],
            "last_frame_sha256": frame_rows[-1]["sha256"],
            "mp4_path": str(paths["side_grasp.mp4"].relative_to(REPO)),
            "mp4_sha256": sha256_file(paths["side_grasp.mp4"]),
            "mp4_bytes": paths["side_grasp.mp4"].stat().st_size,
            "ffmpeg_command": command,
            "decode": {
                "metadata": decode_metadata,
                "decoded_frame_count": decoded_frames,
                "full_decode_pass": decode_pass,
            },
            "renderer": {
                "basic_writer_used": False,
                "rgb_annotator_synchronous_get_data": True,
                "warmup_render_updates_not_written": warmup_render_updates,
                "written_render_updates": len(frame_rows),
                "timeline_stopped_before_context_new_stage": bool(
                    not clock_baseline["timeline_is_playing"]
                ),
                "clock_baseline": clock_baseline,
                "clock_final": clock_final,
                "clock_audits": clock_audits,
                "actual_app_update_count": sum(
                    row["label"].startswith("app.update:") for row in clock_audits
                ),
                "actual_replicator_step_count": sum(
                    row["label"].startswith("rep.orchestrator.step:")
                    for row in clock_audits
                ),
                "observed_physics_step_event_count": int(
                    physics_step_observation["event_count"]
                ),
                "observed_physics_step_event_dt_sum_s": float(
                    physics_step_observation["dt_sum_s"]
                ),
                "observed_simulation_manager_step_delta": int(
                    clock_final["simulation_manager_num_physics_steps"]
                    - clock_baseline["simulation_manager_num_physics_steps"]
                ),
                "observed_simulation_manager_time_delta_s": float(
                    clock_final["simulation_manager_time_s"]
                    - clock_baseline["simulation_manager_time_s"]
                ),
                "explicit_physics_api_calls": [],
                "explicit_physics_api_call_count": 0,
                "physics_scene_paths_before_first_render": (
                    physics_scenes_before_first_render
                ),
                "physics_scene_paths_end": physics_scenes_end,
                "zero_physics_observed_pass": clock_contract_pass,
                "state_application": "direct USD body/object transforms from frozen trace",
            },
            "decision_dependencies": {
                "paths": {
                    name: str(path.relative_to(REPO))
                    for name, path in render_dependency_paths.items()
                },
                "sha256_at_start": render_dependency_hashes_start,
                "sha256_at_end": render_dependency_hashes_end,
                "equal": dependency_hashes_equal,
                "physics_finalize_to_render_start": physics_finalize_cross_bind,
                "three_way_physics_finalize_render_start_end_equal": bool(
                    physics_finalize_cross_bind["pass"] and dependency_hashes_equal
                ),
            },
            "pass": bool(
                decode_pass and clock_contract_pass and dependency_hashes_equal
                and physics_finalize_cross_bind["pass"]
                and len(frame_rows) == TOTAL_STEPS // VIDEO_STEP_STRIDE
            ),
        }
        write_json_x(paths["rgb_frames_manifest.json"], manifest)
        append_phase(
            paths["render_phase.jsonl"],
            "render_trace_durable",
            manifest_sha256=sha256_file(paths["rgb_frames_manifest.json"]),
            mp4_sha256=sha256_file(paths["side_grasp.mp4"]),
            observed_physics_step_events=int(physics_step_observation["event_count"]),
            observed_simulation_manager_step_delta=int(
                clock_final["simulation_manager_num_physics_steps"]
                - clock_baseline["simulation_manager_num_physics_steps"]
            ),
        )
        physics_step_subscription = None
        print(
            f"[{LOG}] RENDER_TRACE_COMPLETE profile={profile} frames={len(frame_rows)} "
            f"mp4={paths['side_grasp.mp4']}",
            flush=True,
        )
        return 0
    except BaseException as exc:
        # Forward-only evidence stays in place for diagnosis; never delete or
        # overwrite a partial frame directory/MP4 under this tag.
        record_render_failure_evidence(profile, paths, exc)
        raise
    finally:
        if simulation_app is not None:
            simulation_app.close()


def _linux_pgid_members(pgid: int) -> list[int]:
    members: list[int] = []
    for stat_path in Path("/proc").glob("[0-9]*/stat"):
        try:
            raw = stat_path.read_text()
            closing = raw.rfind(")")
            fields = raw[closing + 2 :].split()
            if closing > 0 and len(fields) >= 3 and int(fields[2]) == pgid:
                members.append(int(stat_path.parent.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
            continue
    return sorted(members)


def _gpu_pid_set(csv_text: str) -> set[int]:
    result: set[int] = set()
    for line in csv_text.splitlines():
        token = line.split(",", 1)[0].strip()
        if token:
            result.add(int(token))
    return result


CHILD_LIFECYCLE_KEYS = {
    "label", "attempt_index", "attempt_count", "command", "pid", "pgid",
    "sid", "tty", "start_time_unix", "end_time_unix", "elapsed_seconds",
    "timeout_seconds", "timed_out", "signal_actions", "supervisor_signal",
    "group_members_after_reap", "group_reaped", "raw_wait_status",
    "wifexited", "exit_code", "wifsignaled", "signal_number",
    "signal_name", "core_dumped", "normalized_returncode",
}


def _strict_json_int(value: Any) -> bool:
    return type(value) is int


def _strict_json_float(value: Any) -> bool:
    return type(value) is float and math.isfinite(value)


def _json_type_value_exact(actual: Any, expected: Any) -> bool:
    """Compare JSON-shaped values without Python's ``False == 0`` aliasing."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return bool(
            set(actual) == set(expected)
            and all(
                _json_type_value_exact(actual[key], expected[key])
                for key in expected
            )
        )
    if isinstance(expected, list):
        return bool(
            len(actual) == len(expected)
            and all(
                _json_type_value_exact(a_value, e_value)
                for a_value, e_value in zip(actual, expected)
            )
        )
    if isinstance(expected, float):
        return bool(math.isfinite(actual) and math.isfinite(expected) and actual == expected)
    return actual == expected


def _read_self_pid_namespace_evidence() -> dict[str, Any]:
    """Output-free check of this process's two accessible procfs namespace aliases.

    Equality here is intentionally only self versus ``/proc/<own-pid>``.  It is not
    evidence that this process shares PID 1's namespace.  PID-1 command bytes and the
    complete visible ancestor walk remain the sandbox-rejection authority.
    """
    own_pid = os.getpid()
    self_path = "/proc/self/ns/pid"
    own_path = f"/proc/{own_pid}/ns/pid"
    self_readlink = os.readlink(self_path)
    own_readlink = os.readlink(own_path)
    self_stat = os.stat(self_path)
    own_stat = os.stat(own_path)
    self_device = int(self_stat.st_dev)
    own_device = int(own_stat.st_dev)
    self_inode = int(self_stat.st_ino)
    own_inode = int(own_stat.st_ino)
    consistent = bool(
        own_pid > 1
        and self_device > 0
        and own_device > 0
        and self_device == own_device
        and self_inode > 0
        and own_inode > 0
        and self_inode == own_inode
        and self_readlink == own_readlink
        and self_readlink == f"pid:[{self_inode}]"
    )
    evidence = {
        "supervisor_pid": own_pid,
        "self_pid_namespace_path": self_path,
        "own_pid_namespace_path": own_path,
        "self_pid_namespace_readlink": self_readlink,
        "own_pid_namespace_readlink": own_readlink,
        "self_pid_namespace_device": self_device,
        "own_pid_namespace_device": own_device,
        "self_pid_namespace_inode": self_inode,
        "own_pid_namespace_inode": own_inode,
        "supervisor_self_namespace_consistent": consistent,
        "pid_namespace_evidence_scope": (
            "supervisor_self_and_own_pid_alias_only__not_pid1_namespace_comparison"
        ),
        "namespace_consistency_is_not_pid1_or_host_proof": True,
    }
    if not consistent:
        raise RuntimeError(f"SELF_PID_NAMESPACE_PRELAUNCH_FAIL {evidence!r}")
    return evidence


def _validate_host_launch_context(
    value: Any, expected_supervisor_pid: int | None = None
) -> bool:
    """Strictly validate Supervisor V7's honest host/self-namespace record."""
    expected_keys = {
        "artifact", "authorization_boundary", "pid1_cmdline",
        "pid1_cmdline_hex", "pid1_cmdline_sha256",
        "supervisor_pid", "self_pid_namespace_path", "own_pid_namespace_path",
        "self_pid_namespace_readlink", "own_pid_namespace_readlink",
        "self_pid_namespace_device", "own_pid_namespace_device",
        "self_pid_namespace_inode", "own_pid_namespace_inode",
        "supervisor_self_namespace_consistent", "pid_namespace_evidence_scope",
        "namespace_consistency_is_not_pid1_or_host_proof",
        "sandbox_rejection_authority", "boot_id", "forbidden_tokens",
        "forbidden_matches", "pass",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        return False
    try:
        raw = bytes.fromhex(value["pid1_cmdline_hex"])
    except (TypeError, ValueError):
        return False
    decoded = raw.replace(b"\0", b" ").decode("utf-8", errors="replace")
    boot_id = value.get("boot_id")
    boot_parts = boot_id.split("-") if isinstance(boot_id, str) else []
    supervisor_pid = value.get("supervisor_pid")
    return bool(
        value.get("artifact") == "T3U_HOST_LAUNCH_CONTEXT_V2"
        and value.get("authorization_boundary") == "require_escalated_exec_command"
        and isinstance(value.get("pid1_cmdline"), str)
        and value["pid1_cmdline"] == decoded
        and raw
        and value.get("pid1_cmdline_sha256")
        == hashlib.sha256(raw).hexdigest()
        and _strict_json_int(supervisor_pid)
        and supervisor_pid > 1
        and (
            expected_supervisor_pid is None
            or supervisor_pid == expected_supervisor_pid
        )
        and value.get("self_pid_namespace_path") == "/proc/self/ns/pid"
        and value.get("own_pid_namespace_path")
        == f"/proc/{supervisor_pid}/ns/pid"
        and isinstance(value.get("self_pid_namespace_readlink"), str)
        and isinstance(value.get("own_pid_namespace_readlink"), str)
        and _strict_json_int(value.get("self_pid_namespace_device"))
        and value["self_pid_namespace_device"] > 0
        and _strict_json_int(value.get("own_pid_namespace_device"))
        and value["own_pid_namespace_device"] > 0
        and value["self_pid_namespace_device"]
        == value["own_pid_namespace_device"]
        and _strict_json_int(value.get("self_pid_namespace_inode"))
        and value["self_pid_namespace_inode"] > 0
        and _strict_json_int(value.get("own_pid_namespace_inode"))
        and value["own_pid_namespace_inode"] > 0
        and value["self_pid_namespace_inode"]
        == value["own_pid_namespace_inode"]
        and value["self_pid_namespace_readlink"]
        == value["own_pid_namespace_readlink"]
        == f"pid:[{value['self_pid_namespace_inode']}]"
        and value.get("supervisor_self_namespace_consistent") is True
        and value.get("pid_namespace_evidence_scope")
        == "supervisor_self_and_own_pid_alias_only__not_pid1_namespace_comparison"
        and value.get("namespace_consistency_is_not_pid1_or_host_proof") is True
        and value.get("sandbox_rejection_authority")
        == "pid1_cmdline_plus_complete_visible_ancestry_forbidden_token_gate"
        and len(boot_parts) == 5
        and [len(part) for part in boot_parts] == [8, 4, 4, 4, 12]
        and all(part and all(char in "0123456789abcdef" for char in part.lower())
                for part in boot_parts)
        and value.get("forbidden_tokens")
        == ["bwrap", "--die-with-parent", "codex-linux-sandbox"]
        and value.get("forbidden_matches") == []
        and not any(
            token in decoded
            for token in ("bwrap", "--die-with-parent", "codex-linux-sandbox")
        )
        and value.get("pass") is True
    )


def _strict_all_true_bool_map(value: Any, expected_keys: set[str]) -> bool:
    return bool(
        isinstance(value, dict)
        and set(value) == expected_keys
        and all(type(item) is bool and item is True for item in value.values())
    )


def _strict_pid(value: Any) -> bool:
    return _strict_json_int(value) and value > 1


def _read_pid_file_or_invalid(path: Path) -> int:
    try:
        token = path.read_text().strip()
        if not token.isascii() or not token.isdecimal():
            return -1
        value = int(token)
        return value if value > 1 else -1
    except BaseException:
        return -1


def _strict_signal_actions(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    for row in value:
        if not (
            isinstance(row, dict)
            and set(row)
            == {"signal", "time_unix", "members_before", "sent", "error", "reason"}
            and row.get("signal") in {"SIGTERM", "SIGKILL"}
            and _strict_json_float(row.get("time_unix"))
            and isinstance(row.get("members_before"), list)
            and all(_strict_pid(pid) for pid in row["members_before"])
            and row["members_before"] == sorted(set(row["members_before"]))
            and type(row.get("sent")) is bool
            and (row.get("error") is None or isinstance(row.get("error"), str))
            and isinstance(row.get("reason"), str)
            and bool(row["reason"])
        ):
            return False
    return True


def _strict_child_lifecycle(
    child: Any,
    *,
    label: str,
    command: list[str],
    pid: int,
    supervisor_sid: int,
    require_success: bool,
) -> bool:
    """Recompute raw wait semantics with strict JSON numeric types."""
    try:
        if not (
            isinstance(child, dict)
            and set(child) == CHILD_LIFECYCLE_KEYS
            and child.get("label") == label
            and child.get("command") == command
            and all(isinstance(item, str) for item in child.get("command", []))
            and _strict_json_int(child.get("attempt_index"))
            and child["attempt_index"] == 0
            and _strict_json_int(child.get("attempt_count"))
            and child["attempt_count"] == 1
            and _strict_pid(pid)
            and _strict_pid(supervisor_sid)
            and _strict_pid(child.get("pid"))
            and child["pid"] == pid
            and _strict_pid(child.get("pgid"))
            and child["pgid"] == pid
            and _strict_pid(child.get("sid"))
            and child["sid"] == supervisor_sid
            and isinstance(child.get("tty"), dict)
            and set(child["tty"]) == {"stdin", "stdout", "stderr"}
            and all(child["tty"].get(name) is False for name in ("stdin", "stdout", "stderr"))
            and _strict_json_float(child.get("start_time_unix"))
            and child["start_time_unix"] > 0.0
            and _strict_json_float(child.get("end_time_unix"))
            and child["end_time_unix"] >= child["start_time_unix"]
            and _strict_json_float(child.get("elapsed_seconds"))
            and child["elapsed_seconds"] >= 0.0
            and _strict_json_float(child.get("timeout_seconds"))
            and child["timeout_seconds"] == 7200.0
            and type(child.get("timed_out")) is bool
            and _strict_signal_actions(child.get("signal_actions"))
            and child.get("supervisor_signal") is None
            and child.get("group_members_after_reap") == []
            and child.get("group_reaped") is True
            and _strict_json_int(child.get("raw_wait_status"))
            and child["raw_wait_status"] >= 0
            and type(child.get("wifexited")) is bool
            and type(child.get("wifsignaled")) is bool
            and type(child.get("core_dumped")) is bool
            and _strict_json_int(child.get("normalized_returncode"))
        ):
            return False
        raw = child["raw_wait_status"]
        exited = os.WIFEXITED(raw)
        signaled = os.WIFSIGNALED(raw)
        signal_number = os.WTERMSIG(raw) if signaled else None
        exit_code = os.WEXITSTATUS(raw) if exited else None
        normalized = exit_code if exited else 128 + int(signal_number or 0)
        decoded_exact = bool(
            child["wifexited"] is exited
            and child["wifsignaled"] is signaled
            and (
                (_strict_json_int(child.get("exit_code")) and child["exit_code"] == exit_code)
                if exited else child.get("exit_code") is None
            )
            and (
                _strict_json_int(child.get("signal_number"))
                and child["signal_number"] == signal_number
                if signaled else child.get("signal_number") is None
            )
            and child.get("signal_name")
            == (signal.Signals(signal_number).name if signal_number is not None else None)
            and child["core_dumped"]
            is (bool(os.WCOREDUMP(raw)) if signaled else False)
            and child["normalized_returncode"] == normalized
        )
        if not decoded_exact:
            return False
        if require_success:
            return bool(
                raw == 0
                and exited
                and exit_code == 0
                and not signaled
                and child["normalized_returncode"] == 0
                and child["timed_out"] is False
                and child["signal_actions"] == []
            )
        return True
    except (KeyError, TypeError, ValueError):
        return False


def _strict_attempts(value: Any, *, render_count: int) -> bool:
    return bool(
        isinstance(value, dict)
        and set(value) == {"physics", "render", "automatic_retry_count"}
        and all(_strict_json_int(value.get(name)) for name in value)
        and value["physics"] == 1
        and value["render"] == render_count
        and value["automatic_retry_count"] == 0
    )


def _strict_supervisor_identity(value: Any, *, pid: int, pgid: int) -> bool:
    return bool(
        isinstance(value, dict)
        and set(value)
        == {
            "pid", "pgid", "sid", "tty", "group_members_before_exit",
            "self_only_before_exit", "signal_received", "cleanup_actions",
            "active_child_at_exit",
        }
        and _strict_pid(pid)
        and _strict_pid(pgid)
        and pid == pgid
        and _strict_pid(value.get("pid"))
        and value["pid"] == pid
        and _strict_pid(value.get("pgid"))
        and value["pgid"] == pgid
        and _strict_pid(value.get("sid"))
        and value["sid"] == pid
        and isinstance(value.get("tty"), dict)
        and set(value["tty"]) == {"stdin", "stdout", "stderr"}
        and all(value["tty"].get(name) is False for name in ("stdin", "stdout", "stderr"))
        and isinstance(value.get("group_members_before_exit"), list)
        and len(value["group_members_before_exit"]) == 1
        and _strict_pid(value["group_members_before_exit"][0])
        and value["group_members_before_exit"][0] == pid
        and value.get("self_only_before_exit") is True
        and value.get("signal_received") is None
        and value.get("cleanup_actions") == []
        and value.get("active_child_at_exit") is None
    )


def _strict_outcome_times(value: Any, children: list[dict[str, Any]]) -> bool:
    if not (
        isinstance(value, dict)
        and _strict_json_float(value.get("start_time_unix"))
        and value["start_time_unix"] > 0.0
        and _strict_json_float(value.get("end_time_unix"))
        and value["end_time_unix"] >= value["start_time_unix"]
        and _strict_json_float(value.get("elapsed_seconds"))
        and value["elapsed_seconds"] >= 0.0
    ):
        return False
    previous = value["start_time_unix"]
    for child in children:
        if not (
            isinstance(child, dict)
            and _strict_json_float(child.get("start_time_unix"))
            and _strict_json_float(child.get("end_time_unix"))
            and previous <= child["start_time_unix"] <= child["end_time_unix"]
        ):
            return False
        previous = child["end_time_unix"]
    return previous <= value["end_time_unix"]


def _strict_gpu_pid_list(value: Any) -> bool:
    return bool(
        isinstance(value, list)
        and all(_strict_pid(pid) for pid in value)
        and value == sorted(set(value))
    )


def _strict_gpu_summary(
    value: Any, *, before: set[int], supervisor_end: set[int]
) -> bool:
    expected_before = sorted(before)
    expected_end = sorted(supervisor_end)
    return bool(
        isinstance(value, dict)
        and set(value)
        == {"before_pids", "supervisor_end_pids", "fresh_pid_delta", "no_fresh_pid_delta"}
        and _strict_gpu_pid_list(value.get("before_pids"))
        and _strict_gpu_pid_list(value.get("supervisor_end_pids"))
        and _strict_gpu_pid_list(value.get("fresh_pid_delta"))
        and value["before_pids"] == expected_before
        and value["supervisor_end_pids"] == expected_end
        and value["fresh_pid_delta"] == []
        and value.get("no_fresh_pid_delta") is True
    )


def validate_authoritative_trace_cadence(trace: dict[str, np.ndarray]) -> bool:
    """Regenerate the complete step/time/phase authority from frozen constants."""
    expected_physics_steps = np.arange(1, TOTAL_STEPS + 1, dtype=np.int64)
    expected_sim_time = expected_physics_steps.astype(np.float64) * DT_S
    expected_phase_ids = np.concatenate(
        [
            np.full(steps, phase_id, dtype=np.int16)
            for phase_id, steps in enumerate(PHASE_STEPS.values())
        ]
    )
    expected_phase_steps = np.concatenate(
        [np.arange(steps, dtype=np.int32) for steps in PHASE_STEPS.values()]
    )
    expected = {
        "physics_step": expected_physics_steps,
        "sim_time_s": expected_sim_time,
        "phase_id": expected_phase_ids,
        "phase_step": expected_phase_steps,
    }
    return bool(
        all(
            isinstance(trace.get(name), np.ndarray)
            and trace[name].dtype == value.dtype
            and trace[name].shape == (TOTAL_STEPS,)
            and np.array_equal(trace[name], value)
            for name, value in expected.items()
        )
    )


def validate_render_manifest_semantics(
    profile: str,
    paths: dict[str, Path],
    manifest: dict[str, Any],
    results: dict[str, Any],
    plan: dict[str, Any],
) -> dict[str, bool]:
    """Recompute render claims from files; never promote on ``pass`` alone."""
    expected_top_keys = {
        "artifact", "profile", "argv", "scientific_authoritative",
        "render_is_posthoc_observability_only", "source_trace_path",
        "source_trace_sha256", "source_results_sha256", "source_plan_sha256",
        "executed_source_sha256", "representative_binding", "cadence",
        "resolution", "frame_count", "frames", "first_frame_sha256",
        "last_frame_sha256", "mp4_path", "mp4_sha256", "mp4_bytes",
        "ffmpeg_command", "decode", "renderer", "decision_dependencies", "pass",
    }
    expected_renderer_keys = {
        "basic_writer_used", "rgb_annotator_synchronous_get_data",
        "warmup_render_updates_not_written", "written_render_updates",
        "timeline_stopped_before_context_new_stage", "clock_baseline",
        "clock_final", "clock_audits", "actual_app_update_count",
        "actual_replicator_step_count", "observed_physics_step_event_count",
        "observed_physics_step_event_dt_sum_s",
        "observed_simulation_manager_step_delta",
        "observed_simulation_manager_time_delta_s", "explicit_physics_api_calls",
        "explicit_physics_api_call_count", "physics_scene_paths_before_first_render",
        "physics_scene_paths_end", "zero_physics_observed_pass", "state_application",
    }
    clock_keys = {
        "timeline_is_playing", "timeline_time_s",
        "simulation_manager_num_physics_steps", "simulation_manager_time_s",
        "physics_step_event_count", "physics_step_event_dt_sum_s",
    }
    frame_keys = {
        "frame_index", "source_trace_index", "physics_step", "sim_time_s",
        "phase_id", "phase", "path", "sha256", "bytes", "clock",
        "state_fidelity_pre", "state_fidelity_post",
    }
    fidelity_keys = {
        "moving_body_transform_max_abs", "object_transform_max_abs",
        "joint_pos_deg", "joint_target_deg", "joint_source_finite",
        "gate_max_abs", "pass",
    }
    renderer = manifest.get("renderer", {})
    baseline = renderer.get("clock_baseline", {})
    final_clock = renderer.get("clock_final", {})

    def strict_int(value: Any) -> bool:
        return type(value) is int

    def strict_float(value: Any) -> bool:
        return type(value) is float and math.isfinite(value)

    def clock_exact(value: Any) -> bool:
        return bool(
            isinstance(value, dict)
            and set(value) == clock_keys
            and value.get("timeline_is_playing") is False
            and strict_float(value.get("timeline_time_s"))
            and strict_int(value.get("simulation_manager_num_physics_steps"))
            and strict_float(value.get("simulation_manager_time_s"))
            and strict_int(value.get("physics_step_event_count"))
            and strict_float(value.get("physics_step_event_dt_sum_s"))
        )

    clock_rows = renderer.get("clock_audits", [])
    expected_clock_labels = [
        "context.new_stage",
        *(["app.update:new_stage_settle"] * 3),
        *(["app.update:robot_reference_settle"] * 6),
        "app.update:stopped_clock_counterexample",
        *[
            label
            for warmup_index in range(6)
            for label in (
                f"rep.orchestrator.step:warmup:{warmup_index}",
                f"app.update:warmup:{warmup_index}",
            )
        ],
        *[
            label
            for frame_index in range(TOTAL_STEPS // VIDEO_STEP_STRIDE)
            for label in (
                f"rep.orchestrator.step:frame:{frame_index}",
                f"app.update:frame:{frame_index}",
            )
        ],
    ]
    clock_rows_exact = bool(
        isinstance(clock_rows, list)
        and len(clock_rows) == 491
        and [
            row.get("label") if isinstance(row, dict) else None
            for row in clock_rows
        ] == expected_clock_labels
        and all(
            isinstance(row, dict)
            and set(row) == {"label", "before", "after", "pass"}
            and isinstance(row.get("label"), str)
            and clock_exact(row.get("before"))
            and clock_exact(row.get("after"))
            and row.get("before") == baseline
            and row.get("after") == baseline
            and row.get("pass") is True
            for row in clock_rows
        )
    )
    app_count = sum(
        isinstance(row, dict) and str(row.get("label", "")).startswith("app.update:")
        for row in clock_rows
    )
    replicator_count = sum(
        isinstance(row, dict)
        and str(row.get("label", "")).startswith("rep.orchestrator.step:")
        for row in clock_rows
    )
    stopped_counterexample_present = any(
        isinstance(row, dict)
        and row.get("label") == "app.update:stopped_clock_counterexample"
        and row.get("before") == baseline
        and row.get("after") == baseline
        and row.get("pass") is True
        for row in clock_rows
    )
    clock_zero_recomputed = bool(
        clock_exact(baseline)
        and clock_exact(final_clock)
        and final_clock == baseline
        and strict_int(baseline.get("physics_step_event_count"))
        and baseline.get("physics_step_event_count") == 0
        and strict_float(baseline.get("physics_step_event_dt_sum_s"))
        and baseline.get("physics_step_event_dt_sum_s") == 0.0
        and strict_int(renderer.get("observed_physics_step_event_count"))
        and renderer.get("observed_physics_step_event_count") == 0
        and strict_float(renderer.get("observed_physics_step_event_dt_sum_s"))
        and renderer.get("observed_physics_step_event_dt_sum_s") == 0.0
        and strict_int(renderer.get("observed_simulation_manager_step_delta"))
        and renderer.get("observed_simulation_manager_step_delta") == 0
        and strict_float(renderer.get("observed_simulation_manager_time_delta_s"))
        and renderer.get("observed_simulation_manager_time_delta_s") == 0.0
        and renderer.get("explicit_physics_api_calls") == []
        and strict_int(renderer.get("explicit_physics_api_call_count"))
        and renderer.get("explicit_physics_api_call_count") == 0
        and renderer.get("physics_scene_paths_before_first_render") == []
        and renderer.get("physics_scene_paths_end") == []
        and renderer.get("timeline_stopped_before_context_new_stage") is True
        and renderer.get("zero_physics_observed_pass") is True
        and clock_rows_exact
        and stopped_counterexample_present
        and strict_int(renderer.get("actual_app_update_count"))
        and renderer.get("actual_app_update_count") == app_count
        and strict_int(renderer.get("actual_replicator_step_count"))
        and renderer.get("actual_replicator_step_count") == replicator_count
        and app_count == 250
        and replicator_count == 240
    )

    frames = manifest.get("frames", [])
    frame_indices = np.arange(
        VIDEO_STEP_STRIDE - 1, TOTAL_STEPS, VIDEO_STEP_STRIDE, dtype=np.int64
    )
    frame_rows_exact = False
    trace_cadence_exact = False
    first_hash = last_hash = None
    if isinstance(frames, list) and len(frames) == len(frame_indices):
        with np.load(paths["trace.npz"], allow_pickle=False) as archive:
            trace = {name: archive[name] for name in archive.files}
        trace_cadence_exact = bool(
            validate_authoritative_trace_cadence(trace)
            and np.array_equal(
                trace["physics_step"][frame_indices],
                np.arange(
                    VIDEO_STEP_STRIDE,
                    TOTAL_STEPS + 1,
                    VIDEO_STEP_STRIDE,
                    dtype=np.int64,
                ),
            )
        )
        rep_slot_raw = results.get("representative_binding", {}).get("environment_slot")
        rep_slot = rep_slot_raw if strict_int(rep_slot_raw) else -1
        rep_slot_valid = bool(
            trace.get("joint_pos_deg") is not None
            and 0 <= rep_slot < int(trace["joint_pos_deg"].shape[1])
        )
        trace_rep_slot = rep_slot if rep_slot_valid else 0
        phase_names = tuple(PHASE_STEPS)
        row_passes: list[bool] = []
        for output_index, (row, step_index_raw) in enumerate(zip(frames, frame_indices)):
            step_index = int(step_index_raw)
            frame_path = CASE_DIR / f"t3u_{profile}_rgb_frames/frame_{output_index:04d}.png"
            expected_relative = str(frame_path.relative_to(REPO))

            def fidelity_exact(value: Any) -> bool:
                if not isinstance(value, dict) or set(value) != fidelity_keys:
                    return False
                body_errors = value.get("moving_body_transform_max_abs", {})
                gate = value.get("gate_max_abs")
                joint_pos = value.get("joint_pos_deg")
                joint_target = value.get("joint_target_deg")
                return bool(
                    isinstance(body_errors, dict)
                    and set(body_errors) == set(MOVING_BODIES)
                    and strict_float(gate)
                    and gate == 1.0e-12
                    and all(
                        strict_float(error)
                        and 0.0 <= error <= gate
                        for error in body_errors.values()
                    )
                    and strict_float(value.get("object_transform_max_abs"))
                    and 0.0 <= value["object_transform_max_abs"] <= gate
                    and isinstance(joint_pos, list)
                    and len(joint_pos) == len(JOINT_ORDER)
                    and all(strict_float(item) for item in joint_pos)
                    and isinstance(joint_target, list)
                    and len(joint_target) == len(JOINT_ORDER)
                    and all(strict_float(item) for item in joint_target)
                    and value.get("joint_source_finite") is True
                    and value.get("pass") is True
                    and np.array_equal(
                        np.asarray(value.get("joint_pos_deg"), dtype=np.float64),
                        np.asarray(trace["joint_pos_deg"][step_index, trace_rep_slot], dtype=np.float64),
                    )
                    and np.array_equal(
                        np.asarray(value.get("joint_target_deg"), dtype=np.float64),
                        np.asarray(trace["joint_target_deg"][step_index, trace_rep_slot], dtype=np.float64),
                    )
                )

            row_clock = row.get("clock", {}) if isinstance(row, dict) else {}
            row_passes.append(
                bool(
                    isinstance(row, dict)
                    and rep_slot_valid
                    and trace_cadence_exact
                    and set(row) == frame_keys
                    and strict_int(row.get("frame_index"))
                    and row.get("frame_index") == output_index
                    and strict_int(row.get("source_trace_index"))
                    and row.get("source_trace_index") == step_index
                    and strict_int(row.get("physics_step"))
                    and row.get("physics_step") == int(trace["physics_step"][step_index])
                    and strict_float(row.get("sim_time_s"))
                    and row.get("sim_time_s") == float(trace["sim_time_s"][step_index])
                    and strict_int(row.get("phase_id"))
                    and row.get("phase_id") == int(trace["phase_id"][step_index])
                    and row.get("phase") == phase_names[int(trace["phase_id"][step_index])]
                    and row.get("path") == expected_relative
                    and frame_path.is_file()
                    and row.get("sha256") == sha256_file(frame_path)
                    and strict_int(row.get("bytes"))
                    and row.get("bytes") > 0
                    and row.get("bytes") == frame_path.stat().st_size
                    and isinstance(row_clock, dict)
                    and set(row_clock) == {"before", "after_replicator", "after_app_update", "pass"}
                    and clock_exact(row_clock.get("before"))
                    and clock_exact(row_clock.get("after_replicator"))
                    and clock_exact(row_clock.get("after_app_update"))
                    and row_clock.get("before") == baseline
                    and row_clock.get("after_replicator") == baseline
                    and row_clock.get("after_app_update") == baseline
                    and row_clock.get("pass") is True
                    and fidelity_exact(row.get("state_fidelity_pre"))
                    and fidelity_exact(row.get("state_fidelity_post"))
                )
            )
        frame_rows_exact = bool(all(row_passes))
        if frames:
            first_hash = frames[0].get("sha256")
            last_hash = frames[-1].get("sha256")

    dependency_paths_now, dependency_hashes_now = render_dependency_snapshot(profile)
    dependencies = manifest.get("decision_dependencies", {})
    expected_dependency_paths = {
        name: str(path.relative_to(REPO)) for name, path in dependency_paths_now.items()
    }
    physics_finalize_cross_bind_now = render_physics_finalize_cross_bind(
        results, dependency_hashes_now
    )
    dependency_exact = bool(
        isinstance(dependencies, dict)
        and set(dependencies)
        == {
            "paths", "sha256_at_start", "sha256_at_end", "equal",
            "physics_finalize_to_render_start",
            "three_way_physics_finalize_render_start_end_equal",
        }
        and dependencies.get("paths") == expected_dependency_paths
        and dependencies.get("sha256_at_start") == dependency_hashes_now
        and dependencies.get("sha256_at_end") == dependency_hashes_now
        and dependencies.get("equal") is True
        and dependencies.get("physics_finalize_to_render_start")
        == physics_finalize_cross_bind_now
        and physics_finalize_cross_bind_now["pass"] is True
        and dependencies.get("three_way_physics_finalize_render_start_end_equal") is True
    )
    decode = manifest.get("decode", {})
    metadata = decode.get("metadata", {}) if isinstance(decode, dict) else {}
    decode_exact = bool(
        isinstance(decode, dict)
        and set(decode) == {"metadata", "decoded_frame_count", "full_decode_pass"}
        and strict_int(decode.get("decoded_frame_count"))
        and decode.get("decoded_frame_count") == len(frame_indices)
        and decode.get("full_decode_pass") is True
        and isinstance(metadata.get("size"), (list, tuple))
        and len(metadata.get("size")) == 2
        and all(strict_int(value) for value in metadata["size"])
        and tuple(metadata["size"]) == (VIDEO_WIDTH, VIDEO_HEIGHT)
        and strict_float(metadata.get("fps"))
        and abs(metadata["fps"] - VIDEO_FPS) < 1.0e-9
    )
    source_exact = bool(
        manifest.get("source_trace_path") == str(paths["trace.npz"].relative_to(REPO))
        and manifest.get("source_trace_sha256") == sha256_file(paths["trace.npz"])
        and manifest.get("source_results_sha256") == sha256_file(paths["results.json"])
        and manifest.get("source_plan_sha256") == sha256_file(paths["plan.json"])
        and manifest.get("executed_source_sha256") == sha256_file(Path(__file__))
        and manifest.get("representative_binding") == results.get("representative_binding")
        == plan.get("representative_binding")
    )
    mp4_exact = bool(
        paths["side_grasp.mp4"].is_file()
        and manifest.get("mp4_path") == str(paths["side_grasp.mp4"].relative_to(REPO))
        and manifest.get("mp4_sha256") == sha256_file(paths["side_grasp.mp4"])
        and strict_int(manifest.get("mp4_bytes"))
        and manifest.get("mp4_bytes") > 0
        and manifest.get("mp4_bytes") == paths["side_grasp.mp4"].stat().st_size
        and isinstance(manifest.get("ffmpeg_command"), list)
        and bool(manifest["ffmpeg_command"])
        and manifest["ffmpeg_command"][-1] == str(paths["side_grasp.mp4"])
        and "-n" in manifest["ffmpeg_command"]
    )
    checks = {
        "top_level_exact": bool(
            set(manifest) == expected_top_keys
            and manifest.get("artifact") == "T3U_ISOLATED_POSTHOC_RGB_TRACE_REPLAY_V1"
            and manifest.get("profile") == profile
            and manifest.get("argv")
            == [str(Path(__file__).resolve()), "--render_trace", profile]
            and manifest.get("scientific_authoritative") is False
            and manifest.get("render_is_posthoc_observability_only") is True
        ),
        "source_and_representative_bindings_recomputed": source_exact,
        "cadence_and_resolution_exact": bool(
            manifest.get("cadence")
            == {
                "physics_hz": int(round(1.0 / DT_S)),
                "video_fps": VIDEO_FPS,
                "physics_step_stride": VIDEO_STEP_STRIDE,
                "mapping": "frame k -> trace index (k+1)*10-1; physics steps 10..2340",
            }
            and manifest.get("resolution") == [VIDEO_WIDTH, VIDEO_HEIGHT]
            and all(strict_int(value) for value in manifest.get("resolution", []))
            and strict_int(manifest.get("frame_count"))
            and manifest.get("frame_count") == len(frame_indices)
        ),
        "renderer_schema_exact": bool(
            isinstance(renderer, dict)
            and set(renderer) == expected_renderer_keys
            and renderer.get("basic_writer_used") is False
            and renderer.get("rgb_annotator_synchronous_get_data") is True
            and strict_int(renderer.get("warmup_render_updates_not_written"))
            and renderer.get("warmup_render_updates_not_written") == 6
            and strict_int(renderer.get("written_render_updates"))
            and renderer.get("written_render_updates") == len(frame_indices)
            and renderer.get("state_application")
            == "direct USD body/object transforms from frozen trace"
        ),
        "zero_physics_clocks_callbacks_scenes_recomputed": clock_zero_recomputed,
        "trace_full_cadence_and_sample_schedule_recomputed": trace_cadence_exact,
        "all_frame_files_trace_mappings_clocks_and_state_exact": frame_rows_exact,
        "first_last_frame_hashes_exact": bool(
            manifest.get("first_frame_sha256") == first_hash
            and manifest.get("last_frame_sha256") == last_hash
        ),
        "mp4_file_hash_and_no_overwrite_command_exact": mp4_exact,
        "full_decode_semantics_exact": decode_exact,
        "decision_dependency_start_end_current_hashes_exact": dependency_exact,
        "manifest_pass_recomputed_not_trusted": False,
    }
    checks["manifest_pass_recomputed_not_trusted"] = bool(
        manifest.get("pass") is True
        and all(
            value for name, value in checks.items()
            if name != "manifest_pass_recomputed_not_trusted"
        )
    )
    return checks


def validate_joint_limit_readback_semantics(
    report: Any, expected_envs: int
) -> bool:
    if not isinstance(report, dict):
        return False
    parsed = parse_urdf_limits()
    expected_names = list(URDF_JOINT_MAP)
    expected_deg = np.asarray(
        [parsed[label] for label in JOINT_ORDER], dtype=np.float64
    )
    expected_rad = np.radians(expected_deg)
    try:
        soft = np.asarray(report["soft_joint_pos_limits_rad"], dtype=np.float64)
        cached = np.asarray(report["cached_lower_upper_limits_rad"], dtype=np.float64)
        rows = report["composed_revolute_joint_rows"]
        soft_tol = float(report["soft_limit_abs_tolerance_rad"])
        usd_tol = float(report["usd_limit_abs_tolerance_deg"])
    except (KeyError, TypeError, ValueError):
        return False
    rows_exact = bool(
        isinstance(rows, list)
        and len(rows) == expected_envs * len(JOINT_ORDER)
        and all(
            isinstance(row, dict)
            and set(row)
            == {
                "env_index", "joint_index", "joint_name", "joint_label", "path",
                "lower_deg", "upper_deg", "lower_error_deg", "upper_error_deg",
                "pass",
            }
            and row.get("env_index") == env_index
            and row.get("joint_index") == joint_index
            and row.get("joint_name") == expected_names[joint_index]
            and row.get("joint_label") == JOINT_ORDER[joint_index]
            and row.get("path")
            == (
                f"/World/envs/env_{env_index}/Robot/joints/"
                f"{expected_names[joint_index]}"
            )
            and math.isfinite(float(row.get("lower_deg", math.nan)))
            and math.isfinite(float(row.get("upper_deg", math.nan)))
            and math.isclose(
                float(row["lower_deg"]), float(expected_deg[joint_index, 0]),
                rel_tol=0.0, abs_tol=usd_tol,
            )
            and math.isclose(
                float(row["upper_deg"]), float(expected_deg[joint_index, 1]),
                rel_tol=0.0, abs_tol=usd_tol,
            )
            and math.isclose(
                float(row.get("lower_error_deg", math.inf)),
                abs(float(row["lower_deg"]) - float(expected_deg[joint_index, 0])),
                rel_tol=0.0, abs_tol=1.0e-15,
            )
            and math.isclose(
                float(row.get("upper_error_deg", math.inf)),
                abs(float(row["upper_deg"]) - float(expected_deg[joint_index, 1])),
                rel_tol=0.0, abs_tol=1.0e-15,
            )
            and row.get("pass") is True
            for env_index in range(expected_envs)
            for joint_index, row in [
                (joint_index, rows[env_index * len(JOINT_ORDER) + joint_index])
                for joint_index in range(len(JOINT_ORDER))
            ]
        )
    )
    return bool(
        set(report)
        == {
            "authority", "urdf_sha256", "joint_names", "joint_labels",
            "expected_urdf_limits_deg", "expected_urdf_limits_rad",
            "soft_limit_abs_tolerance_rad", "usd_limit_abs_tolerance_deg",
            "soft_joint_pos_limits_shape", "soft_joint_pos_limits_rad",
            "soft_joint_pos_limits_max_abs_error_rad",
            "cached_lower_upper_limits_rad", "cached_limits_max_abs_error_rad",
            "composed_revolute_joint_rows", "soft_limits_pass",
            "cached_limits_pass", "composed_usd_limits_pass", "pass",
        }
        and report.get("authority")
        == "parsed_frozen_urdf_cross_checked_against_composed_usd_and_runtime_soft_limits"
        and report.get("urdf_sha256") == URDF_SHA256
        and report.get("joint_names") == expected_names
        and report.get("joint_labels") == list(JOINT_ORDER)
        and np.array_equal(
            np.asarray(report.get("expected_urdf_limits_deg"), dtype=np.float64),
            expected_deg,
        )
        and np.array_equal(
            np.asarray(report.get("expected_urdf_limits_rad"), dtype=np.float64),
            expected_rad,
        )
        and soft_tol == 5.0e-7
        and usd_tol == 2.0e-5
        and soft.shape == (expected_envs, len(JOINT_ORDER), 2)
        and cached.shape == (len(JOINT_ORDER), 2)
        and np.isfinite(soft).all()
        and np.isfinite(cached).all()
        and float(np.max(np.abs(soft - expected_rad[None, :, :]))) <= soft_tol
        and float(np.max(np.abs(cached - expected_rad))) <= soft_tol
        and math.isclose(
            float(report.get("soft_joint_pos_limits_max_abs_error_rad", math.inf)),
            float(np.max(np.abs(soft - expected_rad[None, :, :]))),
            rel_tol=0.0, abs_tol=1.0e-15,
        )
        and math.isclose(
            float(report.get("cached_limits_max_abs_error_rad", math.inf)),
            float(np.max(np.abs(cached - expected_rad))),
            rel_tol=0.0, abs_tol=1.0e-15,
        )
        and rows_exact
        and report.get("soft_limits_pass") is True
        and report.get("cached_limits_pass") is True
        and report.get("composed_usd_limits_pass") is True
        and report.get("pass") is True
    )


def validate_fixed_base_readback_semantics(report: Any, expected_envs: int) -> bool:
    if not isinstance(report, dict):
        return False
    rows = report.get("root_fixed_joint_rows")
    return bool(
        set(report)
        == {
            "authority", "urdf_conceptual_fixed_body", "composed_fixed_body",
            "urdf_to_composed_mapping", "isaaclab_is_fixed_base", "body_names",
            "expected_clone_count", "actual_clone_count", "root_fixed_joint_rows",
            "pass",
        }
        and report.get("authority")
        == "isaaclab_physx_metatype_plus_composed_enabled_root_fixed_joint"
        and report.get("urdf_conceptual_fixed_body") == "base_link"
        and report.get("composed_fixed_body") == FIXED_BASE_BODY
        and report.get("urdf_to_composed_mapping")
        == "world+base_link_merged_to_Robot/world"
        and report.get("isaaclab_is_fixed_base") is True
        and report.get("body_names") == list(SELF_CONTACT_BODIES)
        and report.get("expected_clone_count") == expected_envs
        and report.get("actual_clone_count") == expected_envs
        and isinstance(rows, list)
        and len(rows) == expected_envs
        and all(
            row
            == {
                "env_index": env_index,
                "joint_path": f"/World/envs/env_{env_index}/Robot/root_joint",
                "joint_type": "PhysicsFixedJoint",
                "joint_enabled": True,
                "body0_targets": [],
                "body1_targets": [
                    f"/World/envs/env_{env_index}/Robot/{FIXED_BASE_BODY}"
                ],
                "fixed_body_path": (
                    f"/World/envs/env_{env_index}/Robot/{FIXED_BASE_BODY}"
                ),
                "fixed_body_rigid_api_present": True,
                "pass": True,
            }
            for env_index, row in enumerate(rows)
        )
        and report.get("pass") is True
    )


def validate_self_contact_filter_identity_semantics(
    report: Any,
    expected_envs: int,
    expected_epoch: str,
) -> bool:
    """Recompute the 15 rigid-contact-view identities without trusting PASS."""
    if not isinstance(report, dict) or expected_epoch not in {
        "precontrol", "postcontrol_pre_task"
    }:
        return False
    expected_keys = [f"{a}__{b}" for a, b in SELF_PAIRS]
    expected_scene = {
        "replicate_physics": True,
        "filter_collisions": True,
        "clone_in_fabric": False,
    }
    scene = report.get("scene_clone_configuration")
    clock = report.get("clock_before_control")
    rows = report.get("pair_rows")
    checks = report.get("checks")
    if not (
        set(report)
        == {
            "artifact", "authority", "audit_epoch",
            "scene_clone_configuration", "expected_env_count",
            "expected_pair_count", "clock_before_control", "pair_rows",
            "checks", "pass",
        }
        and report.get("artifact") == "T3U_SELF_CONTACT_FILTER_IDENTITY_V1"
        and report.get("authority") == "actual_rigid_contact_view_runtime_identity"
        and report.get("audit_epoch") == expected_epoch
        and scene
        == {"actual": expected_scene, "expected": expected_scene, "pass": True}
        and all(type(value) is bool for value in scene["actual"].values())
        and type(report.get("expected_env_count")) is int
        and report["expected_env_count"] == expected_envs
        and type(report.get("expected_pair_count")) is int
        and report["expected_pair_count"] == len(SELF_PAIRS) == 15
        and isinstance(clock, dict)
        and set(clock)
        == {
            "simulation_manager_num_physics_steps", "simulation_manager_time_s",
            "simulation_context_step_index", "simulation_context_time_s",
            "env_sim_step_counter", "common_step_counter", "episode_length_buf",
        }
        and type(clock["simulation_manager_num_physics_steps"]) is int
        and type(clock["simulation_context_step_index"]) is int
        and type(clock["simulation_manager_time_s"]) is float
        and math.isfinite(clock["simulation_manager_time_s"])
        and type(clock["simulation_context_time_s"]) is float
        and math.isfinite(clock["simulation_context_time_s"])
        and type(clock["env_sim_step_counter"]) is int
        and clock["env_sim_step_counter"] == 0
        and type(clock["common_step_counter"]) is int
        and clock["common_step_counter"] == 0
        and clock["episode_length_buf"] == [0] * expected_envs
        and all(type(value) is int for value in clock["episode_length_buf"])
        and isinstance(rows, dict)
        and len(rows) == len(expected_keys)
        and set(rows) == set(expected_keys)
        and isinstance(checks, dict)
        and set(checks)
        == {
            "scene_clone_contract_exact", "pair_inventory_exact",
            "pair_count_exactly_15", "all_pair_views_pass",
            "task_counters_zero_before_control",
        }
        and all(type(value) is bool and value is True for value in checks.values())
        and report.get("pass") is True
    ):
        return False

    for body_a, body_b in SELF_PAIRS:
        key = f"{body_a}__{body_b}"
        row = rows.get(key)
        expected_subject = [
            f"/World/envs/env_{index}/Robot/{body_a}"
            for index in range(expected_envs)
        ]
        expected_filter = [
            f"/World/envs/env_{index}/Robot/{body_b}"
            for index in range(expected_envs)
        ]
        filter_expr = f"/World/envs/env_.*/Robot/{body_b}"
        filter_glob = filter_expr.replace(".*", "*")
        if not (
            isinstance(row, dict)
            and set(row)
            == {
                "pair", "subject_expression", "expected_sensor_paths",
                "actual_sensor_paths", "sensor_stage_paths", "sensor_count",
                "filter_count", "raw_contact_count_shape",
                "configured_max_contact_data_count_per_prim",
                "expected_max_contact_data_count", "actual_max_contact_data_count",
                "filter_identity", "checks", "pass",
            }
            and row.get("pair") == [body_a, body_b]
            and row.get("subject_expression")
            == f"/World/envs/env_.*/Robot/{body_a}"
            and row.get("expected_sensor_paths") == expected_subject
            and row.get("actual_sensor_paths") == expected_subject
            and isinstance(row.get("sensor_stage_paths"), list)
            and len(row["sensor_stage_paths"]) == expected_envs
            and row["sensor_stage_paths"]
            == [
                {"path": path, "valid": True,
                 "type": row["sensor_stage_paths"][index].get("type")}
                for index, path in enumerate(expected_subject)
            ]
            and all(
                isinstance(item.get("type"), str) and item["type"]
                for item in row["sensor_stage_paths"]
            )
            and type(row.get("sensor_count")) is int
            and row["sensor_count"] == expected_envs
            and type(row.get("filter_count")) is int
            and row["filter_count"] == 1
            and row.get("raw_contact_count_shape") == [expected_envs, 1]
            and type(row.get("configured_max_contact_data_count_per_prim")) is int
            and row["configured_max_contact_data_count_per_prim"] == 256
            and type(row.get("expected_max_contact_data_count")) is int
            and row["expected_max_contact_data_count"] == 256 * expected_envs
            and type(row.get("actual_max_contact_data_count")) is int
            and row["actual_max_contact_data_count"] == 256 * expected_envs
            and isinstance(row.get("filter_identity"), dict)
            and set(row["filter_identity"])
            == {
                "label", "force_matrix_shape", "filter_count", "actual_filter_paths",
                "expected_concrete_env0_representative",
                "physx_replicated_filter_representation",
                "expected_filter_expression", "accepted_physx_glob",
                "resolved_stage_paths_from_expression", "expected_stage_paths", "pass",
            }
            and row["filter_identity"].get("label") == f"self:{key}"
            and row["filter_identity"].get("force_matrix_shape")
            == [expected_envs, 1, 1, 3]
            and row["filter_identity"].get("filter_count") == 1
            and row["filter_identity"].get("actual_filter_paths")
            == [expected_filter[0]]
            and row["filter_identity"].get(
                "expected_concrete_env0_representative"
            ) == expected_filter[0]
            and row["filter_identity"].get(
                "physx_replicated_filter_representation"
            ) == "single_logical_filter_as_env0_concrete_representative"
            and row["filter_identity"].get("expected_filter_expression")
            == filter_expr
            and row["filter_identity"].get("accepted_physx_glob") == filter_glob
            and row["filter_identity"].get("resolved_stage_paths_from_expression")
            == sorted(expected_filter)
            and isinstance(
                row["filter_identity"].get("expected_stage_paths"), list
            )
            and len(row["filter_identity"]["expected_stage_paths"])
            == expected_envs
            and row["filter_identity"].get("expected_stage_paths")
            == [
                {"path": path, "valid": True,
                 "type": row["filter_identity"]["expected_stage_paths"][index].get("type")}
                for index, path in enumerate(expected_filter)
            ]
            and all(
                isinstance(item.get("type"), str) and item["type"]
                for item in row["filter_identity"]["expected_stage_paths"]
            )
            and row["filter_identity"].get("pass") is True
            and isinstance(row.get("checks"), dict)
            and set(row["checks"])
            == {
                "base_filter_identity_pass", "sensor_count_exact",
                "filter_count_exact", "sensor_paths_ordered_exact",
                "sensor_stage_paths_valid", "raw_contact_count_shape_exact",
                "max_contact_data_count_exact",
                "configured_per_prim_capacity_exact",
            }
            and all(
                type(value) is bool and value is True
                for value in row["checks"].values()
            )
            and row.get("pass") is True
        ):
            return False
    return True


def validate_self_collision_behavioral_control_semantics(
    report: Any,
    expected_envs: int,
) -> bool:
    """Recompute the exact two-frame behavioral control without trusting PASS."""
    if not isinstance(report, dict):
        return False
    snapshot_keys = {
        "simulation_manager_num_physics_steps",
        "simulation_manager_time_s",
        "simulation_context_step_index",
        "simulation_context_time_s",
        "env_sim_step_counter",
        "common_step_counter",
        "episode_length_buf",
    }
    expected_pair_keys = {f"{a}__{b}" for a, b in SELF_PAIRS}
    expected_positive_keys = {
        f"{a}__{b}" for a, b in SELF_COLLISION_POSITIVE_EXPECTED_PAIRS
    }
    phase_check_keys = {
        "pair_inventory_exact", "all_pair_rows_pass",
        "robot_object_raw_zero", "support_raw_zero",
        "robot_object_force_lte_1e_minus_8",
        "support_force_lte_1e_minus_8", "manager_step_delta_one",
        "manager_time_delta_callback_dt", "simulation_context_step_delta_one",
        "simulation_context_time_delta_callback_dt", "task_counters_remain_zero",
    }
    report_check_keys = {
        "runtime_geometry_all8_pass", "negative_runtime_geometry_all8_pass",
        "precontrol_filter_identity_pass",
        "precontrol_clock_equals_behavior_start",
        "positive_pass", "negative_pass",
        "callback_count_exactly_two", "callback_dt_both_finite_nominal",
        "total_manager_step_delta_two",
        "total_manager_time_delta_callback_fsum",
        "total_sim_context_step_delta_two",
        "total_sim_context_time_delta_callback_fsum",
        "task_counters_zero_after_both",
    }
    callback_dts = report.get("callback_dts_s")
    callback_dts_exact = bool(
        isinstance(callback_dts, list)
        and len(callback_dts) == 2
        and all(
            type(item) is float
            and math.isfinite(item)
            and math.isclose(
                item, DT_S, rel_tol=0.0,
                abs_tol=CLOCK_CALLBACK_NOMINAL_DT_ABS_TOL_S,
            )
            for item in callback_dts
        )
    )

    def snapshot_exact(value: Any) -> bool:
        return bool(
            isinstance(value, dict)
            and set(value) == snapshot_keys
            and type(value.get("simulation_manager_num_physics_steps")) is int
            and type(value.get("simulation_context_step_index")) is int
            and type(value.get("simulation_manager_time_s")) is float
            and math.isfinite(value["simulation_manager_time_s"])
            and type(value.get("simulation_context_time_s")) is float
            and math.isfinite(value["simulation_context_time_s"])
            and type(value.get("env_sim_step_counter")) is int
            and value["env_sim_step_counter"] == 0
            and type(value.get("common_step_counter")) is int
            and value["common_step_counter"] == 0
            and isinstance(value.get("episode_length_buf"), list)
            and value["episode_length_buf"] == [0] * expected_envs
            and all(type(item) is int for item in value["episode_length_buf"])
        )

    def phase_exact(value: Any, label: str, callback_dt: float) -> bool:
        if not isinstance(value, dict) or set(value) != {
            "label", "before", "after", "pair_rows",
            "object_force_max_n_per_env", "support_force_max_n_per_env",
            "checks", "pass",
        }:
            return False
        before, after = value.get("before"), value.get("after")
        rows = value.get("pair_rows")
        checks = value.get("checks")
        if not (
            value.get("label") == label
            and snapshot_exact(before) and snapshot_exact(after)
            and after["simulation_manager_num_physics_steps"]
            - before["simulation_manager_num_physics_steps"] == 1
            and after["simulation_context_step_index"]
            - before["simulation_context_step_index"] == 1
            and math.isclose(
                after["simulation_manager_time_s"]
                - before["simulation_manager_time_s"],
                callback_dt, rel_tol=0.0,
                abs_tol=_clock_elapsed_abs_tolerance_s(
                    callback_dt,
                    after["simulation_manager_time_s"]
                    - before["simulation_manager_time_s"],
                ),
            )
            and math.isclose(
                after["simulation_context_time_s"]
                - before["simulation_context_time_s"],
                callback_dt, rel_tol=0.0,
                abs_tol=_clock_elapsed_abs_tolerance_s(
                    callback_dt,
                    after["simulation_context_time_s"]
                    - before["simulation_context_time_s"],
                ),
            )
            and isinstance(rows, dict) and set(rows) == expected_pair_keys
            and isinstance(checks, dict) and set(checks) == phase_check_keys
            and all(type(item) is bool and item is True for item in checks.values())
            and value.get("pass") is True
        ):
            return False
        for key, row in rows.items():
            contact_expected = label == "positive" and key in expected_positive_keys
            if not (
                isinstance(row, dict)
                and set(row) == {
                    "expected_contact", "raw_count_shape", "raw_count_per_env",
                    "raw_count_total", "actual_max_contact_data_count",
                    "raw_count_total_strictly_below_actual_capacity",
                    "force_norm_n_per_env", "pass",
                }
                and row.get("expected_contact") is contact_expected
                and row.get("raw_count_shape") == [expected_envs, 1]
                and isinstance(row.get("raw_count_per_env"), list)
                and len(row["raw_count_per_env"]) == expected_envs
                and all(type(item) is int for item in row["raw_count_per_env"])
                and type(row.get("raw_count_total")) is int
                and row["raw_count_total"] == sum(row["raw_count_per_env"])
                and type(row.get("actual_max_contact_data_count")) is int
                and row["actual_max_contact_data_count"] == 256 * expected_envs
                and row.get("raw_count_total_strictly_below_actual_capacity") is True
                and row["raw_count_total"] < row["actual_max_contact_data_count"]
                and isinstance(row.get("force_norm_n_per_env"), list)
                and len(row["force_norm_n_per_env"]) == expected_envs
                and all(
                    type(item) is float and math.isfinite(item)
                    for item in row["force_norm_n_per_env"]
                )
                and (
                    all(1 <= item < 256 for item in row["raw_count_per_env"])
                    and all(item > CONTACT_GATE_N for item in row["force_norm_n_per_env"])
                    if contact_expected
                    else all(item == 0 for item in row["raw_count_per_env"])
                    and all(
                        item <= SELF_COLLISION_NEGATIVE_FORCE_GATE_N
                        for item in row["force_norm_n_per_env"]
                    )
                )
                and row.get("pass") is True
            ):
                return False
        for force_key in (
            "object_force_max_n_per_env", "support_force_max_n_per_env"
        ):
            force = value.get(force_key)
            if not (
                isinstance(force, list) and len(force) == expected_envs
                and all(
                    type(item) is float and math.isfinite(item)
                    and item <= SELF_COLLISION_NEGATIVE_FORCE_GATE_N
                    for item in force
                )
            ):
                return False
        return True

    runtime = report.get("runtime_geometry")
    runtime_rows = runtime.get("rows") if isinstance(runtime, dict) else None
    negative_runtime = report.get("negative_runtime_geometry_before_step")
    negative_runtime_rows = (
        negative_runtime.get("rows") if isinstance(negative_runtime, dict) else None
    )
    precontrol = report.get("precontrol_self_contact_filter_identity")
    before, after = report.get("before"), report.get("after")
    checks = report.get("checks")
    return bool(
        set(report)
        == {
            "artifact", "authority", "behavioral_proof_scope",
            "deprecated_dynamic_control_queries", "diagnostic_physics_steps",
            "task_physics_steps", "configured_contact_capacity_per_prim",
            "positive_raw_count_saturation_forbidden", "callback_dts_s",
            "before", "after", "precontrol_self_contact_filter_identity",
            "runtime_geometry", "negative_runtime_geometry_before_step",
            "positive", "negative", "checks", "pass",
        }
        and report.get("artifact")
        == "T3U_SAME_PROCESS_SELF_COLLISION_BEHAVIORAL_CONTROL_V1"
        and report.get("authority")
        == "actual_contact_sensor_force_matrix_and_raw_physx_contact_data"
        and report.get("behavioral_proof_scope")
        == (
            "two_preregistered_overlap_pairs_detected_and_HOME_all_15_pairs_clear; "
            "not a proof of every pose or manifold"
        )
        and type(report.get("deprecated_dynamic_control_queries")) is int
        and report["deprecated_dynamic_control_queries"] == 0
        and type(report.get("diagnostic_physics_steps")) is int
        and report["diagnostic_physics_steps"] == 2
        and type(report.get("task_physics_steps")) is int
        and report["task_physics_steps"] == 0
        and type(report.get("configured_contact_capacity_per_prim")) is int
        and report["configured_contact_capacity_per_prim"] == 256
        and report.get("positive_raw_count_saturation_forbidden") is True
        and callback_dts_exact
        and snapshot_exact(before) and snapshot_exact(after)
        and validate_self_contact_filter_identity_semantics(
            precontrol, expected_envs, "precontrol"
        )
        and _json_type_value_exact(
            precontrol.get("clock_before_control"), before
        )
        and after["simulation_manager_num_physics_steps"]
        - before["simulation_manager_num_physics_steps"] == 2
        and after["simulation_context_step_index"]
        - before["simulation_context_step_index"] == 2
        and math.isclose(
            after["simulation_manager_time_s"] - before["simulation_manager_time_s"],
            math.fsum(callback_dts), rel_tol=0.0,
            abs_tol=_clock_elapsed_abs_tolerance_s(
                math.fsum(callback_dts),
                after["simulation_manager_time_s"]
                - before["simulation_manager_time_s"],
            ),
        )
        and math.isclose(
            after["simulation_context_time_s"] - before["simulation_context_time_s"],
            math.fsum(callback_dts), rel_tol=0.0,
            abs_tol=_clock_elapsed_abs_tolerance_s(
                math.fsum(callback_dts),
                after["simulation_context_time_s"]
                - before["simulation_context_time_s"],
            ),
        )
        and isinstance(runtime, dict) and set(runtime) == {"rows", "pass"}
        and isinstance(runtime_rows, list) and len(runtime_rows) == expected_envs
        and all(
            isinstance(row, dict)
            and set(row) == {
                "env_index", "actual_q_max_abs_error_rad",
                "pair_inradius_mm", "pass",
            }
            and type(row.get("env_index")) is int and row["env_index"] == index
            and type(row.get("actual_q_max_abs_error_rad")) is float
            and math.isfinite(row["actual_q_max_abs_error_rad"])
            and row["actual_q_max_abs_error_rad"] <= 1.0e-7
            and isinstance(row.get("pair_inradius_mm"), dict)
            and set(row["pair_inradius_mm"]) == expected_positive_keys
            and all(
                type(item) is float and math.isfinite(item)
                and item >= SELF_COLLISION_POSITIVE_INRADIUS_GATE_MM
                for item in row["pair_inradius_mm"].values()
            )
            and row.get("pass") is True
            for index, row in enumerate(runtime_rows)
        )
        and runtime.get("pass") is True
        and isinstance(negative_runtime, dict)
        and set(negative_runtime) == {"authority", "rows", "pass"}
        and negative_runtime.get("authority")
        == (
            "direct_root_physx_view_dof_positions_and_link_transforms_"
            "after_HOME_write_before_negative_step"
        )
        and isinstance(negative_runtime_rows, list)
        and len(negative_runtime_rows) == expected_envs
        and all(
            isinstance(row, dict)
            and set(row)
            == {
                "env_index", "actual_q_max_abs_error_rad", "positive_pair_set",
                "pair_minimum_separating_face_margin_mm", "checks", "pass",
            }
            and type(row.get("env_index")) is int and row["env_index"] == index
            and type(row.get("actual_q_max_abs_error_rad")) is float
            and math.isfinite(row["actual_q_max_abs_error_rad"])
            and row["actual_q_max_abs_error_rad"] <= 1.0e-7
            and row.get("positive_pair_set") == []
            and isinstance(
                row.get("pair_minimum_separating_face_margin_mm"), dict
            )
            and set(row["pair_minimum_separating_face_margin_mm"])
            == expected_pair_keys
            and all(
                type(value) is float and math.isfinite(value) and value >= 0.0
                for value in row[
                    "pair_minimum_separating_face_margin_mm"
                ].values()
            )
            and row["pair_minimum_separating_face_margin_mm"]["link2__link4"]
            >= SELF_COLLISION_NEGATIVE_SEPARATION_GATE_MM
            and isinstance(row.get("checks"), dict)
            and set(row["checks"])
            == {
                "actual_q_equals_written_HOME",
                "all_15_pair_runtime_overlap_set_empty",
                "all_15_pair_separations_finite_nonnegative",
                "link2_link4_runtime_separation_gte_60mm",
            }
            and all(
                type(value) is bool and value is True
                for value in row["checks"].values()
            )
            and row.get("pass") is True
            for index, row in enumerate(negative_runtime_rows)
        )
        and negative_runtime.get("pass") is True
        and phase_exact(report.get("positive"), "positive", callback_dts[0])
        and phase_exact(report.get("negative"), "negative", callback_dts[1])
        and isinstance(checks, dict) and set(checks) == report_check_keys
        and all(type(item) is bool and item is True for item in checks.values())
        and report.get("pass") is True
    )


def validate_self_collision_readback_semantics(
    report: Any,
    expected_envs: int,
) -> bool:
    """Independently reject fallback-only, alias-typed, or runtime-false rows."""
    if not isinstance(report, dict):
        return False
    try:
        source = report["source_asset"]
        composed = report["composed_usd"]
        root_view = report["root_physx_view"]
        geometry = report["geometry_certificate"]
        behavioral = report["behavioral_control"]
    except (KeyError, TypeError):
        return False

    source_checks = {
        "root_exactly_one", "root_suffix_exact", "usd_articulation_root_api",
        "physx_articulation_api", "attribute_valid", "attribute_type_bool",
        "authored_value_opinion", "resolved_value_strict_false",
        "pinned_physics_spec_exactly_one",
        "pinned_physics_spec_typed_explicit_false",
    }
    row_checks = {
        "container_valid", "articulation_root_exactly_one", "root_suffix_exact",
        "usd_articulation_root_api", "physx_articulation_api", "attribute_valid",
        "attribute_type_bool", "authored_value_opinion",
        "resolved_value_strict_true", "strongest_explicit_value_strict_true",
        "pinned_source_explicit_false_exactly_one",
        "strong_true_precedes_pinned_false",
    }
    root_view_checks = {
        "backend_present", "view_check_strict_true", "count_strict_int",
        "count_exact", "prim_paths_are_strings",
        "prim_paths_exact_ordered_identity",
    }
    def strict_true_checks(value: Any, keys: set[str]) -> bool:
        return bool(
            isinstance(value, dict)
            and set(value) == keys
            and all(type(value[key]) is bool and value[key] is True for key in keys)
        )

    def stack_row_exact(row: Any, index: int) -> bool:
        return bool(
            isinstance(row, dict)
            and set(row)
            == {
                "strength_index", "layer_identifier", "layer_real_path",
                "spec_path", "type_name", "default_authored", "default_value",
                "default_python_type",
            }
            and type(row.get("strength_index")) is int
            and row["strength_index"] == index
            and isinstance(row.get("layer_identifier"), str)
            and row["layer_identifier"]
            and (row.get("layer_real_path") is None
                 or isinstance(row.get("layer_real_path"), str))
            and isinstance(row.get("spec_path"), str)
            and isinstance(row.get("type_name"), str)
            and type(row.get("default_authored")) is bool
            and (
                (row["default_authored"] is False
                 and row.get("default_value") is None
                 and row.get("default_python_type") is None)
                or (
                    row["default_authored"] is True
                    and row.get("type_name") == "bool"
                    and row.get("default_python_type") == "bool"
                    and type(row.get("default_value")) is bool
                )
            )
        )

    source_stack = source.get("property_stack_strong_to_weak", []) \
        if isinstance(source, dict) else []
    source_stack_exact = bool(
        isinstance(source_stack, list)
        and source_stack
        and all(stack_row_exact(row, index) for index, row in enumerate(source_stack))
        and len(
            [
                row for row in source_stack
                if row.get("layer_real_path") == source.get("source_physics_layer_path")
                and row.get("default_authored") is True
                and row.get("type_name") == "bool"
                and row.get("default_python_type") == "bool"
                and type(row.get("default_value")) is bool
                and row["default_value"] is False
            ]
        ) == 1
    )
    source_exact = bool(
        isinstance(source, dict)
        and set(source)
        == {
            "source_usd_path", "source_physics_layer_path", "root_candidates",
            "root_path", "property_stack_strong_to_weak", "checks", "pass",
        }
        and source.get("source_usd_path")
        == str(ATTEMPT3_ROOT_PATH.resolve())
        and source.get("source_physics_layer_path")
        == str(ATTEMPT3_PHYSICS_PATH.resolve())
        and source.get("root_candidates") == [f"/roarm_m3{ARTICULATION_ROOT_SUFFIX}"]
        and source.get("root_path") == f"/roarm_m3{ARTICULATION_ROOT_SUFFIX}"
        and source_stack_exact
        and strict_true_checks(source.get("checks"), source_checks)
        and source.get("pass") is True
    )

    composed_rows = composed.get("rows", []) if isinstance(composed, dict) else []
    composed_rows_exact = bool(
        isinstance(composed_rows, list)
        and len(composed_rows) == expected_envs
        and all(
            isinstance(row, dict)
            and set(row)
            == {
                "env_index", "container_path", "articulation_root_candidates",
                "root_path", "expected_root_path", "attribute_name",
                "attribute_type_name", "resolved_value",
                "property_stack_strong_to_weak",
                "strongest_authored_strength_index",
                "pinned_source_false_strength_index", "checks", "pass",
            }
            and type(row.get("env_index")) is int
            and row["env_index"] == env_index
            and row.get("container_path") == f"/World/envs/env_{env_index}/Robot"
            and row.get("root_path")
            == f"/World/envs/env_{env_index}/Robot{ARTICULATION_ROOT_SUFFIX}"
            and row.get("expected_root_path") == row.get("root_path")
            and row.get("articulation_root_candidates") == [row.get("root_path")]
            and row.get("attribute_name") == SELF_COLLISION_ATTR
            and row.get("attribute_type_name") == "bool"
            and type(row.get("resolved_value")) is bool
            and row["resolved_value"] is True
            and type(row.get("strongest_authored_strength_index")) is int
            and row["strongest_authored_strength_index"] >= 0
            and type(row.get("pinned_source_false_strength_index")) is int
            and row["pinned_source_false_strength_index"]
            > row["strongest_authored_strength_index"]
            and isinstance(row.get("property_stack_strong_to_weak"), list)
            and row["property_stack_strong_to_weak"]
            and all(
                stack_row_exact(stack_row, index)
                for index, stack_row in enumerate(
                    row["property_stack_strong_to_weak"]
                )
            )
            and row["property_stack_strong_to_weak"]
                [row["strongest_authored_strength_index"]]["default_value"] is True
            and row["property_stack_strong_to_weak"]
                [row["pinned_source_false_strength_index"]]["layer_real_path"]
                == composed.get("pinned_source_layer_path")
            and row["property_stack_strong_to_weak"]
                [row["pinned_source_false_strength_index"]]["default_authored"] is True
            and row["property_stack_strong_to_weak"]
                [row["pinned_source_false_strength_index"]]["type_name"] == "bool"
            and row["property_stack_strong_to_weak"]
                [row["pinned_source_false_strength_index"]]["default_python_type"]
                == "bool"
            and type(
                row["property_stack_strong_to_weak"]
                [row["pinned_source_false_strength_index"]]["default_value"]
            ) is bool
            and row["property_stack_strong_to_weak"]
                [row["pinned_source_false_strength_index"]]["default_value"] is False
            and strict_true_checks(row.get("checks"), row_checks)
            and row.get("pass") is True
            for env_index, row in enumerate(composed_rows)
        )
    )
    expected_root_paths = [
        f"/World/envs/env_{index}/Robot{ARTICULATION_ROOT_SUFFIX}"
        for index in range(expected_envs)
    ]
    composed_exact = bool(
        isinstance(composed, dict)
        and set(composed)
        == {
            "authority", "attribute", "expected_clone_count",
            "actual_clone_count", "root_suffix", "pinned_source_layer_path",
            "rows", "pass",
        }
        and composed.get("authority")
        == "composed_usd_authorship_and_property_stack"
        and composed.get("attribute") == SELF_COLLISION_ATTR
        and type(composed.get("expected_clone_count")) is int
        and composed["expected_clone_count"] == expected_envs
        and type(composed.get("actual_clone_count")) is int
        and composed["actual_clone_count"] == expected_envs
        and composed.get("root_suffix") == ARTICULATION_ROOT_SUFFIX
        and composed.get("pinned_source_layer_path")
        == source.get("source_physics_layer_path")
        and composed_rows_exact
        and composed.get("pass") is True
    )

    root_view_exact = bool(
        isinstance(root_view, dict)
        and set(root_view)
        == {
            "authority", "backend_python_type", "frontend_python_type", "check",
            "expected_count", "actual_count", "expected_prim_paths",
            "actual_prim_paths", "checks", "pass",
        }
        and root_view.get("authority")
        == "isaaclab_root_physx_view_runtime_identity"
        and isinstance(root_view.get("backend_python_type"), str)
        and root_view["backend_python_type"]
        and isinstance(root_view.get("frontend_python_type"), str)
        and type(root_view.get("check")) is bool and root_view["check"] is True
        and type(root_view.get("expected_count")) is int
        and root_view["expected_count"] == expected_envs
        and type(root_view.get("actual_count")) is int
        and root_view["actual_count"] == expected_envs
        and root_view.get("expected_prim_paths") == expected_root_paths
        and root_view.get("actual_prim_paths") == expected_root_paths
        and strict_true_checks(root_view.get("checks"), root_view_checks)
        and root_view.get("pass") is True
    )

    expected_positive_keys = [
        f"{a}__{b}" for a, b in SELF_COLLISION_POSITIVE_EXPECTED_PAIRS
    ]
    expected_pair_keys = {f"{a}__{b}" for a, b in SELF_PAIRS}
    geometry_checks = geometry.get("checks", {}) if isinstance(geometry, dict) else {}
    positive_geometry = geometry.get("positive", {}) if isinstance(geometry, dict) else {}
    negative_geometry = geometry.get("negative", {}) if isinstance(geometry, dict) else {}
    positive_pair_rows = positive_geometry.get("pair_rows", {}) \
        if isinstance(positive_geometry, dict) else {}
    negative_pair_rows = negative_geometry.get("pair_rows", {}) \
        if isinstance(negative_geometry, dict) else {}
    def geometry_pose_exact(value: Any, expected_positive: list[str]) -> bool:
        if not (
            isinstance(value, dict)
            and set(value) == {
                "positive_pair_set", "pair_rows", "moving_min_z_mm",
                "object_cylinder_vertex_proxy_separation_mm",
            }
            and value.get("positive_pair_set") == expected_positive
            and isinstance(value.get("pair_rows"), dict)
            and set(value["pair_rows"]) == expected_pair_keys
            and type(value.get("moving_min_z_mm")) is float
            and math.isfinite(value["moving_min_z_mm"])
            and type(value.get("object_cylinder_vertex_proxy_separation_mm")) is float
            and math.isfinite(value["object_cylinder_vertex_proxy_separation_mm"])
        ):
            return False
        derived_positive: list[str] = []
        for key, row in value["pair_rows"].items():
            if not (
                isinstance(row, dict)
                and set(row) == {
                    "positive_intersection_count", "max_intersection_inradius_mm",
                    "minimum_separating_face_margin_mm", "intersections",
                }
                and type(row.get("positive_intersection_count")) is int
                and row["positive_intersection_count"] >= 0
                and isinstance(row.get("intersections"), list)
                and len(row["intersections"]) == row["positive_intersection_count"]
            ):
                return False
            for intersection in row["intersections"]:
                if not (
                    isinstance(intersection, dict)
                    and set(intersection) == {"part_a", "part_b", "inradius_mm"}
                    and isinstance(intersection.get("part_a"), str)
                    and isinstance(intersection.get("part_b"), str)
                    and type(intersection.get("inradius_mm")) is float
                    and math.isfinite(intersection["inradius_mm"])
                    and intersection["inradius_mm"] > 0.0
                ):
                    return False
            if row["positive_intersection_count"]:
                derived_positive.append(key)
                if not (
                    type(row.get("max_intersection_inradius_mm")) is float
                    and row["max_intersection_inradius_mm"]
                    == max(item["inradius_mm"] for item in row["intersections"])
                    and (
                        row.get("minimum_separating_face_margin_mm") is None
                        or (
                            type(row.get("minimum_separating_face_margin_mm")) is float
                            and math.isfinite(row["minimum_separating_face_margin_mm"])
                            and row["minimum_separating_face_margin_mm"] >= 0.0
                        )
                    )
                ):
                    return False
            elif not (
                row.get("max_intersection_inradius_mm") is None
                and type(row.get("minimum_separating_face_margin_mm")) is float
                and math.isfinite(row["minimum_separating_face_margin_mm"])
                and row["minimum_separating_face_margin_mm"] >= 0.0
            ):
                return False
        return derived_positive == expected_positive

    limit_rows = geometry.get("joint_limit_rows", {}) \
        if isinstance(geometry, dict) else {}
    limit_rows_exact = bool(
        isinstance(limit_rows, dict)
        and set(limit_rows) == set(JOINT_ORDER)
        and all(
            isinstance(limit_rows[name], dict)
            and set(limit_rows[name])
            == {"value_deg", "lower_deg", "upper_deg", "pass"}
            and type(limit_rows[name].get("value_deg")) is float
            and limit_rows[name]["value_deg"]
            == float(SELF_COLLISION_POSITIVE_Q_DEG[index])
            and type(limit_rows[name].get("lower_deg")) is float
            and limit_rows[name]["lower_deg"] == float(parse_urdf_limits()[name][0])
            and type(limit_rows[name].get("upper_deg")) is float
            and limit_rows[name]["upper_deg"] == float(parse_urdf_limits()[name][1])
            and limit_rows[name].get("pass") is True
            for index, name in enumerate(JOINT_ORDER)
        )
    )
    geometry_exact = bool(
        isinstance(geometry, dict)
        and set(geometry) == {
            "artifact", "authority", "scipy_version", "positive_control_pair",
            "positive_expected_overlap_pairs", "positive_q_deg", "negative_q_deg",
            "positive", "negative", "joint_limit_rows", "checks", "pass",
        }
        and geometry.get("artifact")
        == "T3U_FROZEN_CONVEX_SELF_COLLISION_GEOMETRY_CERTIFICATE_V1"
        and geometry.get("authority")
        == "composed_attempt3_enabled_convex_hulls_plus_exact_decimal_urdf_fk"
        and isinstance(geometry.get("scipy_version"), str)
        and geometry.get("positive_control_pair") == list(SELF_COLLISION_CONTROL_PAIR)
        and geometry.get("positive_expected_overlap_pairs")
        == [list(pair) for pair in SELF_COLLISION_POSITIVE_EXPECTED_PAIRS]
        and geometry.get("positive_q_deg") == SELF_COLLISION_POSITIVE_Q_DEG.tolist()
        and geometry.get("negative_q_deg") == SELF_COLLISION_NEGATIVE_Q_DEG.tolist()
        and positive_geometry.get("positive_pair_set") == expected_positive_keys
        and negative_geometry.get("positive_pair_set") == []
        and geometry_pose_exact(positive_geometry, expected_positive_keys)
        and geometry_pose_exact(negative_geometry, [])
        and limit_rows_exact
        and set(positive_pair_rows) == expected_pair_keys
        and set(negative_pair_rows) == expected_pair_keys
        and positive_pair_rows["link2__link4"].get("max_intersection_inradius_mm")
        is not None
        and positive_pair_rows["link2__link4"]["max_intersection_inradius_mm"]
        >= SELF_COLLISION_POSITIVE_INRADIUS_GATE_MM
        and positive_pair_rows["link2__link5"].get("max_intersection_inradius_mm")
        is not None
        and positive_pair_rows["link2__link5"]["max_intersection_inradius_mm"]
        >= SELF_COLLISION_POSITIVE_INRADIUS_GATE_MM
        and positive_geometry.get("moving_min_z_mm", -math.inf) >= 71.0
        and positive_geometry.get("object_cylinder_vertex_proxy_separation_mm", -math.inf)
        >= 395.0
        and negative_pair_rows["link2__link4"].get("minimum_separating_face_margin_mm", -math.inf)
        >= SELF_COLLISION_NEGATIVE_SEPARATION_GATE_MM
        and isinstance(geometry_checks, dict)
        and set(geometry_checks)
        == {
            "model_body_inventory_exact",
            "positive_q_inside_exact_urdf_limits",
            "positive_overlap_pair_set_exact",
            "positive_link2_link4_inradius_gte_5mm",
            "positive_link2_link5_inradius_gte_5mm",
            "positive_all_moving_floor_clearance_gte_71mm",
            "positive_object_cylinder_vertex_proxy_separation_gte_395mm",
            "negative_home_has_zero_positive_intersections",
            "negative_link2_link4_signed_margin_lte_minus_60mm",
        }
        and all(type(value) is bool and value is True for value in geometry_checks.values())
        and geometry.get("pass") is True
    )

    precontrol = report.get("precontrol_self_contact_filter_identity")
    behavioral_exact = bool(
        validate_self_contact_filter_identity_semantics(
            precontrol, expected_envs, "precontrol"
        )
        and validate_self_collision_behavioral_control_semantics(
            behavioral, expected_envs
        )
        and _json_type_value_exact(
            behavioral.get("precontrol_self_contact_filter_identity"),
            precontrol,
        )
    )
    return bool(
        set(report)
        == {
            "authority", "attribute", "source_asset", "composed_usd",
            "root_physx_view", "geometry_certificate",
            "precontrol_self_contact_filter_identity", "behavioral_control",
            "deprecated_dynamic_control_removed",
            "behavioral_proof_scope_limited_to_preregistered_poses",
            "pairwise_contact_evidence_authority", "task_physics_steps_before_gate",
            "diagnostic_physics_steps_before_gate", "pass",
        }
        and report.get("authority")
        == "usd_authorship_root_view_plus_same_process_behavioral_control"
        and report.get("attribute") == SELF_COLLISION_ATTR
        and source_exact and composed_exact and root_view_exact
        and geometry_exact and behavioral_exact
        and report.get("deprecated_dynamic_control_removed") is True
        and report.get("behavioral_proof_scope_limited_to_preregistered_poses") is True
        and report.get("pairwise_contact_evidence_authority")
        == "full_step_trace.self_contact_force_w_n_and_self_raw_contact_count"
        and type(report.get("task_physics_steps_before_gate")) is int
        and report["task_physics_steps_before_gate"] == 0
        and type(report.get("diagnostic_physics_steps_before_gate")) is int
        and report["diagnostic_physics_steps_before_gate"] == 2
        and report.get("pass") is True
    )

def validate_task_epoch_reports_semantics(
    instrumentation: Any,
    behavioral_control: Any,
    expected_envs: int,
) -> bool:
    """Recompute the diagnostic/rebaseline/first-step/final clock contracts."""
    if not isinstance(instrumentation, dict):
        return False
    rebaseline = instrumentation.get("post_diagnostic_task_rebaseline")
    first = instrumentation.get("first_task_step_freshness")
    clocks = instrumentation.get("physics_clock_accounting")
    if not all(isinstance(row, dict) for row in (rebaseline, first, clocks)):
        return False
    baseline = rebaseline.get("task_baseline")
    authoritative = rebaseline.get("authoritative_physx_view")
    authoritative_shapes = rebaseline.get("authoritative_physx_view_shapes")
    reset_sensors = rebaseline.get("sensor_rows")
    fresh_sensors = first.get("sensor_rows")
    final = clocks.get("task_final")
    if not all(
        isinstance(row, dict)
        for row in (
            baseline, authoritative, authoritative_shapes,
            reset_sensors, fresh_sensors, final,
        )
    ):
        return False
    expected_reset_sensor_names = {
        "t3u_object_contact",
        *{f"t3u_{body}_support" for body in MOVING_BODIES},
        *{f"t3u_self_{a}__{b}" for a, b in SELF_PAIRS},
    }
    expected_fresh_sensor_names = {
        "object",
        *{f"support:{body}" for body in MOVING_BODIES},
        *{f"self:{a}__{b}" for a, b in SELF_PAIRS},
    }
    reset_sensor_exact = bool(
        set(reset_sensors) == expected_reset_sensor_names
        and all(
            isinstance(row, dict)
            and row.get("timestamp_all_zero") is True
            and row.get("outdated_all_true") is True
            and row.get("raw_contact_counts_pre_task_not_asserted") is True
            and row.get("pass") is True
            and isinstance(row.get("managed_fields"), dict)
            and all(
                value is None
                or (
                    isinstance(value, dict)
                    and value.get("finite") is True
                    and value.get("all_zero") is True
                )
                for value in row["managed_fields"].values()
            )
            and (
                row.get("contact_position") is None
                or row["contact_position"].get("all_nan") is True
            )
            and (
                row.get("contact_position_aggregate") is None
                or row["contact_position_aggregate"].get("all_nan") is True
            )
            for row in reset_sensors.values()
        )
    )
    fresh_sensor_exact = bool(
        set(fresh_sensors) == expected_fresh_sensor_names
        and all(
            isinstance(row, dict)
            and row
            == {
                "timestamp_all_dt": True,
                "last_update_all_dt": True,
                "outdated_all_false": True,
                "managed_public_force_buffers_finite": True,
                "raw_counts_finite_nonnegative_integer": True,
                "raw_counts_within_capacity": True,
                "pass": True,
            }
            for row in fresh_sensors.values()
        )
    )
    authoritative_exact = bool(
        set(authoritative)
        == {
            "robot_q_max_abs_error_rad", "robot_qd_max_abs_rad_s",
            "robot_position_target_max_abs_error_rad",
            "robot_stiffness_max_abs_error", "robot_damping_max_abs_error",
            "object_transform_xyzw_max_abs_error", "object_velocity_max_abs",
        }
        and all(
            type(value) is float and math.isfinite(value) and value >= 0.0
            for value in authoritative.values()
        )
        and authoritative["robot_q_max_abs_error_rad"] <= 1.0e-7
        and authoritative["robot_qd_max_abs_rad_s"] <= 1.0e-7
        and authoritative["robot_position_target_max_abs_error_rad"] <= 1.0e-7
        and authoritative["robot_stiffness_max_abs_error"] <= 1.0e-6
        and authoritative["robot_damping_max_abs_error"] <= 1.0e-6
        and authoritative["object_transform_xyzw_max_abs_error"] <= 1.0e-7
        and authoritative["object_velocity_max_abs"] <= 1.0e-7
        and authoritative_shapes
        == {
            "robot_q": [expected_envs, len(JOINT_ORDER)],
            "robot_qd": [expected_envs, len(JOINT_ORDER)],
            "robot_position_targets": [expected_envs, len(JOINT_ORDER)],
            "robot_stiffness": [expected_envs, len(JOINT_ORDER)],
            "robot_damping": [expected_envs, len(JOINT_ORDER)],
            "object_transforms_xyzw": [expected_envs, 7],
            "object_velocities": [expected_envs, 6],
        }
    )
    diagnostic_after = behavioral_control.get("after", {}) \
        if isinstance(behavioral_control, dict) else {}
    baseline_exact = bool(
        set(baseline)
        == {
            "simulation_manager_num_physics_steps",
            "simulation_manager_time_s", "simulation_context_step_index",
            "simulation_context_time_s", "robot_data_timestamp_s",
            "object_data_timestamp_s",
        }
        and baseline["simulation_manager_num_physics_steps"]
        == diagnostic_after.get("simulation_manager_num_physics_steps")
        and baseline["simulation_context_step_index"]
        == diagnostic_after.get("simulation_context_step_index")
        and math.isclose(
            baseline["simulation_manager_time_s"],
            diagnostic_after.get("simulation_manager_time_s", math.nan),
            rel_tol=0.0, abs_tol=1.0e-12,
        )
        and math.isclose(
            baseline["simulation_context_time_s"],
            diagnostic_after.get("simulation_context_time_s", math.nan),
            rel_tol=0.0, abs_tol=1.0e-12,
        )
        and type(baseline["robot_data_timestamp_s"]) is float
        and math.isfinite(baseline["robot_data_timestamp_s"])
        and baseline["robot_data_timestamp_s"] >= 0.0
        and type(baseline["object_data_timestamp_s"]) is float
        and math.isfinite(baseline["object_data_timestamp_s"])
        and baseline["object_data_timestamp_s"] >= 0.0
    )
    rebaseline_exact = bool(
        rebaseline.get("artifact")
        == "T3U_POST_DIAGNOSTIC_FULL_TASK_REBASELINE_V1"
        and rebaseline.get("physics_steps_added") == 0
        and rebaseline.get("operations")
        == [
            "env.reset", "write_object_pose_velocity",
            "write_robot_joint_state_and_position_target",
            "write_robot_stiffness_damping", "clear_task_counters_latches_actions",
            "scene.write_data_to_sim",
            "sim.forward_zero_physics", "compute_intermediate_values",
            "reset_all_scene_sensors",
        ]
        and isinstance(rebaseline.get("checks"), dict)
        and len(rebaseline["checks"]) == 15
        and rebaseline["checks"].get(
            "reward_buf_absent_before_first_task_step"
        ) is True
        and all(value is True for value in rebaseline["checks"].values())
        and baseline_exact and authoritative_exact and reset_sensor_exact
        and rebaseline.get("pass") is True
    )
    behavioral_exact = validate_self_collision_behavioral_control_semantics(
        behavioral_control, expected_envs
    )
    diagnostic_before = (
        behavioral_control.get("before", {})
        if isinstance(behavioral_control, dict) else {}
    )
    diagnostic_after = (
        behavioral_control.get("after", {})
        if isinstance(behavioral_control, dict) else {}
    )
    diagnostic_dts = (
        behavioral_control.get("callback_dts_s")
        if isinstance(behavioral_control, dict) else None
    )
    task_dts = clocks.get("task_callback_dts_s")

    def callback_vector_exact(value: Any, expected_count: int) -> bool:
        return bool(
            isinstance(value, list)
            and len(value) == expected_count
            and all(
                type(item) is float
                and math.isfinite(item)
                and math.isclose(
                    item, DT_S, rel_tol=0.0,
                    abs_tol=CLOCK_CALLBACK_NOMINAL_DT_ABS_TOL_S,
                )
                for item in value
            )
        )

    diagnostic_dts_exact = callback_vector_exact(diagnostic_dts, 2)
    task_dts_exact = callback_vector_exact(task_dts, TOTAL_STEPS)
    if not (diagnostic_dts_exact and task_dts_exact):
        return False
    diagnostic_fsum = float(math.fsum(diagnostic_dts))
    task_fsum = float(math.fsum(task_dts))
    combined_fsum = float(math.fsum([*diagnostic_dts, *task_dts]))

    first_dts = first.get("physics_callback_dts_s")
    first_before = first.get("clock_before")
    first_after = first.get("clock_after")
    first_comparisons = first.get("elapsed_comparisons")
    first_clock_keys = {
        "simulation_manager_num_physics_steps", "simulation_manager_time_s",
        "simulation_context_step_index", "simulation_context_time_s",
    }
    first_clock_schema_exact = bool(
        isinstance(first_before, dict) and set(first_before) == first_clock_keys
        and isinstance(first_after, dict) and set(first_after) == first_clock_keys
        and all(
            type(row.get("simulation_manager_num_physics_steps")) is int
            and type(row.get("simulation_context_step_index")) is int
            and type(row.get("simulation_manager_time_s")) is float
            and math.isfinite(row["simulation_manager_time_s"])
            and type(row.get("simulation_context_time_s")) is float
            and math.isfinite(row["simulation_context_time_s"])
            for row in (first_before, first_after)
        )
    )
    if first_clock_schema_exact and callback_vector_exact(first_dts, 1):
        first_manager_delta = float(
            first_after["simulation_manager_time_s"]
            - first_before["simulation_manager_time_s"]
        )
        first_context_delta = float(
            first_after["simulation_context_time_s"]
            - first_before["simulation_context_time_s"]
        )
        expected_first_comparisons = {
            "manager": _clock_elapsed_comparison(
                first_manager_delta, first_dts[0]
            ),
            "context": _clock_elapsed_comparison(
                first_context_delta, first_dts[0]
            ),
        }
        expected_first_tolerance = _clock_elapsed_abs_tolerance_s(
            first_dts[0], first_manager_delta, first_context_delta
        )
    else:
        expected_first_comparisons = {}
        expected_first_tolerance = math.nan
    first_check_keys = {
        "task_callback_exactly_one", "task_callback_dt_finite_nominal",
        "manager_clock_new_epoch_plus_one",
        "simulation_context_new_epoch_plus_one", "task_counters_exactly_one",
        "task_reset_flags_false", "asset_data_timestamps_fresh_dt",
        "all_sensor_epochs_and_public_data_fresh",
        "reward_buf_created_from_return_exact_zero_finite",
    }
    first_exact = bool(
        set(first)
        == {
            "artifact", "local_task_step", "nominal_dt_s_informational",
            "physics_callback_dts_s", "elapsed_abs_tolerance_s",
            "clock_before", "clock_after", "elapsed_comparisons",
            "sensor_rows", "support_positive_control_authority",
            "reward_lifecycle", "checks", "pass",
        }
        and first.get("artifact") == "T3U_FIRST_TASK_STEP_FRESH_EPOCH_GATE_V1"
        and first.get("local_task_step") == 1
        and first.get("nominal_dt_s_informational") == DT_S
        and first_dts == [task_dts[0]]
        and first_clock_schema_exact
        and first_before
        == {
            key: baseline[key] for key in first_clock_keys
        }
        and first_after["simulation_manager_num_physics_steps"]
        - first_before["simulation_manager_num_physics_steps"] == 1
        and first_after["simulation_context_step_index"]
        - first_before["simulation_context_step_index"] == 1
        and first_before["simulation_manager_num_physics_steps"]
        == first_before["simulation_context_step_index"]
        and first_before["simulation_manager_time_s"]
        == first_before["simulation_context_time_s"]
        and first_after["simulation_manager_num_physics_steps"]
        == first_after["simulation_context_step_index"]
        and first_after["simulation_manager_time_s"]
        == first_after["simulation_context_time_s"]
        and first.get("elapsed_abs_tolerance_s") == expected_first_tolerance
        and _json_type_value_exact(
            first_comparisons, expected_first_comparisons
        )
        and all(row["pass"] is True for row in expected_first_comparisons.values())
        and first.get("support_positive_control_authority")
        == "unchanged_median_of_final_60_settle_samples_not_first_frame"
        and first.get("reward_lifecycle")
        == {
            "returned_shape": [expected_envs],
            "env_reward_buf_shape": [expected_envs],
            "returned_finite": True,
            "returned_all_zero": True,
            "returned_equals_env_reward_buf": True,
        }
        and isinstance(first.get("checks"), dict)
        and set(first["checks"]) == first_check_keys
        and all(
            type(value) is bool and value is True
            for value in first["checks"].values()
        )
        and fresh_sensor_exact and first.get("pass") is True
    )

    final_schema_exact = bool(
        set(final)
        == {
            "simulation_manager_num_physics_steps",
            "simulation_manager_time_s", "simulation_context_step_index",
            "simulation_context_time_s", "env_sim_step_counter",
            "common_step_counter", "episode_length_buf",
        }
        and type(final.get("simulation_manager_num_physics_steps")) is int
        and type(final.get("simulation_context_step_index")) is int
        and type(final.get("simulation_manager_time_s")) is float
        and math.isfinite(final["simulation_manager_time_s"])
        and type(final.get("simulation_context_time_s")) is float
        and math.isfinite(final["simulation_context_time_s"])
        and type(final.get("env_sim_step_counter")) is int
        and type(final.get("common_step_counter")) is int
        and isinstance(final.get("episode_length_buf"), list)
        and all(type(item) is int for item in final["episode_length_buf"])
    )
    if not final_schema_exact:
        return False
    expected_deltas = {
        "diagnostic_manager_s": float(
            diagnostic_after["simulation_manager_time_s"]
            - diagnostic_before["simulation_manager_time_s"]
        ),
        "diagnostic_context_s": float(
            diagnostic_after["simulation_context_time_s"]
            - diagnostic_before["simulation_context_time_s"]
        ),
        "task_manager_s": float(
            final["simulation_manager_time_s"]
            - baseline["simulation_manager_time_s"]
        ),
        "task_context_s": float(
            final["simulation_context_time_s"]
            - baseline["simulation_context_time_s"]
        ),
        "combined_manager_s": float(
            final["simulation_manager_time_s"]
            - diagnostic_before["simulation_manager_time_s"]
        ),
        "combined_context_s": float(
            final["simulation_context_time_s"]
            - diagnostic_before["simulation_context_time_s"]
        ),
    }
    expected_tolerances = {
        "diagnostic_s": _clock_elapsed_abs_tolerance_s(
            diagnostic_fsum, expected_deltas["diagnostic_manager_s"],
            expected_deltas["diagnostic_context_s"],
        ),
        "task_s": _clock_elapsed_abs_tolerance_s(
            task_fsum, expected_deltas["task_manager_s"],
            expected_deltas["task_context_s"],
        ),
        "combined_s": _clock_elapsed_abs_tolerance_s(
            combined_fsum, expected_deltas["combined_manager_s"],
            expected_deltas["combined_context_s"],
        ),
    }
    expected_comparisons = {
        "diagnostic_manager": _clock_elapsed_comparison(
            expected_deltas["diagnostic_manager_s"], diagnostic_fsum
        ),
        "diagnostic_context": _clock_elapsed_comparison(
            expected_deltas["diagnostic_context_s"], diagnostic_fsum
        ),
        "task_manager": _clock_elapsed_comparison(
            expected_deltas["task_manager_s"], task_fsum
        ),
        "task_context": _clock_elapsed_comparison(
            expected_deltas["task_context_s"], task_fsum
        ),
        "combined_manager": _clock_elapsed_comparison(
            expected_deltas["combined_manager_s"], combined_fsum
        ),
        "combined_context": _clock_elapsed_comparison(
            expected_deltas["combined_context_s"], combined_fsum
        ),
    }
    expected_clock_checks = {
        "diagnostic_callback_count_2": len(diagnostic_dts) == 2,
        "task_callback_count_2340": len(task_dts) == TOTAL_STEPS,
        "combined_callback_count_2342": (
            len(diagnostic_dts) + len(task_dts) == TOTAL_STEPS + 2
        ),
        "all_callback_dts_finite_nominal": (
            diagnostic_dts_exact and task_dts_exact
        ),
        "manager_diagnostic_step_delta_2": (
            diagnostic_after["simulation_manager_num_physics_steps"]
            - diagnostic_before["simulation_manager_num_physics_steps"] == 2
        ),
        "manager_diagnostic_time_delta_callback_fsum": (
            expected_comparisons["diagnostic_manager"]["pass"] is True
        ),
        "simulation_context_diagnostic_step_delta_2": (
            diagnostic_after["simulation_context_step_index"]
            - diagnostic_before["simulation_context_step_index"] == 2
        ),
        "simulation_context_diagnostic_time_delta_callback_fsum": (
            expected_comparisons["diagnostic_context"]["pass"] is True
        ),
        "manager_task_step_delta_2340": (
            final["simulation_manager_num_physics_steps"]
            - baseline["simulation_manager_num_physics_steps"] == TOTAL_STEPS
        ),
        "manager_task_time_delta_callback_fsum": (
            expected_comparisons["task_manager"]["pass"] is True
        ),
        "simulation_context_task_step_delta_2340": (
            final["simulation_context_step_index"]
            - baseline["simulation_context_step_index"] == TOTAL_STEPS
        ),
        "simulation_context_task_time_delta_callback_fsum": (
            expected_comparisons["task_context"]["pass"] is True
        ),
        "task_counters_2340": bool(
            final["env_sim_step_counter"] == TOTAL_STEPS
            and final["common_step_counter"] == TOTAL_STEPS
            and final["episode_length_buf"] == [TOTAL_STEPS] * expected_envs
        ),
        "combined_manager_steps_2342": (
            final["simulation_manager_num_physics_steps"]
            - diagnostic_before["simulation_manager_num_physics_steps"]
            == TOTAL_STEPS + 2
        ),
        "combined_manager_time_delta_callback_fsum": (
            expected_comparisons["combined_manager"]["pass"] is True
        ),
        "combined_simulation_context_steps_2342": (
            final["simulation_context_step_index"]
            - diagnostic_before["simulation_context_step_index"]
            == TOTAL_STEPS + 2
        ),
        "combined_simulation_context_time_delta_callback_fsum": (
            expected_comparisons["combined_context"]["pass"] is True
        ),
        "manager_context_task_baselines_exact": bool(
            _clock_manager_context_equal(
                baseline["simulation_manager_num_physics_steps"],
                baseline["simulation_manager_time_s"],
                baseline["simulation_context_step_index"],
                baseline["simulation_context_time_s"],
            )
        ),
        "manager_context_diagnostic_before_after_exact": bool(
            _clock_manager_context_equal(
                diagnostic_before["simulation_manager_num_physics_steps"],
                diagnostic_before["simulation_manager_time_s"],
                diagnostic_before["simulation_context_step_index"],
                diagnostic_before["simulation_context_time_s"],
            )
            and _clock_manager_context_equal(
                diagnostic_after["simulation_manager_num_physics_steps"],
                diagnostic_after["simulation_manager_time_s"],
                diagnostic_after["simulation_context_step_index"],
                diagnostic_after["simulation_context_time_s"],
            )
        ),
        "manager_context_finals_exact": bool(
            _clock_manager_context_equal(
                final["simulation_manager_num_physics_steps"],
                final["simulation_manager_time_s"],
                final["simulation_context_step_index"],
                final["simulation_context_time_s"],
            )
        ),
    }
    clock_checks = clocks.get("checks")
    clocks_exact = bool(
        set(clocks)
        == {
            "artifact", "elapsed_time_authority", "diagnostic_physics_steps",
            "task_physics_steps", "combined_physics_steps",
            "task_local_step_range", "nominal_dt_s_informational",
            "nominal_task_duration_s_informational",
            "nominal_combined_duration_s_informational",
            "callback_nominal_dt_abs_tolerance_s", "elapsed_ulp_multiplier",
            "diagnostic_callback_dts_s", "task_callback_dts_s",
            "task_callback_count", "task_callback_dt_min_s",
            "task_callback_dt_max_s", "callback_fsum_s",
            "observed_elapsed_deltas_s", "elapsed_abs_tolerance_s",
            "elapsed_comparisons", "task_baseline", "task_final", "checks",
            "pass",
        }
        and clocks.get("artifact")
        == "T3U_DIAGNOSTIC_AND_TASK_PHYSICS_CLOCK_ACCOUNTING_V2"
        and clocks.get("elapsed_time_authority")
        == "math.fsum_of_durable_physics_callback_step_size_vectors"
        and clocks.get("diagnostic_physics_steps") == 2
        and clocks.get("task_physics_steps") == TOTAL_STEPS
        and clocks.get("combined_physics_steps") == TOTAL_STEPS + 2
        and clocks.get("task_local_step_range") == [1, TOTAL_STEPS]
        and clocks.get("nominal_dt_s_informational") == DT_S
        and clocks.get("nominal_task_duration_s_informational")
        == TOTAL_STEPS * DT_S
        and clocks.get("nominal_combined_duration_s_informational")
        == (TOTAL_STEPS + 2) * DT_S
        and clocks.get("callback_nominal_dt_abs_tolerance_s")
        == CLOCK_CALLBACK_NOMINAL_DT_ABS_TOL_S
        and clocks.get("elapsed_ulp_multiplier")
        == CLOCK_ELAPSED_ULP_MULTIPLIER
        and clocks.get("diagnostic_callback_dts_s") == diagnostic_dts
        and clocks.get("task_callback_count") == TOTAL_STEPS
        and clocks.get("task_callback_dt_min_s") == min(task_dts)
        and clocks.get("task_callback_dt_max_s") == max(task_dts)
        and clocks.get("callback_fsum_s")
        == {
            "diagnostic": diagnostic_fsum,
            "task": task_fsum,
            "combined": combined_fsum,
        }
        and clocks.get("observed_elapsed_deltas_s") == expected_deltas
        and clocks.get("elapsed_abs_tolerance_s") == expected_tolerances
        and _json_type_value_exact(
            clocks.get("elapsed_comparisons"), expected_comparisons
        )
        and all(row["pass"] is True for row in expected_comparisons.values())
        and _json_type_value_exact(clocks.get("task_baseline"), baseline)
        and isinstance(clock_checks, dict)
        and _json_type_value_exact(clock_checks, expected_clock_checks)
        and all(
            type(value) is bool and value is True
            for value in clock_checks.values()
        )
        and clocks.get("pass") is True
    )
    return bool(
        behavioral_exact and rebaseline_exact and first_exact and clocks_exact
    )


def validate_result_semantics(
    profile: str,
    paths: dict[str, Path],
    results: dict[str, Any],
    plan: dict[str, Any],
) -> dict[str, bool]:
    """Recompute runtime/plan integrity from durable JSON and authoritative NPZ."""
    expected_envs = 8 if profile == PREFLIGHT_PROFILE else 64
    expected_planned = 5 if profile == PREFLIGHT_PROFILE else 40
    expected_active = 5 if profile == PREFLIGHT_PROFILE else 10
    expected_prereg_sha = (
        PREFLIGHT_PREREG_SHA256
        if profile == PREFLIGHT_PROFILE else CANONICAL_PREREG_SHA256
    )
    try:
        retired_preflight2_exact = _json_type_value_exact(
            plan.get("retired_preflight2"), validate_preflight2_retirement()
        )
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError):
        retired_preflight2_exact = False
    try:
        retired_preflight3_exact = _json_type_value_exact(
            plan.get("retired_preflight3_launch"),
            validate_preflight3_launch_retirement(),
        )
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError):
        retired_preflight3_exact = False
    try:
        retired_preflight4_exact = _json_type_value_exact(
            plan.get("retired_preflight4_launch"),
            validate_preflight4_launch_retirement(),
        )
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError):
        retired_preflight4_exact = False
    try:
        retired_preflight5_exact = _json_type_value_exact(
            plan.get("retired_preflight5_dynamic_control_abort"),
            validate_preflight5_retirement(),
        )
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError):
        retired_preflight5_exact = False
    try:
        retired_preflight6_exact = _json_type_value_exact(
            plan.get("retired_preflight6_post_activation_dynamic_control_abort"),
            validate_preflight6_retirement(),
        )
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError):
        retired_preflight6_exact = False
    try:
        retired_preflight7_exact = _json_type_value_exact(
            plan.get("retired_preflight7_filter_representation_abort"),
            validate_preflight7_retirement(),
        )
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError):
        retired_preflight7_exact = False
    try:
        retired_preflight8_exact = _json_type_value_exact(
            plan.get("retired_preflight8_reward_buffer_lifecycle_abort"),
            validate_preflight8_retirement(),
        )
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError):
        retired_preflight8_exact = False
    try:
        retired_preflight9_exact = _json_type_value_exact(
            plan.get("retired_preflight9_clock_accounting_abort"),
            validate_preflight9_retirement(),
        )
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError):
        retired_preflight9_exact = False
    expected_result_keys = {
        "tool", "case", "tag", "profile", "scientific_authoritative",
        "scientific_verdict_preclose_candidate", "internal_verdict", "object",
        "fixed_controls", "gates", "plan_counts", "representative_binding",
        "classifications", "classification_summary", "metrics", "instrumentation",
        "runtime_instrumentation_pass", "stage_collision_audit",
        "static_collision_geometry", "pinch_calibration",
        "instrumentation_witness_source", "rerun", "decision_snapshot",
        "provenance", "artifact_hashes_preclose", "required_postphysics_pending",
        "wall_seconds", "scope_warning",
    }
    recomputed_trial_contract = validate_trial_set_contract(
        plan, profile, "terminal_recompute", hard_fail=False
    )
    stored_trial_contracts = plan.get("trial_set_contracts", {})
    trial_contract_exact = bool(
        recomputed_trial_contract.get("pass") is True
        and set(stored_trial_contracts) == {"ik_frame", "post_static_clearance"}
        and stored_trial_contracts["ik_frame"].get("pass") is True
        and stored_trial_contracts["post_static_clearance"].get("pass") is True
        and stored_trial_contracts["ik_frame"].get("actual_planned_ordered")
        == recomputed_trial_contract["actual_planned_ordered"]
        and stored_trial_contracts["ik_frame"].get("actual_active_ordered")
        == recomputed_trial_contract["actual_active_ordered"]
        and stored_trial_contracts["post_static_clearance"].get("actual_planned_ordered")
        == recomputed_trial_contract["actual_planned_ordered"]
        and stored_trial_contracts["post_static_clearance"].get("actual_active_ordered")
        == recomputed_trial_contract["actual_active_ordered"]
    )

    stage_readback = results.get("object", {}).get("stage_readback", {})
    stage_units = stage_readback.get("stage_units", {})
    tolerances = stage_readback.get("tolerances", {})
    object_rows = stage_readback.get("object_clones", [])
    linear_tol = tolerances.get("linear_si_m_abs")
    mass_tol = tolerances.get("mass_si_kg_abs")
    material_tol = tolerances.get("material_abs")
    units_tol = tolerances.get("stage_unit_abs")
    tolerances_valid = bool(
        linear_tol == 1.0e-9 and mass_tol == 5.0e-9
        and material_tol == 1.0e-6 and units_tol == 1.0e-12
    )
    object_rows_exact = bool(
        isinstance(object_rows, list)
        and len(object_rows) == expected_envs
        and tolerances_valid
        and all(
            isinstance(row, dict)
            and row.get("env_index") == env_index
            and row.get("root_path") == f"/World/envs/env_{env_index}/Sponge"
            and row.get("root_type") == "Xform"
            and row.get("cylinder_path")
            == f"/World/envs/env_{env_index}/Sponge/geometry/mesh"
            and row.get("cylinder_type") == "Cylinder"
            and row.get("axis") == "Z"
            and math.isclose(float(row.get("radius_si_m", math.nan)), OBJ_RADIUS_M,
                             rel_tol=0.0, abs_tol=float(linear_tol))
            and math.isclose(float(row.get("height_si_m", math.nan)), OBJ_HEIGHT_M,
                             rel_tol=0.0, abs_tol=float(linear_tol))
            and math.isclose(float(row.get("mass_si_kg", math.nan)), OBJ_MASS_KG,
                             rel_tol=0.0, abs_tol=float(mass_tol))
            and row.get("collision_api_present") is True
            and row.get("collision_enabled") is True
            and row.get("mass_api_path") == f"/World/envs/env_{env_index}/Sponge"
            and row.get("mass_api_present") is True
            and row.get("rigid_body_api_present") is True
            and row.get("physx_rigid_body_api_present") is True
            and row.get("rigid_body_enabled") is True
            and row.get("disable_gravity") is False
            and row.get("material_binding_kind") == "computed_bound_physics"
            and row.get("material_binding_relationship_path")
            == (
                f"/World/envs/env_{env_index}/Sponge/geometry/mesh"
                ".material:binding:physics"
            )
            and row.get("material_path")
            == f"/World/envs/env_{env_index}/Sponge/geometry/material"
            and row.get("material_path") == row.get("expected_material_path")
            and row.get("material_type") == "Material"
            and row.get("material_api_present") is True
            and row.get("physx_material_api_present") is True
            and math.isclose(float(row.get("static_friction", math.nan)), STATIC_FRICTION,
                             rel_tol=0.0, abs_tol=float(material_tol))
            and math.isclose(float(row.get("dynamic_friction", math.nan)), DYNAMIC_FRICTION,
                             rel_tol=0.0, abs_tol=float(material_tol))
            and math.isclose(float(row.get("restitution", math.nan)), RESTITUTION,
                             rel_tol=0.0, abs_tol=float(material_tol))
            and row.get("friction_combine_mode") == "average"
            and row.get("restitution_combine_mode") == "average"
            and row.get("pass") is True
            for env_index, row in enumerate(object_rows)
        )
    )
    object_readback_exact = bool(
        stage_readback.get("pass") is True
        and stage_readback.get("authority")
        == "composed_stage_schema_readback_before_task_step"
        and stage_readback.get("clone_count_expected") == expected_envs
        and stage_readback.get("clone_count_actual") == expected_envs
        and stage_readback.get("effective_pair_friction")
        == "not_computed_not_measured_not_claimed"
        and stage_units
        == {
            "meters_per_unit": 1.0,
            "kilograms_per_unit": 1.0,
            "expected_each": 1.0,
            "pass": True,
        }
        and object_rows_exact
        and isinstance(stage_readback.get("all_stage_physics_materials"), list)
        and stage_readback.get("all_stage_physics_materials")
        and results.get("instrumentation", {}).get("object_stage_readback")
        == stage_readback
    )

    instrumentation = results.get("instrumentation", {})
    support_filter = instrumentation.get("support_filter_audit", {})
    self_filter = instrumentation.get("self_filter_audit", {})

    def filter_row_exact(
        row: Any, expected_label: str, expected_expr: str, expected_paths: list[str]
    ) -> bool:
        if not isinstance(row, dict):
            return False
        expected_glob = expected_expr.replace(".*", "*")
        stage_rows = row.get("expected_stage_paths", [])
        return bool(
            set(row)
            == {
                "label", "force_matrix_shape", "filter_count", "actual_filter_paths",
                "expected_concrete_env0_representative",
                "physx_replicated_filter_representation",
                "expected_filter_expression", "accepted_physx_glob",
                "resolved_stage_paths_from_expression", "expected_stage_paths", "pass",
            }
            and row.get("label") == expected_label
            and row.get("force_matrix_shape") == [expected_envs, 1, 1, 3]
            and row.get("filter_count") == 1
            and (
                row.get("actual_filter_paths") == [expected_paths[0]]
                if "/Robot/" in expected_expr
                else row.get("actual_filter_paths")
                in ([expected_expr], [expected_glob])
            )
            and row.get("expected_concrete_env0_representative") == expected_paths[0]
            and row.get("physx_replicated_filter_representation")
            == (
                "single_logical_filter_as_env0_concrete_representative"
                if "/Robot/" in expected_expr
                else "authored_expression_or_physx_glob"
            )
            and row.get("expected_filter_expression") == expected_expr
            and row.get("accepted_physx_glob") == expected_glob
            and row.get("resolved_stage_paths_from_expression") == sorted(expected_paths)
            and isinstance(stage_rows, list)
            and [stage_row.get("path") for stage_row in stage_rows] == expected_paths
            and all(stage_row.get("valid") is True for stage_row in stage_rows)
            and row.get("pass") is True
        )

    # The exact discovered collider is recorded in every support row; require all
    # six moving-link support reporters to agree rather than reconstructing cfg.
    support_ground_paths = {
        row.get("expected_filter_expression")
        for row in support_filter.values() if isinstance(row, dict)
    }
    support_filters_exact = bool(
        set(support_filter) == set(MOVING_BODIES)
        and len(support_ground_paths) == 1
        and all(
            filter_row_exact(
                support_filter[body], f"support:{body}",
                support_filter[body].get("expected_filter_expression"),
                [support_filter[body].get("expected_filter_expression")],
            )
            for body in MOVING_BODIES
        )
    )
    expected_self_keys = {f"{a}__{b}" for a, b in SELF_PAIRS}
    self_filters_exact = bool(
        set(self_filter) == expected_self_keys
        and all(
            filter_row_exact(
                self_filter[f"{a}__{b}"], f"self:{a}__{b}",
                f"/World/envs/env_.*/Robot/{b}",
                [f"/World/envs/env_{index}/Robot/{b}" for index in range(expected_envs)],
            )
            for a, b in SELF_PAIRS
        )
    )

    filter_reuse = instrumentation.get("self_contact_filter_identity_reuse", {})
    filter_reuse_checks = filter_reuse.get("checks", {}) \
        if isinstance(filter_reuse, dict) else {}
    precontrol_filter = filter_reuse.get("precontrol", {}) \
        if isinstance(filter_reuse, dict) else {}
    postcontrol_filter = filter_reuse.get("postcontrol_pre_task", {}) \
        if isinstance(filter_reuse, dict) else {}
    filter_identity_reuse_exact = bool(
        isinstance(filter_reuse, dict)
        and set(filter_reuse)
        == {"artifact", "precontrol", "postcontrol_pre_task", "checks", "pass"}
        and filter_reuse.get("artifact")
        == "T3U_SELF_CONTACT_FILTER_IDENTITY_REUSE_V1"
        and validate_self_contact_filter_identity_semantics(
            precontrol_filter, expected_envs, "precontrol"
        )
        and validate_self_contact_filter_identity_semantics(
            postcontrol_filter, expected_envs, "postcontrol_pre_task"
        )
        and _json_type_value_exact(
            precontrol_filter.get("scene_clone_configuration"),
            postcontrol_filter.get("scene_clone_configuration"),
        )
        and _json_type_value_exact(
            precontrol_filter.get("pair_rows"),
            postcontrol_filter.get("pair_rows"),
        )
        and _json_type_value_exact(
            precontrol_filter,
            results.get("stage_collision_audit", {}).get(
                "self_collision_readback", {}
            ).get("precontrol_self_contact_filter_identity"),
        )
        and _json_type_value_exact(
            postcontrol_filter.get("clock_before_control"),
            results.get("stage_collision_audit", {}).get(
                "self_collision_readback", {}
            ).get("behavioral_control", {}).get("after"),
        )
        and isinstance(filter_reuse_checks, dict)
        and set(filter_reuse_checks)
        == {
            "precontrol_pass", "postcontrol_pass",
            "scene_clone_configuration_equal", "pair_rows_exactly_equal",
            "expected_counts_equal", "postcontrol_clock_equals_rebaseline_input",
        }
        and all(
            type(value) is bool and value is True
            for value in filter_reuse_checks.values()
        )
        and filter_reuse.get("pass") is True
        and _json_type_value_exact(
            results.get("stage_collision_audit", {}).get(
                "self_contact_filter_identity_reuse"
            ),
            filter_reuse,
        )
    )

    stage_collision = results.get("stage_collision_audit", {})
    reporter = stage_collision.get("cloned_object_and_moving_body_reporters", {})
    reporter_exact = bool(
        reporter
        == {
            "pass": True,
            "checked": expected_envs * (1 + len(CONTACT_REPORT_BODIES)),
            "expected": expected_envs * (1 + len(CONTACT_REPORT_BODIES)),
            "subjects_per_clone": ["object", *CONTACT_REPORT_BODIES],
            "threshold": 0.0,
        }
    )
    self_collision = stage_collision.get("self_collision_readback", {})
    self_collision_exact = bool(
        validate_self_collision_readback_semantics(
            self_collision, expected_envs
        )
        and instrumentation.get("self_collision_readback")
        == self_collision
        and plan.get("effective_self_collision_readback")
        == self_collision
        and results.get("fixed_controls", {}).get(
            "self_collision_setting_authority"
        ) == "stage_collision_audit.self_collision_readback"
        and results.get("fixed_controls", {}).get(
            "self_collision_setting_not_physical_contact_proof"
        ) is True
        and results.get("fixed_controls", {}).get(
            "self_collision_behavioral_control_scope"
        ) == "positive_two_pairs_then_negative_HOME__not_all_pose_proof"
    )
    task_epoch_exact = bool(
        stage_collision.get("pass") is True
        and
        validate_task_epoch_reports_semantics(
            instrumentation,
            self_collision.get("behavioral_control"),
            expected_envs,
        )
        and _json_type_value_exact(
            instrumentation.get("post_diagnostic_task_rebaseline"),
            stage_collision.get("post_diagnostic_task_rebaseline"),
        )
        and _json_type_value_exact(
            instrumentation.get("first_task_step_freshness"),
            stage_collision.get("first_task_step_freshness"),
        )
        and _json_type_value_exact(
            instrumentation.get("physics_clock_accounting"),
            stage_collision.get("physics_clock_accounting"),
        )
    )
    joint_limits_readback = stage_collision.get("joint_limits_readback", {})
    joint_limits_exact = bool(
        validate_joint_limit_readback_semantics(
            joint_limits_readback, expected_envs
        )
        and instrumentation.get("joint_limits_readback")
        == joint_limits_readback
        and plan.get("effective_joint_limits_readback")
        == joint_limits_readback
        and results.get("fixed_controls", {}).get("joint_limit_authority")
        == "stage_collision_audit.joint_limits_readback"
    )
    fixed_base_readback = stage_collision.get("fixed_base_readback", {})
    fixed_base_readback_exact = bool(
        validate_fixed_base_readback_semantics(
            fixed_base_readback, expected_envs
        )
        and instrumentation.get("fixed_base_readback")
        == fixed_base_readback
        and plan.get("fixed_base_runtime_readback")
        == fixed_base_readback
        and results.get("fixed_controls", {}).get("fixed_base_authority")
        == "stage_collision_audit.fixed_base_readback"
    )

    with np.load(paths["trace.npz"], allow_pickle=False) as archive:
        trace = {name: archive[name] for name in archive.files}
    required_finite_trace = {
        "physics_step", "sim_time_s", "phase_id", "phase_step",
        "joint_pos_deg", "joint_planned_target_deg", "joint_target_deg",
        "joint_vel_rad_s",
        "object_pos_m", "object_quat_wxyz", "object_lin_vel_m_s",
        "object_ang_vel_rad_s", "tcp_pos_m", "moving_body_pos_m",
        "moving_body_quat_wxyz", "moving_body_lin_vel_m_s",
        "moving_body_ang_vel_rad_s", "object_force_w_n",
        "moving_link_support_force_w_n", "self_contact_force_w_n",
        "self_contact_body_pos_m",
        "fixed_base_pos_m", "fixed_base_quat_wxyz",
        "fixed_base_lin_vel_m_s", "fixed_base_ang_vel_rad_s",
        "object_raw_contact_count", "support_raw_contact_count",
        "self_raw_contact_count", "object_tilt_deg",
        "witness_moving_support_force_w_n", "witness_joint_pos_deg",
        "witness_joint_target_deg",
    }
    trace_shape_exact = bool(
        required_finite_trace.issubset(trace)
        and trace.get("physics_step", np.empty(0)).shape == (TOTAL_STEPS,)
        and all(
            trace[name].shape[0] == TOTAL_STEPS
            for name in required_finite_trace if name in trace
        )
        and trace.get("joint_pos_deg", np.empty((0, 0))).shape[1] == expected_active
        and trace.get("joint_pos_deg", np.empty((0, 0, 0))).shape
        == (TOTAL_STEPS, expected_active, len(JOINT_ORDER))
        and trace.get("joint_planned_target_deg", np.empty((0, 0, 0))).shape
        == (TOTAL_STEPS, expected_active, len(JOINT_ORDER))
        and trace.get("joint_target_deg", np.empty((0, 0, 0))).shape
        == (TOTAL_STEPS, expected_active, len(JOINT_ORDER))
        and trace.get("self_contact_force_w_n", np.empty((0, 0, 0, 0))).shape
        == (TOTAL_STEPS, expected_active, len(SELF_PAIRS), 3)
        and trace.get("self_raw_contact_count", np.empty((0, 0, 0))).shape
        == (TOTAL_STEPS, expected_active, len(SELF_PAIRS))
        and trace.get("self_contact_body_pos_m", np.empty((0, 0, 0, 0))).shape
        == (TOTAL_STEPS, expected_active, len(SELF_CONTACT_BODIES), 3)
        and trace.get("fixed_base_pos_m", np.empty((0, 0, 0))).shape
        == (TOTAL_STEPS, expected_active, 3)
        and trace.get("fixed_base_quat_wxyz", np.empty((0, 0, 0))).shape
        == (TOTAL_STEPS, expected_active, 4)
        and trace.get("fixed_base_lin_vel_m_s", np.empty((0, 0, 0))).shape
        == (TOTAL_STEPS, expected_active, 3)
        and trace.get("fixed_base_ang_vel_rad_s", np.empty((0, 0, 0))).shape
        == (TOTAL_STEPS, expected_active, 3)
    )
    trace_cadence_exact = bool(
        trace_shape_exact
        and validate_authoritative_trace_cadence(trace)
    )
    trace_finite_exact = bool(
        trace_shape_exact and trace_cadence_exact
        and all(np.isfinite(trace[name]).all() for name in required_finite_trace)
    )
    counts_exact = False
    contact_positions_exact = False
    quaternion_exact = False
    joint_limit_application_exact = False
    fixed_base_stability_exact = False
    if trace_shape_exact:
        counts_exact = True
        for name in (
            "object_raw_contact_count", "support_raw_contact_count",
            "self_raw_contact_count",
        ):
            value = trace[name]
            counts_exact = bool(
                counts_exact
                and np.isfinite(value).all()
                and (value >= 0).all()
                and np.equal(value, np.round(value)).all()
                and (value <= 256).all()
            )
        counts_exact = bool(
            counts_exact
            and (trace["object_raw_contact_count"].sum(axis=-1) <= 256).all()
        )
        object_positions = trace.get("object_contact_pos_m")
        if object_positions is not None:
            object_counts = trace["object_raw_contact_count"]
            contact_positions_exact = bool(
                not np.isinf(object_positions).any()
                and np.logical_or(
                    object_counts <= 0,
                    np.isfinite(object_positions).all(axis=-1),
                ).all()
            )
        object_q_norm = np.linalg.norm(trace["object_quat_wxyz"], axis=-1)
        body_q_norm = np.linalg.norm(trace["moving_body_quat_wxyz"], axis=-1)
        quaternion_exact = bool(
            np.isfinite(object_q_norm).all() and np.isfinite(body_q_norm).all()
            and np.max(np.abs(object_q_norm - 1.0)) <= 1.0e-3
            and np.max(np.abs(body_q_norm - 1.0)) <= 1.0e-3
        )
        planned_target_deg = trace["joint_planned_target_deg"]
        applied_target_deg = trace["joint_target_deg"]
        actual_joint_deg = trace["joint_pos_deg"]
        expected_limits_deg = np.asarray(
            joint_limits_readback.get("expected_urdf_limits_deg", []),
            dtype=np.float64,
        )
        planned_applied_tol_deg = math.degrees(1.0e-7)
        actual_limit_tol_deg = math.degrees(1.0e-5)
        joint_limit_application_exact = bool(
            joint_limits_exact
            and expected_limits_deg.shape == (len(JOINT_ORDER), 2)
            and np.max(np.abs(planned_target_deg - applied_target_deg))
            <= planned_applied_tol_deg
            and (applied_target_deg >= expected_limits_deg[:, 0]).all()
            and (applied_target_deg <= expected_limits_deg[:, 1]).all()
            and (
                actual_joint_deg
                >= expected_limits_deg[:, 0] - actual_limit_tol_deg
            ).all()
            and (
                actual_joint_deg
                <= expected_limits_deg[:, 1] + actual_limit_tol_deg
            ).all()
        )
        fixed_base_pos = trace["fixed_base_pos_m"]
        fixed_base_quat = trace["fixed_base_quat_wxyz"]
        fixed_base_pos_ref = fixed_base_pos[0:1]
        fixed_base_quat_ref = fixed_base_quat[0:1]
        fixed_base_quat_error = np.minimum(
            np.max(np.abs(fixed_base_quat - fixed_base_quat_ref), axis=-1),
            np.max(np.abs(fixed_base_quat + fixed_base_quat_ref), axis=-1),
        )
        fixed_base_stability_exact = bool(
            fixed_base_readback_exact
            and np.max(np.abs(fixed_base_pos - fixed_base_pos_ref)) <= 1.0e-7
            and np.max(fixed_base_quat_error) <= 1.0e-7
            and np.max(np.abs(trace["fixed_base_lin_vel_m_s"])) <= 1.0e-7
            and np.max(np.abs(trace["fixed_base_ang_vel_rad_s"])) <= 1.0e-7
        )

    def json_numbers_finite(value: Any) -> bool:
        if isinstance(value, dict):
            return all(json_numbers_finite(item) for item in value.values())
        if isinstance(value, list):
            return all(json_numbers_finite(item) for item in value)
        if isinstance(value, bool) or value is None or isinstance(value, str):
            return True
        if isinstance(value, (int, float)):
            return math.isfinite(float(value))
        return False

    numeric_report = instrumentation.get("numeric_integrity", {})
    numeric_failure_counts = numeric_report.get("failure_counts_by_check", {})
    numeric_count_rows = numeric_report.get("count_integrity", {})
    numeric_quaternion_rows = numeric_report.get("quaternion_norm", {})
    numeric_joint_limit_row = numeric_report.get("joint_limit_application", {})
    numeric_fixed_base_row = numeric_report.get("fixed_base_stability", {})
    numeric_exact = bool(
        trace_finite_exact and counts_exact and contact_positions_exact and quaternion_exact
        and joint_limit_application_exact
        and fixed_base_stability_exact
        and numeric_report.get("pass") is True
        and numeric_report.get("pass_all_active") is True
        and numeric_report.get("pass_all_envs") is True
        and numeric_report.get("conditional_contact_position_pass") is True
        and numeric_report.get("metrics_finite") is True
        and numeric_report.get("per_env") == [True] * expected_envs
        and numeric_report.get("active_per_env") == [True] * expected_active
        and isinstance(numeric_failure_counts, dict) and numeric_failure_counts
        and all(value == 0 for value in numeric_failure_counts.values())
        and set(numeric_count_rows)
        == {
            "object_raw_contact_count", "support_raw_contact_count",
            "self_raw_contact_count",
        }
        and all(row.get("pass") is True for row in numeric_count_rows.values())
        and set(numeric_quaternion_rows) == {"object", "moving_body", "fixed_base"}
        and all(row.get("pass") is True for row in numeric_quaternion_rows.values())
        and numeric_joint_limit_row
        == {
            "readback_pass": True,
            "planned_applied_target_abs_tolerance_rad": 1.0e-7,
            "actual_position_limit_abs_tolerance_rad": 1.0e-5,
            "planned_equals_applied_failure_count": 0,
            "applied_inside_limit_failure_count": 0,
            "actual_inside_limit_failure_count": 0,
        }
        and numeric_fixed_base_row
        == {
            "readback_pass": True,
            "position_abs_tolerance_m": 1.0e-7,
            "quaternion_component_abs_tolerance_sign_invariant": 1.0e-7,
            "linear_angular_velocity_abs_tolerance": 1.0e-7,
            "position_drift_failure_count": 0,
            "orientation_drift_failure_count": 0,
            "linear_velocity_failure_count": 0,
            "angular_velocity_failure_count": 0,
        }
        and numeric_report.get(
            "trace_nonfinite_failures_excluding_conditionally_allowed_contact_positions"
        ) == {}
        and json_numbers_finite(results.get("metrics", {}))
        and results.get("metrics", {}).get("numeric_integrity") == [True] * expected_active
        and results.get("metrics", {}).get("measurement_valid") == [True] * expected_active
    )
    provenance = results.get("provenance", {})
    try:
        _, dependency_hashes_current = render_dependency_snapshot(profile)
    except BaseException:
        dependency_hashes_current = {}
    source_and_dependencies_exact = bool(
        provenance.get("source_sha256") == sha256_file(Path(__file__))
        and provenance.get("source_stable") is True
        and provenance.get("p15_sha256") == P15_CANDIDATES_SHA256
        and provenance.get("prereg_sha256") == expected_prereg_sha
        and provenance.get("urdf_sha256") == URDF_SHA256
        and provenance.get("dependency_hashes_equal") is True
        and provenance.get("dependency_hashes_at_start")
        == provenance.get("dependency_hashes_at_finalize")
        == dependency_hashes_current
        and _json_type_value_exact(
            provenance.get("retired_preflight2"),
            plan.get("retired_preflight2"),
        )
        and _json_type_value_exact(
            provenance.get("retired_preflight3_launch"),
            plan.get("retired_preflight3_launch"),
        )
        and _json_type_value_exact(
            provenance.get("retired_preflight4_launch"),
            plan.get("retired_preflight4_launch"),
        )
        and _json_type_value_exact(
            provenance.get("retired_preflight5_dynamic_control_abort"),
            plan.get("retired_preflight5_dynamic_control_abort"),
        )
        and _json_type_value_exact(
            provenance.get("retired_preflight6_post_activation_dynamic_control_abort"),
            plan.get("retired_preflight6_post_activation_dynamic_control_abort"),
        )
        and _json_type_value_exact(
            provenance.get("retired_preflight7_filter_representation_abort"),
            plan.get("retired_preflight7_filter_representation_abort"),
        )
        and _json_type_value_exact(
            provenance.get("retired_preflight8_reward_buffer_lifecycle_abort"),
            plan.get("retired_preflight8_reward_buffer_lifecycle_abort"),
        )
        and _json_type_value_exact(
            provenance.get("retired_preflight9_clock_accounting_abort"),
            plan.get("retired_preflight9_clock_accounting_abort"),
        )
        and retired_preflight2_exact
        and retired_preflight3_exact
        and retired_preflight4_exact
        and retired_preflight5_exact
        and retired_preflight6_exact
        and retired_preflight7_exact
        and retired_preflight8_exact
        and retired_preflight9_exact
    )
    metrics_for_classification = {
        name: np.asarray(value)
        for name, value in results.get("metrics", {}).items()
    }
    try:
        recomputed_verdict, recomputed_labels, recomputed_classification_summary = (
            classify(metrics_for_classification)
        )
    except (KeyError, RuntimeError, TypeError, ValueError):
        recomputed_verdict = "CLASSIFICATION_RECOMPUTE_ERROR"
        recomputed_labels = []
        recomputed_classification_summary = {}
    expected_classification_rows = [
        {"trial_id": row["trial_id"], "label": recomputed_labels[index]}
        for index, row in enumerate(
            [row for row in plan.get("trials", []) if row.get("feasible")]
        )
        if index < len(recomputed_labels)
    ]
    classification_exact = bool(
        len(expected_classification_rows) == expected_active
        and results.get("classifications") == expected_classification_rows
        and results.get("classification_summary")
        == recomputed_classification_summary
        and recomputed_classification_summary.get("partition_exactly_once") is True
        and results.get("scientific_verdict_preclose_candidate")
        == (None if profile == PREFLIGHT_PROFILE else recomputed_verdict)
    )
    static_alignment = results.get("stage_collision_audit", {}).get(
        "static_fk_alignment", {}
    )
    authored_alignment = static_alignment.get("authored_rest_alignment", {})
    runtime_alignment = static_alignment.get(
        "same_epoch_runtime_articulation_alignment", {}
    )
    static_geometry = results.get("static_collision_geometry", {})
    exact_urdf_clearance_alignment = bool(
        static_alignment.get("pass") is True
        and static_alignment.get("clearance_fk_authority")
        == "exact_decimal_parsed_frozen_urdf"
        and static_alignment.get("p10_role")
        == "IK_only__not_clearance_frame_authority"
        and authored_alignment.get("semantic_scope")
        == "Usd default-time authored/rest q=0"
        and authored_alignment.get("expected_joint_state_deg") == [0.0] * 6
        and authored_alignment.get("translation_gate_m") == 1.0e-6
        and authored_alignment.get("rotation_matrix_max_abs_gate") == 1.0e-6
        and authored_alignment.get("pass") is True
        and set(authored_alignment.get("bodies", {})) == set(MOVING_BODIES)
        and all(
            row.get("pass") is True
            for row in authored_alignment.get("bodies", {}).values()
        )
        and authored_alignment.get("joint_coordinate_derivation", {}).get("pass")
        is True
        and runtime_alignment.get("tensor_schema_pass") is True
        and runtime_alignment.get("single_epoch_pass") is True
        and runtime_alignment.get("articulation_data_sim_timestamp_before")
        == runtime_alignment.get("articulation_data_sim_timestamp_after")
        and runtime_alignment.get("translation_gate_m") == 5.0e-6
        and runtime_alignment.get("rotation_matrix_max_abs_gate") == 1.0e-5
        and runtime_alignment.get("quaternion_norm_abs_gate") == 1.0e-6
        and runtime_alignment.get("pass") is True
        and len(runtime_alignment.get("rows", []))
        == expected_envs * len(MOVING_BODIES)
        and all(row.get("pass") is True for row in runtime_alignment.get("rows", []))
        and set(static_alignment.get("legacy_p10_rounded_chain_diagnostic_at_q0", {}))
        == set(MOVING_BODIES[:-1])
        and static_geometry.get("clearance_fk_authority")
        == "exact frozen URDF decimal origins/rpy/axes parsed at runtime; p10 excluded"
        and static_geometry.get("urdf_kinematic_chain")
        == parse_urdf_kinematic_chain()
    )
    checks = {
        "result_top_level_profile_and_mode_exact": bool(
            set(results) == expected_result_keys
            and results.get("tool")
            == "p16_g0b_t3u_cyld29h50_side_midpoint_parallel_physics_v9"
            and results.get("case") == "g0b_d420"
            and results.get("tag") == f"t3u_{profile}"
            and results.get("profile") == profile
            and results.get("scientific_authoritative")
            is (profile == CANONICAL_PROFILE)
        ),
        "plan_exact_ordered_trial_set_recomputed": bool(
            trial_contract_exact
            and plan.get("n_planned") == expected_planned
            and plan.get("n_feasible") == expected_active
            and results.get("plan_counts")
            == {"planned": expected_planned, "feasible": expected_active}
        ),
        "composed_object_material_mass_units_all_clones_exact": object_readback_exact,
        "support_filter_actual_identity_exact": support_filters_exact,
        "self_filter_actual_identity_exact": self_filters_exact,
        "precontrol_and_postcontrol_self_filter_identity_reuse_exact": (
            filter_identity_reuse_exact
        ),
        "cylinder_plus_base_and_six_reporters_every_clone_exact": reporter_exact,
        "self_collision_stage_readback_exact": self_collision_exact,
        "diagnostic_one_frame_rebaseline_and_task_epoch_exact": task_epoch_exact,
        "authored_rest_and_runtime_pose_exact_urdf_clearance_alignment": (
            exact_urdf_clearance_alignment
        ),
        "fixed_base_metatype_joint_and_full_step_stability_exact": bool(
            fixed_base_readback_exact and fixed_base_stability_exact
        ),
        "composed_and_runtime_joint_limits_equal_frozen_urdf": (
            joint_limits_exact and joint_limit_application_exact
        ),
        "numeric_trace_metrics_quaternions_counts_recomputed": numeric_exact,
        "authoritative_trace_step_time_phase_cadence_exact": trace_cadence_exact,
        "causal_classification_masks_counts_and_verdict_recomputed": (
            classification_exact
        ),
        "source_prereg_p15_and_dependency_pins_exact": source_and_dependencies_exact,
        "retired_preflight2_failure_provenance_exact_and_nonpromotable": (
            retired_preflight2_exact
        ),
        "retired_preflight3_launch_provenance_exact_and_nonpromotable": (
            retired_preflight3_exact
        ),
        "retired_preflight4_launch_provenance_exact_and_nonpromotable": (
            retired_preflight4_exact
        ),
        "retired_preflight5_dynamic_control_abort_exact_and_nonpromotable": (
            retired_preflight5_exact
        ),
        "retired_preflight6_post_activation_dynamic_control_abort_exact_and_nonpromotable": (
            retired_preflight6_exact
        ),
        "retired_preflight7_filter_representation_abort_exact_and_nonpromotable": (
            retired_preflight7_exact
        ),
        "retired_preflight8_reward_buffer_lifecycle_abort_exact_and_nonpromotable": (
            retired_preflight8_exact
        ),
        "retired_preflight9_clock_accounting_abort_exact_and_nonpromotable": (
            retired_preflight9_exact
        ),
        "runtime_instrumentation_recomputed_not_trusted": bool(
            results.get("runtime_instrumentation_pass") is True
            and instrumentation.get("contact_buffers_ok") is True
            and instrumentation.get("saturated_sensors") == []
            and instrumentation.get("instrumentation_witness", {}).get("pass") is True
            and instrumentation.get("moving_bodies") == list(MOVING_BODIES)
            and instrumentation.get("self_contact_bodies")
            == list(SELF_CONTACT_BODIES)
            and instrumentation.get("nonadjacent_self_pairs")
            == [list(pair) for pair in SELF_PAIRS]
            and instrumentation.get("adjacent_self_pair_exclusions")
            == [list(pair) for pair in ADJACENT_SELF_PAIR_EXCLUSIONS]
            and instrumentation.get(
                "fixed_base_support_contact_excluded_from_task_gate"
            ) is True
            and set(instrumentation.get("object_filter_map", {}))
            == {"support", *MOVING_BODIES}
            and sorted(instrumentation.get("object_filter_map", {}).values())
            == list(range(1 + len(MOVING_BODIES)))
            and numeric_exact and support_filters_exact and self_filters_exact
            and filter_identity_reuse_exact
            and reporter_exact and self_collision_exact
            and task_epoch_exact
            and object_readback_exact and joint_limits_exact
            and joint_limit_application_exact and fixed_base_readback_exact
            and fixed_base_stability_exact and exact_urdf_clearance_alignment
        ),
    }
    if set(checks) != RESULT_SEMANTIC_CHECK_KEYS:
        raise RuntimeError(
            "RESULT_SEMANTIC_CHECK_KEYSET_INTERNAL_DRIFT "
            f"expected={sorted(RESULT_SEMANTIC_CHECK_KEYS)} "
            f"actual={sorted(checks)}"
        )
    return checks


def _terminal_render_abort_attest_mode(
    profile: str,
    paths: dict[str, Path],
    external: dict[str, Path],
    *,
    verify_only: bool,
) -> int | dict[str, Any] | None:
    """Attest a completed physics preclose followed by a failed render."""
    if profile not in EXECUTABLE_PROFILES or not external["supervisor_outcome"].is_file():
        return None
    supervisor_outcome = json.loads(external["supervisor_outcome"].read_text())
    physics = supervisor_outcome.get("physics", {})
    render = supervisor_outcome.get("render")
    if not isinstance(render, dict):
        return None
    helper = load_module(
        f"p16_terminal_render_abort_supervisor_helper_{profile}", SUPERVISOR_PATH
    )
    prefix = f"{TAG}_{profile}"
    recomputed_physics_gate = helper._physics_preclose_semantic_gate(
        profile, prefix, paths, physics, external["stdout"]
    )
    recomputed_render_gate = helper._render_posthoc_semantic_gate(
        profile, prefix, paths, render, external["stdout"]
    )
    if not (
        recomputed_physics_gate.get("pass") is True
        and recomputed_render_gate.get("pass") is False
    ):
        return None
    if not verify_only and (
        paths["terminal_attestation.json"].exists() or external["gpu_after"].exists()
    ):
        raise RuntimeError("TERMINAL_RENDER_ABORT_ATTEST_FORWARD_ONLY_OUTPUT_EXISTS")

    required = {
        **{
            name: paths[name]
            for name in (
                "results.json", "plan.json", "trace.npz", "timeline.rrd",
                "timeline.rbl", "rerun_validation.json", "decision_snapshot.png",
                "inspection.png", "script.py.txt", "argv.txt", "phase.jsonl",
                "preclose_sentinel.json", "exit_status.txt",
            )
        },
        "stdout.log": external["stdout"],
        "supervisor_launcher.log": external["supervisor_launcher"],
        "supervisor_pid.txt": external["supervisor_pid"],
        "physics_python_pid.txt": external["physics_python_pid"],
        "render_python_pid.txt": external["render_python_pid"],
        "pgid.txt": external["pgid"],
        "supervisor_contract.json": external["supervisor_contract"],
        "supervisor_outcome.json": external["supervisor_outcome"],
        "nvidia_smi_before.csv": external["gpu_before"],
        "nvidia_smi_supervisor_end.csv": external["gpu_supervisor_end"],
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"TERMINAL_RENDER_ABORT_REQUIRED_MISSING {missing}")

    supervisor_pid = _read_pid_file_or_invalid(external["supervisor_pid"])
    physics_pid = _read_pid_file_or_invalid(external["physics_python_pid"])
    render_pid = _read_pid_file_or_invalid(external["render_python_pid"])
    pgid = _read_pid_file_or_invalid(external["pgid"])
    before_gpu = _gpu_pid_set(external["gpu_before"].read_text())
    supervisor_end_gpu = _gpu_pid_set(external["gpu_supervisor_end"].read_text())
    contract = json.loads(external["supervisor_contract"].read_text())
    phase_rows = [
        json.loads(line)
        for line in paths["phase.jsonl"].read_text().splitlines()
        if line.strip()
    ]
    phase_names = [row.get("phase") for row in phase_rows]
    render_phase_parse_error: str | None = None
    try:
        render_phase_loaded = (
            [
                json.loads(line)
                for line in paths["render_phase.jsonl"].read_text().splitlines()
                if line.strip()
            ]
            if paths["render_phase.jsonl"].is_file()
            else []
        )
        if not all(isinstance(row, dict) for row in render_phase_loaded):
            raise TypeError("render_phase_jsonl_rows_must_be_objects")
        render_phase_rows = render_phase_loaded
    except BaseException as exc:
        render_phase_rows = []
        render_phase_parse_error = f"{type(exc).__name__}: {exc}"
    render_phase_names = [row.get("phase") for row in render_phase_rows]
    expected_contract = {
        "artifact": "T3U_DETACHED_PHYSICS_THEN_RENDER_SUPERVISOR_V12",
        "automatic_retry_count": 0,
        "detached": True,
        "physics_timeout_seconds": 7200,
        "render_timeout_seconds": 7200,
        "term_signal": "TERM",
        "kill_after_seconds": 20,
        "physics_then_render_only_on_raw_zero_and_preclose_semantic_gate": True,
        "physics_semantic_gate_artifact": "T3U_PHYSICS_PRECLOSE_SEMANTIC_GATE_V1",
        "render_success_requires_raw_zero_and_posthoc_semantic_gate": True,
        "render_semantic_gate_artifact": "T3U_RENDER_POSTHOC_SEMANTIC_GATE_V1",
        "semantic_gate_failure_exit_status": 125,
        "raw_waitpid_status_authority": True,
        "bounded_waitpid_only": True,
        "supervisor_signal_cleanup": (
            "SIGTERM_SIGINT__active_child_pgid_TERM_20s_then_KILL_20s"
        ),
        "child_parent_death_signal": "SIGTERM",
        "child_preexec_signal_state": (
            "SIGTERM_SIGINT_SIGHUP_SIG_DFL__empty_mask__expected_parent_pid_recheck"
        ),
        "host_launch_boundary": (
            "require_escalated_exec_command__outside_bwrap_die_with_parent"
        ),
        "forbidden_sandbox_ancestor_gate": True,
    }
    host_launch_context = contract.get("host_launch_context") if isinstance(contract, dict) else None
    expected_contract["host_launch_context"] = host_launch_context
    host_launch_context_exact = _validate_host_launch_context(
        host_launch_context, supervisor_pid
    )
    expected_outcome_keys = {
        "artifact", "profile", "argv", "supervisor_source_sha256",
        "p16_source_sha256", "candidates_sha256", "start_time_unix",
        "end_time_unix", "elapsed_seconds", "supervisor", "attempts",
        "physics", "physics_artifact_gate", "render", "render_artifact_gate",
        "render_started_iff_physics_success", "combined_exit_status", "gpu",
        "bindings", "contract", "host_launch_context", "pass",
    }
    expected_physics_command = [
        ISAAC_PYTHON, str(Path(__file__).resolve()), "--run_label", profile,
        "--candidates_sha256", P15_CANDIDATES_SHA256,
    ]
    expected_render_command = [
        ISAAC_PYTHON, str(Path(__file__).resolve()), "--render_trace", profile,
    ]

    physics_lifecycle = _strict_child_lifecycle(
        physics,
        label="physics",
        command=expected_physics_command,
        pid=physics_pid,
        supervisor_sid=supervisor_pid,
        require_success=True,
    )
    render_lifecycle = _strict_child_lifecycle(
        render,
        label="render",
        command=expected_render_command,
        pid=render_pid,
        supervisor_sid=supervisor_pid,
        require_success=False,
    )
    physics_raw_success = bool(
        physics_lifecycle
        and physics.get("raw_wait_status") == 0
        and physics.get("timed_out") is False
        and physics.get("signal_actions") == []
    )
    render_raw_return = (
        render["normalized_returncode"]
        if isinstance(render, dict)
        and _strict_json_int(render.get("normalized_returncode"))
        else -1
    )
    expected_combined = (
        render_raw_return if render_raw_return != 0 else 125
    )

    physics_prefixes = [
        [
            "run_claim", "results_durable", "preclose_sentinel_durable",
            "simulation_app_close_start", "simulation_app_close_returned",
        ],
        [
            "run_claim", "results_durable", "preclose_sentinel_durable",
            "simulation_app_close_start",
        ],
    ]
    matched_prefix = next(
        (candidate for candidate in physics_prefixes if phase_names == candidate), None
    )
    render_tail = render_phase_names
    render_tail_valid = bool(
        render_tail in ([], ["render_failure"], ["render_trace_durable"])
    )
    render_failure_parse_error: str | None = None
    try:
        failure_loaded = (
            json.loads(paths["render_failure.json"].read_text())
            if paths["render_failure.json"].is_file()
            else None
        )
        if failure_loaded is not None and not isinstance(failure_loaded, dict):
            raise TypeError("render_failure_root_must_be_object")
        failure = failure_loaded
    except BaseException as exc:
        failure = None
        render_failure_parse_error = f"{type(exc).__name__}: {exc}"
    render_failure_rows = [
        row for row in render_phase_rows if row.get("phase") == "render_failure"
    ]
    if render_phase_parse_error is not None or render_failure_parse_error is not None:
        render_failure_evidence_exact = bool(
            recomputed_render_gate.get("pass") is False
            and recomputed_render_gate.get("parse_error") is not None
        )
    elif failure is None:
        render_failure_evidence_exact = bool(
            render_tail in ([], ["render_trace_durable"])
            and not paths["render_failure.json"].exists()
        )
    else:
        physics_last_time = (
            float(phase_rows[-1].get("time_unix", math.nan))
            if phase_rows else math.nan
        )
        render_failure_evidence_exact = bool(
            set(failure)
            == {"type", "message", "traceback", "phase", "profile", "source_sha256"}
            and failure.get("phase") == "render_trace"
            and failure.get("profile") == profile
            and failure.get("source_sha256") == sha256_file(Path(__file__))
            and isinstance(failure.get("traceback"), str)
            and failure.get("traceback")
            and len(render_failure_rows) == 1
            and set(render_failure_rows[0])
            == {"time_unix", "phase", "type", "message", "failure_sha256"}
            and isinstance(render_failure_rows[0].get("time_unix"), (int, float))
            and not isinstance(render_failure_rows[0].get("time_unix"), bool)
            and math.isfinite(float(render_failure_rows[0]["time_unix"]))
            and float(render_failure_rows[0]["time_unix"]) >= physics_last_time
            and render_failure_rows[0].get("type") == failure.get("type")
            and render_failure_rows[0].get("message") == failure.get("message")
            and render_failure_rows[0].get("failure_sha256")
            == sha256_file(paths["render_failure.json"])
        )

    expected_binding_paths: dict[str, Path] = {}
    for suffix, path in paths.items():
        if suffix != "terminal_attestation.json" and path.is_file():
            expected_binding_paths[suffix] = path
    for name, path in external.items():
        if name not in {"supervisor_outcome", "gpu_after", "supervisor_failure"} and path.is_file():
            expected_binding_paths[path.name.removeprefix(f"{TAG}_{profile}_")] = path
    expected_binding_paths["supervisor_launcher.log"] = external["supervisor_launcher"]
    outcome_bindings = supervisor_outcome.get("bindings", {})
    bindings_valid = bool(
        isinstance(outcome_bindings, dict)
        and set(outcome_bindings) == set(expected_binding_paths)
        and all(
            outcome_bindings[name]
            == {"path": str(path.relative_to(REPO)), "sha256": sha256_file(path)}
            for name, path in expected_binding_paths.items()
        )
    )
    # Create the one-shot gpu_after output only after every pure JSON/phase/raw-wait/
    # binding recomputation above has completed, so malformed evidence cannot brick a
    # retry by leaving gpu_after without a terminal attestation.
    if verify_only:
        if not external["gpu_after"].is_file():
            raise RuntimeError("TERMINAL_RENDER_ABORT_GPU_AFTER_MISSING")
        gpu_after_text = external["gpu_after"].read_text()
        after_gpu = _gpu_pid_set(gpu_after_text)
    else:
        try:
            query = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,process_name,used_gpu_memory",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=15.0,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("TERMINAL_RENDER_ABORT_NVIDIA_SMI_TIMEOUT_15S") from exc
        if query.returncode != 0:
            raise RuntimeError(f"TERMINAL_RENDER_ABORT_NVIDIA_SMI_FAIL {query.stderr}")
        gpu_after_text = query.stdout
        after_gpu = _gpu_pid_set(gpu_after_text)
    checks = {
        "physics_preclose_gate_recomputed_pass": bool(
            _json_type_value_exact(
                supervisor_outcome.get("physics_artifact_gate"),
                recomputed_physics_gate,
            )
            and recomputed_physics_gate.get("pass") is True
        ),
        "render_gate_recomputed_failed_not_trusted": bool(
            _json_type_value_exact(
                supervisor_outcome.get("render_artifact_gate"),
                recomputed_render_gate,
            )
            and recomputed_render_gate.get("pass") is False
        ),
        "render_failure_evidence_and_phase_exact": bool(
            matched_prefix is not None
            and render_tail_valid
            and render_failure_evidence_exact
        ),
        "supervisor_schema_sources_contract_exact": bool(
            set(supervisor_outcome) == expected_outcome_keys
            and supervisor_outcome.get("artifact")
            == "T3U_DETACHED_SUPERVISOR_OUTCOME_V12"
            and supervisor_outcome.get("profile") == profile
            and supervisor_outcome.get("argv")
            == [
                str(SUPERVISOR_PATH), "--profile", profile,
                "--candidates_sha256", P15_CANDIDATES_SHA256,
            ]
            and supervisor_outcome.get("supervisor_source_sha256") == SUPERVISOR_SHA256
            and supervisor_outcome.get("p16_source_sha256") == sha256_file(Path(__file__))
            and supervisor_outcome.get("candidates_sha256") == P15_CANDIDATES_SHA256
            and _json_type_value_exact(contract, expected_contract)
            and _json_type_value_exact(
                supervisor_outcome.get("contract"), expected_contract
            )
            and host_launch_context_exact
            and _json_type_value_exact(
                supervisor_outcome.get("host_launch_context"), host_launch_context
            )
        ),
        "supervisor_identity_attempts_and_timestamps_exact": bool(
            _strict_supervisor_identity(
                supervisor_outcome.get("supervisor"), pid=supervisor_pid, pgid=pgid
            )
            and _strict_attempts(supervisor_outcome.get("attempts"), render_count=1)
            and supervisor_outcome.get("render_started_iff_physics_success") is True
            and _strict_outcome_times(supervisor_outcome, [physics, render])
        ),
        "physics_raw_success_and_render_raw_lifecycle_exact": bool(
            physics_raw_success and render_lifecycle
        ),
        "reserved_or_raw_nonzero_combined_exit_exact": bool(
            expected_combined != 0
            and _strict_json_int(supervisor_outcome.get("combined_exit_status"))
            and supervisor_outcome.get("combined_exit_status") == expected_combined
            and paths["exit_status.txt"].read_text().strip() == str(expected_combined)
            and supervisor_outcome.get("pass") is False
        ),
        "all_processes_and_groups_reaped": bool(
            not Path(f"/proc/{supervisor_pid}").exists()
            and not Path(f"/proc/{physics_pid}").exists()
            and not Path(f"/proc/{render_pid}").exists()
            and _linux_pgid_members(pgid) == []
            and _linux_pgid_members(physics_pid) == []
            and _linux_pgid_members(render_pid) == []
        ),
        "gpu_no_fresh_process": bool(
            supervisor_end_gpu - before_gpu == set()
            and after_gpu - before_gpu == set()
            and _strict_gpu_summary(
                supervisor_outcome.get("gpu"),
                before=before_gpu,
                supervisor_end=supervisor_end_gpu,
            )
        ),
        "frozen_source_and_physics_argv_exact": bool(
            sha256_file(paths["script.py.txt"]) == sha256_file(Path(__file__))
            and paths["argv.txt"].read_text().splitlines()
            == expected_physics_command[1:]
        ),
        "outcome_file_bindings_exact": bindings_valid,
        "no_supervisor_failure_marker": not external["supervisor_failure"].exists(),
    }
    attestation_valid = bool(all(checks.values()))
    if not verify_only:
        write_bytes_x(external["gpu_after"], gpu_after_text.encode("utf-8"))
    binding_paths = {
        **required,
        "gpu_after": external["gpu_after"],
        **(
            {"render_failure.json": paths["render_failure.json"]}
            if paths["render_failure.json"].is_file() else {}
        ),
        **(
            {"rgb_frames_manifest.json": paths["rgb_frames_manifest.json"]}
            if paths["rgb_frames_manifest.json"].is_file() else {}
        ),
        **(
            {"side_grasp.mp4": paths["side_grasp.mp4"]}
            if paths["side_grasp.mp4"].is_file() else {}
        ),
    }
    attestation = {
        "artifact": "T3U_EXTERNAL_TERMINAL_RENDER_ABORT_ATTESTATION_V1",
        "profile": profile,
        "argv": [str(Path(__file__).resolve()), "--terminal_attest", profile],
        "upstream_failure_stage": "postphysics_render",
        "upstream_failure_type": (
            failure.get("type") if isinstance(failure, dict)
            else "RENDER_POSTHOC_SEMANTIC_GATE_FAIL"
        ),
        "physics_artifact_gate": recomputed_physics_gate,
        "render_artifact_gate": recomputed_render_gate,
        "evidence_checks": checks,
        "processes": {
            "supervisor_pid": supervisor_pid,
            "physics_pid": physics_pid,
            "render_pid": render_pid,
            "pgid": pgid,
            "combined_exit_status": expected_combined,
        },
        "gpu": {
            "before_pids": sorted(before_gpu),
            "supervisor_end_pids": sorted(supervisor_end_gpu),
            "after_pids": sorted(after_gpu),
        },
        "dependency_pins": {
            "p16_source_sha256": sha256_file(Path(__file__)),
            "supervisor_source_sha256": SUPERVISOR_SHA256,
            "selected_prereg_sha256": (
                PREFLIGHT_PREREG_SHA256
                if profile == PREFLIGHT_PROFILE else CANONICAL_PREREG_SHA256
            ),
            "preflight3_source_sha256": PREFLIGHT3_SOURCE_SHA256,
            "preflight3_supervisor_sha256": PREFLIGHT3_SUPERVISOR_SHA256,
            "preflight3_prereg_sha256": PREFLIGHT3_PREREG_SHA256,
            "preflight3_canonical_prereg_sha256": (
                PREFLIGHT3_CANONICAL_PREREG_SHA256
            ),
            "preflight3_zero_launcher_sha256": PREFLIGHT3_LAUNCHER_SHA256,
            "preflight3_posthoc_audit_failure_sha256": (
                PREFLIGHT3_POSTHOC_AUDIT_FAILURE_SHA256
            ),
            "preflight4_source_sha256": PREFLIGHT4_SOURCE_SHA256,
            "preflight4_supervisor_sha256": PREFLIGHT4_SUPERVISOR_SHA256,
            "preflight4_prereg_sha256": PREFLIGHT4_PREREG_SHA256,
            "preflight4_canonical_prereg_sha256": (
                PREFLIGHT4_CANONICAL_PREREG_SHA256
            ),
            "preflight4_failure_sha256": PREFLIGHT4_FAILURE_SHA256,
            "preflight4_launcher_sha256": PREFLIGHT4_LAUNCHER_SHA256,
            **PREFLIGHT5_DEPENDENCY_PINS,
            **PREFLIGHT6_DEPENDENCY_PINS,
            **PREFLIGHT7_DEPENDENCY_PINS,
            **PREFLIGHT8_DEPENDENCY_PINS,
            **PREFLIGHT9_DEPENDENCY_PINS,
        },
        "bindings": {
            name: {"path": str(path.relative_to(REPO)), "sha256": sha256_file(path)}
            for name, path in binding_paths.items()
        },
        "attestation_valid": attestation_valid,
        "scientific_artifacts_complete": False,
        "promotion_allowed": False,
        "pass": False,
        "verdict": (
            "ATTESTED_POSTPHYSICS_RENDER_ABORT__NO_PROMOTION"
            if attestation_valid
            else "POSTPHYSICS_RENDER_ABORT_EVIDENCE_INVALID__NO_PROMOTION"
        ),
    }
    if verify_only:
        return attestation
    write_json_x(paths["terminal_attestation.json"], attestation)
    print(
        f"[{LOG}] TERMINAL_RENDER_ABORT_ATTEST profile={profile} "
        f"valid={attestation_valid} promotion=False",
        flush=True,
    )
    return 0 if attestation_valid else 1


def _terminal_abort_attest_mode(
    profile: str,
    paths: dict[str, Path],
    external: dict[str, Path],
    *,
    verify_only: bool,
) -> int | dict[str, Any] | None:
    """Attest an upstream-aborted preflight without inventing science files."""
    if profile not in EXECUTABLE_PROFILES or not external["supervisor_outcome"].is_file():
        return None
    supervisor_outcome = json.loads(external["supervisor_outcome"].read_text())
    upstream_failed = bool(
        paths["failure.json"].is_file()
        or supervisor_outcome.get("combined_exit_status") != 0
        or supervisor_outcome.get("pass") is not True
    )
    if not upstream_failed:
        return None
    if not verify_only and (
        paths["terminal_attestation.json"].exists() or external["gpu_after"].exists()
    ):
        raise RuntimeError("TERMINAL_ABORT_ATTEST_FORWARD_ONLY_OUTPUT_EXISTS")

    required = {
        "failure": paths["failure.json"],
        "script": paths["script.py.txt"],
        "argv": paths["argv.txt"],
        "phase": paths["phase.jsonl"],
        "exit_status": paths["exit_status.txt"],
        "stdout": external["stdout"],
        "supervisor_launcher": external["supervisor_launcher"],
        "supervisor_pid": external["supervisor_pid"],
        "physics_python_pid": external["physics_python_pid"],
        "pgid": external["pgid"],
        "supervisor_contract": external["supervisor_contract"],
        "supervisor_outcome": external["supervisor_outcome"],
        "gpu_before": external["gpu_before"],
        "gpu_supervisor_end": external["gpu_supervisor_end"],
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"TERMINAL_ABORT_ATTEST_REQUIRED_MISSING {missing}")
    if external["render_python_pid"].exists():
        raise RuntimeError("TERMINAL_ABORT_ATTEST_RENDER_PID_MUST_BE_ABSENT")

    failure_loaded = json.loads(paths["failure.json"].read_text())
    failure_shape_exact = isinstance(failure_loaded, dict)
    failure = failure_loaded if failure_shape_exact else {}
    contract = json.loads(external["supervisor_contract"].read_text())
    physics = supervisor_outcome.get("physics", {})
    helper = load_module(
        f"p16_terminal_abort_supervisor_helper_{profile}", SUPERVISOR_PATH
    )
    recomputed_gate = helper._physics_preclose_semantic_gate(
        profile, f"{TAG}_{profile}", paths, physics, external["stdout"]
    )
    phase_rows_loaded = [
        json.loads(line)
        for line in paths["phase.jsonl"].read_text().splitlines()
        if line.strip()
    ]
    phase_rows_shape_exact = all(isinstance(row, dict) for row in phase_rows_loaded)
    phase_rows = phase_rows_loaded if phase_rows_shape_exact else []
    phase_names = [row.get("phase") for row in phase_rows]
    supervisor_pid = _read_pid_file_or_invalid(external["supervisor_pid"])
    physics_pid = _read_pid_file_or_invalid(external["physics_python_pid"])
    pgid = _read_pid_file_or_invalid(external["pgid"])
    before_gpu = _gpu_pid_set(external["gpu_before"].read_text())
    supervisor_end_gpu = _gpu_pid_set(external["gpu_supervisor_end"].read_text())
    raw_return = (
        physics["normalized_returncode"]
        if isinstance(physics, dict)
        and _strict_json_int(physics.get("normalized_returncode"))
        else -1
    )
    expected_combined = raw_return if raw_return != 0 else 125
    expected_contract = {
        "artifact": "T3U_DETACHED_PHYSICS_THEN_RENDER_SUPERVISOR_V12",
        "automatic_retry_count": 0,
        "detached": True,
        "physics_timeout_seconds": 7200,
        "render_timeout_seconds": 7200,
        "term_signal": "TERM",
        "kill_after_seconds": 20,
        "physics_then_render_only_on_raw_zero_and_preclose_semantic_gate": True,
        "physics_semantic_gate_artifact": "T3U_PHYSICS_PRECLOSE_SEMANTIC_GATE_V1",
        "render_success_requires_raw_zero_and_posthoc_semantic_gate": True,
        "render_semantic_gate_artifact": "T3U_RENDER_POSTHOC_SEMANTIC_GATE_V1",
        "semantic_gate_failure_exit_status": 125,
        "raw_waitpid_status_authority": True,
        "bounded_waitpid_only": True,
        "supervisor_signal_cleanup": (
            "SIGTERM_SIGINT__active_child_pgid_TERM_20s_then_KILL_20s"
        ),
        "child_parent_death_signal": "SIGTERM",
        "child_preexec_signal_state": (
            "SIGTERM_SIGINT_SIGHUP_SIG_DFL__empty_mask__expected_parent_pid_recheck"
        ),
        "host_launch_boundary": (
            "require_escalated_exec_command__outside_bwrap_die_with_parent"
        ),
        "forbidden_sandbox_ancestor_gate": True,
    }
    host_launch_context = contract.get("host_launch_context") if isinstance(contract, dict) else None
    expected_contract["host_launch_context"] = host_launch_context
    host_launch_context_exact = _validate_host_launch_context(
        host_launch_context, supervisor_pid
    )
    expected_outcome_keys = {
        "artifact", "profile", "argv", "supervisor_source_sha256",
        "p16_source_sha256", "candidates_sha256", "start_time_unix",
        "end_time_unix", "elapsed_seconds", "supervisor", "attempts",
        "physics", "physics_artifact_gate", "render", "render_artifact_gate",
        "render_started_iff_physics_success", "combined_exit_status", "gpu",
        "bindings", "contract", "host_launch_context", "pass",
    }
    expected_physics_command = [
        ISAAC_PYTHON, str(Path(__file__).resolve()), "--run_label", profile,
        "--candidates_sha256", P15_CANDIDATES_SHA256,
    ]
    expected_physics_argv = expected_physics_command[1:]
    expected_supervisor_argv = [
        str(SUPERVISOR_PATH), "--profile", profile,
        "--candidates_sha256", P15_CANDIDATES_SHA256,
    ]
    physics_lifecycle_valid = _strict_child_lifecycle(
        physics,
        label="physics",
        command=expected_physics_command,
        pid=physics_pid,
        supervisor_sid=supervisor_pid,
        require_success=False,
    )
    expected_binding_paths: dict[str, Path] = {}
    for suffix, path in paths.items():
        if suffix != "terminal_attestation.json" and path.is_file():
            expected_binding_paths[suffix] = path
    for name, path in external.items():
        if name not in {"supervisor_outcome", "gpu_after", "supervisor_failure"} and path.is_file():
            expected_binding_paths[path.name.removeprefix(f"{TAG}_{profile}_")] = path
    expected_binding_paths["supervisor_launcher.log"] = external["supervisor_launcher"]
    outcome_bindings = supervisor_outcome.get("bindings", {})
    bindings_valid = bool(
        isinstance(outcome_bindings, dict)
        and set(outcome_bindings) == set(expected_binding_paths)
        and all(
            _json_type_value_exact(
                outcome_bindings[name],
                {
                    "path": str(path.relative_to(REPO)),
                    "sha256": sha256_file(path),
                },
            )
            for name, path in expected_binding_paths.items()
        )
    )
    # Delay the one-shot output until all pure upstream JSON/phase/lifecycle and
    # binding computations have completed.
    if verify_only:
        if not external["gpu_after"].is_file():
            raise RuntimeError("TERMINAL_ABORT_ATTEST_GPU_AFTER_MISSING")
        gpu_after_text = external["gpu_after"].read_text()
        after_gpu = _gpu_pid_set(gpu_after_text)
    else:
        try:
            query = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,process_name,used_gpu_memory",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=15.0,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("TERMINAL_ABORT_NVIDIA_SMI_TIMEOUT_15S") from exc
        if query.returncode != 0:
            raise RuntimeError(f"TERMINAL_ABORT_NVIDIA_SMI_FAIL {query.stderr}")
        gpu_after_text = query.stdout
        after_gpu = _gpu_pid_set(gpu_after_text)
    checks = {
        "upstream_failure_marker_bound": bool(
            failure_shape_exact
            and failure.get("profile") == profile
            and failure.get("source_sha256") == sha256_file(Path(__file__))
            and isinstance(failure.get("traceback"), str)
            and failure.get("traceback")
        ),
        "frozen_source_and_argv_bound": bool(
            sha256_file(paths["script.py.txt"]) == sha256_file(Path(__file__))
            and paths["argv.txt"].read_text().splitlines() == expected_physics_argv
        ),
        "phase_is_aborted_before_preclose": bool(
            phase_rows_shape_exact
            and phase_names.count("run_claim") == 1
            and phase_names.count("failure") == 1
            and "preclose_sentinel_durable" not in phase_names
            and phase_names.index("run_claim") < phase_names.index("failure")
            and phase_names.count("simulation_app_close_start") in (0, 1)
            and (
                "simulation_app_close_start" not in phase_names
                or phase_names.index("failure")
                < phase_names.index("simulation_app_close_start")
            )
        ),
        "supervisor_schema_identity_and_contract_exact": bool(
            set(supervisor_outcome) == expected_outcome_keys
            and supervisor_outcome.get("artifact")
            == "T3U_DETACHED_SUPERVISOR_OUTCOME_V12"
            and supervisor_outcome.get("profile") == profile
            and supervisor_outcome.get("argv") == expected_supervisor_argv
            and supervisor_outcome.get("supervisor_source_sha256") == SUPERVISOR_SHA256
            and supervisor_outcome.get("p16_source_sha256") == sha256_file(Path(__file__))
            and supervisor_outcome.get("candidates_sha256") == P15_CANDIDATES_SHA256
            and _json_type_value_exact(contract, expected_contract)
            and _json_type_value_exact(
                supervisor_outcome.get("contract"), expected_contract
            )
            and host_launch_context_exact
            and _json_type_value_exact(
                supervisor_outcome.get("host_launch_context"), host_launch_context
            )
        ),
        "physics_child_raw_lifecycle_valid": physics_lifecycle_valid,
        "supervisor_identity_and_timestamps_exact": bool(
            _strict_supervisor_identity(
                supervisor_outcome.get("supervisor"),
                pid=supervisor_pid,
                pgid=pgid,
            )
            and _strict_outcome_times(supervisor_outcome, [physics])
        ),
        "semantic_gate_recomputed_failed_exact": bool(
            _json_type_value_exact(
                supervisor_outcome.get("physics_artifact_gate"), recomputed_gate
            )
            and recomputed_gate.get("pass") is False
        ),
        "render_not_started_after_semantic_failure": bool(
            supervisor_outcome.get("render") is None
            and supervisor_outcome.get("render_artifact_gate") is None
            and _strict_attempts(
                supervisor_outcome.get("attempts"), render_count=0
            )
            and supervisor_outcome.get("render_started_iff_physics_success") is True
            and not external["render_python_pid"].exists()
        ),
        "reserved_combined_exit_bound": bool(
            _strict_json_int(supervisor_outcome.get("combined_exit_status"))
            and supervisor_outcome.get("combined_exit_status") == expected_combined
            and expected_combined != 0
            and paths["exit_status.txt"].read_text().strip() == str(expected_combined)
            and supervisor_outcome.get("pass") is False
        ),
        "supervisor_and_child_reaped": bool(
            not Path(f"/proc/{supervisor_pid}").exists()
            and not Path(f"/proc/{physics_pid}").exists()
            and _linux_pgid_members(pgid) == []
            and _linux_pgid_members(physics_pid) == []
        ),
        "gpu_no_fresh_process": bool(
            supervisor_end_gpu - before_gpu == set()
            and after_gpu - before_gpu == set()
            and _strict_gpu_summary(
                supervisor_outcome.get("gpu"),
                before=before_gpu,
                supervisor_end=supervisor_end_gpu,
            )
        ),
        "outcome_file_bindings_exact": bindings_valid,
        "no_supervisor_failure_marker": not external["supervisor_failure"].exists(),
    }
    attestation_valid = bool(all(checks.values()))
    if not verify_only:
        write_bytes_x(external["gpu_after"], gpu_after_text.encode("utf-8"))
    binding_paths = {**required, "gpu_after": external["gpu_after"]}
    attestation = {
        "artifact": "T3U_EXTERNAL_TERMINAL_ABORT_ATTESTATION_V2",
        "profile": profile,
        "argv": [str(Path(__file__).resolve()), "--terminal_attest", profile],
        "upstream_failure_type": failure.get("type"),
        "upstream_failure_message": failure.get("message"),
        "physics_steps_claimed": None,
        "physics_step_count_authority": "absent_on_aborted_run__no_step_count_claim",
        "supervisor_combined_exit_status": expected_combined,
        "physics_artifact_gate": recomputed_gate,
        "evidence_checks": checks,
        "processes": {
            "supervisor_pid": supervisor_pid,
            "physics_pid": physics_pid,
            "render_pid": None,
            "pgid": pgid,
        },
        "gpu": {
            "before_pids": sorted(before_gpu),
            "supervisor_end_pids": sorted(supervisor_end_gpu),
            "after_pids": sorted(after_gpu),
        },
        "dependency_pins": {
            "p16_source_sha256": sha256_file(Path(__file__)),
            "supervisor_source_sha256": SUPERVISOR_SHA256,
            "selected_prereg_sha256": (
                PREFLIGHT_PREREG_SHA256
                if profile == PREFLIGHT_PROFILE else CANONICAL_PREREG_SHA256
            ),
            "preflight1_failure_sha256": PREFLIGHT1_FAILURE_SHA256,
            "preflight3_source_sha256": PREFLIGHT3_SOURCE_SHA256,
            "preflight3_supervisor_sha256": PREFLIGHT3_SUPERVISOR_SHA256,
            "preflight3_prereg_sha256": PREFLIGHT3_PREREG_SHA256,
            "preflight3_canonical_prereg_sha256": (
                PREFLIGHT3_CANONICAL_PREREG_SHA256
            ),
            "preflight3_zero_launcher_sha256": PREFLIGHT3_LAUNCHER_SHA256,
            "preflight3_posthoc_audit_failure_sha256": (
                PREFLIGHT3_POSTHOC_AUDIT_FAILURE_SHA256
            ),
            "preflight4_source_sha256": PREFLIGHT4_SOURCE_SHA256,
            "preflight4_supervisor_sha256": PREFLIGHT4_SUPERVISOR_SHA256,
            "preflight4_prereg_sha256": PREFLIGHT4_PREREG_SHA256,
            "preflight4_canonical_prereg_sha256": (
                PREFLIGHT4_CANONICAL_PREREG_SHA256
            ),
            "preflight4_failure_sha256": PREFLIGHT4_FAILURE_SHA256,
            "preflight4_launcher_sha256": PREFLIGHT4_LAUNCHER_SHA256,
            **PREFLIGHT5_DEPENDENCY_PINS,
            **PREFLIGHT6_DEPENDENCY_PINS,
            **PREFLIGHT7_DEPENDENCY_PINS,
            **PREFLIGHT8_DEPENDENCY_PINS,
            **PREFLIGHT9_DEPENDENCY_PINS,
        },
        "bindings": {
            name: {"path": str(path.relative_to(REPO)), "sha256": sha256_file(path)}
            for name, path in binding_paths.items()
        },
        "attestation_valid": attestation_valid,
        "scientific_artifacts_complete": False,
        "promotion_allowed": False,
        "pass": False,
        "verdict": (
            "ATTESTED_UPSTREAM_ABORT_BEFORE_SCIENCE__NO_PROMOTION"
            if attestation_valid
            else "UPSTREAM_ABORT_EVIDENCE_INVALID__NO_PROMOTION"
        ),
    }
    if verify_only:
        return attestation
    write_json_x(paths["terminal_attestation.json"], attestation)
    print(
        f"[{LOG}] TERMINAL_ABORT_ATTEST profile={profile} "
        f"valid={attestation_valid} promotion=False",
        flush=True,
    )
    return 0 if attestation_valid else 1


def terminal_attest_mode(
    profile: str, *, verify_only: bool = False
) -> int | dict[str, Any]:
    """Recompute post-reap evidence; optionally verify historical bytes only."""
    prefix = f"{TAG}_{profile}"
    paths = run_paths(prefix)
    external = {
        "stdout": CASE_DIR / f"{prefix}_stdout.log",
        "supervisor_launcher": CASE_DIR / f"{prefix}_supervisor_launcher.log",
        "supervisor_pid": CASE_DIR / f"{prefix}_supervisor_pid.txt",
        "physics_python_pid": CASE_DIR / f"{prefix}_physics_python_pid.txt",
        "render_python_pid": CASE_DIR / f"{prefix}_render_python_pid.txt",
        "pgid": CASE_DIR / f"{prefix}_pgid.txt",
        "supervisor_contract": CASE_DIR / f"{prefix}_supervisor_contract.json",
        "supervisor_outcome": CASE_DIR / f"{prefix}_supervisor_outcome.json",
        "gpu_before": CASE_DIR / f"{prefix}_nvidia_smi_before.csv",
        "gpu_supervisor_end": CASE_DIR / f"{prefix}_nvidia_smi_supervisor_end.csv",
        "gpu_after": CASE_DIR / f"{prefix}_nvidia_smi_after.csv",
        "supervisor_failure": CASE_DIR / f"{prefix}_supervisor_failure.json",
    }
    render_abort_attestation = _terminal_render_abort_attest_mode(
        profile, paths, external, verify_only=verify_only
    )
    if render_abort_attestation is not None:
        return render_abort_attestation
    abort_attestation = _terminal_abort_attest_mode(
        profile, paths, external, verify_only=verify_only
    )
    if abort_attestation is not None:
        return abort_attestation
    required = {
        name: paths[name]
        for name in (
            "results.json", "plan.json", "trace.npz", "timeline.rrd", "timeline.rbl",
            "rerun_validation.json", "decision_snapshot.png", "inspection.png",
            "rgb_frames_manifest.json", "side_grasp.mp4", "script.py.txt", "argv.txt",
            "phase.jsonl", "render_phase.jsonl", "preclose_sentinel.json",
            "exit_status.txt",
        )
    }
    required.update(
        {
            name: path
            for name, path in external.items()
            if name not in {"gpu_after", "supervisor_failure"}
        }
    )
    if verify_only:
        required["gpu_after"] = external["gpu_after"]
    missing = [f"{name}:{path}" for name, path in required.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"TERMINAL_ATTEST_ARTIFACT_MISSING {missing}")
    if not verify_only and (
        paths["terminal_attestation.json"].exists()
        or external["gpu_after"].exists()
    ):
        raise RuntimeError("TERMINAL_ATTEST_FORWARD_ONLY_OUTPUT_EXISTS")

    before_gpu = _gpu_pid_set(external["gpu_before"].read_text())
    supervisor_pid = _read_pid_file_or_invalid(external["supervisor_pid"])
    physics_pid = _read_pid_file_or_invalid(external["physics_python_pid"])
    render_pid = _read_pid_file_or_invalid(external["render_python_pid"])
    pgid = _read_pid_file_or_invalid(external["pgid"])
    exit_status = paths["exit_status.txt"].read_text().strip()
    stdout = external["stdout"].read_text(errors="replace")
    supervisor_contract = json.loads(external["supervisor_contract"].read_text())
    supervisor_outcome = json.loads(external["supervisor_outcome"].read_text())
    results = json.loads(paths["results.json"].read_text())
    plan = json.loads(paths["plan.json"].read_text())
    sentinel = json.loads(paths["preclose_sentinel.json"].read_text())
    rerun_validation = json.loads(paths["rerun_validation.json"].read_text())
    render_manifest = json.loads(paths["rgb_frames_manifest.json"].read_text())
    physics_argv = paths["argv.txt"].read_text().splitlines()
    phase_rows = [
        json.loads(line) for line in paths["phase.jsonl"].read_text().splitlines() if line.strip()
    ]
    phase_names = [row.get("phase") for row in phase_rows]
    render_phase_rows = [
        json.loads(line)
        for line in paths["render_phase.jsonl"].read_text().splitlines()
        if line.strip()
    ]
    render_phase_names = [row.get("phase") for row in render_phase_rows]
    result_semantic_checks = validate_result_semantics(
        profile, paths, results, plan
    )
    result_semantic_exact_all_true = bool(
        isinstance(result_semantic_checks, dict)
        and set(result_semantic_checks) == RESULT_SEMANTIC_CHECK_KEYS
        and all(
            type(value) is bool and value is True
            for value in result_semantic_checks.values()
        )
    )
    render_semantic_checks = validate_render_manifest_semantics(
        profile, paths, render_manifest, results, plan
    )
    warning_tokens = (
        "Traceback (most recent call last)", "CUDA_UNAVAILABLE", "FAILURE_MARKER",
        "G0_ARTIFACT_EXISTS_ABORT", "RuntimeError:", "Segmentation fault", "core dumped",
    )
    expected_supervisor_contract = {
        "artifact": "T3U_DETACHED_PHYSICS_THEN_RENDER_SUPERVISOR_V12",
        "automatic_retry_count": 0,
        "detached": True,
        "physics_timeout_seconds": 7200,
        "render_timeout_seconds": 7200,
        "term_signal": "TERM",
        "kill_after_seconds": 20,
        "physics_then_render_only_on_raw_zero_and_preclose_semantic_gate": True,
        "physics_semantic_gate_artifact": "T3U_PHYSICS_PRECLOSE_SEMANTIC_GATE_V1",
        "render_success_requires_raw_zero_and_posthoc_semantic_gate": True,
        "render_semantic_gate_artifact": "T3U_RENDER_POSTHOC_SEMANTIC_GATE_V1",
        "semantic_gate_failure_exit_status": 125,
        "raw_waitpid_status_authority": True,
        "bounded_waitpid_only": True,
        "supervisor_signal_cleanup": (
            "SIGTERM_SIGINT__active_child_pgid_TERM_20s_then_KILL_20s"
        ),
        "child_parent_death_signal": "SIGTERM",
        "child_preexec_signal_state": (
            "SIGTERM_SIGINT_SIGHUP_SIG_DFL__empty_mask__expected_parent_pid_recheck"
        ),
        "host_launch_boundary": (
            "require_escalated_exec_command__outside_bwrap_die_with_parent"
        ),
        "forbidden_sandbox_ancestor_gate": True,
    }
    host_launch_context = (
        supervisor_contract.get("host_launch_context")
        if isinstance(supervisor_contract, dict) else None
    )
    expected_supervisor_contract["host_launch_context"] = host_launch_context
    host_launch_context_exact = _validate_host_launch_context(
        host_launch_context, supervisor_pid
    )
    expected_supervisor_argv = [
        str(SUPERVISOR_PATH), "--profile", profile,
        "--candidates_sha256", P15_CANDIDATES_SHA256,
    ]
    expected_physics_command = [
        ISAAC_PYTHON, str(Path(__file__).resolve()), "--run_label", profile,
        "--candidates_sha256", P15_CANDIDATES_SHA256,
    ]
    expected_render_command = [
        ISAAC_PYTHON, str(Path(__file__).resolve()), "--render_trace", profile,
    ]
    expected_outcome_keys = {
        "artifact", "profile", "argv", "supervisor_source_sha256",
        "p16_source_sha256", "candidates_sha256", "start_time_unix",
        "end_time_unix", "elapsed_seconds", "supervisor", "attempts",
        "physics", "physics_artifact_gate", "render", "render_artifact_gate",
        "render_started_iff_physics_success",
        "combined_exit_status", "gpu", "bindings", "contract",
        "host_launch_context", "pass",
    }
    physics_outcome = supervisor_outcome.get("physics", {})
    render_outcome = supervisor_outcome.get("render", {})
    supervisor_helper = load_module(
        f"p16_terminal_supervisor_helper_{profile}", SUPERVISOR_PATH
    )
    recomputed_physics_artifact_gate = supervisor_helper._physics_preclose_semantic_gate(
        profile, prefix, paths, physics_outcome, external["stdout"]
    )
    recomputed_render_artifact_gate = supervisor_helper._render_posthoc_semantic_gate(
        profile, prefix, paths, render_outcome, external["stdout"]
    )

    expected_outcome_binding_paths = {
        **{
            name: paths[name]
            for name in (
                "results.json", "plan.json", "trace.npz", "timeline.rrd",
                "timeline.rbl", "rerun_validation.json", "decision_snapshot.png",
                "inspection.png", "rgb_frames_manifest.json", "side_grasp.mp4",
                "script.py.txt", "argv.txt", "phase.jsonl", "render_phase.jsonl",
                "preclose_sentinel.json", "exit_status.txt",
            )
        },
        "stdout.log": external["stdout"],
        "supervisor_pid.txt": external["supervisor_pid"],
        "physics_python_pid.txt": external["physics_python_pid"],
        "render_python_pid.txt": external["render_python_pid"],
        "pgid.txt": external["pgid"],
        "supervisor_contract.json": external["supervisor_contract"],
        "nvidia_smi_before.csv": external["gpu_before"],
        "nvidia_smi_supervisor_end.csv": external["gpu_supervisor_end"],
        "supervisor_launcher.log": external["supervisor_launcher"],
    }
    outcome_bindings = supervisor_outcome.get("bindings", {})
    outcome_bindings_exact = bool(
        isinstance(outcome_bindings, dict)
        and set(outcome_bindings) == set(expected_outcome_binding_paths)
        and all(
            isinstance(row, dict)
            and set(row) == {"path", "sha256"}
            and row["path"] == str(expected_outcome_binding_paths[name].relative_to(REPO))
            and expected_outcome_binding_paths[name].is_file()
            and sha256_file(expected_outcome_binding_paths[name]) == row["sha256"]
            for name, row in outcome_bindings.items()
        )
    )
    gpu_supervisor_end = external["gpu_supervisor_end"].read_text()
    # All JSON/NPZ/render/phase/raw-wait and binding recomputation above is pure.
    # Only now create the one-shot gpu_after file, preventing malformed evidence from
    # leaving a retry-blocking output without a terminal attestation.
    if verify_only:
        gpu_after_text = external["gpu_after"].read_text()
        after_gpu = _gpu_pid_set(gpu_after_text)
    else:
        try:
            query = subprocess.run(
                [
                    "nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=15.0,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("TERMINAL_NVIDIA_SMI_TIMEOUT_15S") from exc
        if query.returncode != 0:
            raise RuntimeError(f"TERMINAL_NVIDIA_SMI_FAIL {query.stderr}")
        gpu_after_text = query.stdout
        after_gpu = _gpu_pid_set(gpu_after_text)
    supervisor_outcome_checks = {
        "top_level_schema_and_exact_keys": bool(
            set(supervisor_outcome) == expected_outcome_keys
            and supervisor_outcome.get("artifact")
            == "T3U_DETACHED_SUPERVISOR_OUTCOME_V12"
            and supervisor_outcome.get("profile") == profile
        ),
        "supervisor_argv_exact": supervisor_outcome.get("argv") == expected_supervisor_argv,
        "source_and_candidate_pins_exact": bool(
            supervisor_outcome.get("supervisor_source_sha256") == SUPERVISOR_SHA256
            and supervisor_outcome.get("p16_source_sha256")
            == sha256_file(Path(__file__))
            and supervisor_outcome.get("candidates_sha256") == P15_CANDIDATES_SHA256
        ),
        "contract_exact_and_bound": bool(
            _json_type_value_exact(
                supervisor_outcome.get("contract"), expected_supervisor_contract
            )
            and _json_type_value_exact(
                supervisor_contract, expected_supervisor_contract
            )
            and host_launch_context_exact
            and _json_type_value_exact(
                supervisor_outcome.get("host_launch_context"), host_launch_context
            )
        ),
        "supervisor_identity_exact": bool(
            _strict_supervisor_identity(
                supervisor_outcome.get("supervisor"),
                pid=supervisor_pid,
                pgid=pgid,
            )
        ),
        "attempt_counts_exact_no_retry": _strict_attempts(
            supervisor_outcome.get("attempts"), render_count=1
        ),
        "physics_raw_wait_success_exact": _strict_child_lifecycle(
            physics_outcome,
            label="physics",
            command=expected_physics_command,
            pid=physics_pid,
            supervisor_sid=supervisor_pid,
            require_success=True,
        ),
        "physics_preclose_semantic_gate_recomputed_exact": bool(
            _json_type_value_exact(
                supervisor_outcome.get("physics_artifact_gate"),
                recomputed_physics_artifact_gate,
            )
            and recomputed_physics_artifact_gate.get("pass") is True
        ),
        "render_raw_wait_success_exact": _strict_child_lifecycle(
            render_outcome,
            label="render",
            command=expected_render_command,
            pid=render_pid,
            supervisor_sid=supervisor_pid,
            require_success=True,
        ),
        "render_posthoc_semantic_gate_recomputed_exact": bool(
            _json_type_value_exact(
                supervisor_outcome.get("render_artifact_gate"),
                recomputed_render_artifact_gate,
            )
            and recomputed_render_artifact_gate.get("pass") is True
        ),
        "render_iff_physics_success_exact": bool(
            supervisor_outcome.get("render_started_iff_physics_success") is True
            and render_outcome
        ),
        "combined_exit_exact": bool(
            _strict_json_int(supervisor_outcome.get("combined_exit_status"))
            and supervisor_outcome.get("combined_exit_status") == 0
            and exit_status == "0"
        ),
        "child_process_groups_empty": bool(
            _linux_pgid_members(physics_pid) == []
            and _linux_pgid_members(render_pid) == []
        ),
        "gpu_before_to_supervisor_end_no_fresh_pid": bool(
            _strict_gpu_summary(
                supervisor_outcome.get("gpu"),
                before=before_gpu,
                supervisor_end=_gpu_pid_set(gpu_supervisor_end),
            )
        ),
        "outcome_file_bindings_exact": outcome_bindings_exact,
        "outcome_pass_recomputed_not_trusted": supervisor_outcome.get("pass") is True,
        "timestamps_monotonic": _strict_outcome_times(
            supervisor_outcome, [physics_outcome, render_outcome]
        ),
    }
    supervisor_outcome_checks["outcome_pass_recomputed_not_trusted"] = bool(
        supervisor_outcome.get("pass") is True
        and all(
            value
            for key, value in supervisor_outcome_checks.items()
            if key != "outcome_pass_recomputed_not_trusted"
        )
    )
    artifact_checks = {
        "results_bound_to_sentinel": bool(
            sentinel.get("results_sha256") == sha256_file(paths["results.json"])
            and sentinel.get("trace_sha256") == sha256_file(paths["trace.npz"])
            and sentinel.get("rerun_validation_sha256")
            == sha256_file(paths["rerun_validation.json"])
            and sentinel.get("source_sha256") == sha256_file(Path(__file__))
        ),
        "source_and_frozen_copy_exact": bool(
            sha256_file(paths["script.py.txt"]) == sha256_file(Path(__file__))
            == results.get("provenance", {}).get("source_sha256")
        ),
        "physics_argv_exact": bool(
            physics_argv[1:]
            == [
                "--run_label", profile, "--candidates_sha256",
                results.get("provenance", {}).get("p15_sha256"),
            ]
        ),
        "plan_representative_binding_exact": bool(
            plan.get("representative_binding") == results.get("representative_binding")
        ),
        "preclose_artifact_hashes_exact": bool(
            all(
                results.get("artifact_hashes_preclose", {}).get(name)
                == sha256_file(paths[name])
                for name in (
                    "plan.json", "trace.npz", "timeline.rrd", "timeline.rbl",
                    "rerun_validation.json", "decision_snapshot.png", "inspection.png",
                )
            )
        ),
        "rerun_validation_pass": bool(rerun_validation.get("pass")),
        "results_rerun_technical_pass": bool(results.get("rerun", {}).get("technical_pass")),
        "results_runtime_instrumentation_pass": bool(
            results.get("runtime_instrumentation_pass") is True
            and result_semantic_exact_all_true
        ),
        "render_manifest_pass": bool(
            render_manifest.get("pass") is True
            and all(render_semantic_checks.values())
        ),
        "render_argv_exact": bool(
            render_manifest.get("argv", [])[1:] == ["--render_trace", profile]
        ),
        "render_source_bindings_exact": bool(
            render_manifest.get("source_results_sha256")
            == sha256_file(paths["results.json"])
            and render_manifest.get("source_plan_sha256")
            == sha256_file(paths["plan.json"])
            and render_manifest.get("executed_source_sha256")
            == sha256_file(Path(__file__))
            and render_manifest.get("representative_binding")
            == results.get("representative_binding")
        ),
        "render_trace_hash_exact": bool(
            render_manifest.get("source_trace_sha256") == sha256_file(paths["trace.npz"])
        ),
        "render_mp4_hash_exact": bool(
            render_manifest.get("mp4_sha256") == sha256_file(paths["side_grasp.mp4"])
        ),
        "render_zero_physics_steps": render_semantic_checks[
            "zero_physics_clocks_callbacks_scenes_recomputed"
        ],
        "render_dependency_physics_finalize_start_end_exact": render_semantic_checks[
            "decision_dependency_start_end_current_hashes_exact"
        ],
        "render_all_frame_state_fidelity_exact": render_semantic_checks[
            "all_frame_files_trace_mappings_clocks_and_state_exact"
        ],
        "render_full_decode_pass": bool(
            render_manifest.get("decode", {}).get("full_decode_pass") is True
            and render_manifest.get("decode", {}).get("decoded_frame_count")
            == render_manifest.get("frame_count") == TOTAL_STEPS // VIDEO_STEP_STRIDE
        ),
    }
    lifecycle_checks = {
        "supervisor_contract_exact": _json_type_value_exact(
            supervisor_contract, expected_supervisor_contract
        ),
        "combined_exit_status_zero": exit_status == "0",
        "failure_marker_absent": not paths["failure.json"].exists(),
        "render_failure_marker_absent": not paths["render_failure.json"].exists(),
        "supervisor_failure_marker_absent": not external["supervisor_failure"].exists(),
        "supervisor_reaped": not Path(f"/proc/{supervisor_pid}").exists(),
        "physics_python_reaped": not Path(f"/proc/{physics_pid}").exists(),
        "render_python_reaped": not Path(f"/proc/{render_pid}").exists(),
        "process_group_empty": _linux_pgid_members(pgid) == [],
        "physics_pid_not_on_gpu": physics_pid not in after_gpu,
        "render_pid_not_on_gpu": render_pid not in after_gpu,
        "fresh_gpu_pid_delta_empty": not (after_gpu - before_gpu),
        "stdout_has_no_failure_tokens": not any(token in stdout for token in warning_tokens),
        "supervisor_launcher_has_no_failure_tokens": not any(
            token in external["supervisor_launcher"].read_text(errors="replace")
            for token in warning_tokens
        ),
        "stdout_one_physics_preclose": stdout.count(f"[{LOG}] PRECLOSE") == 1,
        "stdout_one_render_completion": stdout.count(f"[{LOG}] RENDER_TRACE_COMPLETE") == 1,
        "phase_order_complete": bool(
            phase_names
            in (
                [
                    "run_claim", "results_durable", "preclose_sentinel_durable",
                    "simulation_app_close_start",
                ],
                [
                    "run_claim", "results_durable", "preclose_sentinel_durable",
                    "simulation_app_close_start", "simulation_app_close_returned",
                ],
            )
            and render_phase_names == ["render_trace_durable"]
        ),
        "phase_result_hash_binding": bool(
            phase_rows[1].get("results_sha256") == sha256_file(paths["results.json"])
            if len(phase_rows) >= 2 else False
        ),
        "phase_sentinel_hash_binding": bool(
            phase_rows[2].get("sentinel_sha256")
            == sha256_file(paths["preclose_sentinel.json"])
            if len(phase_rows) >= 3 else False
        ),
        "phase_render_hash_binding": bool(
            render_phase_rows[0].get("manifest_sha256")
            == sha256_file(paths["rgb_frames_manifest.json"])
            and render_phase_rows[0].get("mp4_sha256")
            == sha256_file(paths["side_grasp.mp4"])
            if len(render_phase_rows) == 1 else False
        ),
    }
    all_pass = bool(
        all(artifact_checks.values())
        and all(lifecycle_checks.values())
        and all(supervisor_outcome_checks.values())
        and result_semantic_exact_all_true
        and all(render_semantic_checks.values())
    )
    if not verify_only:
        write_bytes_x(external["gpu_after"], gpu_after_text.encode("utf-8"))
    attestation_binding_paths = {**required, "gpu_after": external["gpu_after"]}
    attestation = {
        "artifact": "T3U_EXTERNAL_TERMINAL_ATTESTATION_V4",
        "profile": profile,
        "argv": [str(Path(__file__).resolve()), "--terminal_attest", profile],
        "artifact_checks": artifact_checks,
        "lifecycle_checks": lifecycle_checks,
        "supervisor_outcome_checks": supervisor_outcome_checks,
        "result_semantic_checks": result_semantic_checks,
        "render_semantic_checks": render_semantic_checks,
        "processes": {
            "supervisor_pid": supervisor_pid,
            "physics_python_pid": physics_pid,
            "render_python_pid": render_pid,
            "pgid": pgid,
            "pgid_members_after_exit": _linux_pgid_members(pgid),
            "combined_exit_status": exit_status,
        },
        "gpu": {
            "before_pids": sorted(before_gpu), "after_pids": sorted(after_gpu),
            "new_pids": sorted(after_gpu - before_gpu),
            "before_sha256": sha256_file(external["gpu_before"]),
            "after_sha256": sha256_file(external["gpu_after"]),
        },
        "dependency_pins": {
            "p16_source_sha256": sha256_file(Path(__file__)),
            "supervisor_source_sha256": SUPERVISOR_SHA256,
            "preflight_prereg_sha256": PREFLIGHT_PREREG_SHA256,
            "canonical_prereg_sha256": CANONICAL_PREREG_SHA256,
            "p15_source_sha256": P15_SHA256,
            "p15_prereg_sha256": P15_PREREG_SHA256,
            "p15_candidates_sha256": P15_CANDIDATES_SHA256,
            "preflight2_terminal_attestation_sha256": (
                PREFLIGHT2_TERMINAL_ATTESTATION_SHA256
            ),
            "preflight2_failure_sha256": PREFLIGHT2_FAILURE_SHA256,
            "preflight2_supervisor_outcome_sha256": PREFLIGHT2_OUTCOME_SHA256,
            "preflight2_phase_sha256": PREFLIGHT2_PHASE_SHA256,
            "preflight2_exit_status_sha256": PREFLIGHT2_EXIT_STATUS_SHA256,
            "retired_canonical_prereg_sha256": RETIRED_CANONICAL_PREREG_SHA256,
            "preflight3_source_sha256": PREFLIGHT3_SOURCE_SHA256,
            "preflight3_supervisor_sha256": PREFLIGHT3_SUPERVISOR_SHA256,
            "preflight3_prereg_sha256": PREFLIGHT3_PREREG_SHA256,
            "preflight3_canonical_prereg_sha256": (
                PREFLIGHT3_CANONICAL_PREREG_SHA256
            ),
            "preflight3_zero_launcher_sha256": PREFLIGHT3_LAUNCHER_SHA256,
            "preflight3_posthoc_audit_failure_sha256": (
                PREFLIGHT3_POSTHOC_AUDIT_FAILURE_SHA256
            ),
            "preflight4_source_sha256": PREFLIGHT4_SOURCE_SHA256,
            "preflight4_supervisor_sha256": PREFLIGHT4_SUPERVISOR_SHA256,
            "preflight4_prereg_sha256": PREFLIGHT4_PREREG_SHA256,
            "preflight4_canonical_prereg_sha256": (
                PREFLIGHT4_CANONICAL_PREREG_SHA256
            ),
            "preflight4_failure_sha256": PREFLIGHT4_FAILURE_SHA256,
            "preflight4_launcher_sha256": PREFLIGHT4_LAUNCHER_SHA256,
            **PREFLIGHT5_DEPENDENCY_PINS,
            **PREFLIGHT6_DEPENDENCY_PINS,
            **PREFLIGHT7_DEPENDENCY_PINS,
            **PREFLIGHT8_DEPENDENCY_PINS,
            **PREFLIGHT9_DEPENDENCY_PINS,
        },
        "bindings": {
            name: {"path": str(path.relative_to(REPO)), "sha256": sha256_file(path)}
            for name, path in attestation_binding_paths.items()
        },
        "manual_visual_inspection_still_required": True,
        "scientific_or_preflight_promotion": False,
        "pass": all_pass,
        "verdict": (
            "TERMINAL_ATTESTED_PENDING_MANUAL_VISUAL"
            if all_pass else "EXTERNAL_TERMINAL_ATTESTATION_FAIL"
        ),
    }
    if verify_only:
        return attestation
    write_json_x(paths["terminal_attestation.json"], attestation)
    print(
        f"[{LOG}] TERMINAL_ATTEST profile={profile} pass={all_pass} ",
        f"path={paths['terminal_attestation.json']}",
        flush=True,
    )
    return 0 if all_pass else 1


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prelaunch_guard", choices=list(EXECUTABLE_PROFILES))
    mode.add_argument("--launch_liveness_guard", choices=list(EXECUTABLE_PROFILES))
    mode.add_argument("--run_label", choices=list(EXECUTABLE_PROFILES))
    mode.add_argument("--render_trace", choices=list(EXECUTABLE_PROFILES))
    mode.add_argument("--terminal_attest", choices=list(EXECUTABLE_PROFILES))
    parser.add_argument("--candidates_sha256")
    parser.add_argument("--supervisor_pid", type=int)
    return parser


def prelaunch_guard_mode(profile: str) -> int:
    """Pure, output-free pin/G0 check used before the host wrapper redirects."""
    if profile not in EXECUTABLE_PROFILES:
        raise RuntimeError(f"PRELAUNCH_PROFILE_INVALID {profile}")
    # The shell's PID-1 check catches the observed Codex namespace.  Walk every
    # visible ancestor as well so a sandbox that exposes host PID 1 but retains a
    # bubblewrap/codex sandbox ancestor still fails before launcher redirection.
    pid = os.getpid()
    seen: set[int] = set()
    forbidden: list[dict[str, Any]] = []
    reached_pid1 = False
    for _ in range(64):
        if pid <= 0 or pid in seen:
            break
        seen.add(pid)
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        argv = [
            token.decode("utf-8", errors="replace")
            for token in raw.split(b"\0")
            if token
        ]
        executable = Path(argv[0]).name if argv else ""
        if executable in {"bwrap", "codex-linux-sandbox"}:
            forbidden.append({"pid": pid, "token": executable})
        if "--die-with-parent" in argv:
            forbidden.append({"pid": pid, "token": "--die-with-parent"})
        if pid == 1:
            reached_pid1 = True
            break
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        close = stat.rfind(")")
        fields = stat[close + 2 :].split() if close >= 0 else []
        if len(fields) < 2:
            raise RuntimeError(f"PRELAUNCH_HOST_ANCESTRY_STAT_FAIL pid={pid}")
        pid = int(fields[1])
    if not reached_pid1:
        raise RuntimeError("PRELAUNCH_HOST_ANCESTRY_INCOMPLETE")
    if forbidden:
        raise RuntimeError(
            "PRELAUNCH_HOST_REQUIRED__SANDBOX_ANCESTOR_FORBIDDEN "
            f"matches={forbidden!r}"
        )
    # Exercise exactly the two namespace paths V7 will use, before any redirect or
    # G0 target is consumed.  This proves only procfs accessibility/self consistency.
    _read_self_pid_namespace_evidence()
    selected = PREFLIGHT_PREREG if profile == PREFLIGHT_PROFILE else CANONICAL_PREREG
    selected_sha = (
        PREFLIGHT_PREREG_SHA256
        if profile == PREFLIGHT_PROFILE else CANONICAL_PREREG_SHA256
    )
    pins = (
        (SUPERVISOR_PATH, SUPERVISOR_SHA256),
        (selected, selected_sha),
        (PREFLIGHT_PREREG, PREFLIGHT_PREREG_SHA256),
        (CANONICAL_PREREG, CANONICAL_PREREG_SHA256),
        (CANDIDATES_PATH, P15_CANDIDATES_SHA256),
        (PREFLIGHT3_SOURCE_PATH, PREFLIGHT3_SOURCE_SHA256),
        (PREFLIGHT3_SUPERVISOR_PATH, PREFLIGHT3_SUPERVISOR_SHA256),
        (PREFLIGHT3_PREREG, PREFLIGHT3_PREREG_SHA256),
        (PREFLIGHT3_CANONICAL_PREREG, PREFLIGHT3_CANONICAL_PREREG_SHA256),
        (PREFLIGHT3_LAUNCHER_PATH, PREFLIGHT3_LAUNCHER_SHA256),
        (
            PREFLIGHT3_POSTHOC_AUDIT_FAILURE_PATH,
            PREFLIGHT3_POSTHOC_AUDIT_FAILURE_SHA256,
        ),
        (PREFLIGHT4_SOURCE_PATH, PREFLIGHT4_SOURCE_SHA256),
        (PREFLIGHT4_SUPERVISOR_PATH, PREFLIGHT4_SUPERVISOR_SHA256),
        (PREFLIGHT4_PREREG, PREFLIGHT4_PREREG_SHA256),
        (PREFLIGHT4_CANONICAL_PREREG, PREFLIGHT4_CANONICAL_PREREG_SHA256),
        (PREFLIGHT4_FAILURE_PATH, PREFLIGHT4_FAILURE_SHA256),
        (PREFLIGHT4_LAUNCHER_PATH, PREFLIGHT4_LAUNCHER_SHA256),
        (PREFLIGHT5_SOURCE_PATH, PREFLIGHT5_SOURCE_SHA256),
        (PREFLIGHT5_SUPERVISOR_PATH, PREFLIGHT5_SUPERVISOR_SHA256),
        (
            PREFLIGHT5_CANONICAL_PREREG,
            PREFLIGHT5_CANONICAL_PREREG_SHA256,
        ),
        *tuple(
            (
                CASE_DIR / f"t3u_side_preflight5_{suffix}",
                expected_sha,
            )
            for suffix, expected_sha in PREFLIGHT5_EVIDENCE_SHA256.items()
        ),
        (PREFLIGHT6_SOURCE_PATH, PREFLIGHT6_SOURCE_SHA256),
        (PREFLIGHT6_SUPERVISOR_PATH, PREFLIGHT6_SUPERVISOR_SHA256),
        (PREFLIGHT6_PREREG, PREFLIGHT6_PREREG_SHA256),
        (
            PREFLIGHT6_CANONICAL_PREREG,
            PREFLIGHT6_CANONICAL_PREREG_SHA256,
        ),
        *tuple(
            (
                CASE_DIR / f"t3u_side_preflight6_{suffix}",
                expected_sha,
            )
            for suffix, expected_sha in PREFLIGHT6_EVIDENCE_SHA256.items()
        ),
        (PREFLIGHT7_SOURCE_PATH, PREFLIGHT7_SOURCE_SHA256),
        (PREFLIGHT7_SUPERVISOR_PATH, PREFLIGHT7_SUPERVISOR_SHA256),
        (PREFLIGHT7_PREREG, PREFLIGHT7_PREREG_SHA256),
        (PREFLIGHT7_CANONICAL_PREREG, PREFLIGHT7_CANONICAL_PREREG_SHA256),
        *tuple(
            (CASE_DIR / f"t3u_side_preflight7_{suffix}", expected_sha)
            for suffix, expected_sha in PREFLIGHT7_EVIDENCE_SHA256.items()
        ),
        (PREFLIGHT8_SOURCE_PATH, PREFLIGHT8_SOURCE_SHA256),
        (PREFLIGHT8_SUPERVISOR_PATH, PREFLIGHT8_SUPERVISOR_SHA256),
        (PREFLIGHT8_PREREG, PREFLIGHT8_PREREG_SHA256),
        (PREFLIGHT8_CANONICAL_PREREG, PREFLIGHT8_CANONICAL_PREREG_SHA256),
        *tuple(
            (CASE_DIR / f"t3u_side_preflight8_{suffix}", expected_sha)
            for suffix, expected_sha in PREFLIGHT8_EVIDENCE_SHA256.items()
        ),
        (PREFLIGHT9_SOURCE_PATH, PREFLIGHT9_SOURCE_SHA256),
        (PREFLIGHT9_SUPERVISOR_PATH, PREFLIGHT9_SUPERVISOR_SHA256),
        (PREFLIGHT9_PREREG, PREFLIGHT9_PREREG_SHA256),
        (PREFLIGHT9_CANONICAL_PREREG, PREFLIGHT9_CANONICAL_PREREG_SHA256),
        *tuple(
            (CASE_DIR / f"t3u_side_preflight9_{suffix}", expected_sha)
            for suffix, expected_sha in PREFLIGHT9_EVIDENCE_SHA256.items()
        ),
    )
    for path, expected in pins:
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"PRELAUNCH_PIN_DRIFT path={path} expected={expected} actual={actual}"
            )
    validate_preflight3_launch_retirement()
    validate_preflight4_launch_retirement()
    validate_preflight5_retirement()
    validate_preflight6_retirement()
    validate_preflight7_retirement()
    validate_preflight8_retirement()
    validate_object_physx_world_frame_regression()
    prefix = f"{TAG}_{profile}"
    external_suffixes = (
        "stdout.log", "supervisor_launcher.log", "supervisor_pid.txt",
        "physics_python_pid.txt", "render_python_pid.txt", "pgid.txt",
        "supervisor_contract.json", "supervisor_outcome.json",
        "nvidia_smi_before.csv", "nvidia_smi_supervisor_end.csv",
        "nvidia_smi_after.csv", "supervisor_failure.json",
    )
    targets = [
        *run_paths(prefix).values(),
        *(CASE_DIR / f"{prefix}_{suffix}" for suffix in external_suffixes),
        CASE_DIR / f"{prefix}_rgb_frames",
    ]
    present = [path for path in targets if path.exists()]
    if present:
        raise RuntimeError(
            "PRELAUNCH_G0_TARGET_EXISTS "
            f"{[str(path.relative_to(REPO)) for path in present]}"
        )
    return 0


def launch_liveness_guard_mode(profile: str, supervisor_pid: int) -> int:
    """Output-free second host check, called only after >=2 seconds of survival."""
    if profile not in EXECUTABLE_PROFILES or type(supervisor_pid) is not int:
        raise RuntimeError("LAUNCH_LIVENESS_ARGUMENT_INVALID")
    if supervisor_pid <= 1:
        raise RuntimeError("LAUNCH_LIVENESS_PID_INVALID")
    prefix = f"{TAG}_{profile}"
    pid_path = CASE_DIR / f"{prefix}_supervisor_pid.txt"
    pgid_path = CASE_DIR / f"{prefix}_pgid.txt"
    contract_path = CASE_DIR / f"{prefix}_supervisor_contract.json"
    recorded_pid = _read_pid_file_or_invalid(pid_path)
    recorded_pgid = _read_pid_file_or_invalid(pgid_path)
    if recorded_pid != supervisor_pid or recorded_pgid != supervisor_pid:
        raise RuntimeError("LAUNCH_LIVENESS_PID_PGID_BINDING_FAIL")
    if not Path(f"/proc/{supervisor_pid}").is_dir():
        raise RuntimeError("LAUNCH_LIVENESS_PROCESS_ABSENT")
    if os.getpgid(supervisor_pid) != supervisor_pid or os.getsid(supervisor_pid) != supervisor_pid:
        raise RuntimeError("LAUNCH_LIVENESS_SESSION_LEADER_FAIL")
    raw_cmdline = Path(f"/proc/{supervisor_pid}/cmdline").read_bytes()
    actual_argv = [
        token.decode("utf-8", errors="strict")
        for token in raw_cmdline.split(b"\0")
        if token
    ]
    expected_argv = [
        ISAAC_PYTHON, str(SUPERVISOR_PATH), "--profile", profile,
        "--candidates_sha256", P15_CANDIDATES_SHA256,
    ]
    if actual_argv != expected_argv:
        raise RuntimeError(
            f"LAUNCH_LIVENESS_ARGV_FAIL expected={expected_argv} actual={actual_argv}"
        )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    host_context = contract.get("host_launch_context") if isinstance(contract, dict) else None
    expected_contract = {
        "artifact": "T3U_DETACHED_PHYSICS_THEN_RENDER_SUPERVISOR_V12",
        "automatic_retry_count": 0,
        "detached": True,
        "physics_timeout_seconds": 7200,
        "render_timeout_seconds": 7200,
        "term_signal": "TERM",
        "kill_after_seconds": 20,
        "physics_then_render_only_on_raw_zero_and_preclose_semantic_gate": True,
        "physics_semantic_gate_artifact": "T3U_PHYSICS_PRECLOSE_SEMANTIC_GATE_V1",
        "render_success_requires_raw_zero_and_posthoc_semantic_gate": True,
        "render_semantic_gate_artifact": "T3U_RENDER_POSTHOC_SEMANTIC_GATE_V1",
        "semantic_gate_failure_exit_status": 125,
        "raw_waitpid_status_authority": True,
        "bounded_waitpid_only": True,
        "supervisor_signal_cleanup": (
            "SIGTERM_SIGINT__active_child_pgid_TERM_20s_then_KILL_20s"
        ),
        "child_parent_death_signal": "SIGTERM",
        "child_preexec_signal_state": (
            "SIGTERM_SIGINT_SIGHUP_SIG_DFL__empty_mask__expected_parent_pid_recheck"
        ),
        "host_launch_boundary": (
            "require_escalated_exec_command__outside_bwrap_die_with_parent"
        ),
        "forbidden_sandbox_ancestor_gate": True,
        "host_launch_context": host_context,
    }
    if (
        not _validate_host_launch_context(host_context, supervisor_pid)
        or not _json_type_value_exact(contract, expected_contract)
    ):
        raise RuntimeError("LAUNCH_LIVENESS_CONTRACT_FAIL")
    return 0


def main() -> int:
    args = build_argparser().parse_args()
    if args.launch_liveness_guard is not None:
        expected_argv = [
            "--launch_liveness_guard", args.launch_liveness_guard,
            "--supervisor_pid", str(args.supervisor_pid),
        ]
        if sys.argv[1:] != expected_argv:
            raise RuntimeError(f"LAUNCH_LIVENESS_GUARD_ARGV_DRIFT argv={sys.argv}")
        if args.supervisor_pid is None or args.candidates_sha256 is not None:
            raise RuntimeError("LAUNCH_LIVENESS_GUARD_ARGUMENT_FAIL")
        return launch_liveness_guard_mode(
            args.launch_liveness_guard, args.supervisor_pid
        )
    if args.supervisor_pid is not None:
        raise RuntimeError("SUPERVISOR_PID_ONLY_FOR_LAUNCH_LIVENESS_GUARD")
    if args.prelaunch_guard is not None:
        if sys.argv[1:] != ["--prelaunch_guard", args.prelaunch_guard]:
            raise RuntimeError(f"PRELAUNCH_GUARD_ARGV_DRIFT argv={sys.argv}")
        if args.candidates_sha256 is not None:
            raise RuntimeError("PRELAUNCH_GUARD_FORBIDS_PHYSICS_ARGUMENTS")
        return prelaunch_guard_mode(args.prelaunch_guard)
    if args.terminal_attest is not None:
        if sys.argv[1:] != ["--terminal_attest", args.terminal_attest]:
            raise RuntimeError(f"TERMINAL_ATTEST_ARGV_DRIFT argv={sys.argv}")
        if args.candidates_sha256 is not None:
            raise RuntimeError("TERMINAL_ATTEST_FORBIDS_PHYSICS_ARGUMENTS")
        return terminal_attest_mode(args.terminal_attest)
    if args.render_trace is not None:
        if sys.argv[1:] != ["--render_trace", args.render_trace]:
            raise RuntimeError(f"RENDER_TRACE_ARGV_DRIFT argv={sys.argv}")
        if args.candidates_sha256 is not None:
            raise RuntimeError("RENDER_TRACE_FORBIDS_PHYSICS_ARGUMENTS")
        try:
            return render_trace_mode(args.render_trace)
        except BaseException as exc:
            # Source-gate failures occur before render_trace_mode's Kit try/finally;
            # retain the same durable failure authority for that earlier boundary.
            render_failure_paths = run_paths(f"{TAG}_{args.render_trace}")
            render_frame_dir = CASE_DIR / f"{TAG}_{args.render_trace}_rgb_frames"
            if not (
                render_failure_paths["rgb_frames_manifest.json"].exists()
                or render_failure_paths["side_grasp.mp4"].exists()
                or render_frame_dir.exists()
            ):
                record_render_failure_evidence(
                    args.render_trace, render_failure_paths, exc
                )
            raise
    if args.candidates_sha256 is None:
        raise RuntimeError("PHYSICS_RUN_REQUIRES_CANDIDATES_SHA256")
    profile = args.run_label
    if sys.argv[1:] != [
        "--run_label", profile, "--candidates_sha256", args.candidates_sha256
    ]:
        raise RuntimeError(f"PHYSICS_ARGV_DRIFT argv={sys.argv}")
    expected_envs = 8 if profile == PREFLIGHT_PROFILE else 64
    args.num_envs = expected_envs
    args.contact_capacity = 256
    args.headless = True
    prefix = f"{TAG}_{profile}"
    paths = run_paths(prefix)
    existing = [str(path.relative_to(REPO)) for path in paths.values() if path.exists()]
    if existing:
        print(f"[{LOG}] G0_ARTIFACT_EXISTS_ABORT {existing}", flush=True)
        return 3

    source_start = Path(__file__).read_bytes()
    source_sha = hashlib.sha256(source_start).hexdigest()
    prereg = PREFLIGHT_PREREG if profile == PREFLIGHT_PROFILE else CANONICAL_PREREG
    prereg_expected = PREFLIGHT_PREREG_SHA256 if profile == PREFLIGHT_PROFILE else CANONICAL_PREREG_SHA256
    if sha256_file(prereg) != prereg_expected:
        raise RuntimeError(
            f"PREREG_SHA256_MISMATCH expected={prereg_expected} actual={sha256_file(prereg)}"
        )
    for path, expected in (
        (P14_PATH, P14_SHA256), (P10_PATH, P10_SHA256),
        (P15_PATH, P15_SHA256), (P15_PREREG_PATH, P15_PREREG_SHA256),
        (SUPERVISOR_PATH, SUPERVISOR_SHA256),
        (PREFLIGHT1_PREREG, PREFLIGHT1_PREREG_SHA256),
        (PREFLIGHT1_FAILURE_PATH, PREFLIGHT1_FAILURE_SHA256),
        (PREFLIGHT1_OUTCOME_PATH, PREFLIGHT1_OUTCOME_SHA256),
        (PREFLIGHT2_SOURCE_PATH, PREFLIGHT2_SOURCE_SHA256),
        (PREFLIGHT2_SUPERVISOR_PATH, PREFLIGHT2_SUPERVISOR_SHA256),
        (PREFLIGHT2_PREREG, PREFLIGHT2_PREREG_SHA256),
        (PREFLIGHT2_FAILURE_PATH, PREFLIGHT2_FAILURE_SHA256),
        (PREFLIGHT2_OUTCOME_PATH, PREFLIGHT2_OUTCOME_SHA256),
        (PREFLIGHT2_PHASE_PATH, PREFLIGHT2_PHASE_SHA256),
        (PREFLIGHT2_EXIT_STATUS_PATH, PREFLIGHT2_EXIT_STATUS_SHA256),
        (
            PREFLIGHT2_TERMINAL_ATTESTATION_PATH,
            PREFLIGHT2_TERMINAL_ATTESTATION_SHA256,
        ),
        (RETIRED_CANONICAL_PREREG, RETIRED_CANONICAL_PREREG_SHA256),
        (PREFLIGHT3_SOURCE_PATH, PREFLIGHT3_SOURCE_SHA256),
        (PREFLIGHT3_SUPERVISOR_PATH, PREFLIGHT3_SUPERVISOR_SHA256),
        (PREFLIGHT3_PREREG, PREFLIGHT3_PREREG_SHA256),
        (
            PREFLIGHT3_CANONICAL_PREREG,
            PREFLIGHT3_CANONICAL_PREREG_SHA256,
        ),
        (PREFLIGHT3_LAUNCHER_PATH, PREFLIGHT3_LAUNCHER_SHA256),
        (
            PREFLIGHT3_POSTHOC_AUDIT_FAILURE_PATH,
            PREFLIGHT3_POSTHOC_AUDIT_FAILURE_SHA256,
        ),
        (PREFLIGHT4_SOURCE_PATH, PREFLIGHT4_SOURCE_SHA256),
        (PREFLIGHT4_SUPERVISOR_PATH, PREFLIGHT4_SUPERVISOR_SHA256),
        (PREFLIGHT4_PREREG, PREFLIGHT4_PREREG_SHA256),
        (PREFLIGHT4_CANONICAL_PREREG, PREFLIGHT4_CANONICAL_PREREG_SHA256),
        (PREFLIGHT4_FAILURE_PATH, PREFLIGHT4_FAILURE_SHA256),
        (PREFLIGHT4_LAUNCHER_PATH, PREFLIGHT4_LAUNCHER_SHA256),
        (PREFLIGHT5_SOURCE_PATH, PREFLIGHT5_SOURCE_SHA256),
        (PREFLIGHT5_SUPERVISOR_PATH, PREFLIGHT5_SUPERVISOR_SHA256),
        (
            PREFLIGHT5_CANONICAL_PREREG,
            PREFLIGHT5_CANONICAL_PREREG_SHA256,
        ),
        *tuple(
            (
                CASE_DIR / f"t3u_side_preflight5_{suffix}",
                expected_sha,
            )
            for suffix, expected_sha in PREFLIGHT5_EVIDENCE_SHA256.items()
        ),
        (PREFLIGHT7_SOURCE_PATH, PREFLIGHT7_SOURCE_SHA256),
        (PREFLIGHT7_SUPERVISOR_PATH, PREFLIGHT7_SUPERVISOR_SHA256),
        (PREFLIGHT7_PREREG, PREFLIGHT7_PREREG_SHA256),
        (PREFLIGHT7_CANONICAL_PREREG, PREFLIGHT7_CANONICAL_PREREG_SHA256),
        *tuple(
            (CASE_DIR / f"t3u_side_preflight7_{suffix}", expected_sha)
            for suffix, expected_sha in PREFLIGHT7_EVIDENCE_SHA256.items()
        ),
        (PREFLIGHT8_SOURCE_PATH, PREFLIGHT8_SOURCE_SHA256),
        (PREFLIGHT8_SUPERVISOR_PATH, PREFLIGHT8_SUPERVISOR_SHA256),
        (PREFLIGHT8_PREREG, PREFLIGHT8_PREREG_SHA256),
        (PREFLIGHT8_CANONICAL_PREREG, PREFLIGHT8_CANONICAL_PREREG_SHA256),
        *tuple(
            (CASE_DIR / f"t3u_side_preflight8_{suffix}", expected_sha)
            for suffix, expected_sha in PREFLIGHT8_EVIDENCE_SHA256.items()
        ),
        (PREFLIGHT_PREREG, PREFLIGHT_PREREG_SHA256),
        (CANONICAL_PREREG, CANONICAL_PREREG_SHA256),
        (WITNESS_RESULTS_PATH, WITNESS_RESULTS_SHA256),
        (WITNESS_PLAN_PATH, WITNESS_PLAN_SHA256),
        (JAW_PATH, JAW_SHA256), (URDF_PATH, URDF_SHA256),
    ):
        if sha256_file(path) != expected:
            raise RuntimeError(f"PINNED_SOURCE_DRIFT path={path} expected={expected}")
    if prereg.resolve() in {
        RETIRED_CANONICAL_PREREG.resolve(),
        PREFLIGHT3_PREREG.resolve(),
        PREFLIGHT3_CANONICAL_PREREG.resolve(),
        PREFLIGHT4_PREREG.resolve(),
        PREFLIGHT4_CANONICAL_PREREG.resolve(),
        (CASE_DIR / "t3u_side_preflight5_prereg.md").resolve(),
        PREFLIGHT5_CANONICAL_PREREG.resolve(),
        PREFLIGHT6_PREREG.resolve(),
        PREFLIGHT6_CANONICAL_PREREG.resolve(),
        PREFLIGHT7_PREREG.resolve(),
        PREFLIGHT7_CANONICAL_PREREG.resolve(),
        PREFLIGHT8_PREREG.resolve(),
        PREFLIGHT8_CANONICAL_PREREG.resolve(),
        PREFLIGHT9_PREREG.resolve(),
        PREFLIGHT9_CANONICAL_PREREG.resolve(),
    }:
        raise RuntimeError("RETIRED_CANONICAL_PREREG_IS_NOT_EXECUTABLE")
    preflight2_retirement = validate_preflight2_retirement()
    preflight3_retirement = validate_preflight3_launch_retirement()
    preflight4_retirement = validate_preflight4_launch_retirement()
    preflight5_retirement = validate_preflight5_retirement()
    preflight6_retirement = validate_preflight6_retirement()
    preflight7_retirement = validate_preflight7_retirement()
    preflight8_retirement = validate_preflight8_retirement()
    preflight9_retirement = validate_preflight9_retirement()
    p14 = load_module("p16_pinned_p14", P14_PATH)
    p14._version_gate()
    pinned_local_sources = p14._verify_pinned_local_sources()
    p10 = p14._import_p10()
    asset = p14._asset_gate(p10)
    limits = parse_urdf_limits()
    urdf_chain = parse_urdf_kinematic_chain()
    candidates_path = CANDIDATES_PATH
    handoff = load_candidates(candidates_path.resolve(), args.candidates_sha256)
    p15_observability = validate_p15_observability(handoff)
    witness_source = validate_witness_source()
    dependency_paths = profile_decision_dependency_paths(
        profile, pinned_local_sources, asset
    )
    preflight_bindings: dict[str, Any] = {}

    if profile == CANONICAL_PROFILE:
        preflight_prefix = f"t3u_{PREFLIGHT_PROFILE}"
        preflight_paths = run_paths(preflight_prefix)
        preflight_external = {
            "stdout": CASE_DIR / f"{preflight_prefix}_stdout.log",
            "supervisor_launcher": CASE_DIR / f"{preflight_prefix}_supervisor_launcher.log",
            "supervisor_pid": CASE_DIR / f"{preflight_prefix}_supervisor_pid.txt",
            "physics_python_pid": CASE_DIR / f"{preflight_prefix}_physics_python_pid.txt",
            "render_python_pid": CASE_DIR / f"{preflight_prefix}_render_python_pid.txt",
            "pgid": CASE_DIR / f"{preflight_prefix}_pgid.txt",
            "supervisor_contract": CASE_DIR / f"{preflight_prefix}_supervisor_contract.json",
            "supervisor_outcome": CASE_DIR / f"{preflight_prefix}_supervisor_outcome.json",
            "gpu_before": CASE_DIR / f"{preflight_prefix}_nvidia_smi_before.csv",
            "gpu_supervisor_end": CASE_DIR / f"{preflight_prefix}_nvidia_smi_supervisor_end.csv",
            "gpu_after": CASE_DIR / f"{preflight_prefix}_nvidia_smi_after.csv",
            "supervisor_failure": CASE_DIR / f"{preflight_prefix}_supervisor_failure.json",
        }
        preflight_required_core = {
            name: preflight_paths[name]
            for name in (
                "results.json", "plan.json", "trace.npz", "timeline.rrd",
                "timeline.rbl", "rerun_validation.json", "decision_snapshot.png",
                "inspection.png", "rgb_frames_manifest.json", "side_grasp.mp4",
                "script.py.txt", "argv.txt", "phase.jsonl", "render_phase.jsonl",
                "preclose_sentinel.json", "exit_status.txt",
                "terminal_attestation.json", "manual_visual_inspection.json",
            )
        }
        preflight_required_external = {
            name: path for name, path in preflight_external.items()
            if name != "supervisor_failure"
        }
        preflight_all_required = {
            **preflight_required_core, **preflight_required_external
        }
        if not all(path.is_file() for path in preflight_all_required.values()):
            raise RuntimeError("CANONICAL_BLOCKED_ON_PREFLIGHT_ARTIFACTS")
        if (
            preflight_paths["failure.json"].exists()
            or preflight_paths["render_failure.json"].exists()
            or preflight_external["supervisor_failure"].exists()
        ):
            raise RuntimeError("CANONICAL_BLOCKED_ON_PREFLIGHT_FAILURE_MARKER")

        preflight_doc = json.loads(preflight_paths["results.json"].read_text())
        preflight_plan_doc = json.loads(preflight_paths["plan.json"].read_text())
        preflight_render_doc = json.loads(
            preflight_paths["rgb_frames_manifest.json"].read_text()
        )
        preflight_attest_doc = json.loads(
            preflight_paths["terminal_attestation.json"].read_text()
        )
        preflight_visual_doc = json.loads(
            preflight_paths["manual_visual_inspection.json"].read_text()
        )
        preflight_outcome_doc = json.loads(
            preflight_external["supervisor_outcome"].read_text()
        )
        preflight_contract_doc = json.loads(
            preflight_external["supervisor_contract"].read_text()
        )
        if not all(
            isinstance(document, dict)
            for document in (
                preflight_doc,
                preflight_plan_doc,
                preflight_render_doc,
                preflight_attest_doc,
                preflight_visual_doc,
                preflight_outcome_doc,
                preflight_contract_doc,
            )
        ):
            raise RuntimeError("CANONICAL_BLOCKED_ON_PREFLIGHT_JSON_ROOT_SHAPE")
        # Re-run the same raw terminal verifier used to create the historical
        # attestation.  Exact equality prevents canonical promotion from
        # trusting forged all-true maps or a declarative supervisor outcome.
        recomputed_terminal_attestation = terminal_attest_mode(
            PREFLIGHT_PROFILE, verify_only=True
        )
        if (
            not isinstance(recomputed_terminal_attestation, dict)
            or not _json_type_value_exact(
                preflight_attest_doc, recomputed_terminal_attestation
            )
            or recomputed_terminal_attestation.get("pass") is not True
        ):
            raise RuntimeError(
                "CANONICAL_BLOCKED_ON_PREFLIGHT_RAW_TERMINAL_RECOMPUTE_FAIL"
            )
        recomputed_result_checks = validate_result_semantics(
            PREFLIGHT_PROFILE, preflight_paths, preflight_doc, preflight_plan_doc
        )
        recomputed_render_checks = validate_render_manifest_semantics(
            PREFLIGHT_PROFILE, preflight_paths, preflight_render_doc,
            preflight_doc, preflight_plan_doc,
        )
        expected_attestation_keys = {
            "artifact", "profile", "argv", "artifact_checks", "lifecycle_checks",
            "supervisor_outcome_checks", "result_semantic_checks",
            "render_semantic_checks", "processes", "gpu", "dependency_pins",
            "bindings", "manual_visual_inspection_still_required",
            "scientific_or_preflight_promotion", "pass", "verdict",
        }
        expected_artifact_check_keys = {
            "results_bound_to_sentinel", "source_and_frozen_copy_exact",
            "physics_argv_exact", "plan_representative_binding_exact",
            "preclose_artifact_hashes_exact", "rerun_validation_pass",
            "results_rerun_technical_pass", "results_runtime_instrumentation_pass",
            "render_manifest_pass", "render_argv_exact", "render_source_bindings_exact",
            "render_trace_hash_exact", "render_mp4_hash_exact",
            "render_zero_physics_steps", "render_dependency_physics_finalize_start_end_exact",
            "render_all_frame_state_fidelity_exact", "render_full_decode_pass",
        }
        expected_lifecycle_check_keys = {
            "supervisor_contract_exact", "combined_exit_status_zero",
            "failure_marker_absent", "render_failure_marker_absent",
            "supervisor_failure_marker_absent",
            "supervisor_reaped", "physics_python_reaped", "render_python_reaped",
            "process_group_empty", "physics_pid_not_on_gpu", "render_pid_not_on_gpu",
            "fresh_gpu_pid_delta_empty", "stdout_has_no_failure_tokens",
            "supervisor_launcher_has_no_failure_tokens", "stdout_one_physics_preclose",
            "stdout_one_render_completion", "phase_order_complete",
            "phase_result_hash_binding", "phase_sentinel_hash_binding",
            "phase_render_hash_binding",
        }
        expected_outcome_check_keys = {
            "top_level_schema_and_exact_keys", "supervisor_argv_exact",
            "source_and_candidate_pins_exact", "contract_exact_and_bound",
            "supervisor_identity_exact", "attempt_counts_exact_no_retry",
            "physics_raw_wait_success_exact",
            "physics_preclose_semantic_gate_recomputed_exact",
            "render_raw_wait_success_exact",
            "render_posthoc_semantic_gate_recomputed_exact",
            "render_iff_physics_success_exact", "combined_exit_exact",
            "child_process_groups_empty", "gpu_before_to_supervisor_end_no_fresh_pid",
            "outcome_file_bindings_exact", "outcome_pass_recomputed_not_trusted",
            "timestamps_monotonic",
        }
        expected_dependency_pins = {
            "p16_source_sha256": source_sha,
            "supervisor_source_sha256": SUPERVISOR_SHA256,
            "preflight_prereg_sha256": PREFLIGHT_PREREG_SHA256,
            "canonical_prereg_sha256": CANONICAL_PREREG_SHA256,
            "p15_source_sha256": P15_SHA256,
            "p15_prereg_sha256": P15_PREREG_SHA256,
            "p15_candidates_sha256": P15_CANDIDATES_SHA256,
            "preflight2_terminal_attestation_sha256": (
                PREFLIGHT2_TERMINAL_ATTESTATION_SHA256
            ),
            "preflight2_failure_sha256": PREFLIGHT2_FAILURE_SHA256,
            "preflight2_supervisor_outcome_sha256": PREFLIGHT2_OUTCOME_SHA256,
            "preflight2_phase_sha256": PREFLIGHT2_PHASE_SHA256,
            "preflight2_exit_status_sha256": PREFLIGHT2_EXIT_STATUS_SHA256,
            "retired_canonical_prereg_sha256": RETIRED_CANONICAL_PREREG_SHA256,
            "preflight3_source_sha256": PREFLIGHT3_SOURCE_SHA256,
            "preflight3_supervisor_sha256": PREFLIGHT3_SUPERVISOR_SHA256,
            "preflight3_prereg_sha256": PREFLIGHT3_PREREG_SHA256,
            "preflight3_canonical_prereg_sha256": (
                PREFLIGHT3_CANONICAL_PREREG_SHA256
            ),
            "preflight3_zero_launcher_sha256": PREFLIGHT3_LAUNCHER_SHA256,
            "preflight3_posthoc_audit_failure_sha256": (
                PREFLIGHT3_POSTHOC_AUDIT_FAILURE_SHA256
            ),
            "preflight4_source_sha256": PREFLIGHT4_SOURCE_SHA256,
            "preflight4_supervisor_sha256": PREFLIGHT4_SUPERVISOR_SHA256,
            "preflight4_prereg_sha256": PREFLIGHT4_PREREG_SHA256,
            "preflight4_canonical_prereg_sha256": (
                PREFLIGHT4_CANONICAL_PREREG_SHA256
            ),
            "preflight4_failure_sha256": PREFLIGHT4_FAILURE_SHA256,
            "preflight4_launcher_sha256": PREFLIGHT4_LAUNCHER_SHA256,
            **PREFLIGHT5_DEPENDENCY_PINS,
            **PREFLIGHT6_DEPENDENCY_PINS,
            **PREFLIGHT7_DEPENDENCY_PINS,
            **PREFLIGHT8_DEPENDENCY_PINS,
            **PREFLIGHT9_DEPENDENCY_PINS,
        }
        expected_attestation_binding_paths = {
            **{
                name: preflight_paths[name]
                for name in (
                    "results.json", "plan.json", "trace.npz", "timeline.rrd",
                    "timeline.rbl", "rerun_validation.json", "decision_snapshot.png",
                    "inspection.png", "rgb_frames_manifest.json", "side_grasp.mp4",
                    "script.py.txt", "argv.txt", "phase.jsonl", "render_phase.jsonl",
                    "preclose_sentinel.json", "exit_status.txt",
                )
            },
            **{
                name: preflight_external[name]
                for name in (
                    "stdout", "supervisor_launcher", "supervisor_pid",
                    "physics_python_pid", "render_python_pid", "pgid",
                    "supervisor_contract", "supervisor_outcome", "gpu_before",
                    "gpu_supervisor_end", "gpu_after",
                )
            },
        }
        attestation_bindings = preflight_attest_doc.get("bindings", {})
        attestation_bindings_exact = bool(
            isinstance(attestation_bindings, dict)
            and set(attestation_bindings) == set(expected_attestation_binding_paths)
            and all(
                isinstance(attestation_bindings[name], dict)
                and set(attestation_bindings[name]) == {"path", "sha256"}
                and attestation_bindings[name]["path"]
                == str(path.relative_to(REPO))
                and attestation_bindings[name]["sha256"] == sha256_file(path)
                for name, path in expected_attestation_binding_paths.items()
            )
        )
        terminal_semantics_exact = bool(
            set(preflight_attest_doc) == expected_attestation_keys
            and preflight_attest_doc.get("artifact")
            == "T3U_EXTERNAL_TERMINAL_ATTESTATION_V4"
            and preflight_attest_doc.get("profile") == PREFLIGHT_PROFILE
            and preflight_attest_doc.get("argv")
            == [str(Path(__file__).resolve()), "--terminal_attest", PREFLIGHT_PROFILE]
            and _strict_all_true_bool_map(
                preflight_attest_doc.get("artifact_checks"),
                expected_artifact_check_keys,
            )
            and _strict_all_true_bool_map(
                preflight_attest_doc.get("lifecycle_checks"),
                expected_lifecycle_check_keys,
            )
            and _strict_all_true_bool_map(
                preflight_attest_doc.get("supervisor_outcome_checks"),
                expected_outcome_check_keys,
            )
            and _json_type_value_exact(
                preflight_attest_doc.get("result_semantic_checks"),
                recomputed_result_checks,
            )
            and _strict_all_true_bool_map(
                recomputed_result_checks, set(RESULT_SEMANTIC_CHECK_KEYS)
            )
            and _json_type_value_exact(
                preflight_attest_doc.get("render_semantic_checks"),
                recomputed_render_checks,
            )
            and _strict_all_true_bool_map(
                recomputed_render_checks, set(recomputed_render_checks)
            )
            and _json_type_value_exact(
                preflight_attest_doc.get("dependency_pins"),
                expected_dependency_pins,
            )
            and attestation_bindings_exact
            and preflight_attest_doc.get("manual_visual_inspection_still_required") is True
            and preflight_attest_doc.get("scientific_or_preflight_promotion") is False
            and preflight_attest_doc.get("pass") is True
            and preflight_attest_doc.get("verdict")
            == "TERMINAL_ATTESTED_PENDING_MANUAL_VISUAL"
        )
        if not terminal_semantics_exact:
            raise RuntimeError("CANONICAL_BLOCKED_ON_PREFLIGHT_TERMINAL_SEMANTIC_FAIL")

        expected_supervisor_contract = {
            "artifact": "T3U_DETACHED_PHYSICS_THEN_RENDER_SUPERVISOR_V12",
            "automatic_retry_count": 0,
            "detached": True,
            "physics_timeout_seconds": 7200,
            "render_timeout_seconds": 7200,
            "term_signal": "TERM",
            "kill_after_seconds": 20,
            "physics_then_render_only_on_raw_zero_and_preclose_semantic_gate": True,
            "physics_semantic_gate_artifact": "T3U_PHYSICS_PRECLOSE_SEMANTIC_GATE_V1",
            "render_success_requires_raw_zero_and_posthoc_semantic_gate": True,
            "render_semantic_gate_artifact": "T3U_RENDER_POSTHOC_SEMANTIC_GATE_V1",
            "semantic_gate_failure_exit_status": 125,
            "raw_waitpid_status_authority": True,
            "bounded_waitpid_only": True,
            "supervisor_signal_cleanup": (
                "SIGTERM_SIGINT__active_child_pgid_TERM_20s_then_KILL_20s"
            ),
            "child_parent_death_signal": "SIGTERM",
            "child_preexec_signal_state": (
                "SIGTERM_SIGINT_SIGHUP_SIG_DFL__empty_mask__expected_parent_pid_recheck"
            ),
            "host_launch_boundary": (
                "require_escalated_exec_command__outside_bwrap_die_with_parent"
            ),
            "forbidden_sandbox_ancestor_gate": True,
        }
        preflight_host_launch_context = preflight_contract_doc.get(
            "host_launch_context"
        )
        expected_supervisor_contract["host_launch_context"] = (
            preflight_host_launch_context
        )
        physics_outcome = preflight_outcome_doc.get("physics", {})
        render_outcome = preflight_outcome_doc.get("render", {})
        supervisor_pids = {
            name: _read_pid_file_or_invalid(preflight_external[name])
            for name in (
                "supervisor_pid", "physics_python_pid", "render_python_pid", "pgid"
            )
        }
        preflight_host_launch_context_exact = _validate_host_launch_context(
            preflight_host_launch_context, supervisor_pids["supervisor_pid"]
        )
        expected_preflight_physics_command = [
            ISAAC_PYTHON,
            str(Path(__file__).resolve()),
            "--run_label",
            PREFLIGHT_PROFILE,
            "--candidates_sha256",
            P15_CANDIDATES_SHA256,
        ]
        expected_preflight_render_command = [
            ISAAC_PYTHON,
            str(Path(__file__).resolve()),
            "--render_trace",
            PREFLIGHT_PROFILE,
        ]
        expected_preflight_outcome_keys = {
            "artifact", "profile", "argv", "supervisor_source_sha256",
            "p16_source_sha256", "candidates_sha256", "start_time_unix",
            "end_time_unix", "elapsed_seconds", "supervisor", "attempts",
            "physics", "physics_artifact_gate", "render", "render_artifact_gate",
            "render_started_iff_physics_success", "combined_exit_status", "gpu",
            "bindings", "contract", "host_launch_context", "pass",
        }
        preflight_before_gpu = _gpu_pid_set(
            preflight_external["gpu_before"].read_text()
        )
        preflight_supervisor_end_gpu = _gpu_pid_set(
            preflight_external["gpu_supervisor_end"].read_text()
        )
        outcome_static_recomputed = bool(
            set(preflight_outcome_doc) == expected_preflight_outcome_keys
            and _json_type_value_exact(
                preflight_contract_doc, expected_supervisor_contract
            )
            and preflight_outcome_doc.get("artifact")
            == "T3U_DETACHED_SUPERVISOR_OUTCOME_V12"
            and preflight_outcome_doc.get("profile") == PREFLIGHT_PROFILE
            and preflight_outcome_doc.get("argv")
            == [
                str(SUPERVISOR_PATH), "--profile", PREFLIGHT_PROFILE,
                "--candidates_sha256", P15_CANDIDATES_SHA256,
            ]
            and preflight_outcome_doc.get("supervisor_source_sha256")
            == SUPERVISOR_SHA256
            and preflight_outcome_doc.get("p16_source_sha256") == source_sha
            and preflight_outcome_doc.get("candidates_sha256") == P15_CANDIDATES_SHA256
            and _json_type_value_exact(
                preflight_outcome_doc.get("contract"), expected_supervisor_contract
            )
            and preflight_host_launch_context_exact
            and _json_type_value_exact(
                preflight_outcome_doc.get("host_launch_context"),
                preflight_host_launch_context,
            )
            and _strict_supervisor_identity(
                preflight_outcome_doc.get("supervisor"),
                pid=supervisor_pids["supervisor_pid"],
                pgid=supervisor_pids["pgid"],
            )
            and _strict_attempts(
                preflight_outcome_doc.get("attempts"), render_count=1
            )
            and isinstance(
                preflight_outcome_doc.get("physics_artifact_gate"), dict
            )
            and preflight_outcome_doc["physics_artifact_gate"].get("pass") is True
            and isinstance(
                preflight_outcome_doc.get("render_artifact_gate"), dict
            )
            and preflight_outcome_doc["render_artifact_gate"].get("pass") is True
            and _strict_child_lifecycle(
                physics_outcome,
                label="physics",
                command=expected_preflight_physics_command,
                pid=supervisor_pids["physics_python_pid"],
                supervisor_sid=supervisor_pids["supervisor_pid"],
                require_success=True,
            )
            and _strict_child_lifecycle(
                render_outcome,
                label="render",
                command=expected_preflight_render_command,
                pid=supervisor_pids["render_python_pid"],
                supervisor_sid=supervisor_pids["supervisor_pid"],
                require_success=True,
            )
            and _strict_outcome_times(
                preflight_outcome_doc, [physics_outcome, render_outcome]
            )
            and _strict_json_int(
                preflight_outcome_doc.get("combined_exit_status")
            )
            and preflight_outcome_doc.get("combined_exit_status") == 0
            and preflight_outcome_doc.get("render_started_iff_physics_success") is True
            and preflight_outcome_doc.get("pass") is True
            and _strict_gpu_summary(
                preflight_outcome_doc.get("gpu"),
                before=preflight_before_gpu,
                supervisor_end=preflight_supervisor_end_gpu,
            )
            and preflight_paths["exit_status.txt"].read_text().strip() == "0"
            and all(
                not Path(f"/proc/{pid}").exists()
                for pid in (
                    supervisor_pids["supervisor_pid"],
                    supervisor_pids["physics_python_pid"],
                    supervisor_pids["render_python_pid"],
                )
            )
            and _linux_pgid_members(supervisor_pids["pgid"]) == []
        )
        if not outcome_static_recomputed:
            raise RuntimeError("CANONICAL_BLOCKED_ON_PREFLIGHT_SUPERVISOR_OUTCOME_FAIL")

        expected_visual_checks = {
            "target_actual_frames_visible_in_decision_snapshot": True,
            "rrd_inspection_opened_and_axes_consistent": True,
            "mp4_opened_and_full_trajectory_visible": True,
            "jaw_object_and_support_relationships_visually_checked": True,
        }
        if not (
            set(preflight_visual_doc)
            == {
                "artifact", "profile", "results_sha256",
                "terminal_attestation_sha256", "rerun_validation_sha256",
                "inspection_png_sha256", "decision_snapshot_png_sha256",
                "rgb_manifest_sha256", "mp4_sha256", "visual_checks",
                "observations", "pass",
            }
            and preflight_visual_doc.get("artifact")
            == "T3U_SIDE_PREFLIGHT_MANUAL_VISUAL_INSPECTION_V1"
            and preflight_visual_doc.get("profile") == PREFLIGHT_PROFILE
            and preflight_visual_doc.get("pass") is True
            and preflight_visual_doc.get("results_sha256")
            == sha256_file(preflight_paths["results.json"])
            and preflight_visual_doc.get("terminal_attestation_sha256")
            == sha256_file(preflight_paths["terminal_attestation.json"])
            and preflight_visual_doc.get("rerun_validation_sha256")
            == sha256_file(preflight_paths["rerun_validation.json"])
            and preflight_visual_doc.get("inspection_png_sha256")
            == sha256_file(preflight_paths["inspection.png"])
            and preflight_visual_doc.get("decision_snapshot_png_sha256")
            == sha256_file(preflight_paths["decision_snapshot.png"])
            and preflight_visual_doc.get("rgb_manifest_sha256")
            == sha256_file(preflight_paths["rgb_frames_manifest.json"])
            and preflight_visual_doc.get("mp4_sha256")
            == sha256_file(preflight_paths["side_grasp.mp4"])
            and _json_type_value_exact(
                preflight_visual_doc.get("visual_checks"), expected_visual_checks
            )
            and isinstance(preflight_visual_doc.get("observations"), list)
            and preflight_visual_doc.get("observations")
            and all(
                isinstance(item, str) and item.strip()
                for item in preflight_visual_doc["observations"]
            )
        ):
            raise RuntimeError("CANONICAL_BLOCKED_ON_PREFLIGHT_VISUAL_FAIL")
        preflight_measurement = preflight_doc.get("metrics", {}).get("measurement_valid", [])
        preflight_provenance = preflight_doc.get("provenance", {})
        preflight_controls = preflight_doc.get("fixed_controls", {})
        if not (
            preflight_doc.get("profile") == PREFLIGHT_PROFILE
            and preflight_doc.get("scientific_authoritative") is False
            and preflight_doc.get("internal_verdict")
            == "INSTRUMENTATION_PREFLIGHT_PASS_PENDING_RENDER_TERMINAL_AND_MANUAL_VISUAL"
            and preflight_provenance.get("source_sha256") == source_sha
            and preflight_provenance.get("p15_sha256") == handoff["sha256"]
            and preflight_provenance.get("prereg_sha256") == PREFLIGHT_PREREG_SHA256
            and preflight_controls.get("q5_open_deg") == Q5_OPEN_DEG
            and preflight_controls.get("q5_close_command_deg") == Q5_CLOSE_COMMAND_DEG
            and preflight_controls.get("phase_steps") == PHASE_STEPS
            and preflight_controls.get("self_collision_setting_authority")
            == "stage_collision_audit.self_collision_readback"
            and preflight_controls.get(
                "self_collision_setting_not_physical_contact_proof"
            ) is True
            and preflight_controls.get("self_collision_behavioral_control_scope")
            == "positive_two_pairs_then_negative_HOME__not_all_pose_proof"
            and preflight_doc.get("plan_counts", {}).get("planned") == 5
            and preflight_doc.get("plan_counts", {}).get("feasible") == 5
            and preflight_doc.get("representative_binding", {}).get("candidate_index")
            == PREFLIGHT_CANDIDATE_INDEX
            and preflight_doc.get("representative_binding", {}).get("pinch_offset_index") == 0
            and preflight_doc.get("runtime_instrumentation_pass") is True
            and preflight_measurement
            and all(bool(value) for value in preflight_measurement)
            and preflight_doc.get("instrumentation", {})
            .get("instrumentation_witness", {})
            .get("pass") is True
            and preflight_doc.get("rerun", {}).get("technical_pass") is True
        ):
            raise RuntimeError("CANONICAL_BLOCKED_ON_PREFLIGHT_INSTRUMENTATION_FAIL")
        preflight_bindings = {
            name: sha256_file(path)
            for name, path in preflight_all_required.items()
        }
        expected_dependency_preflight = _passing_preflight_dependency_paths()
        if expected_dependency_preflight != preflight_all_required:
            raise RuntimeError("CANONICAL_PREFLIGHT_DEPENDENCY_SET_DRIFT")

    dependency_hashes_at_start = {
        name: sha256_file(path) for name, path in dependency_paths.items()
    }

    write_bytes_x(paths["script.py.txt"], source_start)
    write_bytes_x(paths["argv.txt"], ("\n".join(sys.argv) + "\n").encode("utf-8"))
    append_phase(paths["phase.jsonl"], "run_claim", profile=profile,
                 source_sha256=source_sha, prereg_sha256=prereg_expected,
                 candidates_sha256=handoff["sha256"])
    start = time.time()
    os.environ["ROARM_M3_USD_PATH"] = str(p10.ATTEMPT3_USD)
    simulation_app = None
    env = None
    preclose_sentinel_sha: str | None = None
    try:
        from isaaclab.app import AppLauncher

        launcher = AppLauncher(headless=True, enable_cameras=False)
        simulation_app = launcher.app
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA_UNAVAILABLE")
        p14._resolve_attempt3_composition_manifest(asset)
        jaw = p14._import_jaw_extractor_after_kit()
        jaw_asset = jaw.extract_asset()
        calibration = derive_pinch_calibration(jaw, jaw_asset)
        plan = build_plan(p10, handoff, calibration, limits, profile)
        plan["retired_preflight2"] = preflight2_retirement
        plan["retired_preflight3_launch"] = preflight3_retirement
        plan["retired_preflight4_launch"] = preflight4_retirement
        plan["retired_preflight5_dynamic_control_abort"] = preflight5_retirement
        plan["retired_preflight6_post_activation_dynamic_control_abort"] = (
            preflight6_retirement
        )
        plan["retired_preflight7_filter_representation_abort"] = (
            preflight7_retirement
        )
        plan["retired_preflight8_reward_buffer_lifecycle_abort"] = (
            preflight8_retirement
        )
        plan["retired_preflight9_clock_accounting_abort"] = (
            preflight9_retirement
        )
        plan["instrumentation_witness"] = {
            **witness_source,
            "source_trial_id": WITNESS_SOURCE_TRIAL_ID,
            "source_results_path": (
                "claudedocs/runtime_logs/grasp_track/g0b_d420/t3y_workspace1_results.json"
            ),
            "source_plan_path": (
                "claudedocs/runtime_logs/grasp_track/g0b_d420/t3y_workspace1_plan.json"
            ),
            "purpose": "moving-gripper-link/support nonzero contact positive control",
            "excluded_from_scientific_counts": True,
            "q_approach_deg": WITNESS_Q_APPROACH_DEG.tolist(),
            "q_descend_deg": WITNESS_Q_DESCEND_DEG.tolist(),
            "q_close_deg": WITNESS_Q_CLOSE_DEG.tolist(),
            "q_lift_deg": WITNESS_Q_LIFT_DEG.tolist(),
        }
        initial_feasible = [row for row in plan["trials"] if row["feasible"]]
        expected_active = 5 if profile == PREFLIGHT_PROFILE else 10
        if len(initial_feasible) != expected_active:
            # No PhysX result may be claimed if the preregistered candidate set
            # does not reach the fixed base under parsed limits.
            dependency_hashes_at_finalize = {
                name: sha256_file(path) for name, path in dependency_paths.items()
            }
            if (
                dependency_hashes_at_finalize != dependency_hashes_at_start
                or Path(__file__).read_bytes() != source_start
                or p14._asset_gate(p10) != asset
            ):
                raise RuntimeError("NO_IK_DECISION_DEPENDENCY_CHANGED")
            write_json_x(paths["plan.json"], plan)
            output = {
                "tool": "p16_g0b_t3u_cyld29h50_side_midpoint_parallel_physics_v9",
                "profile": profile,
                "scientific_authoritative": profile == CANONICAL_PROFILE,
                "verdict": "NO_IK_FEASIBLE_SDG_SIDE_CANDIDATE",
                "plan_counts": {"planned": plan["n_planned"], "feasible": len(initial_feasible),
                                "expected_feasible": expected_active},
                "physics_executed": False,
                "provenance": {"source_sha256": source_sha, "p15_sha256": handoff["sha256"],
                               "prereg_sha256": prereg_expected, "urdf_sha256": URDF_SHA256,
                               "dependency_hashes_at_start": dependency_hashes_at_start,
                               "dependency_hashes_at_finalize": dependency_hashes_at_finalize,
                               "dependency_hashes_equal": True},
            }
            write_json_x(paths["results.json"], output)
            append_phase(paths["phase.jsonl"], "ik_terminal_no_physics", verdict=output["verdict"])
            return 2

        env = make_env(args, p10)
        if not str(env.device).startswith("cuda"):
            raise RuntimeError(f"GPU_PHYSX_REQUIRED device={env.device}")
        joint_limits_readback = audit_effective_joint_limits(
            env, limits, args.num_envs
        )
        plan["effective_joint_limits_readback"] = joint_limits_readback
        fixed_base_readback = audit_fixed_base_contract(env, args.num_envs)
        plan["fixed_base_runtime_readback"] = fixed_base_readback
        object_stage_readback = audit_object_stage_readback(env, args.num_envs)
        plan["object_stage_readback"] = object_stage_readback
        collision_pass, collision_checks = p10._audit_collision_bodies(env)
        exact_64 = all(
            collision_checks.get(body, {}).get("enabled_total") == 64
            and collision_checks[body].get("enabled_part_count") == 64
            and collision_checks[body].get("disabled_legacy_exact_one") is True
            for body in JAW_BODIES
        )
        if not collision_pass or not exact_64:
            raise RuntimeError(f"ATTEMPT3_64_PLUS_64_RUNTIME_FAIL {collision_checks}")
        collision_vertices, collision_geometry = extract_enabled_collision_vertices(
            env.scene.stage
        )
        static_fk_alignment = audit_static_fk_alignment(
            env.scene.stage, env, collision_geometry, p10, urdf_chain
        )
        plan = apply_planned_clearance_gate(
            plan, collision_vertices, collision_geometry, urdf_chain
        )
        plan["static_fk_alignment"] = static_fk_alignment
        static_trial_set_contract = validate_trial_set_contract(
            plan, profile, "post_static_clearance", hard_fail=False
        )
        plan["trial_set_contracts"]["post_static_clearance"] = static_trial_set_contract
        feasible = [row for row in plan["trials"] if row["feasible"]]
        if not static_trial_set_contract["pass"]:
            dependency_hashes_at_finalize = {
                name: sha256_file(path) for name, path in dependency_paths.items()
            }
            if (
                dependency_hashes_at_finalize != dependency_hashes_at_start
                or Path(__file__).read_bytes() != source_start
                or p14._asset_gate(p10) != asset
            ):
                raise RuntimeError("NO_STATIC_CLEARANCE_DECISION_DEPENDENCY_CHANGED")
            write_json_x(paths["plan.json"], plan)
            output = {
                "tool": "p16_g0b_t3u_cyld29h50_side_midpoint_parallel_physics_v9",
                "profile": profile,
                "scientific_authoritative": profile == CANONICAL_PROFILE,
                "verdict": "NO_PLANNED_STATIC_CLEARANCE_FEASIBLE_SIDE_PATH",
                "plan_counts": {
                    "planned": plan["n_planned"],
                    "ik_frame_feasible": len(initial_feasible),
                    "static_clearance_feasible": len(feasible),
                    "expected_preflight_active": expected_active,
                    "exact_trial_set_contract": static_trial_set_contract,
                },
                "physics_executed": False,
                "provenance": {
                    "source_sha256": source_sha,
                    "p15_sha256": handoff["sha256"],
                    "prereg_sha256": prereg_expected,
                    "urdf_sha256": URDF_SHA256,
                    "dependency_hashes_at_start": dependency_hashes_at_start,
                    "dependency_hashes_at_finalize": dependency_hashes_at_finalize,
                    "dependency_hashes_equal": True,
                },
            }
            write_json_x(paths["results.json"], output)
            append_phase(
                paths["phase.jsonl"], "static_clearance_terminal_no_physics",
                verdict=output["verdict"],
            )
            return 2
        reporter_audit = audit_cloned_reporters(env.scene.stage, args.num_envs)
        self_collision_audit = audit_self_collision_readback(
            env, args.num_envs, p10, collision_geometry, urdf_chain, limits, args
        )
        plan["effective_self_collision_readback"] = self_collision_audit
        representative_rows = [
            row for row in feasible
            if row["candidate_index"] == PREFLIGHT_CANDIDATE_INDEX
            and row["pinch_offset_index"] == 0
        ]
        if len(representative_rows) != 1:
            raise RuntimeError(
                "REPRESENTATIVE_NOMINAL_ROW_NOT_UNIQUELY_FEASIBLE "
                f"count={len(representative_rows)}"
            )
        representative = representative_rows[0]
        representative_slot = feasible.index(representative)
        plan["representative_binding"] = {
            "selected_before_physics": True,
            "trial_id": representative["trial_id"],
            "candidate_id": representative["candidate_id"],
            "candidate_index": PREFLIGHT_CANDIDATE_INDEX,
            "pinch_offset_index": 0,
            "environment_slot": representative_slot,
        }
        write_json_x(paths["plan.json"], plan)
        physics = run_physics(
            args, env, feasible, joint_limits_readback, fixed_base_readback,
            self_collision_audit["behavioral_control"],
            self_collision_audit["precontrol_self_contact_filter_identity"],
        )
        physics["object_stage_readback"] = object_stage_readback
        physics["self_collision_readback"] = self_collision_audit
        trace = physics.pop("trace")
        trace["trial_id"] = np.asarray([row["trial_id"] for row in feasible], dtype="U32")
        trace["representative_environment_slot"] = np.asarray(
            representative_slot, dtype=np.int32
        )
        np.savez_compressed(paths["trace.npz"], **trace)
        decision_snapshot = emit_decision_snapshot(
            paths["decision_snapshot.png"], representative, representative_slot,
            trace, profile,
        )
        rerun = emit_rerun(
            paths, representative, representative_slot, trace,
            collision_vertices, profile,
        )
        dependency_hashes_at_finalize = {
            name: sha256_file(path) for name, path in dependency_paths.items()
        }
        if dependency_hashes_at_finalize != dependency_hashes_at_start:
            raise RuntimeError(
                "DEPENDENCY_CHANGED_DURING_RUN "
                f"start={dependency_hashes_at_start} end={dependency_hashes_at_finalize}"
            )
        if p14._asset_gate(p10) != asset:
            raise RuntimeError("ATTEMPT3_ASSET_MANIFEST_CHANGED_DURING_RUN")
        verdict, labels, classification_summary = classify(physics["metrics"])
        if physics["kinematic_attach_calls"] != 0:
            raise RuntimeError(
                f"KINEMATIC_ATTACH_CALL_GATE_FAIL {physics['kinematic_attach_calls']}"
            )
        runtime_instrumentation_pass = bool(
            np.asarray(physics["metrics"]["measurement_valid"], dtype=bool).all()
            and physics["contact_buffers_ok"] is True
            and physics["instrumentation_witness"]["pass"] is True
            and physics["numeric_integrity"]["pass"] is True
            and joint_limits_readback["pass"] is True
            and fixed_base_readback["pass"] is True
            and self_collision_audit["pass"] is True
            and physics["self_contact_filter_identity_reuse"]["pass"] is True
            and physics["post_diagnostic_task_rebaseline"]["pass"] is True
            and physics["first_task_step_freshness"]["pass"] is True
            and physics["physics_clock_accounting"]["pass"] is True
        )
        if not rerun["technical_pass"]:
            verdict = "MEASUREMENT_INVALID"
        if profile == PREFLIGHT_PROFILE:
            scientific_verdict = None
            internal_verdict = (
                "INSTRUMENTATION_PREFLIGHT_PASS_PENDING_RENDER_TERMINAL_AND_MANUAL_VISUAL"
                if rerun["technical_pass"] and runtime_instrumentation_pass
                else "INSTRUMENTATION_PREFLIGHT_RUNTIME_OR_RERUN_FAIL"
            )
        else:
            scientific_verdict = verdict
            internal_verdict = (
                "SCIENTIFIC_PRECLOSE_READY_PENDING_RENDER_TERMINAL_AND_MANUAL_VISUAL"
                if rerun["technical_pass"] and runtime_instrumentation_pass
                else "SCIENTIFIC_PRECLOSE_MEASUREMENT_INVALID"
            )
        output = {
            "tool": "p16_g0b_t3u_cyld29h50_side_midpoint_parallel_physics_v9",
            "case": "g0b_d420",
            "tag": prefix,
            "profile": profile,
            "scientific_authoritative": profile == CANONICAL_PROFILE,
            "scientific_verdict_preclose_candidate": scientific_verdict,
            "internal_verdict": internal_verdict,
            "object": {"authority": "composed_stage_schema_readback",
                       "stage_readback": object_stage_readback,
                       "initial_center_control_m": OBJECT_CENTER_M.tolist()},
            "fixed_controls": {"object_material_authority": "object.stage_readback",
                               "effective_pair_friction": object_stage_readback[
                                   "effective_pair_friction"
                               ],
                               "q5_open_deg": Q5_OPEN_DEG,
                               "q5_close_command_deg": Q5_CLOSE_COMMAND_DEG,
                               "phase_steps": PHASE_STEPS,
                               "kinematic_attach_disabled": True,
                               "self_collision_setting_authority": (
                                   "stage_collision_audit.self_collision_readback"
                               ),
                               "self_collision_setting_not_physical_contact_proof": True,
                               "self_collision_behavioral_control_scope": (
                                   "positive_two_pairs_then_negative_HOME__not_all_pose_proof"
                               ),
                               "fixed_base_authority": (
                                   "stage_collision_audit.fixed_base_readback"
                               ),
                               "joint_limit_authority": (
                                   "stage_collision_audit.joint_limits_readback"
                               )},
            "gates": {"contact_n_lte": CONTACT_GATE_N,
                      "same_step_bilateral_n_strict_gt": JAW_LOAD_GATE_N,
                      "corrected_lift_mm_strict_gt": LIFT_GATE_MM,
                      "tilt_deg_strict_lt": TIP_GATE_DEG},
            "plan_counts": {"planned": plan["n_planned"], "feasible": len(feasible)},
            "representative_binding": {
                "selected_before_physics": True,
                "trial_id": representative["trial_id"],
                "candidate_id": representative["candidate_id"],
                "candidate_index": PREFLIGHT_CANDIDATE_INDEX,
                "pinch_offset_index": 0,
                "environment_slot": representative_slot,
            },
            "classifications": [{"trial_id": row["trial_id"], "label": labels[i]}
                                for i, row in enumerate(feasible)],
            "classification_summary": classification_summary,
            "metrics": {key: value.tolist() for key, value in physics["metrics"].items()},
            "instrumentation": {key: value for key, value in physics.items() if key != "metrics"},
            "runtime_instrumentation_pass": runtime_instrumentation_pass,
            "stage_collision_audit": {"pass": bool(
                                          reporter_audit["pass"] is True
                                          and self_collision_audit["pass"] is True
                                          and physics[
                                              "self_contact_filter_identity_reuse"
                                          ]["pass"] is True
                                          and fixed_base_readback["pass"] is True
                                          and joint_limits_readback["pass"] is True
                                          and physics[
                                              "post_diagnostic_task_rebaseline"
                                          ]["pass"] is True
                                          and physics[
                                              "first_task_step_freshness"
                                          ]["pass"] is True
                                          and physics[
                                              "physics_clock_accounting"
                                          ]["pass"] is True
                                      ), "exact_64_plus_64": True,
                                      "body_checks": collision_checks,
                                      "static_fk_alignment": static_fk_alignment,
                                      "cloned_object_and_moving_body_reporters": reporter_audit,
                                      "self_collision_readback": self_collision_audit,
                                      "self_contact_filter_identity_reuse": physics[
                                          "self_contact_filter_identity_reuse"
                                      ],
                                      "post_diagnostic_task_rebaseline": physics[
                                          "post_diagnostic_task_rebaseline"
                                      ],
                                      "first_task_step_freshness": physics[
                                          "first_task_step_freshness"
                                      ],
                                      "physics_clock_accounting": physics[
                                          "physics_clock_accounting"
                                      ],
                                      "fixed_base_readback": fixed_base_readback,
                                      "joint_limits_readback": joint_limits_readback},
            "static_collision_geometry": plan["static_collision_geometry"],
            "pinch_calibration": calibration,
            "instrumentation_witness_source": witness_source,
            "rerun": rerun,
            "decision_snapshot": decision_snapshot,
            "provenance": {"source_sha256": source_sha, "source_stable": Path(__file__).read_bytes() == source_start,
                           "p14_sha256": P14_SHA256, "p10_sha256": P10_SHA256,
                           "jaw_sha256": JAW_SHA256, "urdf_sha256": URDF_SHA256,
                           "p15_path": handoff["path"], "p15_sha256": handoff["sha256"],
                           "p15_observability": p15_observability,
                           "retired_preflight2": preflight2_retirement,
                           "retired_preflight3_launch": preflight3_retirement,
                           "retired_preflight4_launch": preflight4_retirement,
                           "retired_preflight5_dynamic_control_abort": (
                               preflight5_retirement
                           ),
                           "retired_preflight6_post_activation_dynamic_control_abort": (
                               preflight6_retirement
                           ),
                           "retired_preflight7_filter_representation_abort": (
                               preflight7_retirement
                           ),
                           "retired_preflight8_reward_buffer_lifecycle_abort": (
                               preflight8_retirement
                           ),
                           "retired_preflight9_clock_accounting_abort": (
                               preflight9_retirement
                           ),
                           "passing_preflight_bindings": preflight_bindings,
                           "prereg_path": str(prereg.relative_to(REPO)),
                           "prereg_sha256": prereg_expected,
                           "p14_pinned_local_sources": pinned_local_sources,
                           "dependency_hashes_at_start": dependency_hashes_at_start,
                           "dependency_hashes_at_finalize": dependency_hashes_at_finalize,
                           "dependency_hashes_equal": True,
                           "attempt3": asset},
            "artifact_hashes_preclose": {
                name: sha256_file(paths[name])
                for name in (
                    "plan.json", "trace.npz", "timeline.rrd", "timeline.rbl",
                    "rerun_validation.json", "decision_snapshot.png", "inspection.png",
                )
            },
            "required_postphysics_pending": [
                "rgb_frames_manifest.json", "side_grasp.mp4",
                "terminal_attestation.json", "manual_visual_inspection.json",
            ],
            "wall_seconds": time.time() - start,
            "scope_warning": (
                "Fixed analytic-cylinder pose, placeholder unmeasured material, parsed URDF limits; "
                "no hardware authority, no real-friction claim, no finite-desk claim, no Isaac Lab learning."
            ),
        }
        if not output["provenance"]["source_stable"]:
            raise RuntimeError("SOURCE_CHANGED_DURING_RUN")
        write_json_x(paths["results.json"], output)
        append_phase(paths["phase.jsonl"], "results_durable",
                     results_sha256=sha256_file(paths["results.json"]),
                     internal_verdict=internal_verdict)
        sentinel = {
            "tag": prefix,
            "results_sha256": sha256_file(paths["results.json"]),
            "source_sha256": source_sha,
            "prereg_sha256": prereg_expected,
            "p15_sha256": handoff["sha256"],
            "trace_sha256": sha256_file(paths["trace.npz"]),
            "rerun_validation_sha256": sha256_file(paths["rerun_validation.json"]),
            "terminal_completion": "PENDING_EXTERNAL_ATTESTATION",
            "observability_completion": "RRD_PNG_COMPLETE__MP4_AND_MANUAL_INSPECTION_PENDING",
        }
        write_json_x(paths["preclose_sentinel.json"], sentinel)
        preclose_sentinel_sha = sha256_file(paths["preclose_sentinel.json"])
        append_phase(paths["phase.jsonl"], "preclose_sentinel_durable",
                     sentinel_sha256=preclose_sentinel_sha)
        print(f"[{LOG}] PRECLOSE profile={profile} verdict={scientific_verdict} "
              f"success={int(np.asarray(physics['metrics']['success']).sum())}/{len(feasible)}", flush=True)
        return 0
    except BaseException as exc:
        failure = {
            "type": type(exc).__name__, "message": str(exc),
            "traceback": traceback.format_exc(), "profile": profile,
            "source_sha256": source_sha,
        }
        if not paths["failure.json"].exists():
            write_json_x(paths["failure.json"], failure)
        if paths["phase.jsonl"].exists():
            append_phase(paths["phase.jsonl"], "failure", type=type(exc).__name__, message=str(exc))
        raise
    finally:
        if env is not None:
            try:
                env.close()
            except BaseException as close_exc:
                if not paths["failure.json"].exists():
                    write_json_x(
                        paths["failure.json"],
                        {
                            "type": type(close_exc).__name__,
                            "message": str(close_exc),
                            "phase": "env_close",
                            "source_sha256": source_sha,
                        },
                    )
                if paths["phase.jsonl"].exists():
                    append_phase(
                        paths["phase.jsonl"], "env_close_failure",
                        type=type(close_exc).__name__, message=str(close_exc),
                    )
        if simulation_app is not None:
            if paths["phase.jsonl"].exists():
                append_phase(
                    paths["phase.jsonl"], "simulation_app_close_start",
                    sentinel_sha256=preclose_sentinel_sha,
                    failure_marker_exists=paths["failure.json"].exists(),
                )
            print(
                f"[{LOG}] SIMULATION_APP_CLOSE_START profile={profile} "
                f"sentinel_sha256={preclose_sentinel_sha}",
                flush=True,
            )
            try:
                simulation_app.close()
            except BaseException as close_exc:
                if not paths["failure.json"].exists():
                    write_json_x(
                        paths["failure.json"],
                        {
                            "type": type(close_exc).__name__,
                            "message": str(close_exc),
                            "phase": "simulation_app_close",
                            "source_sha256": source_sha,
                        },
                    )
                raise
            if paths["phase.jsonl"].exists():
                append_phase(
                    paths["phase.jsonl"], "simulation_app_close_returned",
                    sentinel_sha256=preclose_sentinel_sha,
                )


if __name__ == "__main__":
    raise SystemExit(main())
