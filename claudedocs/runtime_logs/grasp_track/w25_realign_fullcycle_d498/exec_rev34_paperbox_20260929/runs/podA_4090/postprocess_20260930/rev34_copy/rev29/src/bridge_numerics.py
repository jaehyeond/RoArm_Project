"""W13R bridge — 설치본 근거 수치 허용치 체인 (순수 함수 · 물리/솔버/하드웨어 접근 0).

정본 근거 (감사 resume root, **나중 erratum 이 이긴다**)
    `audit/SOURCE_EVIDENCE.md`               — 설치 아카이브/멤버/커널/헤더 해시, 한 호출 duration 상한,
                                               위치 격자·트래커 readback·쿼터니언·엔진시간 항의 분리
    `audit/SOURCE_EVIDENCE_ERRATUM_01.md`    — E1 엄격한 >20 m/s · E2 CUDA 12.8 범위 · E3 delta_q = 64u + 64eta/qmin
    `audit/SOURCE_EVIDENCE_ERRATUM_02.md`    — **E4 물리 각도는 theta_num = 4*asin(delta_q/2)** (E3 의 2*asin 을 대체)
                                               E5 bootstrap(qmin>=0.5, ||omega|| <= pi/D, 기록 의무)
    `audit/SOURCE_EVIDENCE_ERRATUM_03.md`    — 구 rev10 의 round(GetSimTime(),9) 십진 양자화

설계 원칙
    · **어떤 항도 관측 여유에 맞춰 고르지 않는다.** 전부 고정 입력에서 계산한다.
    · 항을 합치기 전에 **따로** 계산하고 따로 기록한다(계약·감사 요구).
    · 필수 고정 입력이 없으면 추정하지 않고 **fail-closed** 로 거절한다.
      특히 `l`·`voxelSize` 가 없으면 위치 격자 항이 증명 불가이므로 preflight NO-GO 다(SOURCE_EVIDENCE §1).

이 모듈이 증명하지 않는 것
    충돌 없음·접촉 거동·목표 도달·배출 성공. 다른 DEME 아카이브·컴파일러·커널·timestep·격자·기하로 일반화되지 않는다.
"""
from __future__ import annotations

import math

import numpy as np

ARTIFACT = "W13R_NUMERIC_ALLOWANCE_V1"

# ── 고정 상수 (전부 출처 명시) ───────────────────────────────────────────────
U_F32 = 2.0 ** -24          # binary32 unit roundoff. ERRATUM_01 E3 의 u
ETA_F32 = 2.0 ** -126       # binary32 최소 정규수. ERRATUM_01 E3 의 eta (ftz=false 면 0 이나 보존)
QMIN_DEFAULT = 0.5          # ERRATUM_02 E5 ①: 트래커 반환 쿼터니언 노름 하한
MAX_SUBVOXEL = 1 << 16      # include/DEM/{VariableTypes.h:19-22, Defines.h:56-58} subVoxelPos_t=uint16

EVIDENCE = {
    "documents": [
        "audit/SOURCE_EVIDENCE.md",
        "audit/SOURCE_EVIDENCE_ERRATUM_01.md",
        "audit/SOURCE_EVIDENCE_ERRATUM_02.md  (LATEST WINS for the rotation factor)",
        "audit/SOURCE_EVIDENCE_ERRATUM_03.md",
        # root `msg_6e53d2635c41`: 위치 격자 항의 정본은 ERRATUM_04 E9 인데 문서 목록에서 빠져 있었다.
        "audit/SOURCE_EVIDENCE_ERRATUM_04.md  (LATEST WINS for the position lattice term)",
    ],
    "audit_root": ("/home/cgxr/orca/workspaces/RoArm_Project/w13-cycle-audit/claudedocs/runtime_logs/"
                   "grasp_track/w13_full_cycle_d484/resume_20260913/audit"),
    "static_archive": ("/home/cgxr/miniconda3/envs/roarm/lib/python3.11/site-packages/lib64/"
                       "libsimulator_multi_gpu.a"),
    "static_archive_sha256": "c47ea1c4f0744a50602a139acb22e3db53df3363e6db23a260d050a1ca620529",
    "members_sha256": {
        "APIPublic.cpp.o": "0082da45ef6fe02805b39588cc93d51f8f3b9cc21543e00d339484378fedcd27",
        "dT.cpp.o": "32771933179c631d64fecbc5ea8dbdfecfa173456ff2c17132257ee69813a55c",
        "APIPrivate.cpp.o": "8b2f5f75dfc3761c93e2ec6edb3cff71f5ae23b5e218255c577b6b92453683ef",
        "AuxClasses.cpp.o": "5a5954df1e7cc5f270cbad3673062806a62c74ccfb08ca8850b82e9ce6786dec",
        "JitKernel_cuda.cpp.o": "6271c1992c3f4f30f86f95964e7f92d8c21ea7de7b6162930caf45b50a0814bc"},
    "kernel_sha256": {
        "DEMIntegrationKernels.cu": "4210e442a528cf70ce8087019049e01acf59cfacd272147146ddd3ad9ab18a90"},
    "header_sha256": {
        "AuxClasses.h": "45f766996fa20a19d8ac7f0de1a81a974f07f80852fc2142f01003e8c3375bf8",
        "API.h": "67be492d974a5adad577e53f078c426b79981abe50c1525cb6dce632b1e10ef0"},
    "cuda": {"nvcc": "12.8, build 12.8.93",
             "libnvrtc.so.12_sha256": "2bb82d1a34b9fefa46aca357299aed66763d4d6613d015d5a591224c00fa7e5a",
             "precision_flags_found_in_objects": "none (no --use_fast_math/--ftz/--prec-div/--prec-sqrt/--fmad)",
             "bound_covers": "both FMA and non-FMA, both default precision and use_fast_math"},
    "lattice_rule": "DEMHelperKernels.cuh:116-159  pos = voxel*voxelSize + subVoxel*l, truncating cast to uint16",
    "tracker_rule": "DEMDynamicThread::getOwnerPos reconstructs binary64 then cvtsd2ss to binary32",
    "duration_rule": "A_final <= nextafter(D + float64(float32(h)), +inf)",
    "motion_rule": "(|v| + r*|omega|) * A_final   [DEMIntegrationKernels.cu:197-233]",
    "rotation_rule": "delta_q = 64u + 64eta/qmin ; theta_num = 4*asin(min(1, delta_q/2)) ; rmax*N*theta_num",
}


class NumericInputsMissing(ValueError):
    """고정 입력이 없다 — 추정하지 않고 거절한다(증명 불가 = NO-GO)."""


# ── 기본 스칼라 ──────────────────────────────────────────────────────────────
def hf_of(timestep_s):
    """`float64(float32(h))`. 설치본이 float 로 저장한 뒤 double 로 올려 쓰는 그 값."""
    h = float(timestep_s)
    if not (math.isfinite(h) and h > 0):
        raise NumericInputsMissing(f"timestep_s must be finite positive, got {h}")
    return float(np.float32(h))


def call_elapsed_upper_bound_s(requested_duration_s, timestep_s):
    """한 `DoDynamicsThenSync(D)` 호출의 엄격한 경과시간 상한 (SOURCE_EVIDENCE §one-call)."""
    D = float(requested_duration_s)
    if not (math.isfinite(D) and D > 0):
        raise NumericInputsMissing(f"requested duration must be finite positive, got {D}")
    return math.nextafter(D + hf_of(timestep_s), math.inf)


def internal_steps(requested_duration_s, timestep_s):
    """설치본 루프가 실제로 도는 N. **누산 규칙 그대로 재현**한다(추정 아님).

    `dT.cpp.o workerThread`: local double 누산기 0 → 적분 1회 → `+= float64(float32(h))` →
    요청 duration > 누산기 인 동안 반복. 따라서 N = 첫 번째로 누산기 >= D 가 되는 횟수.
    """
    D = float(requested_duration_s)
    hf = hf_of(timestep_s)
    acc, n = 0.0, 0
    limit = int(D / hf) + 16
    while acc < D and n <= limit:
        n += 1
        acc += hf
    if acc < D:
        raise NumericInputsMissing(f"internal step reconstruction did not terminate for D={D}, hf={hf}")
    return n, acc


def ulp32(x):
    """|x| 에서의 binary32 ULP."""
    a = abs(float(x))
    if not math.isfinite(a):
        raise NumericInputsMissing(f"ulp32 needs a finite magnitude, got {x}")
    f = np.float32(a)
    return float(np.nextafter(f, np.float32(np.inf)) - f) if a > 0 else float(np.float32(2.0) ** -149)


def delta_q(qmin=QMIN_DEFAULT):
    """ERRATUM_01 E3: 한 적분의 단위 쿼터니언 유클리드 현(chord) 상한."""
    q = float(qmin)
    if not (math.isfinite(q) and q > 0):
        raise NumericInputsMissing(f"qmin must be finite positive, got {qmin}")
    return 64.0 * U_F32 + 64.0 * ETA_F32 / q


def theta_num_rad(qmin=QMIN_DEFAULT):
    """ERRATUM_02 E4: S3 현 → **물리 회전각** = 4*asin(min(1, delta_q/2)). (E3 의 2*asin 을 대체)"""
    dq = delta_q(qmin)
    return 4.0 * math.asin(min(1.0, dq / 2.0))


# ── 네 개의 독립 허용치 항 ───────────────────────────────────────────────────
def ideal_motion_allowance_m(chord_m, theta_rad, r_max_local_m, requested_duration_s, timestep_s,
                             delta_p_m=None, rotvec_rad=None):
    """`(||v32|| + r*||omega32||)*A_final` — **ERRATUM_04 E8 의 binary32 명령 벡터 기준**.

    실제 벡터를 주면 그것으로 계산하고(선호 경로), 주지 않으면 binary64 크기에 보수적 ULP 항
    `A*sqrt(3)*(ulp32(|dp|/D) + r*ulp32(theta/D))` 를 더한다(관측 잔차가 아니라 full ULP).
    """
    D = float(requested_duration_s)
    A = call_elapsed_upper_bound_s(D, timestep_s)
    r = float(r_max_local_m)
    if delta_p_m is not None and rotvec_rad is not None:
        v32, w32 = command_float32_vectors(delta_p_m, rotvec_rad, D)
        return (float(np.linalg.norm(np.asarray(v32, np.float64)))
                + r * float(np.linalg.norm(np.asarray(w32, np.float64)))) * A
    v, w = float(chord_m) / D, float(theta_rad) / D
    return (v + r * w) * A + A * math.sqrt(3.0) * (ulp32(v) + r * ulp32(w))


U_F64 = 2.0 ** -53          # binary64 unit roundoff (ERRATUM_04 E9)


def position_lattice_allowance_m(n_steps, l_m, voxel_size_m=None, coord_bound_m=None,
                                 mode="voxel_fallback"):
    """위치 표현 허용치. **ERRATUM_04 E9 가 `sqrt(3)*N*l` 만 쓰던 주장을 대체한다.**

    E9: 엄격한 `< l` 논증은 주변 binary64 연산(복원 곱·합, LBF 가감, voxel 나눗셈/잔차 나눗셈)을
    따로 세지 않았다. voxel 경계 근방에서 반올림된 나눗셈 뒤 부호 없는 sub-voxel 변환이
    범위를 벗어날 수 있고, 그건 `8*u64*B` 를 더하는 것으로 덮이지 않는다.
    quotient-boundary 안정성 증명이 없으므로 **출처만으로 유한한 fallback**(축당 voxel 1 + sub-voxel 1)을 쓴다:

        sqrt(3) * N * (voxelSize + l + 8*u64*B)

    고정 old-domain/N=4001 에서 `0.0025228147373688847 m`. 이 값은 "작다"고 부르지 않는다 —
    bridge 판정을 바꿀 수 있으므로 epsilon 안에 숨기지 않고 **따로** 보고한다.
    `mode="tiny_binary64"` 는 quotient/변환 증명이 나온 뒤에만 쓴다(그때 `sqrt(3)*N*8*u64*B`).
    """
    if l_m is None:
        raise NumericInputsMissing(
            "position lattice allowance requires the INITIALIZED DEME length unit `l` "
            "(pos = voxel*voxelSize + subVoxel*l). It is not exposed by the installed Python API "
            "(no getter among DEMSolver's 188 methods) and is not present in any preserved old log. "
            "Per SOURCE_EVIDENCE.md section 1 the numeric position envelope is unprovable without it "
            "and preflight is NO-GO. Do not estimate it.")
    l = float(l_m)
    if not (math.isfinite(l) and l > 0):
        raise NumericInputsMissing(f"length unit l must be finite positive, got {l_m}")
    N = float(n_steps)
    if mode == "voxel_fallback":
        if voxel_size_m is None or coord_bound_m is None:
            raise NumericInputsMissing(
                "ERRATUM_04 E9 voxel fallback requires BOTH voxel_size_m and coord_bound_m "
                "(sqrt(3)*N*(voxelSize + l + 8*u64*B)). Do not fall back to the sqrt(3)*N*l-only "
                "claim, which E9 superseded, and do not substitute an empirical epsilon.")
        vs, B = float(voxel_size_m), float(coord_bound_m)
        if not (math.isfinite(vs) and vs > 0 and math.isfinite(B) and B > 0):
            raise NumericInputsMissing(f"voxel_size_m/coord_bound_m must be finite positive, "
                                       f"got {voxel_size_m}/{coord_bound_m}")
        return math.sqrt(3.0) * N * (vs + l + 8.0 * U_F64 * B)
    if mode == "tiny_binary64":
        if coord_bound_m is None:
            raise NumericInputsMissing("tiny_binary64 mode requires coord_bound_m")
        return math.sqrt(3.0) * N * 8.0 * U_F64 * float(coord_bound_m)
    raise NumericInputsMissing(f"unknown position allowance mode {mode!r}")


def position_lattice_constituents(n_steps, l_m, voxel_size_m, coord_bound_m):
    """E9 가 요구한 **네 구성량을 원시로 보존**한다(합계만 남기지 않는다)."""
    return {"n_steps": int(n_steps), "l_m": float(l_m), "voxel_size_m": float(voxel_size_m),
            "coord_bound_B_m": float(coord_bound_m), "u64": U_F64,
            "term_voxel_m": math.sqrt(3.0) * float(n_steps) * float(voxel_size_m),
            "term_l_m": math.sqrt(3.0) * float(n_steps) * float(l_m),
            "term_binary64_m": math.sqrt(3.0) * float(n_steps) * 8.0 * U_F64 * float(coord_bound_m),
            "superseded_sqrt3_N_l_only_m": math.sqrt(3.0) * float(n_steps) * float(l_m),
            "rationale": ("ERRATUM_04 E9: quotient-boundary 안정성 미증명 → 축당 voxel 1 + sub-voxel 1 "
                          "fallback. 관측 여유로 모드를 고르지 않는다.")}


def command_float32_vectors(delta_p_m, rotvec_rad, requested_duration_s):
    """ERRATUM_04 E8: 트래커에 실제로 들어가는 **binary32 명령 벡터**를 만든다.

    `SetVel(float3)`/`SetAngVel(float3)`(AuxClasses.h:242-250)은 성분을 binary32 로 저장한다.
    그래서 상한을 그 벡터 자체로 계산하고, **같은 벡터를 트래커에 넘긴다**
    (컨트롤러의 실효 명령은 바뀌지 않는다 — 바인딩이 어차피 하던 변환을 명시화한 것이다).
    """
    D = float(requested_duration_s)
    if not (math.isfinite(D) and D > 0):
        raise NumericInputsMissing(f"requested duration must be finite positive, got {D}")
    v32 = np.asarray(np.asarray(delta_p_m, float) / D, np.float32)
    w32 = np.asarray(np.asarray(rotvec_rad, float) / D, np.float32)
    if not (np.isfinite(v32).all() and np.isfinite(w32).all()):
        raise NumericInputsMissing("binary32 command vectors must be finite")
    return v32, w32


def tracker_read_allowance_m(domain_max_coord_m, half_ulp=False):
    """트래커 binary64 → binary32 변환 허용치. **샘플 포즈가 아니라 도메인 최대 좌표**에서 계산한다.

    변환 모드가 고정되지 않았으므로 기본은 보수적인 full ULP 다(SOURCE_EVIDENCE §2).
    """
    if domain_max_coord_m is None:
        raise NumericInputsMissing("tracker read allowance requires a source/domain-derived maximum "
                                   "coordinate magnitude (not sampled poses)")
    c = float(domain_max_coord_m)
    if not (math.isfinite(c) and c > 0):
        raise NumericInputsMissing(f"domain_max_coord_m must be finite positive, got {domain_max_coord_m}")
    comp = ulp32(c) * (0.5 if half_ulp else 1.0)
    return math.sqrt(3.0) * comp


def quaternion_rotation_allowance_m(n_steps, r_max_local_m, qmin=QMIN_DEFAULT):
    """ERRATUM_02 E4: `rmax * N * theta_num`. 각 항의 중간값도 같이 낸다."""
    th = theta_num_rad(qmin)
    return float(r_max_local_m) * float(n_steps) * th, th


# ── 묶음 ─────────────────────────────────────────────────────────────────────
def allowance_terms(*, chord_m, theta_rad, r_max_local_m, requested_duration_s, timestep_s,
                    n_steps, l_m, domain_max_coord_m, qmin=QMIN_DEFAULT, half_ulp=False,
                    voxel_size_m=None, delta_p_m=None, rotvec_rad=None,
                    position_mode="voxel_fallback"):
    """네 항을 **따로** 계산해 합계와 함께 돌려준다. 하나라도 고정 입력이 없으면 예외로 거절."""
    ideal = ideal_motion_allowance_m(chord_m, theta_rad, r_max_local_m, requested_duration_s,
                                     timestep_s, delta_p_m=delta_p_m, rotvec_rad=rotvec_rad)
    lattice = position_lattice_allowance_m(n_steps, l_m, voxel_size_m=voxel_size_m,
                                           coord_bound_m=domain_max_coord_m, mode=position_mode)
    tracker = tracker_read_allowance_m(domain_max_coord_m, half_ulp=half_ulp)
    quat, th = quaternion_rotation_allowance_m(n_steps, r_max_local_m, qmin)
    return {"ideal_motion_m": ideal, "position_lattice_m": lattice, "tracker_read_m": tracker,
            "quaternion_rotation_m": quat, "theta_num_rad": th,
            "position_mode": position_mode,
            "ideal_motion_from_binary32_command": bool(delta_p_m is not None and rotvec_rad is not None),
            "total_m": ideal + lattice + tracker + quat}


def bootstrap_checks(*, quat_xyzw, omega_rad_s, requested_duration_s, qmin=QMIN_DEFAULT,
                     unit_norm_tol=1e-6):
    """ERRATUM_02 E5 ①②: 첫 bridge 물리 호출 **전에** 반드시 통과해야 하는 조건.

    ① 트래커 반환 쿼터니언이 유한하고 노름 >= qmin 이며 컨트롤러 단위 허용오차 안.
    ② 명령 국소 각속도가 유한하고 `||omega|| <= pi/D`(principal rotation-vector 구성 상한).
    """
    out = {"qmin": float(qmin), "unit_norm_tol": float(unit_norm_tol)}
    q = np.asarray(quat_xyzw, float)
    ok_shape = q.shape == (4,) and np.isfinite(q).all()
    n = float(np.linalg.norm(q)) if ok_shape else float("nan")
    out["quat_finite_shape_ok"] = bool(ok_shape)
    out["quat_norm"] = n
    out["quat_norm_ge_qmin"] = bool(ok_shape and n >= float(qmin))
    out["quat_within_unit_tolerance"] = bool(ok_shape and abs(n - 1.0) <= float(unit_norm_tol))
    w = np.asarray(omega_rad_s, float)
    ok_w = w.shape == (3,) and np.isfinite(w).all()
    wn = float(np.linalg.norm(w)) if ok_w else float("nan")
    D = float(requested_duration_s)
    lim = math.pi / D if (math.isfinite(D) and D > 0) else float("nan")
    out["omega_finite_shape_ok"] = bool(ok_w)
    out["omega_norm_rad_s"] = wn
    out["omega_bound_rad_s"] = lim
    out["omega_within_principal_bound"] = bool(ok_w and math.isfinite(lim) and wn <= lim)
    out["pass"] = bool(out["quat_finite_shape_ok"] and out["quat_norm_ge_qmin"]
                       and out["quat_within_unit_tolerance"] and out["omega_finite_shape_ok"]
                       and out["omega_within_principal_bound"])
    return out


def pinned_receipt(*, requested_duration_s, timestep_s, l_m, voxel_size_m, domain_max_coord_m,
                   r_max_local_m_by_body, qmin=QMIN_DEFAULT, half_ulp=False,
                   position_mode="voxel_fallback", mode_production=True):
    """immutable preflight 영수증에 들어갈 고정 입력·파생값 전부. 하나라도 없으면 그 자리에 사유를 남긴다."""
    N, acc = internal_steps(requested_duration_s, timestep_s)
    hf = hf_of(timestep_s)
    rec = {
        "artifact": ARTIFACT, "evidence": EVIDENCE,
        "requested_duration_D_s": float(requested_duration_s),
        "timestep_h_s": float(timestep_s), "float64_of_float32_h_s": hf,
        "internal_steps_N": N, "internal_steps_N_source": "DERIVED by replaying the installed "
                                                          "accumulator rule; the installed build exposes "
                                                          "NO directly observed engine step counter",
        "accumulator_at_stop_s": acc,
        "call_elapsed_upper_bound_s": call_elapsed_upper_bound_s(requested_duration_s, timestep_s),
        "qmin": float(qmin), "u_binary32": U_F32, "eta_binary32": ETA_F32,
        "delta_q": delta_q(qmin), "theta_num_rad": theta_num_rad(qmin),
        "N_times_theta_num_rad": N * theta_num_rad(qmin),
        "max_subvoxel": MAX_SUBVOXEL,
        "voxel_size_m": voxel_size_m, "length_unit_l_m": l_m,
        "domain_max_coord_m": domain_max_coord_m,
        "tracker_ulp_mode": "half_ulp" if half_ulp else "full_ulp (conservative; conversion mode not pinned)",
        "r_max_local_m_by_body": dict(r_max_local_m_by_body),
    }
    # ⚠️ root 실측 반례 `msg_6e53d2635c41` / `coordinator/NUMERIC_RECEIPT03_REPRO.json`:
    #    예전 판은 `position_lattice_allowance_m(N, l_m)` 로 불러 **voxel_size_m 과 coord_bound_m 을
    #    전달하지 않았다**. 둘 다 이 함수 인자로 이미 들어와 있는데도 빠뜨려서, 영수증의 위치 항이
    #    항상 None 이 되고 `position_lattice_evidence_gap` 이라는 **거짓 출처 공백**이 생겼다.
    #    같은 완전 입력을 직접 넘기면 0.0025228147373688847 m 가 나온다(root 양성 대조).
    #    이것은 영수증 경로의 결함이다 — 인증 경로(build_static_inputs→certify)는 완전 입력을
    #    넘기고 있었으므로 **실제 bridge 운동식이 틀렸다는 뜻은 아니다**(root 도 그렇게 적었다).
    if mode_production and str(position_mode) == "tiny_binary64":
        raise ValueError(
            "tiny_binary64 위치 모드는 quotient-boundary/변환 증명이 없으므로 **생산에서 거부**한다"
            "(root msg_6e53d2635c41). ERRATUM_04 E9 의 voxel_fallback 만 생산에 쓴다.")
    rec["position_lattice_mode"] = str(position_mode)
    rec["position_lattice_mode_rejected_for_production"] = ["tiny_binary64"]
    try:
        rec["position_lattice_allowance_m"] = position_lattice_allowance_m(
            N, l_m, voxel_size_m=voxel_size_m, coord_bound_m=domain_max_coord_m,
            mode=str(position_mode))
        rec["position_lattice_inputs_forwarded"] = {
            "n_steps": N, "l_m": l_m, "voxel_size_m": voxel_size_m,
            "coord_bound_m": domain_max_coord_m, "mode": str(position_mode)}
    except NumericInputsMissing as exc:
        rec["position_lattice_allowance_m"] = None
        rec["position_lattice_evidence_gap"] = str(exc)
        rec["position_lattice_inputs_forwarded"] = {
            "n_steps": N, "l_m": l_m, "voxel_size_m": voxel_size_m,
            "coord_bound_m": domain_max_coord_m, "mode": str(position_mode)}
    try:
        rec["tracker_read_allowance_m"] = tracker_read_allowance_m(domain_max_coord_m, half_ulp=half_ulp)
    except NumericInputsMissing as exc:
        rec["tracker_read_allowance_m"] = None
        rec["tracker_read_evidence_gap"] = str(exc)
    rec["quaternion_rotation_allowance_m_by_body"] = {
        b: quaternion_rotation_allowance_m(N, r, qmin)[0] for b, r in r_max_local_m_by_body.items()}
    rec["preflight_blocking_gaps"] = [k for k in ("position_lattice_evidence_gap",
                                                  "tracker_read_evidence_gap") if k in rec]
    rec["non_claims"] = [
        "수치 허용치는 표현/반올림 범위일 뿐 충돌 없음·접촉 정확성·도달·배출 성공의 증거가 아니다.",
        "어떤 항도 관측된 여유나 추종 잔차에서 고르지 않았다.",
        "다른 아카이브/컴파일러/커널/timestep/격자/기하로 일반화되지 않는다.",
    ]
    return rec
