"""W25 COMMANDS 생성 — pod 별(attempt_dir 만 다름) JSON 두 개. argv 는 rev34 COMMANDS_w25_template.json variant_paperbox 의
step1 골격 + W19 COMMANDS_w19 의 smoke 3종 골격을 exec 동결본 경로·2차 NPZ 로 채운 것. 판정 문턱은 CLI 인자로 두지 않는다(D492).
실행 금지 항목: 본 실행은 go_required_steps → --allow-go-step 필수."""
import json, sys, time
from pathlib import Path
MAIN = "/home/cgxr/Documents/Robotics/RoArm_Project"
CASE = f"{MAIN}/claudedocs/runtime_logs/grasp_track/w25_realign_fullcycle_d498"
EXEC = f"{CASE}/exec_rev34_paperbox_20260929"
NPZ = f"{CASE}/pile_flat40_20260929/pile_lens6_a4p5_b3p8_c2p5_slab40_outer_310x220_n67737_rho0p503_seed460.npz"
NPZ_SHA = "31dd289731a144de30bbd3054ffd8c233d9897347f3cd4eab2e07f97032583c1"
PY = "/home/cgxr/miniconda3/envs/roarm/bin/python"
SIM = f"{EXEC}/rev34/src/sim_w13_full_cycle.py"
PARAMS = f"{EXEC}/rev34/params_w25_paperbox.json"
NE = f"{EXEC}/rev34/numeric_inputs_w25_paperbox_n67737.json"
CRIT = f"{EXEC}/criteria_w25_paperbox_cap32h.json"
MANIFEST = f"{EXEC}/bundle/BUNDLE_MANIFEST_W25.json"
CAP_HARD, GRACE, CAP_SOFT = 115200, 1200, 114000          # criteria_w25_paperbox_cap32h.json 과 같은 값

def steps(tag):
    R = f"{EXEC}/runs/{tag}"
    return {
        "smoke_import_300": {
            "attempt_dir": f"{R}/smoke_01_import300", "cap_s": 600.0, "grace_s": 60.0, "cwd": f"{EXEC}/rev34/src",
            "note": "W19 smoke_01 과 같은 형태(300알·settle 까지·numeric-evidence 없음). rev34 ON params + 2차 NPZ. import 체인·STL/URDF·DEME JIT(NVRTC, 이 GPU 아키텍처) 확인. 트레이 선언 검사는 전체 NPZ 발자국으로 하므로 축소와 무관(sim_w13_full_cycle.py:153).",
            "argv": [PY, "-B", SIM, "--params", PARAMS, "--pile", NPZ, "--out", f"{R}/smoke_01_import300", "--seed", "460",
                     "--max-particles", "300", "--stop-after-phase", "settle", "--max-wall-s", "540"]},
        "smoke_sphere_regression": {
            "attempt_dir": f"{R}/smoke_02_sphere_regression", "cap_s": 1200.0, "grace_s": 60.0, "cwd": MAIN,
            "note": "W10 G0 구 회귀(W19 와 바이트 동일 입력): diverged=false ∧ n_in_cavity∈[268,362] ∧ stops 전부 servo_stall. 로컬 W10 103.5 s/287개, W19 4090 92.2 s/318·97.8 s/309. 판정 = pod/check_sphere_regression.py(pop 항목은 관측).",
            "argv": [PY, "-u", f"{MAIN}/sim_deme_scoop_s1.py",
                     "--params", f"{MAIN}/claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w3_deme_scoop/b_diverge/params_fixnorm_plunge25.json",
                     "--pile", f"{MAIN}/claudedocs/runtime_logs/sim_deme/pile_practical_fast_d4p16_n18796_seed460.npz",
                     "--out", f"{R}/smoke_02_sphere_regression", "--seed", "460"]},
        "smoke_settle_full": {
            "attempt_dir": f"{R}/smoke_03_settle_full", "cap_s": 2400.0, "grace_s": 120.0, "cwd": f"{EXEC}/rev34/src",
            "note": "67,737알 전량 + numeric-evidence 도메인 게이트 + settle(물리 0.1 s) 까지. **벤치마크 셀**: settle 구간 wall/물리초(pod/settle_speed_ratio.py, sync_wall_elapsed_s 차분)를 pod 간 비교. W19 4090 20k = 809.7/788.9 s/물리초. 67.7k 는 미측정(외삽 금지) → cap 2,400 s.",
            "argv": [PY, "-B", SIM, "--params", PARAMS, "--pile", NPZ, "--out", f"{R}/smoke_03_settle_full", "--seed", "460",
                     "--numeric-evidence", NE, "--stop-after-phase", "settle", "--max-wall-s", "2100"]},
        "run_paperbox_full_cycle": {
            "attempt_dir": f"{R}/run_01", "cap_s": float(CAP_HARD), "grace_s": float(GRACE), "cwd": f"{EXEC}/rev34/src",
            "note": f"본 실행: rev34 규약 A·실물 절차 ON·종이 상자 310×220·벽 230·펠릿면 23.9 cm·travel 45·2차 더미 67,737알(FLAT40_PASS 40.2 mm)·C4 0.135 mm 수용(사용자). 상한 32 h = {CAP_HARD} s(소프트 {CAP_SOFT} + grace {GRACE}) — criteria_w25_paperbox_cap32h.json 과 동일. GO 필요. 연장·재시도 0.",
            "argv": [PY, "-B", SIM, "--params", PARAMS, "--pile", NPZ, "--out", f"{R}/run_01", "--seed", "460",
                     "--numeric-evidence", NE, "--max-wall-s", str(CAP_SOFT)]},
    }

def commands(tag, gpu, extra):
    return {
        "artifact": "W25_RUNPOD_COMMANDS_V1", "pod_tag": tag, "gpu_plan": gpu,
        "note": "pod 에서 로컬과 같은 절대경로로 실행한다. 물성·형상·dt 1 µs·cd_update_freq·서보 정지 모델·보호선 변경 0(rev32 params 바이트 사본 + W25 배치/절차 키, REVISION_PIN physics_params_changed=false). 자동 재시도 없음. 본 실행은 사용자 GO(2026-09-29 병행안) 뒤 --allow-go-step 으로만.",
        "frozen_inputs": {"exec_dir": EXEC, "sim": SIM, "params": PARAMS, "numeric_evidence": NE, "pile_npz": NPZ, "pile_npz_sha256": NPZ_SHA,
                          "criteria": CRIT, "revision_pin": f"{EXEC}/rev34/REVISION_PIN.json"},
        "manifest": MANIFEST, "env": {"PYTHONDONTWRITEBYTECODE": "1"}, "auto_retry": False,
        "gpu_policy": "pod 당 DEME 프로세스 하나. smoke 3종 rc0 후에만 본 실행. DEMSolver() 기본 nGPUs=2 → 2-GPU pod 면 kT/dT 분산(코드 변경 0).",
        "pre_go_steps_order": ["smoke_import_300", "smoke_sphere_regression", "smoke_settle_full"],
        "go_required_steps": ["run_paperbox_full_cycle"],
        "steps": steps(tag),
        "timeout_strategy": {"layer1_sim_soft_cap": f"rev34 sim 은 --max-wall-s {CAP_SOFT} 에서 step 경계 _WallCapStop → 부분 원시 정상 finalize(abort_class WALL_CAP_STOP).",
                             "layer2_runner_hard": f"러너가 cap-grace={CAP_SOFT} s 에 자식 프로세스 그룹 SIGTERM, 남은 grace {GRACE} s 뒤 SIGKILL(cap {CAP_HARD}).",
                             "timeout_is_not_success": True, "watchdog_note": "step 경계 미도달 시 sim 소프트 상한은 미발동(D493) → 러너 하드 상한이 바깥 watchdog."},
        "local_reference_wall_s": {"W19_A_4090_20k_full_cycle": 20977.836, "W19_4090_20k_settle_per_phys_s": [809.7, 788.9],
                                   "W10_G0_sphere_regression_local": 103.51, "W19_4090_G0": [92.16, 97.78],
                                   "W25_estimate_67737_full_cycle_h": [16.4, 23.8], "estimate_note": "세 외삽 모형 범위(D499), 실측 아님"},
        **extra,
    }

A = commands("podA_4090", {"gpu_id": "NVIDIA GeForce RTX 4090", "count": 1, "cloud": "SECURE", "minCudaVersion": "13.0", "price_usd_h": 0.74},
             {"role": "검증된 경로(W19 동일 GPU·드라이버 580 계열). smoke 3종 rc0 → 즉시 GO."})
B = commands("podB_pro6000x2", {"gpu_id": "NVIDIA RTX PRO 6000 Blackwell Server Edition", "count": 2, "cloud": "SECURE", "minCudaVersion": "13.0", "price_usd_h": 4.18},
             {"role": "hedge. Blackwell(sm_120) NVRTC JIT·DEME 정적 커널 호환은 미검증 → smoke 로만 판정.",
              "pre_registered_go_rule": {
                  "metric": "settle_wall_per_phys_s = smoke_03_settle_full 의 settle phase wall_s / sim_s (pod/settle_speed_ratio.py, NPZ sync_wall_elapsed_s 차분)",
                  "ratio": "R = settle_wall_per_phys_s(podA_4090) / settle_wall_per_phys_s(podB_pro6000x2)",
                  "go_if": "smoke 3종 rc0 AND sphere_regression G0 3/3 PASS AND R >= 1.3",
                  "else": "즉시 terminate(본 실행 0). R 은 결과 본 뒤 바꾸지 않는다.",
                  "labeling": "GO 시 podB run_01 은 podA run_01 과 같은 입력의 '장비 혼합 반복(n=2, 4090 vs PRO6000×2)' 로만 표기 — 동일 장비 반복이 아님."}})
out = Path(EXEC)
(out / "COMMANDS_w25_podA_4090.json").write_text(json.dumps(A, ensure_ascii=False, indent=1) + "\n")
(out / "COMMANDS_w25_podB_pro6000x2.json").write_text(json.dumps(B, ensure_ascii=False, indent=1) + "\n")
print("written", [k for k in A["steps"]], "cap", CAP_HARD, CAP_SOFT, GRACE)
