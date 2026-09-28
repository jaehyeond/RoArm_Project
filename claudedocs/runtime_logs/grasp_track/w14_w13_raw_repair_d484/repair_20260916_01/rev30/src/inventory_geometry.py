"""재고 6분류 기하 판정의 **단일 정본**(순수 CPU · numpy 만).

왜 모듈로 뽑았나
----------------
감사 `RAW_CALLSITE_REVIEW_01.md` + 코디네이터 `msg_fb93d27a72b9` 는 관측 분류가
**회전된 알 전체 구체**로 판정돼야 한다고 지적했다. 예전 판은 생산자 `sim_w13_full_cycle.classify`
와 자기검증기 `verify_w13_self.reclass` 두 곳에 **중심점 기준 사본**이 따로 있었다.
사본을 둘 다 고치면 다음에 또 갈라진다 — 감사가 이미 같은 종류의 조용한 불일치를
`contacts_logged`/`bridge_planned_*` 에서 잡았다. 그래서 판정식을 여기 한 곳에 두고
두 곳이 **같은 함수를 호출**한다. 이 모듈이 분류 의미의 정본이다.

rev30 정정 (2026-09-16 저녁, ERRATUM_04 초안·사용자 결정)
--------------------------------------------------
바닥(상자 바닥·용기 바닥)은 열린 경계가 아니라 받침면이다. 하한을 구 최하단 > 바닥−margin(파묻힘 허용 2.5 mm)으로
바꿔 바닥에 놓인 층이 source/receiving_bin 으로 세어지게 한다. 옆벽·상자 윗면·용기 테두리·공구 입구의
안쪽 margin 과 near 밴드·우선순위·속도창·spill 은 그대로다. run_01 의 옛 규약 verdict 는 소급 변경하지 않는다.

rev29 정정 (2026-09-16, W14 raw repair)
------------------------------------
독립 감사 REV28_PRODUCTION_PARTIAL_RAW_AUDIT_01 finding 10: source 상자 안쪽 판정의 z 하한이
구 **최상단**(z+r > floor-margin) 이어서 바닥에 닿은/뚫은 알도 source 로 세었다.
선언(INSIDE_RULE: 모든 구체가 margin 만큼 안쪽)과 어긋난다. rev29 는 z 하한을 구 **최하단**
(z-r > floor+margin) 으로 고친다. 다른 축·near 밴드·우선순위·margin·속도창은 그대로다.
결과: 바닥에 놓인 알은 규약대로 ambiguous(near band) 가 된다 — 규약 자체를 바꾸지 않았다.

동결 유지
--------
`margin`(2.5 mm) · 우선순위 · 속도창(`v_settle`) 값은 **바꾸지 않는다**.
바뀐 것은 "무엇에 대해" 그 임계를 재느냐 뿐이다: 중심점 → 회전된 모든 구체(중심 ± 반지름).
이 분류는 기록/정착 수지용이며 제어·추가 대기·경로에 연결하지 않는다.
옛 raw 의 라벨은 소급 수정하지 않는다.

규약
----
* `in_*`   = **모든 구체가 통째로** margin 만큼 안쪽 (`.all(axis=sphere)`)
* `near_*` = **어느 한 구체라도** margin 밴드에 걸침 → ambiguous (`.any(axis=sphere)`)
* `spill`  = `max(구중심_z + 구반지름) <= spill_rest_z_m` (모든 구체가 통째로 기준 아래)
* 우선순위 tool > near_tool > bin > near_bin > source > near_source > in_flight > spill

구 전개(`p_sphere = p_owner + R(q_owner) @ offset`)는 호출자가 한다. 생산자는
`sim_deme_scoop_s1.expand_spheres` 를, 자기검증기도 **같은 함수**를 쓴다
(그 전개식은 `expand_vs_npz_max_err_m` 로 원시 npz 구 행에 대해 이미 검증된다).
"""
import numpy as np

INV_NAMES = ["source", "receiving_bin", "tool_residual", "spill", "in_flight", "ambiguous"]
INV = {n: i for i, n in enumerate(INV_NAMES)}

REVISION = "rev30"
CONTRACT_VERSION = "RAW_SCHEMA_REQUIRED + ERRATUM_04 (floors are support surfaces; boundary frame at i-1)"
FLOOR_RULE = "support_surface_v2: min(center_z - r) > floor - margin  (source tray floor and bin inner floor)"
SOURCE_FLOOR_RULE = ("rev30: in_source/in_bin lower-z uses every sphere bottom (center_z - radius) > floor - margin "
                     "(penetration allowance = frozen margin); rev29 used floor + margin (strict, floor layer -> ambiguous); "
                     "rev28 compared sphere top > floor - margin (not whole-sphere). Side walls/top/rim keep the strict inward margin.")
GEOMETRY_BASIS = "oriented_clump_all_spheres"
INSIDE_RULE = "all spheres strictly inside by margin, center distance +/- sphere radius"
BOUNDARY_RULE = "any single sphere overlaps the margin band -> ambiguous"
SPILL_RULE = "max(sphere_center_z + sphere_radius) <= spill_rest_z_m"
PRIORITY = "tool > near_tool > bin > near_bin > source > near_source > in_flight > spill"
EXPANSION = "p_sphere = p_owner + R(owner_quat_xyzw) @ offset_m (clump-major)"


def classify_spheres(S, Rr, speed, p_tool, R_tool, cfg):
    """(n, k) 구체 기하 → 길이 n 의 배타·전수 재고 코드(int8).

    S        (n, k, 3) 구체 중심, 세계 좌표 (clump-major)
    Rr       (n, k)    구체 반지름
    speed    (n,)      owner 속력 |v| (m/s). 속도 벡터가 아니라 **속력**을 받는다.
    p_tool   (3,)      고정 립(tool) 실측 위치
    R_tool   (3, 3)    고정 립 실측 회전
    cfg      아래 키를 가진 dict — 전부 동결 파라미터에서 파생된 값이다:
        R_W (3,3) · lip_l5_m (3,) · bowl_center_l5_m (3,) · bowl_r_in_m · cheek_half_y_m
        bin_center_xy_m (2,) · bin_normals (n_theta,2) · bin_apothem_m
        bin_floor_inner_z_m · bin_rim_z_m
        box_bounds_m (3,2) · box_top_m · margin_m · v_settle_m_s · spill_rest_z_m
    """
    S = np.asarray(S, float)
    Rr = np.asarray(Rr, float)
    if S.ndim != 3 or S.shape[2] != 3:
        raise ValueError(f"S 는 (n, k, 3) 이어야 한다: {S.shape}")
    if Rr.shape != S.shape[:2]:
        raise ValueError(f"Rr 는 (n, k) 이어야 한다: {Rr.shape} vs {S.shape[:2]}")
    n, k = Rr.shape
    speed = np.asarray(speed, float)
    if speed.shape != (n,):
        raise ValueError(f"speed 는 (n,) 속력이어야 한다: {speed.shape}")
    if not np.isfinite(S).all() or not np.isfinite(Rr).all() or not np.isfinite(speed).all():
        raise ValueError("분류 입력에 비유한값이 있다 — 조용히 ambiguous 로 덮지 않는다")
    if (Rr < 0).any():
        raise ValueError("구 반지름이 음수다")

    marg = float(cfg["margin_m"])
    code = np.full(n, INV["ambiguous"], np.int8)
    moving = speed >= float(cfg["v_settle_m_s"])
    rest = ~moving

    # ── 툴 공동(link5 프레임): 원통 반경 + 뺨 슬랩 ────────────────────────
    R_W = np.asarray(cfg["R_W"], float)
    L5 = np.asarray(cfg["lip_l5_m"], float)
    C5 = np.asarray(cfg["bowl_center_l5_m"], float)
    q5 = (R_W.T @ (np.asarray(R_tool, float).T
                   @ (S.reshape(-1, 3) - np.asarray(p_tool, float)).T)).T + L5
    q5 = q5.reshape(n, k, 3)
    r_cav = np.hypot(q5[:, :, 0] - C5[0], q5[:, :, 2] - C5[2])
    y_cav = np.abs(q5[:, :, 1])
    r_in, hy = float(cfg["bowl_r_in_m"]), float(cfg["cheek_half_y_m"])
    in_tool = (((r_cav + Rr) < r_in - marg) & ((y_cav + Rr) < hy - marg)).all(1)
    near_tool = (((r_cav - Rr) < r_in + marg) & ((y_cav - Rr) < hy + marg)).any(1)

    # ── 수신 용기: 정다각 반평면 + 바닥/테두리 ────────────────────────────
    bc = np.asarray(cfg["bin_center_xy_m"], float)
    nrm = np.asarray(cfg["bin_normals"], float)
    apo = float(cfg["bin_apothem_m"])
    zf, zr = float(cfg["bin_floor_inner_z_m"]), float(cfg["bin_rim_z_m"])
    d_s = ((S[:, :, :2] - bc) @ nrm.T).max(2)
    in_bin = (((d_s + Rr) < apo - marg)
              & ((S[:, :, 2] - Rr) > zf - marg)                 # rev30 (ERRATUM_04): 용기 바닥도 받침면
              & ((S[:, :, 2] + Rr) < zr - marg)).all(1)
    near_bin = (((d_s - Rr) < apo + marg)
                & ((S[:, :, 2] + Rr) > zf - marg)
                & ((S[:, :, 2] - Rr) < zr + marg)).any(1)

    # ── 더미 상자(원 소스) ────────────────────────────────────────────────
    box = np.asarray(cfg["box_bounds_m"], float)
    box_top = float(cfg["box_top_m"])
    in_src = (((S[:, :, 0] - Rr) > box[0, 0] + marg) & ((S[:, :, 0] + Rr) < box[0, 1] - marg)
              & ((S[:, :, 1] - Rr) > box[1, 0] + marg) & ((S[:, :, 1] + Rr) < box[1, 1] - marg)
              & ((S[:, :, 2] - Rr) > box[2, 0] - marg)          # rev30 (ERRATUM_04): 바닥=받침면, 구 최하단 > 바닥−margin (파묻힘 허용 2.5 mm)
              & ((S[:, :, 2] + Rr) < box_top - marg)).all(1)
    near_src = (((S[:, :, 0] + Rr) > box[0, 0] - marg) & ((S[:, :, 0] - Rr) < box[0, 1] + marg)
                & ((S[:, :, 1] + Rr) > box[1, 0] - marg) & ((S[:, :, 1] - Rr) < box[1, 1] + marg)
                & ((S[:, :, 2] + Rr) > box[2, 0] - marg)
                & ((S[:, :, 2] - Rr) < box_top + marg)).any(1)

    # ── 동결 우선순위 체인 (순서·의미 불변) ───────────────────────────────
    left = np.ones(n, bool)
    code[in_tool] = INV["tool_residual"]; left &= ~in_tool
    m = left & near_tool; code[m] = INV["ambiguous"]; left &= ~near_tool
    m = left & in_bin; code[m & rest] = INV["receiving_bin"]; code[m & moving] = INV["in_flight"]; left &= ~m
    m = left & near_bin; code[m] = INV["ambiguous"]; left &= ~m
    m = left & in_src; code[m & rest] = INV["source"]; code[m & moving] = INV["in_flight"]; left &= ~m
    m = left & near_src; code[m] = INV["ambiguous"]; left &= ~m
    m = left & moving; code[m] = INV["in_flight"]; left &= ~m
    below = (S[:, :, 2] + Rr).max(1) <= float(cfg["spill_rest_z_m"])
    code[left & below] = INV["spill"]
    code[left & ~below] = INV["ambiguous"]
    return code


def semantics_metadata(k_spheres):
    """원시 NPZ metadata 에 그대로 박을 분류 의미 선언. 문자열 정본은 이 모듈이다."""
    return {"classify_geometry_basis": GEOMETRY_BASIS,
            "classify_spheres_per_clump": int(k_spheres),
            "classify_sphere_expansion": EXPANSION,
            "classify_owner_quat_source": "GetOwnerOriQ(0, n_p) at the same particle frame",
            "classify_inside_rule": INSIDE_RULE,
            "classify_boundary_rule": BOUNDARY_RULE,
            "classify_spill_basis": SPILL_RULE,
            "classify_source_floor_rule": SOURCE_FLOOR_RULE,
            "classify_floor_rule": FLOOR_RULE,
            "classify_contract_version": CONTRACT_VERSION,
            "classify_revision": REVISION,
            "classify_predicate_module": "inventory_geometry.classify_spheres"}
