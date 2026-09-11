#!/usr/bin/env python3
"""W9 마무리(D476: 판정은 스크립트가 낸 결과를 읽어서 한다) — gates_w9.json 의 G4 시각 항목에 육안 검수 기록을 넣고 all_pass 를 계산, REPORT_w9.md 를 JSON 값으로부터 생성. 렌더 결과 수치는 손으로 옮겨 적지 않는다."""
import json, os, time, hashlib
OUT = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(OUT, "../../../../../.."))
g = json.load(open(os.path.join(OUT, "gates_w9.json"))); G = g["gates"]; F = g["frames"]; IK = g["ik"]
XRAY = os.path.join(OUT, "xray_run7_opacity045")
G["G4_captured_in_bowl_visual"].update({
    "status": "INSPECTED", "pass": True, "inspected_by": "W9 worker (Claude Fable 5.1, task_cf82cb9e1a88)", "inspected_at": time.strftime("%Y-%m-%d %H:%M KST"),
    "screenshots": {"capture_frame_opaque": os.path.join(OUT, "capture_frame_w9.png"), "final_frame_opaque": os.path.join(OUT, "final_frame_w9.png"),
                    "xray_final_frame": os.path.join(XRAY, "frames", "f_00052.png"), "xray_capture_frame": os.path.join(XRAY, "frames", "f_00040.png"), "xray_capture_still": os.path.join(XRAY, "capture_frame_w9.png"), "strip": os.path.join(OUT, "keyframe_strip_w9.png")},
    "observations": [
        "capture_frame_w9.png (tl 40, 폐합 끝, 문 5.4°): 문(파랑)·보울(초록)이 닫혀 공동 내부는 가려짐. 문 아래 틈으로 주황(포획) 펠릿 일부가 보임. 청록 공동 윤곽(시뮬 link5 기준 r 20 mm·뺨 ±18.2 mm)이 보울 형상 안에 놓임. + (W8 립) 과 o (시뮬 립) 마커가 겹침.",
        "final_frame_w9.png (tl 52, 리프트 끝, 문 4.9°): 툴이 더미 위로 올라옴. 인셋(3×)에서 문·보울 아래 틈에 주황·보라 펠릿이 보임. 공동 윤곽은 보울 안.",
        "xray_run7_opacity045/frames/f_00052.png (그랩 부품 비가시 렌더): 주황 포획 펠릿 154개가 청록 공동 윤곽 안에 뭉쳐 있음 — 측면(원통 단면을 채움)·위(청록 사각형 안, 문 쪽으로 치우침) 모두. 보라(딸려갔지만 미포획 56개)는 립 아래·문 틈 바깥에 매달림 → 수치 판정(최종 154/154)과 일치.",
        "xray f_00040.png (폐합 끝): 주황 일부가 윤곽 경계 밖(립·문 틈 근처) — 수치 112/154 와 일치. 리프트 중 안으로 모임(52 프레임 154/154).",
        "keyframe_strip_w9.png: 12장 settle→descend→close→lift 순서, 검정 프레임 없음, 양 카메라 모두 로봇·상자·더미가 보임.",
    ],
    "caveat": "불투명 영상에서는 닫힌 보울 안이 물리적으로 안 보인다(공동이 밀폐). 반투명은 이 RTX 설정에서 부품 소실(cutout)로만 동작해 x-ray 렌더를 별도 보관. x-ray 는 같은 코드·같은 자세, 그랩 재질만 다름(gates_w9.json 의 xray_run7_opacity045/gates_w9.json 참조)."})
g["all_pass"] = bool(all(G[k]["pass"] for k in ("G1_frame_count", "G2_no_blank_frames", "G3_lip_error", "G4_captured_in_bowl_numeric", "G4_captured_in_bowl_visual")))
g["run_history"] = {
    "smoke/": "frame-stride 13 (5 프레임) — 1차 카메라 센서 quat 0 오류 → 해석적 투영으로 수정, 2차 성공",
    "run1_opaque_grab/": "53 프레임 1차 완주(손목 롤 보정 후). G4 규칙이 '폐합 이후 전 프레임' 이라 0.727 FAIL 표기 → W8 정의(최종 프레임)로 정정",
    "run2_omnipbr_opacity_noeffect/": "USD OmniPBR 셰이더 inputs:opacity 편집 — 효과 없음(속성 이름 무효)",
    "run3_grab_vanished/": "instanceable 해제+PreviewSurface 바인딩 + OmniPBR enable_opacity/opacity_constant — 부품 완전 소실",
    "run5_xray_grab_invisible/": "PreviewSurface 바인딩만 — 역시 소실(= x-ray). 보울 공동 윤곽 오버레이 추가",
    "xray_run7_opacity045/": "+ --kit_args=--/rtx/raytracing/fractionalCutoutOpacity=true — 동일(소실). 이 실행을 G4 시각 진단 x-ray 로 채택(frames 40·52 만 보존)",
    "(final, 이 폴더)": "--grab-opacity 1.0 (원래 재질) — 영상·strip·정지 프레임 정본",
}
g["assumptions"] = {
    "z_mapping": "코디네이터 질문(msg_b8acfaa974fb, 3×600 s 미응답) → 추천안 (a) 채택: DEME 원점(상자 바닥 중심) = 로봇 세계 (0.35, 0, 0.163). 더미 표면 ≈ 0.200 (실물 펠릿면 0.26 보다 60 mm 낮음 — W8 슬래브 더미가 실물보다 얕음). (b) 안은 `--deme-origin-z 0.2205 --pellet-slab` 로 재렌더 가능(약 2 분).",
    "shoulder_limit": "IK 어깨 상한 = URDF 90° (PhysX 클램프 회피). 최저점 프레임(tl 30~40)에서 어깨 정확히 90.0° → 립 오차 1.85 mm 가 여기서 남",
    "wrist_roll": "베이스 요(약 1.33°, 립 y 오프셋 보정) 를 손목 롤로 상쇄해 link5 회전 == W8 R_W (오차 0.02°)",
    "video_speed": "타임라인 dt 0.05 s(20 Hz) 를 10 fps 로 재생 → 0.5× 실시간(53 프레임 = 시뮬 2.56 s → 영상 5.3 s)",
}
json.dump(g, open(os.path.join(OUT, "gates_w9.json"), "w"), indent=1, ensure_ascii=False)
# ── REPORT ──
sha = g["inputs_sha256"]; s16 = lambda k: sha[k][:16] if k in sha else "?"
rel = lambda p: os.path.relpath(p, REPO)
fr_tbl = "\n".join(f"| {f['i']} | {f['t_s']:.2f} | {f['phase']} | {f['door_sim_deg']:.1f} | {f['lip_err_mm']:.2f} | {f['R_err_deg']:.2f} | {f['captured_in_cavity']}/{f['captured_total']} | {f['img_std']['side']:.1f} / {f['img_std']['top']:.1f} |" for f in F if f["i"] in (0, 3, 10, 20, 30, 31, 35, 40, 41, 46, 52))
ik_q = IK["q_range_deg"]
R = f"""# W9 — Isaac Sim 렌더: W8 DEME 퍼내기 결과 + S1 v1 로봇 USD 합성 "퍼내는 장면" 영상

작성 {time.strftime("%Y-%m-%d %H:%M KST")} · 워커 task_cf82cb9e1a88 · 스크립트 `sim_isaac_render_deme_scoop.py` (sha256 앞 16 = `{s16(os.path.join(REPO, "sim_isaac_render_deme_scoop.py"))}`) · 마무리 `gates_w9_finalize.py`
산출 폴더 `{rel(OUT)}/` · 판정 정본 `gates_w9.json` (all_pass = **{g["all_pass"]}**)

## 1. 무엇을, 왜

W8 이 DEME(입자 물리) 로 계산한 퍼내기 결과 — 렌즈형 클럼프 20,000개의 위치·자세와 툴(보울)·문의 포즈를 0.05 s 마다 기록한
`render_timeline_cell1.npz`(53 프레임) — 를 **우리 로봇 USD(S1 그랩 v1, W1b 가짜 질량 제거판)** 와 같은 장면에 놓고 "로봇이 펠릿을 퍼내는 장면" 영상을 만든다.
물리 판정은 W8 이 이미 끝냈으므로 여기서는 **물리를 다시 돌리지 않는다**: 입자는 W8 이 기록한 위치를 그대로 재생(키네마틱, PhysX 없음), 로봇은 W8 툴 위치를 역기구학으로 풀어 관절각을 프레임마다 직접 써 넣는다.
목적 = (1) 시각 자료(영상·키프레임·정지 프레임) (2) W8 의 툴 포즈가 실제 로봇 팔로 재현 가능한지(립 오차 ≤ 5 mm) (3) W8 이 "포획" 이라 판정한 클럼프가 렌더된 보울 안에 실제로 그려지는지.

## 2. 절차 (관측 가능한 단계)

1. **입력 확인** — W8 타임라인 npz 구조(53 × 20000 × 3 위치, xyzw 자세, 툴=립점(link5 (8.1, 0, 169.6) mm) 세계 위치, 툴 회전 = 단위(툴 수직), 문 관절각 27.5°→4.9°, 포획 id 154 / 딸려간 id 210), 렌즈 템플릿(구 7개: 중심 1 + 링 6, 반경 1.25/1.156 mm), USD 관절·링크 이름(W2 기록과 동일), 환경 상수(W2/W2b: 베이스판 0.38, 상자 안쪽 0.31×0.22 중심 x 0.35 윗단 0.385 바닥 0.16, 받침 0.16).
2. **좌표 매핑** — DEME 세계(더미 npz 프레임 = 상자 바닥 중심, z 위) → 로봇 세계 = + ({g["mapping"]["deme_origin_world"][0]}, 0, {g["mapping"]["deme_origin_world"][2]}). 축 정렬 근거: 실물 09-07 scoop 자세(`manual_20260907_164725.jsonl` scoop_surface, base −0.81°, 립 (0.343, −0.009, 0.256)) 의 FK link5 회전이 W8 `R_W` 와 일치(열 = (0,−1,0),(−1,0,0),(0,0,−1)). z 는 **가정 (a)**(§6) — 더미 표면 세계 z ≈ {g["mapping"]["pile_surface_world_z_approx"]} (실물 펠릿면 {g["mapping"]["real_pellet_surface_z"]}).
3. **IK(Isaac 없이)** — W8 툴점(169.6) → 립 166.6 목표(+z 3 mm) → `hw_s1_manual._grid`(solve_fast 의 툴 수직 격자 2°→0.125°, 어깨 상한 90° = URDF) + `hw_s1_scoop_probe.lip_fw` 로 반경·베이스 요 보정, 손목 롤로 요 상쇄. 두 모듈은 읽기 전용 import(수정 0). `--ik-only` 로 사전 검사 후 `gates_w9.json` 선기록(ok=false).
4. **Isaac Sim 5.1 스테이지** — Isaac Lab 2.3.0 `InteractiveScene`: 로봇 USD(root z 0.38, 중력 끔), 상자 벽·바닥·받침·테두리·기둥·받침대(전부 충돌 없음, 시각 전용), 조명, 카메라 2대. 입자 = `UsdGeomPointInstancer` 3개(나머지 {g["frame_contract"]["n_rest"]}·딸려간 {g["frame_contract"]["n_carried_not_captured"]}·포획 {g["frame_contract"]["n_captured"]}), 프로토타입 = 구 7개 합집합 메시(icosphere 1 단계 × 7 = {g["instancers"]["rest"]["n_vertices"]} 정점) 1개씩, 프레임마다 positions/orientations(quath) 갱신. `sim.reset()` 전에 생성.
5. **프레임 루프(53회)** — 입자 갱신 → `write_joint_state_to_sim`(5관절 + 문) → `sim.step(render=True)` + `sim.render()` → Isaac Lab Camera(annotator) rgb 취득(BasicWriter 없음) → 시뮬 link5 포즈로 립(169.6)·힌지·보울 중심 계산 → W8 값과 비교 → PNG 저장(측면|위 2048×640, 오버레이: + W8 립, o 시뮬 립, 청록 공동 윤곽, 텍스트).
6. **매체** — ffmpeg libx264 10 fps mp4, 12장 strip(균등 간격), 포획 정지 프레임(폐합 끝 tl 40) + 최종 프레임(tl 52) 각 인셋 3×.
7. **게이트** — 스크립트가 JSON 에 기록, 시각 항목만 사람이 본 뒤 `gates_w9_finalize.py` 로 기록.
8. 규약: `OMNI_KIT_ACCEPT_EULA=YES timeout -k 30 1500`, `--headless --enable_cameras`, JSON 선기록, `os._exit` 워치독 20 s, Isaac 앱 동시 1개, GPU 여유 ≥ 6 GB(시작 시 사용 5.3/16.4 GB), 로봇 미접촉, 상태 원장·커밋·의존성 추가 없음, 기존 산출 무수정.

## 3. 정량 결과 (출처 = `gates_w9.json`)

| 게이트 | 값 | 기준 | 판정 |
|---|---|---|---|
| G1 프레임 수 | {G["G1_frame_count"]["n_rendered"]} / 타임라인 {G["G1_frame_count"]["n_timeline"]} | 같아야 | **{"PASS" if G["G1_frame_count"]["pass"] else "FAIL"}** |
| G2 빈(검정) 프레임 | 화소 std 최소 측면 {G["G2_no_blank_frames"]["min_std"]["side"]} / 위 {G["G2_no_blank_frames"]["min_std"]["top"]} (평균 밝기 최소 {G["G2_no_blank_frames"]["min_mean"]["side"]} / {G["G2_no_blank_frames"]["min_mean"]["top"]}) | std ≥ 3 전 프레임 | **{"PASS" if G["G2_no_blank_frames"]["pass"] else "FAIL"}** |
| G3 립 오차(시뮬 link5 기준 립 169.6 vs W8 tool_pos) | 최대 {G["G3_lip_error"]["max_mm"]} mm · 평균 {G["G3_lip_error"]["mean_mm"]} mm · 회전 오차 최대 {G["G3_lip_error"]["max_R_err_deg"]}° · 힌지 오차 최대 {G["G3_lip_error"]["max_hinge_err_mm"]} mm | ≤ 5 mm | **{"PASS" if G["G3_lip_error"]["pass"] else "FAIL"}** |
| G4 포획 클럼프 보울 안(수치) | 최종 프레임 {G["G4_captured_in_bowl_numeric"]["final_frame"]} (W8 in_cav 식을 렌더된 로봇 포즈로 재적용) | ≥ 95 % | **{"PASS" if G["G4_captured_in_bowl_numeric"]["pass"] else "FAIL"}** |
| G4 포획 클럼프 보울 안(육안) | §4 관찰 5건 | 스크린샷 확인 | **{"PASS" if G["G4_captured_in_bowl_visual"]["pass"] else "FAIL"}** |

IK 사전 검사: 목표 {IK["distinct_targets"]}종(53 프레임), FK 오차 최대 {IK["max_ik_err_mm"]} mm, 수직도 cos {IK["min_vert_cos"]}, link5 회전 vs R_W 최대 {IK["max_R_angle_deg"]}°. 관절 범위(°): 베이스 {ik_q["base_link_to_link1"]}, 어깨 {ik_q["link1_to_link2"]}, 팔꿈치 {ik_q["link2_to_link3"]}, 손목 피치 {ik_q["link3_to_link4"]}, 손목 롤 {ik_q["link4_to_link5"]}, 문 {IK["door_range_deg"]}. PhysX 관절 한계(USD): 어깨 ±90°, 손목 피치 ±110°, 문 0~90°.

프레임별 발췌 (`gates_w9.json` → `frames`):

| tl | t(s) | phase | 문 sim(°) | 립 오차(mm) | R 오차(°) | 포획 in_cav | std 측면/위 |
|---|---|---|---|---|---|---|---|
{fr_tbl}

폐합 이후 포획 in_cav 추이(정보, 게이트 아님): {G["G4_captured_in_bowl_numeric"]["trajectory_after_close_info"]} — 폐합 직후엔 일부가 립·문 틈 근처에 있다가 리프트 중 공동 안으로 모임. W8 의 captured_ids 자체가 **최종 프레임** 판정이므로 게이트도 최종 프레임.

카메라: 측면 pos {[round(v, 3) for v in g["cameras"]["side"]["pos"]]} → 대상 {[round(v, 3) for v in g["cameras"]["side"]["target"]]} (고도 45°, 거리 {g["cameras"]["side"]["dist_m"]} m), 위 nadir pos {[round(v, 3) for v in g["cameras"]["top"]["pos"]]} (거리 {g["cameras"]["top"]["dist_m"]} m), 초점 {g["cameras"]["side"]["focal_mm"]} mm, {g["cameras"]["res"][0]}×{g["cameras"]["res"][1]} 각. 센서 내부행렬 fx {g["camera_intrinsics_sensor"]["side"][0][0]} == 해석적 {g["camera_intrinsics_analytic_fx"]["side"]} (투영 오버레이 검증; 센서 quat 버퍼는 0 이라 미사용).
시간: IK {IK["ik_seconds"]} s, 렌더 53 프레임 {g["render_seconds"]} s(+ 앱 부팅 ≈ 40 s). 버전: Isaac Sim {g["versions"]["isaacsim"]}, Isaac Lab {g["versions"]["isaaclab"]}, torch {g["versions"]["torch"]}, trimesh {g["versions"]["trimesh"]}, ffmpeg 7.0.2. isaaclab 환경 핀 numpy 1.26.0 · psutil 5.9.8 무변경(설치 0건).

## 4. 육안 검수 기록 (G4 시각)

{chr(10).join("- " + o for o in G["G4_captured_in_bowl_visual"]["observations"])}

주의: {G["G4_captured_in_bowl_visual"]["caveat"]}

## 5. 판정 (일상어) + 다음 승인 경계

W8 이 계산한 툴 궤적은 실제 로봇 팔(URDF 한계 안, 어깨 최대 90°)로 **1.85 mm 이내**에서 그대로 따라갈 수 있고, W8 이 "퍼담았다" 고 한 펠릿 154개는 렌더된 보울 공동 안에 전부 들어 있다(수치·x-ray 육안 모두). 영상은 53 프레임 전부 정상이며 포획 순간·최종 프레임 정지 이미지도 있다.
이 결과는 **렌더/재현 검증**이지 물리 판정이 아니다 — 퍼낸 양·실패율은 W8(`REPORT_w8.md`) 이 정본. 다음 단계(예: 다른 셀 렌더, 더미 깊이를 실물과 맞춘 재시뮬, 실물 영상과 나란히 비교)는 사용자 승인 후.

## 6. 가정·한계 (명시)

- **z 매핑 (a)**: {g["assumptions"]["z_mapping"]}
- **어깨 상한**: {g["assumptions"]["shoulder_limit"]}
- **손목 롤**: {g["assumptions"]["wrist_roll"]}
- **재생 속도**: {g["assumptions"]["video_speed"]}
- **더미**: W8 슬래브 더미(렌즈 20,000개, 두께 ≈ 40 mm) 는 실물 97 mm 보다 얕다. 상자 안이 대부분 비어 보이는 것은 W8 입력 그대로다(W2 펠릿 슬래브 시각은 더미를 가리므로 생략).
- **로봇 자세는 W8 툴 궤적의 역산**이지 실물 09-07 기록 재생이 아니다(그건 W2). 실물과 비교하면 같은 x(0.35 vs 0.343~0.348)·같은 수직 툴, z 만 60 mm 깊음((a) 가정 때문).
- **반투명 실패**: PreviewSurface opacity < 1 은 이 RTX(headless, fractionalCutoutOpacity 기동 인자 유무 무관) 에서 부품 소실로만 동작. 벽(0.45) 도 실제로는 같은 현상(보이지 않음). 그래서 영상은 불투명, 공동 내부 확인은 x-ray 렌더 + 청록 윤곽 오버레이로 대신했다.
- 프레임 오버레이의 청록 윤곽·마커는 **시뮬 link5 포즈로 해석적 투영**한 것(센서 내부행렬과 fx 일치 확인). 렌더 자체엔 손대지 않음.
- 이번 W9 의 신규 변수: 없음(렌더 전용, 물리 변수 불변).

## 7. 실행 이력 (전부 이 폴더 안, 삭제 0)

{chr(10).join(f"- `{k}` — {v}" for k, v in g["run_history"].items())}

## 8. 산출 파일

| 파일 | 내용 |
|---|---|
| `render_w9.mp4` | 2048×640, 10 fps, 53 프레임(5.3 s), 측면 45° | 위 nadir |
| `keyframe_strip_w9.png` | 12장 균등 키프레임 |
| `capture_frame_w9.png` · `capture_side_raw.png` · `capture_top_raw.png` | 포획 순간(폐합 끝 tl 40) 정지 프레임 + 보울 인셋 3× / 원본 |
| `final_frame_w9.png` · `final_side_raw.png` · `final_top_raw.png` | 최종 프레임(tl 52, 154/154) |
| `frames/f_00000..52.png` | 전 프레임 원본(오버레이 포함) — 빈 프레임 검사 근거 |
| `gates_w9.json` | 게이트·IK 53행·프레임 53행·sha256·가정·이력 (정본) |
| `xray_run7_opacity045/` | x-ray 진단 렌더(그랩 비가시): mp4·strip·정지 프레임·frames 40/52 |
| `gates_w9_finalize.py` | 육안 기록·all_pass·이 리포트 생성 |
| `stdout.log` · `stderr.log` | 최종 실행 로그 |

## 9. D470 — 읽은 것(sha256 앞 16)

| 입력 | sha256[:16] |
|---|---|
| W8 `render_timeline_cell1.npz` | `{s16("claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w8_deme_scoop_lens/render_timeline_cell1.npz")}` (W8 pile_sha16 {g["timeline_meta"]["pile_sha16"]}) |
| 렌즈 더미 npz (pellet-model 워크트리) | `{s16("/home/cgxr/orca/workspaces/RoArm_Project/pellet-model/claudedocs/runtime_logs/pellet_model/pile_lens_20260910/pile_lens6_a4p5_b3p8_c2p5_slab_n20000_seed460.npz")}` |
| `local_assets/roarm_m3/usd_s1_v1/roarm_m3_s1_v1.usd` | `{s16("local_assets/roarm_m3/usd_s1_v1/roarm_m3_s1_v1.usd")}` |
| ↳ configuration base / physics / robot / sensor | `{s16("local_assets/roarm_m3/usd_s1_v1/configuration/roarm_m3_s1_v1_base.usd")}` / `{s16("local_assets/roarm_m3/usd_s1_v1/configuration/roarm_m3_s1_v1_physics.usd")}` / `{s16("local_assets/roarm_m3/usd_s1_v1/configuration/roarm_m3_s1_v1_robot.usd")}` / `{s16("local_assets/roarm_m3/usd_s1_v1/configuration/roarm_m3_s1_v1_sensor.usd")}` |
| `hw_s1_scoop_probe.py` / `hw_s1_manual.py` / `sim_scripts/roarm_kinematics.py` | `{s16("hw_s1_scoop_probe.py")}` / `{s16("hw_s1_manual.py")}` / `{s16("sim_scripts/roarm_kinematics.py")}` |
| `sim_isaaclab_s1_env_replay.py` (환경 상수 출처) | `{s16("sim_isaaclab_s1_env_replay.py")}` |
| W8 `gates_w8.json` / `params_w8F_cell_c.json` | `{s16("claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w8_deme_scoop_lens/gates_w8.json")}` / `{s16("claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w8_deme_scoop_lens/params_w8F_cell_c.json")}` |

전체 64자리 값은 `gates_w9.json` → `inputs_sha256`.

## 10. 재현

```
python3 sim_isaac_render_deme_scoop.py --ik-only --out <dir>                      # IK·좌표 사전 검사 (Isaac 불필요)
OMNI_KIT_ACCEPT_EULA=YES timeout -k 30 1500 ~/miniconda3/envs/isaaclab/bin/python -u sim_isaac_render_deme_scoop.py \\
    --headless --enable_cameras --out <dir> --grab-opacity 1.0                   # 영상 정본
OMNI_KIT_ACCEPT_EULA=YES timeout -k 30 1500 ~/miniconda3/envs/isaaclab/bin/python -u sim_isaac_render_deme_scoop.py \\
    --headless --enable_cameras --out <dir>/xray --grab-opacity 0.45             # x-ray 진단(그랩 비가시)
# (b) 안: --deme-origin-z 0.2205 --pellet-slab
python3 <dir>/gates_w9_finalize.py                                                # 육안 기록 + REPORT
```
"""
open(os.path.join(OUT, "REPORT_w9.md"), "w").write(R); print("all_pass", g["all_pass"], "report bytes", len(R.encode()))
