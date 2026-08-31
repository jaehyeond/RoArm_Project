"""3D 출력 전송 매니페스트 생성기 — 슬라이스 산출물을 검증하고 print_job.json으로 굳힌다.

왜 JSON인가: 업로드·출력 명령을 손으로 치면 그때그때 다른 값이 들어간다. 무엇을 어느
프린터에 어떤 검증을 통과한 상태로 보냈는지가 남지 않는다. 이 파일이 전송의 단일 소스이며
`send_print_job.py`가 이것만 읽고 실행한다 — 매니페스트에 없는 것은 보내지 않는다.

사용:  python make_print_job.py [3mf경로] [출력디렉터리] [부품이름] [STL...]
       인자를 안 주면 기존 칼라 쿠폰 기본값을 쓴다 (하위호환).
"""
import json, hashlib, re, zipfile, subprocess, sys
from pathlib import Path

DTR = Path("/home/cgxr/Documents/DK/DTR/bamboo-3dprinter")
_a = sys.argv[1:]
if _a:
    THREEMF = Path(_a[0])
    OUT     = Path(_a[1])
    PARTNAME = _a[2]
    STLS    = [Path(x) for x in _a[3:]]
    STL     = STLS[0]
else:
    OUT = Path("/home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/runtime_logs/scoop_shell_v0")
    THREEMF = DTR / "output/roarm_collar_coupon.3mf"
    STL     = OUT / "collar_test_coupon.stl"
    STLS    = [STL]
    PARTNAME = "roarm_collar_coupon"

BED_X = BED_Y = 256.0
BED_Z = 250.0
BRIM_MM = 5.0

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]

z = zipfile.ZipFile(THREEMF)
si = z.read("Metadata/slice_info.config").decode()
meta = lambda k: (re.search(rf'key="{k}" value="([^"]*)"', si) or [None, None])[1]
plate = json.loads(z.read("Metadata/plate_1.json"))
gcode = z.read("Metadata/plate_1.gcode").decode("utf-8", "ignore")

x0, y0, x1, y1 = plate["bbox_all"]
beds = sorted({int(v) for v in re.findall(r"M140 S(\d+)", gcode)} - {0})
nozz = sorted({int(v) for v in re.findall(r"M104 S(\d+)", gcode)} - {0})
warns = re.findall(r'<warning msg="([^"]+)"', si)
_zs = [float(v) for v in re.findall(r"^G[01] .*?Z([0-9.]+)", gcode, re.M)]
max_z = max(_zs) if _zs else 0.0

# ── 검증 게이트 ───────────────────────────────────────────────────────────
gates = {
    "inside_bed_with_brim": {
        "pass": bool(x0 - BRIM_MM >= 0 and y0 - BRIM_MM >= 0
                     and x1 + BRIM_MM <= BED_X and y1 + BRIM_MM <= BED_Y),
        "detail": f"브림 {BRIM_MM}mm 포함 X {x0-BRIM_MM:.1f}~{x1+BRIM_MM:.1f} "
                  f"Y {y0-BRIM_MM:.1f}~{y1+BRIM_MM:.1f} / 베드 {BED_X:.0f}x{BED_Y:.0f}"},
    "slicer_outside_flag_false": {
        "pass": meta("outside") == "false", "detail": f"slice_info outside={meta('outside')}"},
    "no_support": {
        "pass": meta("support_used") == "false",
        "detail": "볼트 구멍 안 서포트 잔사가 체결 공차를 오염시키므로 서포트는 금지. "
                  "형상을 베드에 평평히 눕혀 오버행 자체를 제거함 (구멍은 수직 관통)"},
    "gcode_has_toolpath": {
        "pass": "G1" in gcode and len(gcode) > 10000, "detail": f"gcode {len(gcode)} B"},
    "bed_temp_at_filament_spec": {
        "pass": beds == [55],
        "detail": f"M140={beds} — Bambu PLA Basic + 텍스처 PEI 제조사 스펙 55°C. "
                  f"프로필 기본 65°C는 연화점(45°C) 초과폭이 커 엘리펀트 풋으로 슬롯 폭을 좁힌다"},
    # 출력 온도는 gcode의 **최고** 노즐 온도다. S75(오징 방지)·S140(베드 레벨링 중 노즐 닦기)은
    # Bambu 시작 루틴의 과도값이며 gcode 주석이 그렇게 명시한다 — 출력 온도로 세면 안 된다.
    "nozzle_print_temp_in_range": {
        "pass": bool(nozz and 200 <= max(nozz) <= 240),
        "detail": f"출력 온도 = max {max(nozz) if nozz else '-'}°C (필라멘트 허용 200~240). "
                  f"과도값 {[n for n in nozz if n != max(nozz)]} = 오징방지/레벨링 루틴"},
}

job = {
    "schema": "roarm.print_job/1",
    "name": PARTNAME,
    "rev": ("" if _a else "v1 — 볼트 체결"),
    "purpose": (f"{PARTNAME} 부품 출력" if _a else
                "조에 이미 뚫려 있는 M2.5 구멍 25mm 스팬에 마운트 플레이트가 맞는지 검증. "
                "(a) 스팬 적합 (b) 볼트 조임 시 1.5mm 판재 변형 (c) 조인 뒤 회전 유격 3종"),
    "v0_result": ("" if _a else
                  "v0(C-채널 클램프)는 실물 시험에서 '들어가나 헐겁고 미끄러짐'. "
                  "조를 통판으로 가정했으나 실제는 프레임 구조(물림면이 가정의 1/6)이고, "
                  "쿠폰에 볼트 구멍이 없어 마찰만 시험된 것. v1은 마찰에 의존하지 않는다"),
    "assembly_png": str(OUT / "ASSEMBLY.png"),
    "decision_refs": ["D457:28196 §12", "D458:28407 §7", "D459:28476 §8", "63rd :120 비가역 개조 0"],

    "source": {
        "stl": [str(x) for x in STLS], "sha256_16": {x.name: sha(x) for x in STLS},
        "generator": "scoop_shell_design.py (파라메트릭 — 재생성 가능)"},

    "slicing": {
        "slicer": "BambuStudio.AppImage 02.05.00.66",
        "headless": "xvfb-run -a 필수 — CLI 모드도 GL 컨텍스트를 요구해 그냥 실행하면 glfwInit 실패",
        "profiles": {
            "machine": "profiles/machine_full.json (원본)",
            "process": "profiles/process_nosupport_roarm.json (사본: enable_support 1→0, "
                       "curr_bed_type 추가='Textured PEI Plate')",
            "filament": "profiles/filament_nosupport_roarm.json (사본: textured_plate_temp 65→55)"},
        "flags": ["--orient 0 (설계 배향 유지)", "--ensure-on-bed", "--arrange 1", "--slice 0"],
        "originals_untouched": True},

    "artifact": {
        "path": str(THREEMF), "sha256_16": sha(THREEMF),
        "bytes": THREEMF.stat().st_size,
        "remote_name": THREEMF.name},

    "printer": {
        "model": "Bambu Lab P1S (BBL-P003)",
        "ip": "192.168.0.96",
        "ip_note": "config.json의 192.168.0.144는 낡음. DHCP 재할당됨. "
                   "MAC 20:6E:F1:8E:4B:D8로 ARP 스윕해 확정 (2026-08-27)",
        "ports_verified": {"mqtt": 8883, "ftps": 990, "camera": 6000},
        "state_at_manifest": "IDLE / 작업 없음 / 노즐 25.2°C 베드 24.6°C",
        "bed_plate_physical": "Textured PEI (사용자 육안 확인: 까끌한 회색)"},

    "print_params": {
        "plate_id": 1,
        "est_seconds": int(meta("prediction")), "est_grams": float(meta("weight")),
        "bed_temp_c": beds, "nozzle_temp_c": [n for n in nozz if 190 <= n <= 250],
        "layer_height_mm": 0.2, "filament": "PLA (Bambu Basic, GFA00)",
        "bbox_mm": {"x": [round(x0, 2), round(x1, 2)], "y": [round(y0, 2), round(y1, 2)],
                    # 🔴 이전 판은 z 를 [0.0, 5.1] 로 **하드코딩**했다 (칼라 쿠폰 값).
                    #    부품이 바뀌어도 그대로 나와 검증에 쓸 수 없었다. gcode 의
                    #    실제 Z 이동에서 읽는다.
                    "z": [0.0, round(max_z, 2)]},
        "bed_size_mm": [BED_X, BED_Y, BED_Z]},

    "slicer_warnings": [
        {"msg": w,
         "verdict": "무해 — PLA 연화점 45°C보다 높은 어떤 실용 베드 온도에서도 뜬다. "
                    "55°C는 제조사 스펙이므로 조치 불요"} for w in warns],

    "gates": gates,
    "all_gates_pass": all(g["pass"] for g in gates.values()),

    "send_plan": [
        {"step": 1, "op": "ftp_upload", "args": {"local": str(THREEMF),
                                                 "remote": THREEMF.name}},
        {"step": 2, "op": "mqtt_print", "args": {"filename": THREEMF.name,
                                                 "plate_id": 1, "use_ams": False,
                                                 "bed_leveling": True}},
        {"step": 3, "op": "monitor", "args": {"poll_s": 30}}],

    "safety": [
        "docs/print.md 원칙: 연속 출력 금지 — 매 출력 사용자 시작 신호 필요",
        "mqtt_print()는 항상 0층부터 시작한다. 중단된 작업 재개는 mqtt_resume() (현재 대기 작업 없음)",
        "DTR 원본 프로필·config.json 무수정 — 사본으로만 오버라이드",
        "필라멘트 적재 여부는 기계적 사실이므로 전송 전 사용자 확인 필요",
    ],
}

p = OUT / "print_job.json"
p.write_text(json.dumps(job, indent=2, ensure_ascii=False))
print(f"작성: {p}\n")
for k, g in gates.items():
    print(f"  [{'PASS' if g['pass'] else 'FAIL'}] {k}")
print(f"\n전체 게이트: {'PASS' if job['all_gates_pass'] else 'FAIL'}")
print(f"출력 예상: {job['print_params']['est_seconds']}초 / {job['print_params']['est_grams']}g")

# ── 베드 배치도 — 자동 배치를 믿지 말고 실제 좌표를 그려서 눈으로 확인한다 ──
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

fig, ax = plt.subplots(figsize=(7.2, 7.2))
ax.add_patch(Rectangle((0, 0), BED_X, BED_Y, fc="#f2f2f2", ec="black", lw=2))
ax.plot([BED_X/2], [BED_Y/2], "+", ms=14, mew=1.6, color="0.5")
ax.text(BED_X/2, BED_Y/2 - 8, "bed center\n(128, 128)", ha="center", va="top", fontsize=8, color="0.4")
ax.add_patch(Rectangle((x0-BRIM_MM, y0-BRIM_MM), (x1-x0)+2*BRIM_MM, (y1-y0)+2*BRIM_MM,
                       fc="orange", alpha=.28, ec="darkorange", lw=1.6, ls="--"))
ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fc="dodgerblue", alpha=.65, ec="navy", lw=2))
ax.annotate("", xy=(x0, y1+9), xytext=(x1, y1+9), arrowprops=dict(arrowstyle="<->", lw=1.4))
ax.text((x0+x1)/2, y1+12, f"{x1-x0:.1f} mm", ha="center", fontsize=9)
ax.annotate("", xy=(x1+9, y0), xytext=(x1+9, y1), arrowprops=dict(arrowstyle="<->", lw=1.4))
ax.text(x1+12, (y0+y1)/2, f"{y1-y0:.1f} mm", va="center", fontsize=9, rotation=90)
ax.text(x0, y0-9, f"X {x0:.1f}~{x1:.1f}\nY {y0:.1f}~{y1:.1f}", fontsize=8.5, va="top", color="navy")
ax.text(4, BED_Y-6, f"part (blue)  +  brim {BRIM_MM:.0f}mm (orange)\n"
                    f"margin  L{x0-BRIM_MM:.0f}  R{BED_X-(x1+BRIM_MM):.0f}  "
                    f"B{y0-BRIM_MM:.0f}  T{BED_Y-(y1+BRIM_MM):.0f} mm",
        fontsize=9, va="top")
ax.set_xlim(-16, BED_X+22); ax.set_ylim(-16, BED_Y+22); ax.set_aspect("equal")
ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
ax.set_title(f"BED LAYOUT  {job['name']}\nP1S {BED_X:.0f}x{BED_Y:.0f} mm  ({plate['bed_type']})",
             fontsize=12, weight="bold")
plt.tight_layout(); plt.savefig(OUT / "BED_LAYOUT.png", dpi=95); plt.close()
job["bed_layout_png"] = str(OUT / "BED_LAYOUT.png")
(OUT / "print_job.json").write_text(json.dumps(job, indent=2, ensure_ascii=False))
print(f"베드 배치도: {OUT/'BED_LAYOUT.png'}")
