"""3D 출력 전송 매니페스트 생성기 — 슬라이스 산출물을 검증하고 print_job.json으로 굳힌다.

왜 JSON인가: 업로드·출력 명령을 손으로 치면 그때그때 다른 값이 들어간다. 무엇을 어느
프린터에 어떤 검증을 통과한 상태로 보냈는지가 남지 않는다. 이 파일이 전송의 단일 소스이며
`send_print_job.py`가 이것만 읽고 실행한다 — 매니페스트에 없는 것은 보내지 않는다.

사용:  python make_print_job.py [3mf경로] [출력디렉터리] [부품이름] [STL...]
       인자를 안 주면 기존 칼라 쿠폰 기본값을 쓴다 (하위호환).
"""
import json, hashlib, math, re, zipfile, subprocess, sys
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
# 🔴 2026-09-01. 게이트는 **내가 넣으려던 값이 아니라 슬라이서가 실제로 쓴 값**을 봐야 한다.
#    이전 판은 프로필(`process_nosupport_roarm.json`)에서 brim_width 를 읽어 8 이라고
#    보고했는데, 3mf 안 project_settings.config 의 실제 값은 **0** 이었다.
#    원인: 프로필 JSON 은 모든 값이 문자열인데 brim_width 만 정수 `8` 로 적혀 있어
#    BambuStudio 가 조용히 무시하고 기본값 0 으로 떨어뜨렸다 (오류 메시지 없음).
#    결과: 브림 0개인 gcode 로 부품 4개를 접지 1.4 cm² 에 세워 출력 -> 전량 탈락.
_proc = json.loads((DTR / "profiles/process_nosupport_roarm.json").read_text())
_used = json.loads(z.read("Metadata/project_settings.config").decode())
brim_type = _used.get("brim_type")
brim_width = float(_used.get("brim_width") or 0)
brim_intended = _proc.get("brim_width")
_zs = [float(v) for v in re.findall(r"^G[01] .*?Z([0-9.]+)", gcode, re.M)]
max_z = max(_zs) if _zs else 0.0

# 브림이 설정에만 있고 툴패스에 없을 수 있다 -> gcode 에서 직접 센다.
brim_moves = len(re.findall(r"^; FEATURE: Brim\s*$", gcode, re.M))

FIL_AREA_MM2 = 3.14159265 * (1.75 / 2) ** 2   # 1.75 mm 필라멘트 단면


def first_layer_contact_mm2(gc):
    """1층이 베드에 실제로 깔아 놓는 면적 (mm²). 브림 포함 — 브림도 부품을 잡아 준다.

    압출 필라멘트 부피 / 층 높이 = 바닥에 닿는 면적. 설정이 아니라 **툴패스**를 적분한다.
    """
    lines = gc.splitlines()
    try:
        s = next(i for i, l in enumerate(lines) if "layer num/total_layer_count: 1/" in l)
    except StopIteration:
        return 0.0, 0.0
    e = next((i for i, l in enumerate(lines[s + 1:], s + 1)
              if l.startswith("; CHANGE_LAYER")), len(lines))
    h = float((re.search(r"^; LAYER_HEIGHT: ([0-9.]+)", "\n".join(lines[:s]), re.M)
               or [None, "0.2"])[1])
    tot = 0.0
    for l in lines[s:e]:
        if not l.startswith("G1"):
            continue
        m = re.search(r"\bE(-?[0-9.]+)", l)
        if m and float(m.group(1)) > 0:
            tot += float(m.group(1))
    return tot * FIL_AREA_MM2 / h, tot


contact_mm2, first_layer_e = first_layer_contact_mm2(gcode)

# 🔴 서포트·오버행·기능축. 전부 **결과**에서 읽는다.
support_used = (meta("support_used") == "true") or ("; FEATURE: Support" in gcode)
on_plate_only = str(_used.get("support_on_build_plate_only", "0")) in ("1", "true")
OVERHANG_FREE_MM2 = 300.0     # 이 아래면 서포트 없이도 실물이 버틴다(잠정 — 성공 사례로 교정할 것)

# 오버행·기능축은 슬라이스가 아니라 **출력 배향 STL** 에서 잰다.
# orient_for_print.py 가 같은 정의로 계산해 orientation.json 에 적어 둔다.
overhang_mm2, func_tilt = float("nan"), float("nan")
for _s in STLS:
    _o = _s.parent / "orientation.json"
    if _o.exists():
        for _r in json.loads(_o.read_text())["parts"]:
            if _r["stl"] == _s.name:
                overhang_mm2 = _r["after"]["overhang"]
                func_tilt = _r["after"]["func_tilt_deg"]
FUNC_TILT_MAX = 10.0


def _support_top_z(gc):
    """서포트가 실제로 도달한 최고 높이 (mm). 설정이 아니라 툴패스에서 읽는다."""
    z, top = 0.0, 0.0
    for l in gc.splitlines():
        m = re.match(r"^; Z_HEIGHT: ([0-9.]+)", l)
        if m:
            z = float(m.group(1))
        elif l.startswith("; FEATURE: Support"):
            top = z
    return top


support_top_z = _support_top_z(gcode)

def _support_in_zone(gc, z_lo):
    """z_lo 위(=기능면 높이)에 실제로 들어간 서포트 압출 점 수.

    설정이 아니라 툴패스를 센다. 같은 설정이라도 배향에 따라 결과가 정반대다.
    """
    n, z, feat = 0, 0.0, ""
    for l in gc.splitlines():
        m = re.match(r"^; Z_HEIGHT: ([0-9.]+)", l)
        if m:
            z = float(m.group(1)); continue
        m2 = re.match(r"^; FEATURE: (.+)$", l)
        if m2:
            feat = m2.group(1).strip(); continue
        if feat.startswith("Support") and l.startswith("G1") and z > z_lo:
            if re.search(r"\bE([0-9.]+)", l):
                n += 1
    return n


# 🔴 2026-09-01 4차. 게이트를 8개 만들면서 **속도는 한 번도 안 봤다.**
#    DK 원본은 Bambu 고속 프로필(내벽 400 mm/s · 가속 10,000 mm/s²)이고 큰 단순
#    부품용이다. 작은 정밀 부품에서는 ① 실 무더기 ② 원형 피처가 다각형 ③ 층간
#    접착 저하(최소 단면 104 mm² 급소가 손으로 부러졌다)로 전부 나타났다.
#    설정이 아니라 **gcode 의 실제 압출 이송속도**를 읽는다.
def _max_extrude_feed(gc):
    """압출을 동반한 이동의 최대 이송속도 (mm/s). F 는 mm/min 이다."""
    best, cur = 0.0, 0.0
    for l in gc.splitlines():
        if not l.startswith("G1"):
            continue
        mf = re.search(r"\bF([0-9.]+)", l)
        if mf:
            cur = float(mf.group(1))
        me = re.search(r"\bE(-?[0-9.]+)", l)
        if me and float(me.group(1)) > 0 and re.search(r"\b[XY]", l):
            best = max(best, cur)
    return best / 60.0


max_feed_mm_s = _max_extrude_feed(gcode)
FEED_MAX_MM_S = 130.0   # 잠정. 400 mm/s 판이 실패했고 90 mm/s 판이 첫 대조군이다

# 기능면(기어) 높이 위에 서포트가 실제로 들어갔는지 — 설정이 아니라 툴패스로.
# 같은 설정이라도 배향에 따라 정반대다: 기어가 바닥이면 1,122점, 꼭대기면 0점.
support_in_gear_zone = _support_in_zone(gcode, support_top_z) if support_used else 0
# 서포트가 못 미치는 높이에 남은 오버행 면적 — STL 에서 직접 잰다
unsupported_mm2, overhang_top_z = overhang_mm2, float("nan")
try:
    # ⚠️ 이 블록의 지역 변수는 모듈 상단 이름과 겹치면 안 된다.
    #    `_a` 를 면적 배열로 쓴 첫 판이 argv(`_a`)를 덮어써서 죽었다.
    import trimesh as _tm
    _mesh = _tm.load(STLS[0])
    _nrm, _cen, _area = _mesh.face_normals, _mesh.triangles.mean(axis=1), _mesh.area_faces
    _ov = _nrm[:, 2] < -math.cos(math.radians(45.0))
    if _ov.any():
        overhang_top_z = float(_cen[_ov][:, 2].max())
        # 🔴 2026-09-01 재정정 (3번째). 정상적으로 받쳐진 오버행은 **항상**
        #    서포트 최고점보다 위에 있다 — 슬라이서가 support_top_z_distance 만큼
        #    일부러 띄워 놓기 때문이다(떼기 쉽게). 그 간격을 허용하지 않으면
        #    **서포트가 잘 붙을수록 FAIL 을 내는** 게이트가 된다.
        #    실제로 g9 에서 오버행 53.0 · 서포트 52.8 = 정확히 0.2 mm 간격인데
        #    522 mm2 를 무방비로 오판했다.
        _zgap = float(_used.get("support_top_z_distance", 0.2) or 0.2)
        unsupported_mm2 = float(
            _area[_ov & (_cen[:, 2] > support_top_z + _zgap + 1e-6)].sum())
except Exception:
    pass
# 🔴 잠정 임계. 근거는 **실패 1건뿐**이고 성공 사례가 아직 없다 (09-01 기준).
#    실패판 = 138 mm² / 높이 52.9 mm = 2.6 mm²/mm -> 첫 층부터 전량 탈락.
#    첫 성공이 나오면 그 값으로 다시 교정할 것. 지금은 안전측으로 10 을 쓴다.
CONTACT_PER_MM = 10.0
contact_need = CONTACT_PER_MM * max_z

# ── 검증 게이트 ───────────────────────────────────────────────────────────
gates = {
    "inside_bed_with_brim": {
        "pass": bool(x0 - BRIM_MM >= 0 and y0 - BRIM_MM >= 0
                     and x1 + BRIM_MM <= BED_X and y1 + BRIM_MM <= BED_Y),
        "detail": f"브림 {BRIM_MM}mm 포함 X {x0-BRIM_MM:.1f}~{x1+BRIM_MM:.1f} "
                  f"Y {y0-BRIM_MM:.1f}~{y1+BRIM_MM:.1f} / 베드 {BED_X:.0f}x{BED_Y:.0f}"},
    "slicer_outside_flag_false": {
        "pass": meta("outside") == "false", "detail": f"slice_info outside={meta('outside')}"},
    # 🔴 2026-09-01 2차 실패로 교체. 이전 게이트는 `support_used == false` 만 봤다.
    #    서포트를 꺼 두면 **항상 통과한다** — 그 형상이 서포트 없이 뽑을 수 있는지는
    #    묻지 않았다. 그래서 오버행 4,892 mm² 를 서포트 0 으로 뽑아 전부 늘어뜨렸다.
    #    이제 묻는 것은 "서포트를 안 썼나"가 아니라 **"늘어질 면이 받쳐졌나"** 다.
    # 🔴 2026-09-01 재정정. 첫 판은 `support_used` 만 봤고 그래서 **통과했다** —
    #    서포트는 z=7.40 mm 에서 끊겼는데 오버행은 57.5 mm 까지 있었다(29% 무방비).
    #    `support_on_build_plate_only=1` 은 모델 위에서 자라지 않으므로 당연한 결과였다.
    #    "서포트가 쓰였나"가 아니라 **"오버행 높이까지 덮나"** 를 묻는다.
    "overhang_supported": {
        "pass": (overhang_mm2 <= OVERHANG_FREE_MM2)
                or (unsupported_mm2 <= OVERHANG_FREE_MM2),
        "detail": (f"45° 초과 오버행 {overhang_mm2:.0f} mm². "
                   f"서포트 도달 높이 {support_top_z:.2f} mm · 오버행 최고 {overhang_top_z:.1f} mm "
                   f"→ 무방비 {unsupported_mm2:.0f} mm² (허용 {OVERHANG_FREE_MM2:.0f}). "
                   f"1차 실패판 4892 mm² 무방비 / g7 판 243 mm² 무방비"),
        "blind_spot": ("높이로만 판정한다. 서포트가 그 높이에서 **평면상 그 자리에** "
                       "있는지는 안 본다. 그리고 임계 300 mm² 는 아직 성공 사례가 "
                       "없는 잠정값이다 — g7 실물(243 mm² 무방비)이 첫 교정 데이터다")},
    # 🔴 2026-09-01 정정 (사용자 승인). 이전 판은 `on_build_plate_only` **설정값**만
    #    봤다. 그러나 오염 위험은 설정이 아니라 **서포트가 기능면 높이에 실제로
    #    들어갔는가**로 정해진다. 기어가 바닥이던 배향에서는 z<6 mm 에 1,122점이
    #    들어갔지만, 기어가 꼭대기인 g9 배향에서는 이빨 구간 **0점**이다.
    #    -> 설정이 아니라 **결과**를 본다. 오늘 아홉 번 반복된 그 교훈이다.
    "support_stays_off_model": {
        "pass": (not support_used) or on_plate_only or (support_in_gear_zone == 0),
        "detail": (f"support_on_build_plate_only={_used.get('support_on_build_plate_only')!r} · "
                   f"기능면(기어) 구간 서포트 {support_in_gear_zone}점. "
                   f"모델 위에서 자라면 볼트 구멍·기어 사이에 잔사가 남아 "
                   f"체결 공차를 오염시킨다 — no_support 금지의 **원래 이유**가 그것이었다")},
    "gcode_has_toolpath": {
        "pass": "G1" in gcode and len(gcode) > 10000, "detail": f"gcode {len(gcode)} B"},
    # 🔴 2026-09-01 정정. 이전 판은 `beds == [55]` 로 **55 를 유일 정답으로 못 박고** 있었다.
    #    그 값은 5.1 mm 납작한 칼라 쿠폰의 엘리펀트 풋을 막으려던 것인데, 높이 59.5 mm 부품에
    #    그대로 적용해 **160/295 층에서 탈락 스파게티**를 냈다. DTR README:192·227 에 이미
    #    "베드 접착 실패 -> PLA 는 55°C 이상 필요" 가 기록돼 있었고 그것을 답습한 것이다.
    #    -> 단일값이 아니라 **부품 높이에 따른 대역**으로 판정한다.
    "bed_temp_vs_part_height": {
        "pass": (all(55 <= b <= 70 for b in beds) and
                 (max_z < 20.0 or all(b >= 60 for b in beds))),
        "detail": (f"M140={beds}, 부품 높이 {max_z:.1f} mm. "
                   f"PLA 최소 55°C(DTR README:227). 높이 20 mm 초과 부품은 냉각 수축으로 "
                   f"모서리가 들리므로 **60°C 이상** 필요. 20 mm 이하 납작한 부품만 55°C 허용"),
        "lesson": "쿠폰 설정을 세로로 긴 부품에 재사용하지 말 것 (09-01 스파게티 실패)"},
    "brim_enabled": {
        "pass": brim_type not in (None, "no_brim", "none") and brim_width > 0,
        "detail": (f"슬라이스에 실제 적용된 값: brim_type={brim_type!r} brim_width={brim_width} "
                   f"(프로필이 넣으려던 값 {brim_intended!r}). "
                   f"🔴 두 번 뚫렸다 — 1차는 brim_type=None 인데 width=5(타입이 꺼져 폭이 무의미), "
                   f"2차는 type=outer_only 인데 width=0(폭이 0). **둘 다 봐야 한다.**"),
        "blind_spot": "설정값만 본다. 툴패스 존재는 brim_in_gcode 가 본다"},
    # 🔴 설정이 맞아도 슬라이서가 브림을 안 뽑을 수 있다. 결과를 직접 센다.
    "brim_in_gcode": {
        "pass": brim_moves > 0,
        "detail": f"gcode 안 '; FEATURE: Brim' 구간 {brim_moves}개. "
                  f"2차 실패판은 brim_type=outer_only 인데 이 값이 **0** 이었다 — "
                  f"설정 게이트는 통과시켰고 부품은 전량 떨어졌다.",
        "blind_spot": "존재만 본다. 브림이 충분히 넓은지는 first_layer_contact_area 가 본다"},
    # 🔴 이번 실패를 유일하게 예측할 수 있었던 수치. 설정이 아니라 1층 툴패스를 적분한다.
    "first_layer_contact_area": {
        "pass": contact_mm2 >= contact_need,
        "detail": (f"1층 접지 {contact_mm2:.0f} mm² (압출 {first_layer_e:.1f} mm, 브림 포함) "
                   f"vs 필요 {contact_need:.0f} mm² = {CONTACT_PER_MM:.0f} × 높이 {max_z:.1f} mm. "
                   f"실패판은 {138} mm² / 52.9 mm = 2.6 mm²/mm 였다"),
        "blind_spot": ("임계 10 mm²/mm 는 **실패 1건에서만** 잡은 잠정값이고 성공 사례가 "
                       "아직 없다. 접지가 넓어도 베드 오염·Z 오프셋·필라멘트 습기는 못 본다")},
    # 출력 온도는 gcode의 **최고** 노즐 온도다. S75(오징 방지)·S140(베드 레벨링 중 노즐 닦기)은
    # Bambu 시작 루틴의 과도값이며 gcode 주석이 그렇게 명시한다 — 출력 온도로 세면 안 된다.
    # 🔴 2026-09-01 2차 실패. 접지만 최대화한 배향이 기어축을 90° 눕혀 이빨을 층으로
    #    쌓았다. 형상도 강도도 무너진다(층간 접착 방향으로 부러짐). 힌지축에는 기어·
    #    피벗보스·핀·로드아이·셸크랭크허브가 전부 동축이라, 이 축 하나로 다 걸린다.
    "functional_axis_vertical": {
        "pass": (func_tilt != func_tilt) or func_tilt <= FUNC_TILT_MAX,   # NaN = 해당없음
        "detail": (f"힌지축 기울기 {func_tilt:.1f}° (허용 {FUNC_TILT_MAX:.0f}°). "
                   f"0° = 이빨이 면내로 찍히고 모든 보어가 진원. "
                   f"2차 실패판은 90° 였다 — 접지만 보고 골라서 그렇게 됐다"),
        "blind_spot": "축 방향만 본다. 이빨 모듈·백래시가 실물에서 맞물리는지는 조립해야 안다"},
    # 🔴 작은 정밀 부품(기어 이빨 모듈 1.0 · 보어 ⌀3 · 최소 단면 104 mm²)에 고속
    #    프로필을 쓰면 실·다각형·층간 약화가 한꺼번에 온다. 실물 3판에서 전부 났다.
    "speed_for_precision": {
        "pass": max_feed_mm_s <= FEED_MAX_MM_S,
        "detail": (f"gcode 최대 압출 이송 {max_feed_mm_s:.0f} mm/s (허용 {FEED_MAX_MM_S:.0f}). "
                   f"DK 원본 고속 프로필은 내벽 400 · 가속 10,000 이라 "
                   f"실 무더기 · 원형 피처 다각형 · 층간 접착 저하를 동시에 낸다"),
        "blind_spot": ("최고 속도만 본다. 가속도·저크·리트랙션은 안 본다 — "
                       "실 무더기의 직접 원인은 리트랙션 설정일 수도 있다. "
                       "그리고 130 mm/s 임계는 아직 성공 사례가 없는 잠정값이다")},
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
