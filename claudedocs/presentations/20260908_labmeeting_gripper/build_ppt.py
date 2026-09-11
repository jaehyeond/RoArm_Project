"""랩미팅 PPT (2026-09-08) — 1. 그리퍼 교체 (S1 그랩) 섹션. 프로포절 PPT 스타일(흰 배경, 좌상단 번호 제목 + 부제, 우상단 목차 알약, 하단 각주·쪽번호).
python-pptx 1.0.2. 그림은 fig/ 와 repo 정본 폴더에서 직접 읽는다.
"""
import os, json
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

ROOT = '/home/cgxr/Documents/Robotics/RoArm_Project/'
G = ROOT + 'claudedocs/runtime_logs/grab_track/g19_servo_direct/'
F = os.path.dirname(os.path.abspath(__file__)) + '/fig/'   # repo 사본 (원본 빌드는 세션 스크래치패드); 재빌드: ~/miniconda3/bin/python build_ppt.py
OUT = os.path.dirname(os.path.abspath(__file__)) + '/labmeeting_20260908_gripper.pptx'
FONT = 'Noto Sans CJK KR'      # 발표 PC 에서 없으면 맑은 고딕으로 대체됨
DARK = RGBColor(0x22, 0x22, 0x22); GRAY = RGBColor(0x6B, 0x6B, 0x6B); GREEN = RGBColor(0x00, 0xB0, 0x50); BLUE = RGBColor(0x44, 0x72, 0xC4); RED = RGBColor(0xC0, 0x39, 0x2B)
SECTIONS = ['1. 그리퍼 교체', '2. 입자 더미 물리 엔진']

prs = Presentation(); prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
page = [0]


def tx(slide, x, y, w, h, text, size=14, bold=False, color=DARK, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, font=FONT):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    lines = text if isinstance(text, list) else [text]
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.alignment = align
        r = p.add_run(); r.text = ln; r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = color; r.font.name = font
    return tb


def bullets(slide, x, y, w, h, items, size=13, color=DARK, gap=6):
    """items: str 또는 (str, level) 또는 (str, level, color)."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = tb.text_frame; tf.word_wrap = True
    for i, it in enumerate(items):
        if isinstance(it, str): it = (it, 0, color)
        if len(it) == 2: it = (it[0], it[1], color)
        s, lvl, col = it
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.level = lvl; p.space_after = Pt(gap)
        r = p.add_run(); r.text = ('• ' if lvl == 0 else '– ') + s; r.font.size = Pt(size - 1.5 * lvl); r.font.color.rgb = col; r.font.name = FONT
    return tb


def pic(slide, path, x, y, w=None, h=None):
    if not os.path.exists(path): tx(slide, x, y, 3, 0.5, f'[missing] {os.path.basename(path)}', 10, color=RED); return None
    kw = {}
    if w: kw['width'] = Inches(w)
    if h: kw['height'] = Inches(h)
    return slide.shapes.add_picture(path, Inches(x), Inches(y), **kw)


def caption(slide, x, y, w, text): tx(slide, x, y, w, 0.35, text, 9, color=GRAY)


def header(slide, num, title, sub='', active=0):
    tx(slide, 0.5, 0.25, 9.5, 0.6, f'{num}  {title}', 24, bold=True)
    if sub: tx(slide, 0.5, 0.85, 10.5, 0.45, sub, 14, color=GRAY)
    # 목차 알약
    x = 9.6
    for i, s in enumerate(SECTIONS):
        shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(0.35), Inches(1.75), Inches(0.36))
        shp.fill.solid(); shp.fill.fore_color.rgb = GREEN if i == active else RGBColor(0xF2, 0xF2, 0xF2); shp.line.fill.background()
        tf = shp.text_frame; tf.margin_top = tf.margin_bottom = 0; p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
        r = p.add_run(); r.text = s; r.font.size = Pt(9); r.font.name = FONT; r.font.color.rgb = RGBColor(255, 255, 255) if i == active else GRAY
        x += 1.85
    # 구분선
    ln = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(1.3), Inches(12.3), Emu(12000)); ln.fill.solid(); ln.fill.fore_color.rgb = RGBColor(0xDD, 0xDD, 0xDD); ln.line.fill.background()


def footer(slide, note=''):
    page[0] += 1
    if note: tx(slide, 0.5, 7.05, 11.5, 0.35, note, 9, color=GRAY)
    tx(slide, 12.3, 7.05, 0.7, 0.35, str(page[0]), 9, color=GRAY, align=PP_ALIGN.RIGHT)


def new(num, title, sub='', active=0, note=''):
    s = prs.slides.add_slide(BLANK); header(s, num, title, sub, active); footer(s, note); return s


# ── 표지 ──
s = prs.slides.add_slide(BLANK); page[0] += 1
tx(s, 1.0, 2.4, 11.3, 1.2, '실시간 상태 변화 인식을 통한 AI 기반 크레인 운전 자동화', 30, bold=True, align=PP_ALIGN.CENTER)
tx(s, 1.0, 3.5, 11.3, 0.6, '중간 보고 — 1. 그리퍼(그랩) 교체 · 2. 입자 더미 물리 엔진 테스트', 18, color=GRAY, align=PP_ALIGN.CENTER)
tx(s, 1.0, 4.6, 11.3, 0.5, '2026-09-08   V2025108  박재현', 14, align=PP_ALIGN.CENTER)

# ── 목차 ──
s = new('', '목차', note='')
bullets(s, 1.2, 1.8, 10, 4.5, ['1. 그리퍼 교체 — 왜 · 설계 경로 · 순정 인터페이스 · 설계 검증 · 시뮬 · 출력 · 조립 · 서보 규약 · 실물 퍼내기 · 남은 문제',
                              '2. 입자 더미 물리 엔진 테스트 (별도 정리)', '3. 반복 시연 영상'], 16, gap=14)

# ── 1-1 왜 교체하나 ──
s = new('1-1.', '왜 그리퍼를 교체하나', '순정 평행 조(그리퍼)로는 연속 입자(펠릿)를 "퍼낼" 수 없다 → 퍼 올리는 집게(그랩) 형태의 엔드이펙터가 필요', note='근거: 08-16 슬리브 조 시험(D452) — 조를 조일수록 배출, 입자 퍼내기 불가 · 프로포절 슬라이드 3 "그랩 = 원료를 퍼 올리는 집게"')
pic(s, G + 'vendor_step_parts/p44_topology_S1.png', 0.5, 1.6, w=6.4)
bullets(s, 7.2, 1.6, 5.8, 5.2, [
    '대상: PP 펠릿(3~5 mm) 더미를 퍼서 → 이송 → 고정 위치 배출',
    '순정 조(집게 2개 평행)는 입자를 밀어낼 뿐 담지 못함 → 반쪽 보울 2개가 닫히는 클램셸 그랩으로 교체',
    '설계 요구', ('입 개구 58 mm (펠릿 대비 충분), 공동 ≥ 6.4 cm³ (한 번에 ~18 g)', 1), ('툴 자중 ≤ 65 g (팔 페이로드 200 g 의 1/3)', 1), ('순정 그리퍼 서보(ST3215-HS, 1.96 N·m)를 그대로 구동원으로 사용', 1), ('3D 프린터(PLA)로 즉시 제작·수정 가능', 1),
    '경로: g18 양쪽 가동(링크·기어·요크) → S1 한쪽 가동·서보축 직결 (09-03 전환)'], 13)

# ── 1-2 설계 경로 ──
s = new('1-2.', '설계 경로 — g18(양쪽 가동) → S1(한쪽 가동 직결)', '"순정 조 위에 얹는" 제약이 사라지자 링크·기어·요크가 전부 불필요해졌다', note='D462~D479 (g18) → D480 (S1). 사용자 결정: 인쇄로 순정 가동 조를 대체하면 셸을 서보 디스크에 직접 붙일 수 있다')
pic(s, G + 's1_v0/s1_v0_design.png', 0.5, 1.5, w=8.3)
bullets(s, 9.0, 1.6, 4.0, 5.2, [
    'g18: 순정 조 양쪽 위에 셸 2개, 4절 링크 + 기어로 대칭 폐합 — 부품 7종, 자중 58.5 g, 조립 불가 판정 2회',
    'S1: 고정 반쪽 보울(고정 조 바깥면 판) + 문(순정 가동 조 자리에 서보 디스크 직결) — 부품 2 + 관절 1',
    '대가: 비대칭 폐합(문 립이 반경 115 mm 호를 그림)', '이득: 시뮬 = 실물 구조(관절 1개), 나사 9개, 자중 43 g',
    '좌: 닫힘(서보 0°) / 중: 열림 29.3° (입 58 mm) / 우: 3D'], 12)

# ── 1-3 순정 인터페이스 ──
s = new('1-3.', '순정 인터페이스 — 벤더 STEP 분해·실물 대조', '문 뺨은 순정 가동 조와 같은 자리(서보 디스크 2장, M3×4 ×4+4, PCD 14)에 붙는다', note='STEP: files.waveshare.com/wiki/RoArm-M3/RoArm-M3_STEP_260310.zip · 실물 라벨 ST3215-HS(20 kg·cm @12 V) · 고정 조 3구멍 ⌀3.2 관통')
pic(s, G + 's1_v0/p46_s1_hub_assembly.png', 0.5, 1.5, w=8.6)
bullets(s, 9.3, 1.6, 3.8, 5.2, [
    'URDF 에는 나사·디스크가 없다 → 벤더 STEP(부품 745개)에서 조립 관계 추출',
    '디스크: 플랜지 ⌀18.8×2.5, 보스 ⌀10.6 은 서보 쪽 → 뺨 창 ⌀11 은 중앙 나사/축 끝 여유 구멍',
    '고정부: 순정 고정 조 바깥면 3구멍(M3 관통·너트)에 판 체결',
    '실물 대조: 서보 라벨 ST3215-HS 1.96 N·m(데이터시트 추정 2.94 는 부적용), 뺨 1.5 mm·포크 폭 39.4 = STEP 일치'], 12)

# ── 1-4 설계 검증 ──
s = new('1-4.', '설계 검증 — 파라메트릭 생성기와 게이트 10/10', '한 스크립트가 형상·질량·간섭·처짐을 계산하고 STL·URDF·인쇄용 분할까지 생성 (scoop_grab_s1_design.py)', note='스윕 간섭 = 문 표면 표본을 1°~30° 회전시키며 link5·서보·베이스와 최소 거리(KD-트리) · 외팔보 = 측판 2장 레일 단면, E 3 GPa')
pic(s, F + 'gates_table.png', 0.5, 1.5, w=7.4)
bullets(s, 8.2, 1.6, 4.9, 5.2, [
    '입력: 벤더 STEP 뺨 윤곽·디스크 구멍 좌표 · 보울 r 20 mm, 벽 1.6, 폭 36.4 · 뺨/판 두께',
    '출력: 볼록 조각 61개(충돌용) + 정확 윤곽 시각 메시 + URDF/USD(Isaac) + 인쇄 STL 3개',
    '게이트가 못 보는 것도 기록: 조각을 이어 붙인 STL 은 구멍을 보장하지 않는다 → v1 부터 최종 STL 에 레이캐스트로 관통 검사 추가',
    '자중 42.9 g(나사 6.2 포함) · 입 58 mm @ 29.3° · 립 힘 최대 17 N'], 12)

# ── 1-5 시뮬 ──
s = new('1-5.', '시뮬레이션 — Isaac Lab 파지 + 3D 조립 애니메이션', 'URDF 관절 1개(문)로 실물과 같은 구조 · 지름 30 mm 구를 바닥에서 집어 올림 (z 0.015 → 0.163 m)', note='sim_isaaclab_grasp_sphere_s1.py (문 effort_limit 1.96 N·m) · p47_s1_assembly_anim.py (pyrender, 16 s 영상) · 1차 실패 교훈: 물체는 고정 립 하강 경로 밖(문 쪽 r+7 mm)')
pic(s, G + 's1_v0/isaaclab_grasp_sphere/strip_s1.png', 0.5, 1.5, w=7.6)
pic(s, G + 's1_v0/assembly_anim/final_keyframes.png', 0.5, 4.3, w=7.6)
bullets(s, 8.4, 1.6, 4.7, 5.2, [
    '위: Isaac Lab 파지 시퀀스(접근 → 열기 → 하강 → 닫기 → 들기), 구가 보울 중심과 3.2 mm 차로 안착',
    '아래: 조립 애니메이션 키프레임 — 순정 조 나사 8개 풀기 → 조 제거 → 고정부(M3×8+너트 3) → 문 삽입 → M3×4 8개 → 0~30° 개폐',
    '시뮬 한계: 입자 물리 없음(구 1개), 팔 토크는 데모값(8 N·m)'], 12)

# ── 1-6 출력 ──
s = new('1-6.', '3D 출력과 조립 — v0 실패 2건 → v1', 'v0(뺨 3 mm) 출력은 성공했으나 조립에서 두 결함 발견 → v1(뺨 2 mm·구멍 3) 재출력', note='P1S, PLA, 0.2 mm, 저속 프로필 + 트리 서포트(빌드플레이트 전용) · 슬라이스 게이트 13/13 · 챔버 카메라로 첫 층·서포트·완주 확인')
pic(s, F + 'print_L025.jpg', 0.5, 1.5, w=4.0); caption(s, 0.5, 3.8, 4.0, '층 25 챔버 카메라 — 트리 서포트 톱니 능선 (gcode 발자국 폭 10.7 → 0.6 mm 테이퍼로 재확인)')
pic(s, F + 'photo_print_door_scale.jpg', 4.7, 1.5, h=2.6); pic(s, F + 'photo_print_fixed_scale.jpg', 6.75, 1.5, h=2.6)
caption(s, 4.7, 4.15, 4.2, 'v0 실측 자중: 문 9.94/9.92 g, 고정부 15.43 g (gcode 순압출 대비 +6/+2 %)')
bullets(s, 9.1, 1.5, 4.0, 5.4, [
    'v0 조립 실패 ①: 뺨 3 mm 에 순정 M3×4 가 디스크에 1 mm 만 물림 → 뺨 2.0 mm 로(물림 2.0)',
    'v0 조립 실패 ②: 고정부 판 구멍 2/3 — 스파인 조각이 팁 구멍을 메움(게이트·미리보기가 못 잡음) → 한 다각형에서 구멍 3개 뺀 뒤 압출, 최종 STL 레이캐스트 3/3',
    'v1: 2 h 14 min, 32.5 g, 게이트 10/10 · 13/13', '조립(09-05): "깔끔하게 들어감", 문 두 쪽은 순간접착'], 12)
pic(s, F + 'photo_cheek_ruler.jpg', 0.5, 4.25, h=2.7); caption(s, 2.7, 5.2, 4.0, '뺨(문 허브) — 네잎 절개 = 창 ⌀11 + 구멍 ⌀3.4 ×4 (PCD 14)')

# ── 1-7 조립·서보 규약 ──
s = new('1-7.', '조립과 서보 구동 규약', '부팅 = 서보 π(닫힘) · 4구멍 패턴이 서보 각을 π 로 인덱싱 · 토크 상한을 낮춘 뒤 0~30° 만 명령', note='hw_s1_door_servo_probe.py · 펌웨어 원문(D479): 맨 {"T":106} 은 리셋이 아니라 조 118.5° 개방 명령 → 금지 · 손목 피치는 펌웨어가 ±90° 로 클램프(SDK 는 110 통과)')
pic(s, F + 'photo_arm_grab.jpg', 0.5, 1.5, h=3.3); pic(s, F + 'servo_test.png', 5.2, 1.5, w=5.0)
bullets(s, 0.5, 5.0, 12.3, 2.0, [
    '개폐 테스트(토크 200): 10/20/30° 는 ±0.4° 로 도달, 0° 만 2.6° 에서 정지 = 문 립이 고정 립에 맞닿는 기계적 닫힘(볼트 여유 ±1.6° 안) · 걸림 없음(영상)',
    '조립 안전: 문을 닫힘 자세로 달았으면 디스크 4구멍(PCD 14, 90° 대칭)이 서보를 π 로 보장 → 부팅 이동 ≤ 1.6°',
    '운용 규약: T:107 로 토크 상한 200(개방)/900(폐합) · 명령은 SDK 그리퍼 각 0~30° 만 · 개구 > 44 mm 면 손목 롤 |r| ≤ 14°'], 12)
caption(s, 10.4, 1.6, 2.7, '우: 서보 각 명령 vs 읽기')

# ── 1-8 실물 퍼내기 ──
s = new('1-8.', '실물 퍼내기·놓기 — 5회 반복', '관절 제어 + URDF FK 로 립 높이를 계산, "펠릿면에서 열고 2.5 cm 잠긴 뒤 토크 900 으로 닫기" 가 재현 조건', note='hw_s1_scoop_probe.py(FK 사전검사) · hw_s1_manual.py(REPL) · s1_cycle.sh [N] · 높이는 줄자 실측 인자(베이스판 38, 펠릿면 26, 상자 윗단 38.5 cm) · 이동은 올리고(45 cm)→옮기고→내리기')
pic(s, F + 'photo_setup_box.jpg', 0.5, 1.5, h=2.55); pic(s, F + 'cycle5_timeline.png', 4.0, 1.45, w=6.2); pic(s, F + 'cycle5_table.png', 10.2, 1.5, w=2.9)
bullets(s, 0.5, 4.2, 12.3, 2.8, [
    '시퀀스: 더미 위 45 cm → 닫힌 채 펠릿면(26)까지 5 cm 단계 하강 → 문 30° → 2.5 cm 잠김 → 닫기(900, 3.6° 초과면 8° 열었다 재닫기) → 펠릿면+8 에서 재닫기 → 45 cm → 툴 세움(P1) → 베이스 +90° → 뻗어 26 cm → open/close → 복귀',
    '진화: v1 허공 개폐(베이스 높이 오판) → v2 닫힘 5.7°(펠릿 끼임) 유출 → v3 토크 900·잠김 2.5 → 닫힘 2.5° "거의 유지" → 5회 반복 5/5, 회당 93 s, 정지 0',
    '적재 신호: 어깨 서보 부하 277~285(빈 하강 133) · 닫힘 2.8~3.5°(기계 2.5) · 정량 계량은 다음 단계'], 12)

# ── 1-9 남은 문제 ──
s = new('1-9.', '남은 문제와 다음 단계', '그리퍼는 수단이 갖춰진 상태 — 다음은 "어디를 퍼면 얼마나 잡히나" 데이터', note='D481 미해결 항목 · 프로포절 해결 과제 ① 양+형상 예측 ② 다음 위치 결정 ③ sim→real GP 보정')
bullets(s, 0.7, 1.7, 6.0, 5.2, [
    ('남은 문제', 0, RED), ('리프트 중 문 1~1.6° 되열림 → 립 틈 3 mm (재닫힘 명령으로 완화)', 1), ('펠릿 끼임(5.7°) 은 토크·깊이로 회피 중 — 기구적 해결 아님', 1), ('문 두 쪽 접착만(타이볼트 미장착)', 1), ('회당 적재 질량 미계량(설계 17.6 g)', 1)], 13)
bullets(s, 7.0, 1.7, 6.0, 5.2, [
    ('다음 단계', 0, GREEN), ('① 회당 적재 질량 n≥5 계량 → 평균·편차 (양 예측의 첫 실측)', 1), ('② s1_v2: 전단(가위) 립 — 문 립이 고정 립 안쪽 2~3 mm 겹침 → 끼임·되열림 제거', 1), ('③ Kinect 높이맵으로 더미 형상 입력 → 퍼낼 위치 결정 루프 시작', 1), ('④ 물리 엔진(DEME) 입자 더미와 실물 대조 (2장)', 1)], 13)

# ── 2. 물리 엔진 (자리) ──
s = new('2.', '입자 더미 물리 엔진 테스트', '(별도 PC 에서 정리 — 이 파일에는 자리만)', active=1)
tx(s, 0.7, 2.5, 12, 1, '여기에 물리 엔진 테스트 슬라이드를 넣는다.', 16, color=GRAY)

# ── 3. 영상 ──
s = new('3.', '반복 시연 영상', '퍼내기 → 이송 → 놓기 반복 (09-07 촬영)', active=1, note='영상 파일: 사용자 촬영본 삽입 (이 슬라이드의 스크린샷 자리에 동영상 삽입)')
pic(s, F + 'scoop_video_sheet.jpg', 2.4, 1.5, h=5.4); caption(s, 2.4, 6.9, 8.5, '자리표시: v1 시행 프레임 시트 — 발표 시 촬영 영상으로 교체')

prs.save(OUT); print('saved', OUT, 'slides', len(prs.slides))
