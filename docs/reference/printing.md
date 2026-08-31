# 3D 출력 레퍼런스 — P1S 파이프라인 (자동 로드 아님. 출력 착수 전에 읽을 것)

> 출처 = `~/Documents/DK/DTR/bamboo-3dprinter` (SO-ARM101 용으로 구축된 headless P1S CLI 스택)
> 를 RoArm 부품 출력에 재사용하며 실측한 것 + 이 repo 의 실패 기록.
> **DK 폴더는 남의 프로젝트다.** 원본(`config.json`, `profiles/*_full.json`, `print_cli.py`)은
> 수정하지 않는다 — 사본으로만 오버라이드한다.
>
> 🔴 **이 파일이 존재하는 이유**: DK README 에 이미 적혀 있던 실패를 우리가 그대로 답습해
> 2026-09-01 에 160/295 층에서 스파게티를 냈다. 남의 폴더에 있는 교훈은 안 읽힌다.
> **우리 폴더에 옮겨 적어야 다음에 안 밟는다.**

## 1. 🔴 답습하지 말 것 — 이미 한 번씩 당한 것들

| 함정 | 증상 | 대책 | 출처 |
|---|---|---|---|
| **베드 온도가 낮다** | 층이 올라가다 부품이 탈락 → 노즐이 허공에 압출(스파게티) | PLA 는 **최소 55 °C**. 높이 20 mm 넘는 부품은 **60 °C 이상** | DK README:192·227 → 우리가 답습(09-01) |
| **브림이 꺼져 있다** | `brim_width` 값은 있는데 접착 면적이 안 늘어남 | `brim_type` 이 `None`/`no_brim` 이면 폭 설정은 **무의미**하다. 반드시 `outer_only` 등으로 켤 것 | 09-01 실측 |
| **쿠폰 설정을 큰 부품에 재사용** | 위 둘의 실제 원인 | 납작한 시험쿠폰용 값(엘리펀트 풋 방지로 낮춘 온도)을 **세로로 긴 부품에 쓰지 마라** | D459 §5 → 09-01 실패 |
| **`xvfb-run` 만으로 부족** | `glfwInit return error, code 65544` / `Wayland: Failed to connect` | `env -u WAYLAND_DISPLAY GDK_BACKEND=x11 QT_QPA_PLATFORM=xcb xvfb-run -a ...`. ⚠️ 에러는 계속 뜨지만 **CLI 슬라이싱은 정상 완료**된다 (GUI 초기화 실패일 뿐) | D459 §2 + 09-01 보강 |
| **프린터 IP 가 DHCP 로 이동** | `config.json` 의 IP 무응답 | MAC `20:6E:F1:8E:4B:D8` 로 ARP 스윕. 2026-08-27 기준 **192.168.0.96**. `config.json` 은 DK 소유라 무수정, 메모리에서만 오버라이드 | D459 §1 |
| **`curr_bed_type` 미설정** | BambuStudio 가 `Cool Plate` 로 오판 | 사본 프로필에 `"curr_bed_type": "Textured PEI Plate"` 명시. 실물은 텍스처 PEI(까끌한 회색) | D459 §4 |
| **`paho-mqtt` 부재** | 전송 스크립트가 ImportError | **base conda 에만 있다.** `~/miniconda3/bin/python` 으로 실행할 것 (`isaaclab`/`roarm` 환경엔 없다) | D459 §7 |
| **`--orient 1`(자동 배향)이 부품을 눕혀준다는 착각** | 오히려 높이가 늘 수 있다 (59.5 → 60.7 mm 실측) | 자동 배향을 믿지 말고 **가장 얇은 축을 직접 Z 로 회전**시켜 STL 을 다시 뽑아라 | 09-01 실측 |
| **매니페스트 하드코딩** | 부품이 바뀌어도 쿠폰 값이 그대로 기록됨 | `bbox z=[0.0,5.1]` 가 박혀 있어 실제 59.5 mm 를 12배 틀리게 적을 뻔했다. gcode 의 실제 Z 이동에서 읽을 것 | 09-01 실측 |
| **PAUSE 작업은 전원 사이클로 소실된다** | `mqtt_resume()` 대상이 없음 | 2026-05-27 PAUSE 작업(layer 199/415)이 사라졌다. 재개 불가 → 0 층부터 재출력 | D459 §3 |
| **`gcode_state` 를 현재 상태로 읽는다** | 대기 중인데 전송이 막힌다 | 그건 **직전 작업의 결말**이다. 현재는 `print_type`/`hms`/`print_error` 로 본다 | 09-01 실측 |
| **P1S 는 탈락을 감지 못 한다** | 스파게티 중에도 `print_error: null`, `state: RUNNING` | **기계를 믿지 말고 눈으로 볼 것.** 첫 층과 20 mm 지점을 확인하라 | 09-01 실측 |
| **FTP 업로드 경로** | `/cache/` 에 올리면 출력이 시작 안 됨 | 루트 `/{filename}` 에 올린다 (printer.py 주석 명시) | D459 §7 |

## 2. 게이트 — 무엇을 못 보는지가 중요하다

`make_print_job.py` 슬라이스 게이트 (전송 전 자동 검사):
```
inside_bed_with_brim        브림 포함 베드 안
slicer_outside_flag_false   슬라이서 자체 판정
no_support                  서포트 잔사가 측정 공차를 오염시킨다
gcode_has_toolpath          빈 gcode 방지
bed_temp_vs_part_height     🆕 높이 20 mm 초과 -> 60 °C 이상 (09-01 신설)
brim_enabled                🆕 brim_type 이 실제로 켜져 있나 (09-01 신설)
nozzle_print_temp_in_range  최고 온도만 검사 (S75 오징방지·S140 노즐닦기는 시작 루틴 과도값)
```
`send_print_job.py` 전송 게이트:
```
all_gates_pass · 3mf 해시 일치 · 프린터 도달 · 안전 상태(IDLE|FINISH) · --yes 명시
```

🔴 **게이트 사각지대가 세 번 연속 났다**:
```
D461 §6   수치 게이트가 공간 배치 오류를 못 잡았다 (두 셸이 겹쳐 있어도 PASS)
D464 §2   볼록 게이트가 퇴화 기하를 못 잡았다 (부피 0 조각 8개가 PASS)
09-01     "브림 폭 포함 베드 안"만 보고 브림이 켜져 있는지는 안 봤다
09-01     프린터 "직전 작업의 결말"만 보고 "지금 가동 중인지"는 안 봤다
```
🔴 **`gcode_state` 는 과거형이다.** 새 작업이 들어오기 전까지 직전 결말(FAILED/FINISH)이
계속 걸려 있다. 프린터가 실제로 대기 중인데 전송이 잘못 차단됐다(09-01, 사용자가 실물을 보고 지적).
현재 가동 여부는 **`print_type` · `hms` · `print_error`** 로 판정한다:
```
gcode_state FAILED · print_type idle · hms [] · print_error 0 · task_id '0'   -> 전송 가능
```
→ **게이트를 추가할 때마다 "이 게이트가 못 보는 것은 무엇인가"를 적어라.**
→ **FAIL 을 못 내는 게이트는 게이트가 아니다.** 회귀를 주입해 실제로 FAIL 이 나는지 확인하라.

## 3. 표준 절차

```bash
# 1) 눕히기 — 가장 얇은 축을 Z 로 (자동 배향 믿지 말 것)
#    새 폴더에 뽑는다 (forward-only: 기존 STL 은 보존)

# 2) 슬라이스
cd ~/Documents/DK/DTR/bamboo-3dprinter
env -u WAYLAND_DISPLAY GDK_BACKEND=x11 QT_QPA_PLATFORM=xcb xvfb-run -a \
  ~/Applications/BambuStudio.AppImage \
  --load-settings "profiles/machine_full.json;profiles/process_nosupport_roarm.json" \
  --load-filaments "profiles/filament_nosupport_roarm.json" \
  --orient 0 --ensure-on-bed --arrange 1 --slice 0 \
  --export-3mf output/<name>.3mf  <STL...>

# 3) 매니페스트 + 게이트
python make_print_job.py <3mf> <출력디렉터리> <부품명> <STL...>

# 4) 전송 (⚠️ base conda 파이썬)
~/miniconda3/bin/python send_print_job.py <출력디렉터리>/print_job.json --yes

# 5) 상태 조회
cd ~/Documents/DK/DTR/bamboo-3dprinter && ~/miniconda3/bin/python -c \
 "import printer;c=printer.load_config();c['printer']['ip']='192.168.0.96';\
  p=printer.P1SPrinter(config=c);p.mqtt_connect();p.mqtt_pushall(wait=8);print(p.mqtt_status(wait=6))"
```

## 4. 안전 규약

- **연속 출력 금지.** 매 출력마다 사용자 시작 신호(`--yes`)가 필요하다.
- **필라멘트 적재 여부와 베드 청소 상태는 기계적 사실이다.** 전송 전 사용자에게 확인받아라.
- 프린터가 `FAILED` 면 전송이 막힌다. **원인을 확인하고 화면에서 해제**한 뒤 재전송한다.
- DK 원본 파일 무수정. 사본(`*_nosupport_roarm.json`)으로만 오버라이드한다.
