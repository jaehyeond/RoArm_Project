# 60th — g0c_d446 `bg1v`: B601 성공 파지 상태-복원 렌더 2장 (물리 0, 재판정 0)

날짜: 2026-08-13 (59th 종료 직후 재개 세션)
이번 case의 신규 변수: **없음** — case 변경 없음. `g0c_d446` 내 시각화 전용
forward-only 패스(신규 태그 `bg1v_*`). 물리 스텝 0, bg1/fg1 산출물 편집 0,
D427~D446 재판정 0, git 커밋 0, 로봇 0.

## 0. 배경 — 사용자 질문과 승인

- 부트 검증: HEAD == origin/master == `9cbd959`, dirty 목록 기대치 정확 일치,
  `g0c_d446/` 17개 산출물 완비(`bg1_failure.json` 부재 = 정상 종료).
- 사용자 질문: "구매 요청 전에 — 이거 쥐는거 성공한 렌더링한게 있어? sim으로."
- 답: **없었음.** bg1은 headless 직접 물리(렌더 0)이고, `bg1_inspection.png`의
  3D 패널(패널 2)은 기본 카메라 프레이밍이 mm 스케일을 벗어나 판독 불가로
  기록된 상태(59th doc §6). RRD는 점군 마커 기반이라 보고용 그림 아님.
- 사용자 승인: **"진행해 — side 1장 + θ=0 top-down 1장"** → bg1v 착수.

## 1. 방법 — 상태 복원 렌더 (`sim_scripts/p19_g0c_bg1v_b601_grasp_render_snapshot.py`)

- 소스 = `bg1_results.json`의 `pose_snaps` **rows 13(`side_phi000`)·21(`top_theta00`)**
  post-CLOSE 시점 world pose(팜/좌우 핑거/물체) verbatim. 물리 재실행 없음.
- preflight 게이트: numpy/psutil 핀(D326) / results sha16 `cb88c549dc459272` +
  88,723 B 일치 / 변형 B USD sha `dbd86576…` = split2 audit 일치 / row 선택 게이트
  (variant==B AND success==true) / write guard(PNG·meta 기존재 시 abort).
- 씬: 변형 B USD 참조 + 받침대/원통(post-CLOSE pose) 저작, 충돌·split2 메쉬
  4개 렌더 숨김, dome+distant light, BBox 자동 프레이밍 카메라(1920×1080).
- 복원 게이트: 저작 후 합성 world 변환 vs 기록값 — **편차 0.0** (허용 1e-9).
- 캡처: `rep.orchestrator.step(rt_subframes=32)` 후 rgb annotator, 이미지
  비평탄 게이트(std/mean), 종료 핀 재검증(results/USD 미변조 확인) PASS.

## 2. 자진 신고 — 시행착오 3건 (각각 진단 후 수정, 부분 산출물 0)

1. **1·2차 침묵 사망**: exit=0·stderr 공백·산출물 0. 원인 = Isaac Sim 5.1
   `SimulationApp.close()`가 예외를 삼키고 프로세스를 exit 0으로 종료 —
   p19 초판에 p18의 `except BaseException` 캡처를 빠뜨림. 캡처+rc stdout 마커
   추가 후 실제 오류 표면화. **exit code로 Isaac 스크립트 성공 판정 금지.**
2. **3차 `POSE_RESTORE_GATE dev=1.0000073`**: 참조된 rig의 원본 xform op가
   `AddTransformOp`(+SetResetXformStack)만으로는 잔존 합성됨. 수정 =
   `SetXformOpOrder([내 op], resetXformStack=True)`로 명시 배제.
3. **4차 `IMAGE_SHAPE (0,)`**: `app.update()` 80회로도 annotator 빈 배열 —
   Replicator 캡처는 `rep.orchestrator.step()` 필요.
- 5차 = 본 실행 성공 (wall 10.3 s). 시행 1~4는 PNG/meta를 만들지 않았고
  write guard 대상 산출물 오염 없음 (`bg1v_stdout.log`는 최종 실행분으로 교체).

## 3. 산출물 (전부 `claudedocs/runtime_logs/grasp_track/g0c_d446/`, forward-only)

| 파일 | sha256(16) | bytes | 내용 |
|---|---|---|---|
| `bg1v_side_phi000_postclose.png` | `aeda6a70bf2fad6e` | 184,972 | row 13 측면 파지 (양측 39.103 N, hang 낙하 0.007 mm) |
| `bg1v_top_theta00_postclose.png` | `c8cb870ce697b441` | 161,600 | row 21 **완전 수직 top-down** (56.335 N, 낙하 0.339 mm) — RoArm D430 원리적 불가 축 |
| `bg1v_render_meta.json` | `9ddf19d3fdf0e15e` | — | 소스 핀·복원 편차 0.0·physics_steps 0·카메라·PNG 해시·종료 핀 재검증 true |
| `bg1v_script.py.txt` / `bg1v_argv.txt` / `bg1v_stdout.log` | — | — | 감사 추적 (최종 실행분) |

## 4. 육안 검수 (Claude 직접, 2/2)

- side: 검은 블레이드 2개 사이에 원통(연분홍)이 물린 상태 명확. 그리퍼 레일·모터
  하우징 시각 메쉬 정상 렌더.
- top-down: 블레이드가 원통 **상단 12 mm만** 물고 있음 — 오류 아님, prereg
  BITE_M=0.012 설계값 그대로.
- 두 장 모두 post-CLOSE 상태라 원통이 받침대 위에 있음. **"쥐고 유지"의 증명은
  그림이 아니라 hang 수치**(받침대 충돌 OFF·240 step·낙하 ≤0.39 mm)가 담당 —
  보고 시 이 캡션 필수.

## 5. D341 정당화

bg1v는 새 공간/시간 **판정을 만들지 않는다** — 이미 판정된(D446) 기록 수치를
그대로 재투영한 D324 단일 프레임 진단 스냅샷. 따라서 RRD 미생성. 권위는 불변:
`bg1_results.json` `cb88c549dc459272` / `bg1_trace.npz` `f9dc41b797fc5fd2`.

## 6. 불변 확인 / 순응

- 종료 핀 재검증: `bg1_results.json`·`bg1_gripper_split2.usd` 미변조 (meta 기록).
- `g0b_d420`/`g0b_d444`/`g0c_d446` 기존 태그 편집 0. bg1 재실행 0. 신규 물리 0.
- git 커밋 0 (사용자 지시 대기 — `.gitignore` g0c whitelist 확장 필요:
  bg1v png/meta 포함 여부 포함).
- stop-hook `/half-clone` 요구 1회 거부 — HARD RULE #11. 거부 누적 **48회 [가정]**.

## 7. 다음 결정 경계 (전부 사용자)

1. B601 구매 품의 (fg1 0/13 vs bg1-B 13/13 + 본 렌더 2장 = 품의 자료 세트).
2. 교수님 보고 패키지 구성 (렌더 포함 여부).
3. 벤더(reBot-Isaacsim) USD 충돌 결함 upstream 제보.
4. full-arm B601 sim (신규 case + prereg 필수).
5. RoArm 트랙 잔여 (fg2 폭-정지 / D≤20 / rim 기움 컨펌) — 병행 가능.
6. git commit/push.
