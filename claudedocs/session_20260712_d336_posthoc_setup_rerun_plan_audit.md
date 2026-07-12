# Session 2026-07-12 (D336 continuation) - setup/rerun/plan 감사 (실험 없음)

이번 case의 신규 변수: `[]` (감사만, 물리/기하 실행 0)

사용자 질문 4건(rerun 어디 갔나 / plan대로 가나 / 왜 실패가 많나·원통 정의 제대로
됐나 / rerun 현업 관행)에 대한 3-agent 병렬 감사 + 직접 코드 확인 결과.
수정은 없음; BACKLOG 등록 1건만 수행.

## 1. Rerun 사용 실태 (repo 감사)

- 구현은 `roarm_rl/viz_debug.py::log_rerun()` 단일 지점. D325 이후 전 런이
  .rrd 저장 (12개). D324는 rerun 미설치로 실패 기록.
- **Live viewer 사용 0회**: D326-D330은 `--live_viewer` CLI 플래그 보유(전부
  미사용), D332-D336은 `live_viewer=False` 하드코딩. `rr.connect/serve` 부재.
- **D333 이후 RRD를 뷰어로 연 증거 없음** — artifact gate가 ok+nonzero만 확인.
  마지막 열람 증거는 D326/D327 `*_rerun_open.png`.
- 내용 갭: 물체는 frame-axes 마커로만 추적(solid 없음); 접촉력은
  `metadata/step_diagnostics` TextDocument로 묻힘(Arrows3D/Scalars 없음);
  D333은 물리 400step 중 target 200만 수록(baseline은 CSV만); 10-trial 프로브는
  trial 1만 RRD.
- trace_steps: D326=1, D327=1/320/320, D328=320x3, D330=320, D332=200,
  D333=200, D334=2, D335=1, D336=1.

## 2. Rerun 현업 관행 (웹 리서치, 출처는 rerun.io docs/blog + Isaac Lab docs)

- 인터랙티브 디버깅 = live (`rr.spawn()`/`connect_grpc`), headless/배치 =
  `rr.save()` 후 `rerun file.rrd` replay — 둘 다 표준.
- 로봇 현업 대표 패턴 = **하이브리드** `rr.set_sinks(GrpcSink(), FileSink(...))`
  ("라이브로 보면서 동시에 아카이브" — 공식 문서의 로봇 사용례).
- Isaac Lab 2.3+ Newton 브랜치는 `--visualizer rerun` + `record_to_rrd` 내장;
  classic PhysX 경로(본 repo)는 수동 SDK 로깅이 정석.
- 주의: rerun-sdk 버전 pin (D326 numpy 사건), .rrd는 인접 minor 호환만 —
  장기 증거는 PNG 병행 유지.
- 우리 상황 판정: zero-step 기하 프로브(D335/D336)는 1프레임이라 post-hoc이
  맞음(단 열람은 해야 함). 200-step 물리 settle 런은 live+file 하이브리드가
  현업 패턴.
- 조치: `claudedocs/BACKLOG.md`에 `rerun_pipeline_upgrade` 등록 (착수 금지).

## 3. 원통 정의 감사 (직접 코드 확인)

- `sim_scripts/cyl34_top_view_d332...py:488-529`: 해석적
  `CylinderCfg(radius=0.017, height=0.090, axis=Z)` — 메시 아님. hppfcl 판정도
  동일 해석 원통 → 판정-스폰 형상 불일치 없음. stage contract가 매 런 테이블
  높이 검증.
- 물리 재질: friction `1.5/1.2` + restitution `0.0` = placeholder (교수님 지시
  8번으로 동결이 맞음). `collision_props` 기본값(숨은 튜닝 없음). 시각 재질
  주황 단색 — G0a는 카메라/렌더 차단이라 무영향.
- 질량 `0.72kg` placeholder — 실물 미실측 (G0b 전제, D331).
- 잔여 리스크 2건 (신규 발견, 결함 아님·확인 필요): ① 물체 prim 슬롯 이름이
  legacy `Sponge` (혼동 유발, 물리 무영향) ② 물리 런에서 PhysX가 Cylinder
  프리미티브를 exact로 처리하는지 convex 근사인지 미감사 — D335/D336(물리 0)
  무관, G0b 접촉 물리 전 확인 가치.

## 4. 실패 원인 분류 + plan 위치 (D322-D336 전수)

- 15세션 중 10-trial 태스크 시도는 5회(전부 0/10), 그중 4회는 잡을 수 없는
  물체(100mm 큐브 vs 개구 40-45mm, D329 발견)로 실행됨.
- 근본원인: (a) 씬 결함 — 이중 지지면 12.117mm 매립 (D332→D333 수리)
  (b) 목표 설계 결함(최대) — D325 family가 그리퍼 실물을 물체 안에 배치
  (D334 확정, D335+D336으로 family 폐쇄) (c) 측정 결함 — 죽은 ContactSensor,
  vacuous 70.000mm fallback, 미실측 질량 spec화, BVH scalar 오독
  (d) 진짜 난이도 — 5-DOF 도달성(D323)뿐. **RL 난이도로 실패한 적 없음(RL
  미시작)**.
- Variable Ladder는 D330 이후 엄격 준수(세션당 0-1 변수). 실패 다발의 원인은
  변수 과다가 아니라 D322 초기 설정 부채의 순차 청산이었고, D336의 exact
  pre-physics gate가 동종 낭비를 구조적으로 차단.
- Plan 위치: G-사다리 1단(G0a) 진행 중, 완료 rung 0. G0a 진단 국면 종료 —
  남은 것은 사용자 선택 (A) wrist/tool-orientation 변수 1개 추가(권장,
  ~4.4mm 근거) (B) r>17mm grasp-depth 재정의.

## 산출물

- 수정 없음. `claudedocs/BACKLOG.md` +1 (`rerun_pipeline_upgrade`).
- 이 문서가 감사의 단일 기록. 상세 근거는 D322-D336 각 세션 doc/DECISIONS.
