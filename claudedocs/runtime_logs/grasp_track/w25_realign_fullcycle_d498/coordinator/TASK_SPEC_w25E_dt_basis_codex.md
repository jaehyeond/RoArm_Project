# TASK_SPEC W25-E — 시간 간격(dt) 선택 근거 메모 + dt 사다리(2·5 µs, 반복 3) 사전 등록·비용 (문서·산술만, 실행 0)

작성 2026-09-28 · 코디네이터 = 메인 Claude Code(Fable 5.1) · 이 과제서가 계약 정본. 보고는 **한국어**, 절차→수치→근거(file:line·sha256)→한계→다음 승인 경계 순, step-by-step.
⚠️ Codex 시작 시 "업데이트 가능" 안내가 뜨면 건너뛰기(설치·업데이트 금지).

## 0. 절대 금지
- GPU 0 · DEME 물리 0 · RunPod 0 · 로봇 0 · 설치 0 · 네트워크 조회 0(문헌 인용은 **repo 안 문서에 이미 있는 것만**, 새 논문·URL 을 지어내지 않는다 — HARD RULE #4) · git 쓰기 0 · 원장 쓰기 0 · 다른 worktree·메인 repo 쓰기 0.
- 산출은 자기 worktree `claudedocs/research/w25_dt_basis_20260928/` 아래에만. 모든 숫자에 file:line/JSON 키. 교과서적 규칙(예: 레일리 시간의 10~20 %)은 "일반 관행(출처 미확인)" 이라고 표기.

## 1. 배경(사실·근거)
- 생산 실행 dt: W10 스쿱 2 µs(541알), W11 스쿱 1 µs(517알), W13/W19 전체 사이클 1 µs. W19B 같은 입력 반복 1 µs: 517/517/489(n=3, D490 "n≥3 없이 차이를 읽지 않는다").
- W15 dt 사다리(D486, `claudedocs/session_20260916_w14_raw_repair_dt_plan.md` §7, `w15-dt-ladder/.../w15_dt_ladder_20260916/REPORT_w15.md`): 10 µs 폭주(재닫기 3.15 s, 엔진 속도 2.17e9 m/s), 100 µs GPU OOM, 1 ms 접촉탐색 커널 assertion. 물리 1초당 실제 경과: 1 µs 866 s · 2 µs 595 s · 10 µs 802 s(10 µs 가 2 µs 보다 느림 — 탐색 여유 ∝ dt, DEME `API.h:196-230`).
- W24 L 10 µs 폭주 원인(`w24-dt10us-cause/claudedocs/research/…/REPORT.md`, 세션 W22 §9): 문 립–렌즈 알 3개–고정 립 층 끼임에서 구-구 16~17쌍 감쇠 명시적 적분 불안정, 증폭률 10 µs 1.50/1.73, 1·2 µs <1, **한계 dt 7.5~8.1 µs**. 메인 표준식 어림 11~20 µs(탄성 29~52 µs 보다 먼저). 레일리 시간(E 5 MPa, 구 r 1.16 mm) ≈ 85 µs.
- 물성(임시값): E 5e6 Pa, ν 0.3, CoR 0.3, μ 0.45, Crr 0.06, 밀도 905, 렌즈 알 = 구 7개(r 1.16 mm 등, 템플릿 `clump_template_json`), 툴 E_mesh 3e9(PLA). params `claudedocs/runtime_logs/grasp_track/w19_runpod_d487/rev32_frozen_copy/params_w13.json`, W11 params 는 `claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w11_dt_sensitivity_20260911/`.
- 비용 실측: 로컬 4090 Laptop 1 µs 2,720.74 s(W11), 2 µs 1,870.9 s(W10); pod 4090 1 µs 1,772.03/1,748.33 s(W19B). 10 µs 는 3.15 s 까지 2,530 s(W15).
- 사용자·교수 질문: "dt 를 왜 그 값으로 정했나, 5 µs 는 왜 안 해봤나" 에 **근거 있는 답**이 필요. "해봤더니 됐다" 식 금지.

## 2. 목표(Change)
1. **dt 선택 근거 메모(교수 대상, 한국어, 2쪽 이내)**: (a) 안정성 한계(W24 L 감쇠 적분 7.5~8.1 µs; 탄성/헤르츠 접촉 시간 — 우리 물성으로 계산: 구 질량, 유효 반경, E*, 접촉 시간 t_c(v 0.1·0.5·1 m/s), 레일리 시간) (b) 정확도/수렴(1 vs 2 µs 포획 517 vs 541 이 실행 간 변동 ±5 % 안이라 n=1 로는 구별 불가) (c) 비용(dt 별 물리 1초당 시간 실측·탐색 여유 효과) (d) 결론 = "dt 는 가장 뻣뻣한 접촉 무리의 안정 한계에 안전계수를 둔 값 + 수렴 검사로 정한다; 5 µs 는 한계의 1.5× 안쪽이라 안정할 수 있으나 검사 없이는 채택 불가" 를 **우리 수치**로. 각 수치에 근거 경로.
2. **dt 사다리 사전 등록(PREREG)**: 같은 입력(W11 params·20k 더미·seed 460·스쿱 셀)에서 2 µs ×2(기존 1회 + 2 → n=3), 5 µs ×3, (선택) 3 µs ×3. 판정 지표 = 포획 알 수·질량, 닫힘/재닫힘 각·정지 사유, 최대 입자 속도·5 m/s 초과 sync, 발산 플래그, 구덩이 높이지도 차이(5 mm 격자) — **결과 보기 전 문턱 고정**(예: dt 간 평균 차 < 실행 간 변동 SD×2 면 "구별 불가", 발산 1회라도 = 그 dt 불가). 중단 규칙. cap = 비관 ×2.
3. **비용·일정**: 셀당 pod 시간 추정(2 µs = 1 µs×0.69 실측비, 5 µs 는 미측정 → 상·하한 2안), pod 1대 순차 vs 2대 병렬, 총 $(0.74 $/h, 확인 필요 표기), 오늘 밤 기동 시 완료 시각(KST) — 기동 시각을 변수로.
4. **RunPod 명령 템플릿**(실행 금지): W19B `launch.json`/params 골격 + dt 만 변경, 부트스트랩 v5 사본, 해시 게이트, setsid watchdog, 회수(`_obj` 포함).

## 3. 산출
- `DT_BASIS_MEMO.md`, `PREREG_dt_ladder.md`, `dt_cost_plan.json` + `dt_cost.py`(순수 파이썬 재현), `RUNPOD_dt_ladder_template.md`, `receipt.json`, `REPORT.md`.

## 4. 환경
- CPU 파이썬 `/home/cgxr/miniconda3/envs/roarm/bin/python`. Orca: 질문은 preamble 의 `ask`. 완료 시 `worker_done` + `--report-path`.
