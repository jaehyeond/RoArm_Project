# QUARANTINE — 포인트 열거 트랙 (DORMANT, 2026-08-05~)

> **상태: 격리(DORMANT).** 사용자 호출("포인트 트랙 열어") 전까지 이 트랙의 문서·수치·판정을
> 능동 작업의 근거로 인용하지 않는다. **삭제·이동·이름변경 없음** — `AGENTS.md` Variable Ladder
> "Folders are forward-only" 규정 때문에 파일은 전부 제자리에 있고, 격리는 **이 색인 1장으로만** 한다.
> 봉인 해시 체인이 경로에 묶여 있어 물리적 이동은 증거 전체를 무효화한다.

## 격리 사유

2026-08-05 교수님 피드백(사용자 전달): **"왜 굳이 이 모든 포인트를 알아야 하냐. sim에서 collider끼리
부딪히면 collision 나오고 두 포인트 나오고, 잡히는 거 하다가 찾으면 되지 않냐."**
→ 접촉점 전수 열거·기하 판정식 노선 **기각**. 목적 적합 파지 자세 1개(수직 상부 접근) 고정 후
물리 시행으로 직행. 상세 = `DECISIONS.md` **D419**.

## 격리 대상 (경로 불변)

| 자산 | 경로 | 무엇을 증명했나 |
|---|---|---|
| D409 attempt1 runtime | `claudedocs/runtime_logs/grasp_track/g0a_d409/attempt1_zero_step_dual_jaw_contact_region_enumeration/` | 1239 pose 전수: A 665 / **B 0 / A∧B 0** / PINCH 1146 / FULL 0. 전 격자 top-rim, `moving_witness_top_margin_mm` 1239/1239 = 0.000 |
| D409 harness 3파일 (sha v2) | `sim_scripts/cyld29h50_d409_zero_step_dual_jaw_contact_region_enumeration_{worker,controller,manual_writer}.py` | worker d82cee18…(2,395줄) / controller 2af457ec…(3,998줄) / writer 7f044583… |
| 봉인 해시 | evidence `ccc8197b…f16750` / region CSV `d5a51cfa…f6cff7` / completion `6ce9218c…d638a8` / tuple `de79bc78…efcc9a60` / prereg `46e31049…` / static `1780ede4…` / attestation `17e16f91…` | attempt1 소모·재실행 금지 |
| d348 collider 분해 | 64+64 파트 (`part_027/029/030/031` + inner17 동결 참조) | **재분해 금지** — 재분해 시 동결 참조 붕괴 (D415 ③) |
| D400~D408 attempt1 | 각 runtime 폴더 | G0a 정렬 수리 이력 |
| D362 33파일 | `sim_scripts/cyl34_top_view_d362_*` | D34×H90·0.72kg·저작 μ 1.5 PhysX. **전이 금지**(D379/D413 ①) |
| 결정 D409~D418 | `claudedocs/DECISIONS.md` | 기하·정정·감사 권위. **물리 verdict 아님** |
| 세션 문서 | `claudedocs/session_20260803~20260805_*.md` (11th~18th) | 상세 이력 |

## 격리와 함께 잠자는 미결 항목

- 승인 대기 ⑥ **판정식 재설계 사전등록(505셀)** — 분류기를 다시 정의할 때만 발동. 현재 DORMANT.
- D418 3축 판별(자세 / 첫접촉-종단 시점 / 분류기 정의) — **미측정 상태로 동결**.
- 승인 A1(`g0a_d418` 개시)·A2(읽기 전용 시리얼) — **철회**. barrel/top-rim 판별용이었고 그 질문이 기각됨.
- `D417-R1 ②` 원통 테이프 기하 비용 미계산 / `D415 ③` mm 여유 재인용 조건.

## 격리에도 불구하고 **살아 있는** 것 (능동 트랙으로 이월)

1. **d348 collider 64/64 유지가 정답** (D415 ③) — 새 트랙의 sim 충돌체도 이걸 쓴다.
2. **동결 FK 재현 게이트** (TCP 3축, 1.9e-13mm 검증) — 자세 계산 신뢰 근거.
3. **`D414 ①` 무효화 범위 규칙** — 그리퍼·팔이 바뀌면(부착물 포함) 동결 증거 전부 무효. **파지 방식과 무관하게 계속 유효.**
4. **HARD RULE #18** 사용자 명시 우선 / **HARD RULE #4** 문헌 검증 / **D341** Rerun 산출 계약.
5. **재인용 금지 4건** (SmolVLA 74ep 100% / SmolVLA+LoRA / hand-eye 2cm / D417 ③ 셀 여유).
6. **문헌 인용 의무**: push-grasping = Dogar 2010/2011·Brost 1988 기성 / cross-embodiment = GraspGen-X `arXiv:2606.00998` / KITE `arXiv:2606.22113`.

## 재개 절차 (사용자 호출 시)

1. 이 문서 재독 → 2. `DECISIONS.md` D409~D418 재독 → 3. `START_HERE.md`의 Frozen 목록 확인 →
4. 재개 사유를 D4xx로 기록한 뒤 Active Case 교체. **Claude 단독 재개 금지.**
