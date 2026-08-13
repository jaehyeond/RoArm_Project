# 57th — git push 복구·증거 오프사이트 백업 + `g0b_d444` flying-gripper case 개시 (prereg까지, 물리 0)

- 날짜: 2026-08-13 KST (56th 직후 동일 날짜 Claude 세션)
- Active case: 개시 시점 `g0b_d420` 동결 확인 → 세션 중 사용자 승인으로 **`g0b_d444` 개시**
- 성격: 부트 재검증 + 사용자 질의 3건 답변 + 승인 집행 (git·case 개시)
- Session progress rule 정당화: 이번 세션의 실패 가능 물리 실험은 **0** — 사유 = 프로젝트
  프로토콜상 물리 실행 전 preregistration이 개시 전제(D416/D418 계열)이고, 사용자 승인이
  세션 후반에 나와 prereg 저작·자산/소스 감사까지가 이번 세션 범위. fg1 물리 실행은
  다음 세션 첫 작업 (START_HERE Next concrete action #1).

## 1. 부트 재검증 (기대 불일치 0)

HEAD == origin/master == `25ee2e2`, master, tracked 4 / untracked 280 / 합 284 (55th
283 + 56th doc 1). START_HERE 56th판 · D441~D443 · 56th doc 정독. 실험 0 상태 확인.

## 2. 사용자 질의 3건 → 답 (근거 = 원본 재확인)

1. **git push 실패 원인** = `g0b_d420/t3s_side_sdg2_raw_candidates.json` **208,941,751 B**
   (GitHub 100 MB 하드 리밋 초과, dirty set 중 유일). 옵션 1(ignore 제외) 사용자 승인.
2. **"병렬 포즈 스윕" 진행 확인** = 53rd t3w (D440 도달 경계 480셀) + 54th t3x (정적 물림
   창) + 54th t3y (D441: 6,144 IK → 3,476 PhysX, 4090 1,024 병렬 env, 62.5 s). 테이블
   접촉은 taxonomy로 계측됨 (`PRECLOSE_COLLISION` 1,217 + `JAW_SUPPORT_CONTACT_FAIL` 139).
3. **최신 렌더 정체** = exact-trace CPU schematic MP4 (RTX 아님; 로컬 렌더 prefix 2 소진 +
   RunPod `/dev/nvidia-modeset` 부재, D443). "grasp 언제 되냐" → 증거 사슬 브리핑:
   수직+상면중심 원리적 불가 (D430) / 기움 θ≥6° bite 양수 + T1 밴드 대응 (D431) / 기운
   자세 IK 도달 가능 (D432) / 그러나 rim-pinch 형태 그대로의 물리 시행은 0회.

## 3. 사용자 승인 + 집행 ①: git 백업/ignore/commit/push

- `.gitignore` 변경 3건: (a) 208.9 MB raw 제외 (`:57-59`), (b) `g0b_d420`
  `*.npz/*.png/*.mp4/*.log/*.csv` whitelist (417파일 235.5 MiB, 최대 단일 9.7 MiB),
  (c) attempt3 `collision_asset/**` whitelist (동결 5-layer USD, 4.9 MB).
- commit `b9020fd` "57th: 54th~56th 증거 커밋 …" — 470파일, 342.0 MiB. push
  `25ee2e2..b9020fd` 성공. working tree clean (0).
- 효과: 단일 사본 리스크 해소 — `t3u_side_preflight13_trace.npz`(sha `ee67d351…`),
  `t3y_workspace1_trace.npz`, D341 inspection/diagnostic PNG 전부, meeting MP4, 동결
  attempt3 USD가 GitHub 오프사이트 사본 확보. 208.9 MB raw는 디스크 원본 보존 +
  필터본 `t3s_side_sdg2_candidates.json`(8행)은 커밋됨.

## 4. 사용자 링크 튜토리얼 감사 → Codex 사용 이력 확인

NVIDIA Grasping SDG 튜토리얼 (Isaac Sim 5.1) = Codex가 8/11~12 `t3s_side_sdg1/sdg2`에서
실사용한 `isaacsim.replicator.grasping` **1.0.9**. 사용 범위 = **antipodal 샘플링만**
(물리 0 스텝, sdg2 prereg 명시). sdg1은 1,024 샘플 → 필터 후 6 < 8로
`SIDE_FILTER_TOO_FEW` 중단, sdg2는 4,096 샘플로 canonical 8행 확보 → 55th P13(0/5)로
소비. 미사용 절반 = GraspingManager 물리 phase 평가 (flying gripper).

## 5. 설치 소스 감사 (읽기 전용) — D444 Evidence

`grasping_manager.py` 직접 열람: pose 주입 공식 지원 (`evaluate_grasp_poses:978`,
YAML poses `:321-330`) / gripper 배치 = root xform 텔레포트 (`:1150-1160`) ⇒ 그리퍼
단독 fixed-root articulation 필수 / phase는 joint drive target만 (`:775-789`) — root
이동·lift 없음 / `object_simulation_phases` 선언만 (`:78,:109`) — 1.0.9 안정성 검사
미구현 ⇒ 자체 hang-test 게이트 설계 / `render=False` 직접 물리 + 임시 격리 씬
(`:666-806`) ⇒ RTX 불필요, 로컬 4090 headless 실행 가능.
한계 확인: antipodal 샘플러는 rim 핀치(비대향 법선)를 원리적으로 제안 불가 — rim pose는
동결 n8 산출물이 소스 (SDG와 상보).

## 6. 사용자 승인 + 집행 ②: `g0b_d444` 개시 (prereg까지)

- 신규 변수 1: `[팔 제거 = 그리퍼 단독 fixed-root articulation]`.
- Prereg: `claudedocs/runtime_logs/grasp_track/g0b_d444/fg1_prereg.md` — 13 pose
  (sdg2 side 8 verbatim + n8 rim-tilt 5 verbatim import), gates = close bilateral
  >0.01 N AND hang(지지면 collider off, 240 steps) 낙하 <6 mm, taxonomy 5종, D442
  lifecycle, D341 계약.
- 분기 의미: 전패 → 그리퍼 기하 병목 확정 / 1+ 성공 → 팔·포즈·궤적 병목.
- DECISIONS append: **D444** (개시 + 소스 사실 3건 + push 백업). LEDGER append 0
  (물리 실행 없음 — 실행 세션에서 fg1 row 기록 예정).
- BACKLOG append 2건: `rim_pinch_tilt_case` (교수님 기움 컨펌 대기) + 8/03
  `isaac_grasping_sdg_grasp_editor_evaluation` 부분 해소 상태 갱신.

## 7. 다음 세션 첫 작업

1. `fg1_gripper_only.usd` 추출 (prereg §3 검증 게이트: 64+64 hull, drive 파라미터
   bit-일치, 참조 메쉬 SHA) → 2. `fg1` 스크립트 저작 (§5 phases/gates, §6 lifecycle,
   §7 D341) → 3. 로컬 4090 실행 → 4. 결과 분기 브리핑 (그리퍼 병목 vs 팔 병목) +
   교수님 보고 자료화. 실패 시 같은 태그 재실행 금지 (fg2 forward-only).

## 8. 불변 확인

D427·D429·D431·D441·D443 재판정 0. `g0b_d420` prefix 편집 0. 로봇 0, RunPod 0,
lerobot-train 0.

## 9. 세션 말미 stop-hook /half-clone 거부 (사후 추가)

브리핑 완료 직후 stop hook이 context 131%를 이유로 `/half-clone`을 요구 → **거부**
(HARD RULE #11 + AGENTS.md Context 95% emergency protocol #4). 상태 문서·커밋·push는
이미 완료 상태였으므로 추가 마감 작업은 본 §9 기록 + continuation prompt 출력뿐.
누적 거부 카운터: 56th [가정 44] 이후 이번이 **45회 [가정 표기 유지]**.
