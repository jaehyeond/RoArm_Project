# Session 2026-05-22 — OpenVLA-OFT v6 Offline Eval — IN PROGRESS at session end

## TL;DR

Track B P2 후속: 5/22 새벽 학습 완료된 12개 OpenVLA-OFT 7B v6 LoRA checkpoint
(2.5K/5K/.../30K) 를 v6 holdout 5ep (45-49, 741 frames) + train_sanity 2ep
(0-1, 207 frames) 로 offline eval. 본 세션 context 126%로 종료, eval은
B200 nohup으로 계속 진행 중. 다음 세션이 결과 pull + best ckpt 분석.

## Verified State at Session End (KST 12:13 ~)

### Eval pipeline (4번 dry-run으로 3개 버그 수정 후 작동)

- **eval script**: `openvla_oft_roarm/eval_offline_v6.py` md5 `3ef67f2b558547623f6b3d03e8e98b4f`
- **launch wrapper**: `openvla_oft_roarm/launch_eval_offline_v6.sh` md5 `8976cf43d41e150679e9663e2e471ebf`
- B200 동일 md5 `scp` 검증됨.
- B200 환경: torch 2.12.0.dev20260407+cu128, peft 0.18.0, transformers 4.57.6,
  lerobot 0.4.4. HARD RULE #15 nightly cu128 준수.

### Dry-run 1 → 4 discovery chain

1. **Dry-run 1 FAIL**: `AttributeError("'OpenVLAForActionPrediction' object has no
   attribute '_supports_sdpa'")`. **원인**: HF cache의
   `modules/transformers_modules/openvla/openvla_hyphen_7b/47a0ec.../modeling_prismatic.py`
   가 **stock** (md5 `0e1ea109`, line 208에 `@property _supports_sdpa`) 이었음.
   Hub snapshot `hub/.../snapshots/47a0ec/` 는 D071 패치되어 있었으나
   transformers는 `openvla_hyphen_7b/` 경로를 사용. **수정**: 4개 cache
   위치 (`_47a0ec`, `openvla_hyphen_7b/47a0ec`, `transformers/`, `hub/`)
   모두 hub snapshot의 패치된 fork (md5 `8c2223ab`)로 덮어쓰기 + .pyc 무효화.
   `.preserve_orig.bak` 백업 보존.
2. **Dry-run 2 FAIL**: Modeling 로드 성공, but action_head load에서
   `RuntimeError: Missing key model.layer_norm1.weight; Unexpected key
   module.model.layer_norm1.weight`. **원인**: action_head 가 DDP wrapper로
   감싸진 채 saved → `module.` prefix. **수정**: state_dict load 전
   `module.` prefix strip.
3. **Dry-run 3 FAIL**: action_head 로드 OK, but inference에서
   `AssertionError: unnorm_key 'roarm_v6_pick' not in norm_stats
   (available: austin_buds, bc_z, bridge_orig, ..., 25 OpenVLA pretraining
   datasets)`. **원인**: `vla.norm_stats = norm_stats` 가 PeftModel wrapper에만
   적용되고 실제 prismatic model까지 propagate 안 됨. **수정**:
   `vla.base_model.model.norm_stats = norm_stats` 명시적 설정 + 양쪽 모두에
   set.
4. **Dry-run 4 PASS**: ckpt 2500 on ep_45 (146 frames) →
   `l2_step0_mean=11.5623°`, `l2_chunk_avg_mean=11.1589°`, 인퍼런스 시간
   ~90s (≈0.62s/frame). Output sha256
   `38c424fe6c6044a62bb367f2dbef4058cd707e49d5c3801f20f2a5e716a2887b`.

### Full eval kicked off (in flight)

- 12 ckpts × (5 holdout ep + 2 train_sanity ep) = 12 × 948 frames
- B200 nohup PID `3507531`
- 출력 JSON path:
  `/NHNHOME/.../JHPark/roarm_b200/outputs/openvla_oft_v6_eval/openvla_oft_v6_eval_20260522_121028.json`
- log: `/tmp/openvla_oft_v6_eval_full_20260522_121028.out`
- ETA ~131분 (12 × 948 × 0.62s + 12 × 70s load) → 종료 예상 ~14:21 KST
- B200 deadline 15:00 KST 안에 들어옴 (~39분 여유)
- 첫 ckpt 결과는 12:20 KST 경 도착 예상

## Decisions / Patches (HARD RULE candidates)

이 세션에서 발견된 3개 inference-time patch는 D-OFT-1~D-OFT-4
(session_20260522_openvla_oft_7b_30k_lora_complete.md) 후속으로 D076
(또는 별도 D-OFT-5/6/7) 후보. 핵심은:

- `_supports_sdpa` 패치는 hub snapshot뿐 아니라 transformers_modules cache의
  **모든 4개 위치** (특히 `openvla_hyphen_7b/<commit>/`) 에 적용 필요
- action_head state_dict는 `module.` DDP prefix strip 필요 (train_roarm_v6.py
  가 DDP wrap 후 저장)
- `norm_stats` injection 은 PeftModel 가 아니라 `vla.base_model.model` 에
  명시적으로 해야 predict_action에서 보임
- `local_files_only=True` + `revision=47a0ec7fc4ec...` 핀으로 fresh
  re-download (processing_prismatic.py를 stock으로 덮어쓰는 사고) 방지

## Files Changed This Session

Local (untracked / modified):
- `openvla_oft_roarm/eval_offline_v6.py` (new, md5 `3ef67f2b...`)
- `openvla_oft_roarm/launch_eval_offline_v6.sh` (new, md5 `8976cf43...`)

B200 (synced):
- `code/openvla_oft_roarm/eval_offline_v6.py` md5 match
- `code/openvla_oft_roarm/launch_eval_offline_v6.sh` md5 match

B200 HF cache patches (durable; will outlive eval):
- `$HF_HOME/modules/transformers_modules/openvla/openvla_hyphen_7b/47a0ec7fc4ec123775a391911046cf33cf9ed83f/{modeling,configuration,processing}_prismatic.py`
  → overwritten with patched fork (md5 `8c2223ab` for modeling)
- `.preserve_orig.bak` 백업 보존
- 동일 patch가 `_47a0ec/`, `transformers/`, `hub/` 4개 위치 모두 동기화됨

## Cross-check Items for Next Session

1. PID 3507531 alive 확인: `ssh JHPark "ps -p 3507531"`
2. 진행 ckpt 수: tail of `/tmp/openvla_oft_v6_eval_full_20260522_121028.out`
3. 완료 시 `ranking by holdout.l2_step0_mean` 출력 확인
4. JSON pull: `scp JHPark:/NHNHOME/.../openvla_oft_v6_eval_20260522_121028.json
   /home/cgxr/Documents/Robotics/RoArm_Project/openvla_oft_b200_pulls/`
5. Best ckpt 분석:
   - holdout `l2_step0_mean` 가장 낮은 ckpt
   - train vs holdout gap (R-OFT-2 overfit 예측 검증; 30K가 train 낮고 holdout
     높으면 confirmed)
   - per-joint MAE / z-score / diversity 패턴

## HARD RULE Compliance

- ✅ #4: 모든 metric은 B200 log + JSON sha256으로 cross-verifiable
- ✅ #11: `/half-clone` 거부 (이 doc + START_HERE update + continuation prompt
  방식)
- ✅ #14: 모든 ssh 명령에 `set -e; source env.sh; [[ -z "$ROARM_B200_ROOT" ]]
  && exit 1` guard 사용
- ✅ #15: torch nightly cu128 + transformers 4.57.6 확인
- ✅ #18: 사용자 명시 정정 없음 (eval scope = "Full holdout 5ep + train
  sanity 2ep", 사용자 선택 그대로 진행)

## Risks / Known Issues

- **R-1**: Eval이 ckpt 중간에 OOM/timeout 가능성. B200 25GB peak during training,
  inference은 보통 lower이지만 PeftModel without merge_and_unload는 LoRA delta
  추가로 메모리 소폭 증가. Monitor에 OOM/Traceback grep 포함.
- **R-2**: B200 deadline 15:00 KST. ETA 14:21 KST이면 39min 여유. 만약
  ckpt당 0.62s/frame이 0.9s/frame로 늦어지면 ~189min = 15:19 KST → 19분 초과.
  중간 진행 점검 시 ETA 재추산 필요.
- **R-3**: HF cache의 patches가 .pyc 캐시 무효화 후에도 영구. 다음 from_pretrained
  호출도 동일 patch 사용. 단, `huggingface-cli download --force-redownload`
  시 patch loss (D-OFT-1 R-OFT-3 와 동일 위험).
