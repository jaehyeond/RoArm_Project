# D359 — D351 historical hash-generator lineage recovery

Date: 2026-07-16 KST

Case: `g0a_d359`

이번 case의 신규 변수:

1. `historical_generator_transcript_lineage`
2. `historical_source_point_id_remap_replay`

신규 physical variable: `[]`

## 1. 무엇을 왜 확인하는가

D358은 현재 authored/raw 계산과 20,736개 등록 recipe를 정상 실행했지만,
D351에 고정된 inner/outer vertex·triangle·patch 해시 6개를 재현하지 못했다.
paired-XZ 2개만 재현됐으므로 결과는
`D358_HASH_PROVENANCE_UNRESOLVED_FAIL_STOP`이었다.

D359의 질문은 다음 하나다.

> 여섯 historical hash를 최초로 출력한 실제 local generator는 무엇을 입력으로
> 읽었고, 정점을 어떤 순서로 remap해 어떤 commit의 D351 상수가 되었는가?

여기서 `remap`은 원래 정점 번호를 작은 patch 전용 정점 번호로 다시 매기는
절차다. 형상이 같아도 이 번호 순서가 다르면 byte stream과 SHA-256은 달라진다.

## 2. 사전 발견과 검증해야 할 가설

읽기 전용 boot 조사에서 보존된 Codex session transcript
`/home/cgxr/.codex/sessions/2026/07/15/
rollout-2026-07-15T15-08-12-019f6463-d763-7761-bb62-68bfe4f993f2.jsonl`
line 1433-1434에 실제 one-off generator call과 output 후보를 찾았다.

후보 call은 다음 두 특성을 가진다.

1. stage path는 D344 attempt3 composite asset
   `claudedocs/runtime_logs/grasp_track/g0a_d344/collision_asset/attempt3/
   roarm_m3_fullmesh_fixed_point_parts/roarm_m3.usd`다.
2. patch vertex 순서는 좌표 row의 lexicographic `np.unique(axis=0)`가 아니라
   `old=np.unique(f[ids].reshape(-1)); u=p[old]`처럼 원본 point ID 오름차순이다.

같은 transcript의 뒤 설명은 local asset path와 coordinate-row unique를 말했다.
따라서 실제 실행 call과 뒤 설명 사이에 불일치가 있었을 가능성을 독립 검증한다.
이 가설은 아직 결과가 아니며 audit replay가 8-field bundle을 exact 재현해야만
채택한다.

## 3. 동결 입력

- Base before edits: `HEAD == origin/master ==
  d4671d4bdefa4f6e5ef1f2f28b8e318c100b7cb5`.
- D351 original harness SHA-256:
  `3c4501885af7590f5883b36666c984ce88728a24d40451ea0a600660a386107d`.
- D358 evidence / completion SHA-256:
  - `6c19cf6c3cd99b9567db65bf065afcb95872c4cfa6940c6584a97717638af3ff`
  - `9ea631942cab32708cbc2f58e2b8351ad03dd2f45ff8c6f699caa44079e875f7`
- Candidate transcript SHA-256:
  `75f9f04762a99dd0a551d1455b6c2c5d0244c8d5453a54084c34f046fcc78ffa`.
- Expected first Git introduction candidate:
  `c2cfa5f41d4c15fec15330cfad38b9b14e4c4f61`.
- D344 and local top-level `roarm_m3.usd` files currently share SHA-256
  `a4be58e87b1f9790f2a2ed600f0620c79d3cfb95c608b3c598308d52f5e46fff`;
  this alone does not prove their composed sublayers are identical.
- D334 user-owned sidecar is read-only and must be exact before/after.

## 4. 등록된 감사 순서

1. `--stage prepare` verifies current Git, frozen hashes, transcript identity,
   bundled standalone core-PXR/OpenUSD 0.24.5 environment, package pins, D334
   sidecar inventory, empty forward-only output, and writes preregistration only.
2. `--stage audit` rechecks preregistration and creates exactly one exclusive
   invocation marker. No retry or overwrite is allowed.
3. Parse the transcript as JSONL and bind the actual custom tool call to its
   output by `call_id`. Record session/subagent identity, line numbers,
   timestamps, exact command SHA-256, source path, remap expression, and the
   six expected output hashes. Also record the later narration/call mismatch.
4. Use `git log --all --reverse -S<hash>` and parent-tree checks to prove the
   first repository commit that introduced all six constants.
5. Load both D344 attempt3 and local composed USD stages with bundled core-PXR.
   Record all used layers and authoritative point/count/index stream hashes.
6. Replay a 2x2 matrix:
   - source: D344 composed stage vs local composed stage;
   - remap: actual historical original-point-ID order vs later
     coordinate-lexicographic order.
7. Independently replay the historical recipe with Python tuple/dict/
   `struct.pack`. Reverse-point-ID and coordinate-remap perturbations must be
   rejected as negative controls.
8. Compare the coordinate-remap row to D358's current authored 8-field bundle
   and explain exactly which axis D358's registered grid omitted.
9. Write evidence/report/completion in forward-only order and recheck the D334
   sidecar. No gate or expected constant is changed.

## 5. 판정 규칙

- `D359_D351_HASH_PROVENANCE_RECOVERED`:
  actual transcript call/output, first Git introduction, historical original-ID
  replay `8/8`, independent replay, D358-current coordinate replay, negative
  controls, and all immutable inputs pass.
- `D359_D351_HASH_PROVENANCE_PARTIAL_FAIL_STOP`:
  generator/source/commit lineage is found but the exact 8-field replay or an
  independent/negative control fails.
- `D359_D351_HASH_PROVENANCE_UNRESOLVED_FAIL_STOP`:
  the actual generator/source/commit cannot be bound to preserved evidence.
- input/runtime/preflight failure uses
  `D359_OFFLINE_INPUT_OR_RUNTIME_FAIL_STOP`.

Even a recovered result does not rewrite D351/D354/D358 artifacts. It may explain
their mismatch and establish the evidence contract inherited by the later physical
case, but it does not decide contact, grasp, or target/IK repair.

## 6. 실행환경과 명령

Python:
`/home/cgxr/miniconda3/envs/isaaclab/bin/python`

Bundled core-PXR `PYTHONPATH`:
`/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/
extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311`

`LD_LIBRARY_PATH`:
`/home/cgxr/miniconda3/envs/isaaclab/lib:/home/cgxr/miniconda3/envs/isaaclab/
lib/python3.11/site-packages/isaacsim/extscache/
omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311/bin`

Registered stages:

```bash
env PYTHONPATH=<registered-core-pxr-root> LD_LIBRARY_PATH=<registered-library-path> \
  /home/cgxr/miniconda3/envs/isaaclab/bin/python -B \
  sim_scripts/cyl34_top_view_d359_d351_historical_hash_generator_lineage.py \
  --stage prepare

env PYTHONPATH=<registered-core-pxr-root> LD_LIBRARY_PATH=<registered-library-path> \
  /home/cgxr/miniconda3/envs/isaaclab/bin/python -B \
  sim_scripts/cyl34_top_view_d359_d351_historical_hash_generator_lineage.py \
  --stage audit
```

Preregistered harness SHA-256:
`961939863649f483f00ce667b347bfe79f38bb623eb713b38bad084930762ea3`.

The sole audit uses a 180-second process alarm. Rerun is intentionally omitted:
this is a pure text/file/hash/schema audit with no spatial or temporal verdict.

## 7. 동결·금지 범위

- Isaac/Kit/SimulationApp/GUI/GPU/RTX/Warp: `0`
- q5 science/state write, physics step, distance/contact/cap-rim query: `0`
- asset/decomposition/gate/hash/tolerance/material/mass/actuator/physics change: `0`
- target/IK/path, settle, trial, hold/lift, G0b, RL/PPO/VLA/ladder: `0`
- package install, D334 sidecar write, commit/push: `0`
- D351-D358 rerun/overwrite: `0`

## 8. Session-progress rule

The 2x2 source/remap replay and reverse-ID/coordinate-remap negative controls can
fail and can change whether the historical lineage is accepted. This is a
failure-capable perturbation evaluation, not validation that cannot affect a
decision.

## 9. Preregistration-time status

This section was written before `prepare` and before the sole audit invocation.
Actual results will be appended below in execution order.

## 10. 실제 실행 순서와 결과

### 10.1 Prepare

등록된 bundled standalone core-PXR 환경에서 `--stage prepare`를 실행했다.
다음 prerequisite가 모두 PASS했다.

- `HEAD == origin/master == d4671d4bdefa4f6e5ef1f2f28b8e318c100b7cb5`
- D351 harness, D358 evidence/completion, transcript, D344/local root hash exact
- OpenUSD `[0,24,5]`, NumPy `1.26.0`, psutil `5.9.8`
- PXR module origin과 등록 `PYTHONPATH`/`LD_LIBRARY_PATH` exact
- forbidden Isaac/Kit/GPU modules `[]`
- D334 sidecar three-file inventory frozen

Preregistration SHA-256은
`97bd47e7f42d8b37a303c9c3d3611b3812d94ddeeeaef60e4d4da70511c02c18`다.

### 10.2 Sole audit invocation

Audit는 정확히 한 번 실행됐고 재시도·watchdog·runtime exception 없이 completion에
도달했다. Evidence elapsed는 `34.150002633s`, completion elapsed는
`34.178689804s`였다. 순방향 phase는 다음 8개다.

1. `audit_started`
2. `transcript_generator_bound`
3. `git_first_introduction_bound`
4. `source_remap_matrix_replayed`
5. `d358_recipe_axis_gap_audited`
6. `authoritative_evidence_written`
7. `report_written`
8. `completion_ready`

Invocation/phase SHA-256은 각각
`923c318a61e5fa42eccdf69fc1e6111b6144536fd4dc1ea6c034e90ba0beff34` /
`de0226bac55b4568578306b77fe7ba2e6c2d89ed9b31b433775fb67fb881cb55`다.

## 11. 실제 generator/source/commit 계보

보존 transcript의 session meta는 subagent `/root/d351_patch_design`, nickname
`Einstein`, 시작 Git `cfd9e7501df89724c3cc2b1038fda05ce0d88e2f`를 기록한다.

- actual generator call: transcript line `1433`, timestamp
  `2026-07-15T06:19:46.955Z`, call ID
  `call_wllbuGOV6T8Td1VIor3SdVoV`
- bound output: line `1434`, timestamp `2026-07-15T06:19:47.297Z`
- later narration: line `1510` event duplicate and line `1511` canonical
  `response_item`, timestamp `2026-07-15T06:28:31.653Z`

Actual call은 D344 attempt3 composed stage를 열고 다음 방식으로 patch를 만들었다.

```python
old = np.unique(f[ids].reshape(-1))
u = p[old]
rem = {int(x): i for i, x in enumerate(old.tolist())}
```

즉 원본 point ID를 오름차순으로 정렬한 뒤 그 ID로 vertex stream과 triangle remap을
만들었다. 이 실행 output에는 frozen expected 8개가 모두 exact 있었다.

그러나 약 8분 45초 뒤 narration은 source를 local asset이라고 설명하고, vertex를
원본 point ID가 아니라 coordinate row의 lexicographic `np.unique(axis=0,
return_inverse=True)`로 재구성하는 코드를 제시하면서 actual-call 해시를 그대로
붙였다. 이후 D351 harness도 상수는 actual-call 값, validator는 coordinate-row
계산을 사용했다.

Git `-S` 추적에서 8개 field의 최초 도입은 모두 commit
`c2cfa5f41d4c15fec15330cfad38b9b14e4c4f61`이다. 그 parent는 위
`cfd9e750...`이며 D351 harness가 없었다. 따라서 transcript generator -> D351
commit의 전진 계보가 연결된다.

## 12. 2x2 replay와 원인 분리

D344 stage와 local stage의 top-level root SHA는 같지만 physics sublayer는 각각
`043a5d35...`(33,705 bytes)와 `1df07d38...`(4,242 bytes)로 달랐다. 따라서 root
file hash만으로 composed stage 전체 동일성을 가정하면 안 된다. 다만 이번 patch의
authoritative points/counts/indices stream은 두 stage에서 모두 exact 같았다.

- points Float32-mm:
  `b89c67e99bd253ae710e6b0a2fcacd0b27263d6ede29fe6f6334ed70247895ed`
- counts Int64:
  `f17eac58b9b109f98f7a69efcc3b1e64b632d805ccca8cc8883cf0349e07cb6c`
- indices Int64:
  `205a08458b895d96c6eb9593d1f04a8815629f7f972a889cce683b86955f2545`

Replay 결과:

| composed source | vertex remap | frozen D351 match |
|---|---|---:|
| D344 attempt3 | original point ID ascending | `8/8` |
| local | original point ID ascending | `8/8` |
| D344 attempt3 | coordinate lexicographic unique | `2/8` |
| local | coordinate lexicographic unique | `2/8` |

Coordinate 방식의 8-field bundle은 D358 current-authored bundle과 `8/8` exact였다.
독립 Python tuple/dict/`struct.pack` original-ID replay도 frozen D351과 `8/8`
exact였고, point-ID reverse 음성 대조군은 의도대로 거부됐다.

그러므로 discriminating cause는 D344/local geometry나 mm/m 변환이 아니라
**vertex remap의 key/order**다. Paired-XZ 2개만 두 방식에서 같았던 이유는 마지막에
XZ 좌표를 다시 정렬해 앞선 vertex-list 순서 차이가 사라지기 때문이다.

D358은 source point ID를 버린 뒤 coordinate tuple만 `lexicographic_unique` 또는
stable-first 방식으로 정렬했다. `original_point_id_ascending` axis가 등록 grid에
없었으므로 20,736 recipe가 여섯 값을 못 찾은 것도 설명된다.

## 13. 최종 verdict와 해석 경계

최종 verdict는
`D359_D351_HASH_PROVENANCE_RECOVERED`다.

Evidence / completion SHA-256은 각각
`9a4c2aa38bfc8e26722852a328d5f228aeccba17e372b017767f4da7c281f822` /
`bfa66efc2e3f36bc7c781558fa118e6cc5f243d8dba948c19f59852fbdf21f85`다.

이 결과는 다음을 뜻한다.

1. D351 historical constants는 임의 숫자나 asset mutation 결과가 아니라 보존된
   one-off generator의 original-point-ID serialization 결과다.
2. D351 validator와 D358 grid가 그 generator와 다른 coordinate-row semantics를
   사용해 mismatch가 발생했다.
3. 과거 D351/D354/D358 artifact와 verdict는 immutable이며 소급 수정하지 않는다.
4. expected hash 교체, gate 완화, asset 변경은 모두 `0`회다.
5. Isaac/Kit/GPU/q5/physics/contact/cap-rim 실행은 모두 `0`회이며
   `g0a_pass=false`다.

이제 사용자가 승인한 두 번째 별도 case에서 실제 PhysX 시간진행으로 moving jaw를
닫고 body-level 접촉력과 원통 움직임을 관찰할 evidence lineage prerequisite는
충족됐다. 그 case도 exact moving face/cap-rim, force closure, grasp/G0a를 자동으로
판정하지 않으며 별도 preregistration을 먼저 작성한다.

## 14. Forward-only postcompletion precision clarification

독립 검토에서 evidence의
`git_lineage.committed_d351_blob_sha256=cd1619e2...13a7` field 이름이 실제
계산 의미보다 강하다는 점을 발견했다. Harness `_run()`이 `git show` stdout에
`.strip()`을 적용한 뒤 SHA-256을 계산했으므로 마지막 LF가 제거된 text diagnostic이지
실제 Git blob/file byte SHA-256이 아니다.

기존 evidence/completion은 덮어쓰지 않았다. 별도 clarification에서 다음을 고정했다.

- Git blob object ID: `9294e00f1c9953f3947e9093529e6bf206b0127c`
- committed file byte SHA-256:
  `3c4501885af7590f5883b36666c984ce88728a24d40451ea0a600660a386107d`
- current D351 file byte SHA-256: 같은 `3c450188...107d`
- transcript narration은 line `1510` duplicate event와 line `1511` canonical
  response 양쪽에 존재

Clarification artifact SHA-256은
`a6645de539e2bc1106e71bd1462fc40da7806a6bd6e8ea8d734091d269794c9b`다.
이 정정은 8/8 replay, first commit, parent path absence, D359 verdict를 바꾸지 않는다.
