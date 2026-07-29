# Session — 2026-07-29 — grasp G0a D408 actual read-only replay observability PASS

## 1. 승인·목적·권위 경계

사용자는 다음 tuple을 정확히 인용해 D408 actual runtime 1회를 승인했다:

`97c7ca51f8116053fcdc59aa9572669231d4abeb66022ed4e59c9e61af28e1ff`

승인량은 controller 1회, software Rerun viewer leg A/B 각 1회,
production manual writer 1회, retry 0이다.

이번 case의 신규 과학 변수: `[]` (0개).

운영·관측성 변수:

1. `d407_clean_view_capture_and_bounded_force_arrow_repair_v1`
2. `prearmed_atomic_manual_writer_pid_phase_handshake_v1`

이 실행은 D407의 동결 trace/RRD를 읽어 화면과 수동 판정 전달을 검증한
fail-capable observability evaluation이다. Isaac/Kit/PhysX import·launch,
새 physics step, q5 read/write, contact query, cylinder spawn, hardware GPU
job, USD/asset write는 승인 범위가 아니며 실제 카운터도 모두 0이다.

D408 PASS가 바꾸지 않는 권위:

- D407 final:
  `D407_SDF_PHYSICS_AB_TIPPING_REMEASURE_FAIL_STOP`
- `d407_retroactive_pass=false`
- `scientific_verdict=null`
- `g0a_pass=false`
- stable grasp, force closure, grasp feasibility, SDF 일반 우월성 등
  9개 scientific null claim

## 2. 실행 직전 재검증

로컬 원본을 다시 읽고 다음을 직접 확인했다.

- `HEAD == origin/master ==
  a69a96d36219268e4bc5e25065cc234da9d99674`
- Git dirty path가 prereg의
  `expected_pre_runtime_dirty_paths`와 exact 131/131
- allowlist 161개 밖 dirty path 0
- D408 physical root는 static file 4개, directory 0개
- 미래 runtime output 30개와 leg directory 2개 전부 부재
- 활성 D408 controller/writer/Rerun process 0
- D407 source regular file 44개/directory 2개 전수 SHA PASS,
  manual JSON 부재
- controller, writer, prereg, attestation, tuple, static-results SHA가
  tuple/static authority와 exact

Static authority:

| 파일 | SHA-256 |
|---|---|
| controller | `00f4317cd12fedec16e23599080e97ec6462c0e54abec872e3444ee6b8603fce` |
| manual writer | `f69d4221f79f0a6cd96a20f81714e4970b0ecc0bd99567037664dbf8a468edf7` |
| preregistration | `0c0f1c03d10210e205d5be0b25fd84c7d94c109fb26387f77fa22f6b984c8d0d` |
| attestation | `fa5a3cf2f1a2bb0a4899e89d26eeb41d7d83b34e2be799056498e7d7fd9d50dd` |
| tuple file | `97c7ca51f8116053fcdc59aa9572669231d4abeb66022ed4e59c9e61af28e1ff` |
| static results | `bfb0f05784f9c01a9a8dccc5126fb8ed12d3b0748eed51e882e89ce5c9962dab` |

## 3. 관찰 가능한 실행 순서

실행 명령:

```bash
/home/cgxr/miniconda3/envs/isaaclab/bin/python3.11 -B \
  sim_scripts/cyl34_top_view_d408_d407_manual_observability_completion_repair_controller.py \
  --stage controller \
  --approved-tuple-sha256 97c7ca51f8116053fcdc59aa9572669231d4abeb66022ed4e59c9e61af28e1ff
```

실행은 2026-07-29 17:35:37~17:37:06 KST에 한 번만 수행됐다.

| phase | 결과 | 경과 |
|---|---|---:|
| controller start | PID 2844592, tuple/static authority 수락 | 기준 |
| writer pre-arm | PID 2844644, READY/HMAC/PID-start binding PASS | 0.035207s |
| leg A replay | historical trace 500행, software viewer 1회 | 7.219554s |
| leg B replay | historical trace 500행, software viewer 1회 | 7.259030s |
| screenshots ready | manual target 5개, aggregate SHA 고정 | start+16.072s |
| manual writer fsync | true publication, deadline 전 | prompt+72.654994s |
| manual received | SHA/size/bytes 재검증 PASS | prompt+72.798467s |
| completion ready | phase sequence 10 | start+89.049754s |

writer는 replay 전에 arm됐고 retry, watchdog, terminate/kill, residual
process가 없었다. writer stdout/stderr log는 0 byte다.

## 4. 실제 다섯 화면의 수동 판독

controller가 출력한 exact manual 대상:

| 대상 | 크기 | SHA-256 |
|---|---:|---|
| A clean spatial | 1120×900 | `6a0b893fcfc07721eab2fbe42317a664d837b332b8e4b246e4fd08034572a562` |
| A decision sheet | 1920×1080 | `3a9e719051f7e9a9e828f6879e1d205af257e997ecf95959656f4d8000597939` |
| B clean spatial | 1120×900 | `98a7175b2bd501d88f1008d96f5efc0c4b5a1d863e5d239907036272f052a5af` |
| B decision sheet | 1920×1080 | `18566b424aaf71d1267e9c2ef2ce09f3e414af816ab8c2522b99aea4fc03058b` |
| A/B comparison | 3840×1080 | `0fefff48f308aab62320c8e67228f2758b84cf70284e0b06733eca784f16ba72` |

원본 해상도에서 직접 관찰한 내용:

1. A/B clean spatial 모두 jaw/gripper와 cylinder가 분명하다.
2. notification, hover legend, text overlap이 없다.
3. A/B q5, body-force, displacement, tilt 축·곡선·숫자가 읽힌다.
4. A final link4/link5/gripper는 모두 0N이다. peak link5는
   23.227865N(row 256), gripper는 43.858340N(row 255)다.
5. B final link5는 165.253139N, gripper는 415.498019N이다.
   B peak link5는 357.175438N(row 358), gripper는
   464.002511N(row 239)다.
6. B gripper glyph의 96px cap과 `display_capped=true`가 명시되고
   화살표·텍스트가 inset 안에 있다.
7. A/B comparison은 양쪽 모두 읽히며 독립 y축이므로 높이가 아니라
   눈금을 비교하라는 경고가 보인다.

따라서 11개 required boolean을 모두 true로 한 strict JSON line을
controller stdin에 정확히 한 번 제출했다.

Manual publication:

- artifact `D408_MANUAL_VISUAL_INSPECTION_V1`
- `received=true`, `pass=true`, 11/11 true
- file size 2,494 bytes
- SHA
  `bf917eb4680d387ea01fe7be6997051005a28f5c05607069f62bea68ea10af18`
- writer fsync는 writer deadline보다 522.345006s 빠름
- writer receipt SHA
  `8e96c684091b2b37e944edb5da2e9bc630088c0bb600a8dfe37eca31ae2bf0f9`

## 5. Rerun·원본 보존 검증

양 leg는 Rerun 0.34.1 Vulkan CPU
`llvmpipe (LLVM 15.0.7, 256 bits)`를 사용했고 hardware GPU job은 0이다.

| leg | semantic rows/cells | force/bbox | presentation RRD | corrected RBL |
|---|---:|---:|---|---|
| A | 12,995 / 57,929 | 2,000 / 8,000 | `ca1ba9e6eca37477f0b5fc17cdf8b3dab1e9980b0109181fbe421a7b5f8b7b8e` | `ca254b65df89b4de68689cc696fc52351797853e023d2802e61b8d9ed6633fe2` |
| B | 13,118 / 58,783 | 2,000 / 8,000 | `e7aad4f73f8997b63e3a37bfee844e0cf563b4c8f021ff22c8170c869d05e802` | `e8ecdac3e0af43f86eb0a482b4711ca99aaeef5b24ac7297640071c8684bb6b7` |

- source recording 1/blueprint 1 → output recording 1/blueprint 0
- force-display entity 3개와 point label만 제거
- added entity 0
- retained semantic digest A
  `77fdd2ff23ff166efedf36f9bcb58765b33998b0d80e1a995eed2e6adb1e8a03`
- retained semantic digest B
  `d60fdceffc915494c702c8c15c26982adeea54cb29c9fcfe59aff2a444cf2ad4`
- source trace/RRD/RBL copy 6/6 bit-exact
- actual force 4,000개와 text bbox 16,000개 전수 PASS
- maximum stored-vs-recomputed norm error:
  A `1.7763568394002505e-15N`,
  B `7.105427357601002e-15N`

## 6. 사후 불변성·종료 감사

- phase chain: exact event/order/hash 10/10 PASS
- terminal phase:
  sequence 10, row SHA
  `ead440b5dace40cb1767a5f985e5f1bf8130ea44e109f6f59e8f712791abeb31`
- D407 source manifest: 44 files/2 directories,
  SHA `8588fe2a67b2534bf1c9351239eeb9d7b2a06736cd81693bbb10007c475d3613`,
  admission부터 completion 전까지 5/5 identical
- 사후 별도 D407 44/44 SHA PASS
- screenshot aggregate SHA
  `a05832741993b0529d670375747aa0d47490a3f3a298f1d64be8944815cbb5db`,
  pre-prompt부터 pre-completion까지 4/4 identical
- 최종 D408 physical tree: regular file 32개, directory 2개,
  symlink/special file 0, 모든 regular file `nlink=1`
- pending manual/terminal file 0
- runtime 종료 직후 Git dirty 151개, prereg allowlist 밖 0
- controller/writer/viewer residual process 0
- 새 repo `__pycache__` 0
- terminal summary가 attempt root의 마지막 write

핵심 최종 SHA:

| artifact | SHA-256 |
|---|---|
| runtime prerequisites | `d3e55f2a3177d30b24810cb17c2af3311a2a24fe0537613da1eef2c0fe73f52d` |
| screenshot manifest | `1ef62c0f6340814070592ce8da7539958e4d2838ed8a3c118d49ab071a2f2362` |
| phase log | `4c7f685010eccc6779fa0d7f62ce9a271adc5fae547a2ca8eb76794c8d4c3b54` |
| source checkpoints | `8d89b873f9c70e4163916da2ad0d7b2d1fe8de8be064cb01f2d5735a93f73274` |
| screenshot checkpoints | `65e20c88309ea6524210e0664b65ab7f3b7c6c9849d950ac9d37100cc256bf3a` |
| terminal summary | `48626366c81c56c9bf2ae0f8c75b6a1291d7d5d3df7906010d93f0919a69dd37` |

## 7. 판정·다음 경계

최종 D408 status:

`D408_D407_MANUAL_OBSERVABILITY_COMPLETION_REPAIR_PASS`

쉬운 말로는 “D407의 기록을 가리지 않고 볼 수 있게 만들고, 사람이 판독한
true/false를 안전하게 한 번 게시하는 수리”가 실제 end-to-end로
성공했다는 뜻이다. 이것은 새 물리 실험이나 D407 과학 PASS가 아니다.

- `observability_repair_pass=true`
- `d408_manual_screenshot_integrity=true`
- `new_controlled_physics_steps=0`
- `d407_retroactive_pass=false`
- `scientific_verdict=null`
- `g0a_pass=false`

이 actual traversal은 handshake, renderer, visual inspection, deadline,
atomic publication 중 어느 단계에서도 실패할 수 있었으므로 session
progress rule의 fail-capable evaluation을 충족한다. 물리 perturbation을
실행하지 않은 이유는 D408의 승인된 단일 목적이 동결 D407 결과의
관측성·수동 publication 수리이기 때문이다.

D408 attempt1은 이 terminal PASS로 동결한다. 같은 경로 재실행, manual
수정, completion 덮어쓰기, D407 소급 PASS는 금지한다. 다음 과학 case나
새 Isaac/PhysX 실행은 새 설계·tuple·사용자 명시 승인 없이는 시작하지 않는다.
