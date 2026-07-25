# 2026-07-25 — Grasp G0a D383 RBL data-value verbosity validation resume

## 1. 무엇을 왜 확인했는가

Case:

`D383 [d382_rbl_data_value_verbosity_validation_resume]`

이번 case의 신규 변수:

`[rbl_data_value_print_verbosity_v1]`

D382는 정적 비교판, 완전한 layout JSON, RBL, presentation RRD를 만들었지만
RBL에 저장된 query/view-name **값**을 summary-only `rrd print -v` 출력에서
찾다가 Viewer 전에 멈췄다. 같은 RBL을 사후 `-vvv`로 읽으면 등록값 `6/6`이
존재했으므로, D383의 질문은 다음 하나였다.

> D382의 네 presentation artifact를 그대로 보존하고 RBL data-value authority만
> `-v -> -vvv`로 바꾸면 strict Rerun validation, Viewer 1회, 원본 해상도
> 육안검사까지 완결되는가?

이 case는 D382 failure에 반응한 observability-only 수리다. geometry, P34 identity,
원통, 접촉, q5 또는 grasp를 다시 측정하는 과학 case가 아니다.

## 2. 승인 범위와 동결 항목

승인된 출력:

`claudedocs/runtime_logs/grasp_track/g0a_d383/attempt1_d382_rbl_data_value_verbosity_validation_resume/`

런타임 입력은 immutable D382 파일 네 개뿐이다.

1. board PNG
2. layout-validation JSON
3. RBL blueprint
4. presentation RRD

동결:

- D382 board pixels/layout/text
- RBL query, display name, view layout
- presentation RRD entity/component/timeline contents
- Viewer command and logical window `1920x1080`
- D380 numeric verdict
- P34 identity=false
- `g0a_pass=false`

금지:

- RRD/RBL/geometry 재생성
- D382 same-path retry/overwrite
- Isaac/Kit/PhysX/USD/asset/collider 실행 또는 변경
- cylinder/physics/public forward/q5/contact
- target/IK/path/pose 및 material/mass/actuator/physics settings 변경
- automatic decomposition sweep
- commit/push

실행 상한:

- offline validation worker 최대 `1`, retry `0`
- Rerun Viewer 최대 `1`, retry `0`
- watchdog `300s`
- Viewer timeout `240s`
- timeout 시에만 D383-owned child process group에 signal

## 3. 부팅과 Git 교차검사

실행 전:

- `HEAD`:
  `b880bc8f28c269f56f05a757dc725619d88c77b1`
- `origin/master`:
  `b880bc8f28c269f56f05a757dc725619d88c77b1`
- subject:
  `모델 change(grap당하는 원기둥)`
- D382의 승인된 미커밋 변경만 상속했다.
- D383은 별도 forward-only 경로와 새 script를 사용했다.

사용 script:

`sim_scripts/cyl34_top_view_d383_d382_rbl_data_value_verbosity_validation_resume.py`

## 4. 실행 전 독립검토와 사전등록

세 번의 read-only 검토를 사용했다.

1. D383 minimal-contract 감사
2. D382 evidence/schema 감사
3. D383 구현 source 사전감사

세 감사 모두 blocker `0`으로 수렴했다. 특히 다음 hidden second variable을
금지했다.

- `/presentation/d381/*`를 D383 이름으로 rename
- RRD/RBL merge 또는 regeneration
- external RBL override 추가
- Viewer window/layout/flags 변경
- strict RRD component-name validator의 `-v`까지 `-vvv`로 변경
- observed entity set을 expected set으로 되돌려 쓰는 자기검증
- strict helper와 별도 command에서 Viewer를 두 번 실행

embedded expected contract는 기존 권위와 정적으로 대조했다.

- exact non-system entities: `70`
- exact component contracts: `70`
- exact timelines: `blueprint`, `log_time`
- 기존 D380 validation과 entity/component/timeline 전부 exact

사전등록 결과:

- checks `20/20`
- negative controls `10/10`
- immutable input hashes `4/4`
- PASS

사전등록:

- `d383_preregistration.json`
- SHA-256:
  `bc2851f6690ee2299ac7290f34f695adf2749b4b06493fd602f18c940ac2ec34`

## 5. 관찰 가능한 순서대로 수행한 절차

### 5.1 네 입력을 bit-exact copy

Worker는 D382 원본을 새 D383 경로로 exclusive-create 복사했다.

| Subject | Bytes | SHA-256 | Result |
|---|---:|---|---|
| board | 230110 | `19bd70781403eb11c4eaefb6adb60ab91a5e6ca9f67f2929548f8afff0b7f06d` | bit-exact |
| layout JSON | 10877 | `7b961cdf8bd606c05438e120728fe243653262de81aa45947a1db0b1c03ab79c` | bit-exact |
| RBL | 69301 | `979ddd6b4a32bfc97e13d75dfb99625af0d0bed90fc1fd9588347667f284b28c` | bit-exact |
| presentation RRD | 305981 | `6c4ad99428f8da0ef842031b161e69db906971084da6d3444c1b76c8c27a7d9a` | bit-exact |

네 row 모두 `regenerated=false`다.

Manifest:

- `d383_bitexact_copy_manifest.json`
- SHA-256:
  `c59955f67bd6fbf030f672e7838b6182ec86624718de62cd239185720b779492`

### 5.2 strict RRD/RBL 구조 검증

`roarm_rl.rerun_contract.validate_rerun_artifact()`를 screenshot 없이 사용했다.
따라서 이 helper가 Viewer를 추가 실행하지 않았다.

결과:

- Rerun CLI exact `0.34.1`
- presentation RRD footer verify PASS
- RBL footer verify PASS
- exact non-system entity `70/70`
- exact component contract `70/70`
- exact timelines `2/2`
  - `blueprint`
  - `log_time`
- unexpected/missing entity `0/0`
- helper headless render attempts `0`

여기서 helper 내부 `rrd print -v`는 component column name을 읽는 기존 용도이므로
변경하지 않았다.

### 5.3 한 변수의 paired 검사

같은 copied RBL에 두 read-only command를 순서대로 적용했다.

음성대조군:

`rerun rrd print -v <copied RBL>`

- return `0`
- registered data-value markers `0/6`
- data-value authority로 사용하지 않음

등록 authority:

`rerun rrd print -vvv <copied RBL>`

- return `0`
- registered data-value markers `6/6`
  - notification-buffer query
  - summary query
  - link5 authored view name
  - link5 cooked view name
  - moving-side authored view name
  - moving-side cooked view name

Payload evidence:

- `d383_rbl_data_value_validation.json`
- SHA-256:
  `43ed74ffece41718d2ce7b312d2f6b880b2cec25419fc012e0944873d12c8313`

### 5.4 Viewer 정확히 1회

loopback preflight가 PASS한 뒤 동결 command를 한 번만 실행했다.

```text
rerun --headless --bind 127.0.0.1 --port auto
      --hide-welcome-screen --window-size 1920x1080
      --screenshot-to <D383 PNG> <copied presentation RRD>
```

결과:

- actual Viewer invocations `1`
- actual Viewer completions `1`
- automatic retry `0`
- return `0`
- elapsed `1.5463953481521457s`
- timeout false
- message-proxy permission error 없음
- non-fatal `libEGL warning: egl: failed to create dri2 screen`이 있었지만
  Viewer는 headless logical `1920x1080`을 기록하고 PNG를 정상 저장했다.

Viewer receipt:

- `d383_viewer_receipt.json`
- SHA-256:
  `94df92567fb3365eea1dbc561ad1df3f2954a6b00ff258cbe309d09c123aaf27`

Combined Rerun validation:

- `d383_rerun_validation.json`
- SHA-256:
  `a9cff4baafab9d73958f352e50632a1a83cf57b1024b6c0ce1d42b27e7909593`

### 5.5 원본 해상도 육안검사

다음 두 파일을 각각 `detail=original`로 직접 검사했다.

1. `d383_d382_board_bitexact_copy_1920x1080.png`
2. `d383_rerun_inspection.png`

정적 board:

- exact `1920x1080`
- geometry subject 두 패널, summary, 두 chart, title/tick/legend/footnote 확인
- decision-obscuring overlap 없음
- canvas clipping 없음
- 표시된 D380 facts 일치

Rerun screenshot:

- physical pixels `3840x2160`
- bytes `6674956`
- SHA-256:
  `e1f9dabec210738a54c80db2d00a8876ec6ae80a0cc71d07528d4bcda0591d6a`
- four geometry views visible
- D380 frozen-result summary visible
- visible timeline `log_time`; Unknown 없음
- notification은 오른쪽 빈 buffer에만 존재
- geometry/summary를 가리는 overlap 없음

`3840x2160`은 HiDPI physical-pixel 결과다. Viewer log의 logical window는
`1920x1080`이다. 정적 board만 exact `1920x1080` gate의 대상이며 Viewer PNG를
거짓으로 exact `1920x1080`이라고 기록하지 않았다.

Manual checks:

- `11/11` PASS
- `d383_manual_visual_inspection.json`
- SHA-256:
  `3ac4ebbdd3ce1c91bcc71c0acf2c01fec3b7e000e4ad634e663144bc9612cde6`

## 6. Worker와 종료 계약

Supervisor authority:

- actual worker `1`
- worker retry `0`
- return `0`
- elapsed `2.169500295072794s`
- actual Viewer start/complete `1/1`
- Viewer retry `0`
- timed out false
- SIGTERM false
- SIGKILL false
- child process-group residue false
- required artifacts 모두 존재

Supervisor:

- `d383_offline_worker_supervisor.json`
- SHA-256:
  `c471a870a29ef5fb39b07cc878819d286c27ff08a9d03d180b7768636311c48f`

단계표식은 prepare부터 supervisor completion까지 forward-only `12`개다.

## 7. 최종 판정

Completion verdict:

`D383_RBL_DATA_VALUE_VERBOSITY_VALIDATION_RESUME_PASS`

Completion checks `25/25` PASS.

Completion:

- `d383_completion_summary.json`
- SHA-256:
  `3d9dd6f0616e604e0f59e4d604d6a15d6e071f8a7ceee302ccc2452338570d97`

초보자용 뜻:

- D382의 화면 파일이 깨져 있던 것이 아니다.
- 잘못은 저장된 값을 보지 못하는 `-v` 출력에서 값을 찾은 검사 방식이었다.
- 값까지 보여주는 `-vvv`로 같은 파일을 읽자 필요한 여섯 값이 모두 확인됐다.
- Rerun 화면도 실제로 한 번 열어 네 형상 화면과 요약을 직접 확인했다.
- 따라서 D380 결과를 보여주는 presentation/observability chain은 이제 완결됐다.

## 8. 과학·물리 상태는 변하지 않았다

D383 PASS가 뜻하지 않는 것:

- P34 collider가 authored geometry와 같다는 뜻이 아니다.
- P34로 원통을 잡을 수 있다는 뜻이 아니다.
- 29×50 원통에 닿거나 쓰러뜨렸다는 뜻이 아니다.
- target/IK/path를 고쳐도 된다는 뜻이 아니다.

보존 상태:

- D380 numeric verdict:
  `D380_FAILED_PART_PROVENANCE_AUDIT_PASS_REPAIR_REQUIRED`
- P34 authored-to-cooked identity: false
- P34 representation repair: null
- repaired live identity: null
- actual OPEN jaw clearance: null
- 29×50 cylinder contact/tipping: null
- q5 closure: null
- grasp feasibility: null
- target/IK/path justification: null
- `g0a_pass=false`

Scope counters:

- Isaac launch `0`
- Kit launch `0`
- PhysX call `0`
- asset/USD read/write `0/0`
- collider regeneration/materialization `0`
- RRD/RBL regeneration `0`
- cylinder create/write `0`
- physics step/public forward `0/0`
- q5 command/sample `0/0`
- contact query `0`
- target/IK/path/pose change `0`
- material/mass/actuator/physics-setting change `0`

## 9. 다음 승인 경계

다음 과학 최소 후보는 아직 미승인이다.

`D384 [p34_failed_part_representation_repair_design]`

권장 범위:

- immutable D379/D380 identity/provenance evidence만 읽는 offline design-only
- 17개 inward-elided P34 part를 대상으로 가장 작은 representation repair 제안
- 신규 repair 변수 한두 개와 rejection gate를 사전등록
- 현재 `0.1mm`/topology-volume gate 완화 금지
- P34를 optimum이라고 부르지 않음
- Isaac/PhysX/asset materialization/live identity/cylinder/physics/q5/contact는 실행하지 않음

그 설계를 사용한 새 asset materialization과 live identity는 다시 별도 승인을 받아야
한다. 29×50 target rebase, mass/COM/inertia, mid-height pose, A64/P34 physics,
hold/lift도 각각 현재 승인 밖이다.
