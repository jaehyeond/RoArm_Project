# D351 — 현재 자세의 zero-step moving-jaw 닫힘 기하 판별

날짜: 2026-07-15 KST  
상태: 사용자 승인 / 독립 실행 전 검토 반영 / 아직 Isaac 실행 전  
이번 case의 신규 변수:
`[moving_jaw_actual_contact_surface_binding, frozen_pose_q5_closure_sweep]`  
신규 물리 변수: `[]`

## 1. 무엇을 왜 확인하는가

D350은 동결된 `(radial,tangent)=(7,11)mm`, q5 `1.5413rad` OPEN 자세에서
실제 fixed jaw의 `link5` surface, raw/live 원통 간격, 실제 Isaac Viewer와
`link5 64 + gripper_link 64` collider를 확인했다. 그러나 moving jaw가 닫힐 때
원통의 어느 feature를 어느 q5에서 처음 만나는지는 아직 측정하지 않았다.

D351의 질문은 다음 두 가지로 제한한다.

1. q0-q4와 원통을 D350 값으로 동결하고 q5만 OPEN→CLOSED로 zero-time 기록할 때,
   moving jaw의 실제 안쪽 날 면이 원통의 `barrel_interior`(위·아래 cap 사이의
   옆면 내부)를 먼저 만나는가, 아니면 top/rim·바깥면·housing이 먼저 닿는가?
2. 그 첫 접촉이 fixed jaw 쪽으로 aperture를 닫는 정확한 부호/순서 관계이며,
   첫 접촉 전 q5 구간에서 table과 엄격히 분리되어 있는가?

이 case는 동결 원통의 moving-jaw 첫 접촉만 본다. fixed jaw에는 D350 raw 기준
`4.2726455mm` gap이 남으므로 동시 양면 접촉, force closure, 물체 밀림, 실제 grasp,
settle 또는 G0a를 주장하지 않는다. fixed-jaw PCA body centerline을 원통 중심으로
강제하지도 않는다.

## 2. 실행 전 독립 검토에서 고친 핵심 오류

초안은 D350처럼 connected component(연결 성분)만 결합하려 했다. 그러나 원본
`gripper_link`는 `13,698` faces가 하나의 vertex/edge-connected component이므로,
이 방식만으로는 안쪽 날과 housing/바깥면을 구별할 수 없다. 실행 전에 다음 exact
surface contract로 교체했다.

- 원본 authored USD Float32/face-order hash를 권위로 고정한다.
- 안쪽 날: faces `672..1164`, `493` faces / `483` vertices,
  authored `y=-5.983415603637695mm`, normal `-local Y`.
- 짝 바깥면 negative control: faces `13205..13697`, 같은 `493/483`,
  authored `y=-4.483415603637695mm`, normal `+local Y`.
- 두 면의 projected XZ topology는 exact 동일하다.
- q5가 감소하는 닫힘 속도는 child local에서 `(y,-x,0)`이다. 안쪽 면은 모든
  점에서 `n·v_close>0`, 바깥면은 모든 점에서 `<0`이므로 새 거리/각도 허용값 없이
  어느 면이 닫힘-facing 안쪽 면인지 결정한다.
- raw first-contact tied faces가 모두 안쪽 face set이면 `intended inner`, 모두
  밖이면 `housing_or_nonpad_first`, 안팎에 걸치면 `AMBIGUOUS_FAIL_STOP`이다.

또한 초안의 overlap endpoint 검증, raw/live contact-order agreement, 실제 움직이는
Rerun jaw, app 종료 후 PNG decode/stability 계약 누락을 모두 실행 전에 보강했다.
최종 독립 검토에서는 last-clear witness 하나만으로 최초 면을 부르는 문제도 잡았다.
따라서 다음을 추가로 사전등록한다.

- raw overlap은 hpp-fcl `Contact.b1` authored triangle ID를 모든 비포화 contact에서
  직접 읽는다. `Contact.pos` 투영을 raw overlap face 권위로 쓰지 않는다.
- live clear witness와 모든 live overlap contact는 동일 q5의 full raw mesh에 투영해
  기존 `0.5mm` 이내인지, nearest tied raw faces가 같은 patch인지, 뒤집지 않은
  callback triangle normal이 frozen inner normal과 같은 반구인지 확인한다.
- raw/live clear와 overlap 네 endpoint의 surface identity가 exact 합의해야 한다.
- raw와 live 각각의 q_clear에서 inner surface를 제외한 complement와 원통의 거리가
  해당 bracket 전체 이동 상한보다 커야 한다. 각 inner surface와 원통 top/bottom cap
  disk도 같은 방식으로 분리되어야 하며, 아니면 최초 면/feature를 확정하지 않고
  `FAIL_STOP`한다.
- live-inner는 새 Qhull 외피가 아니라 D348 callback-topology 832 triangles 자체에서
  frozen inner plane·unflipped `-local Y`·source-vertex 결합·고정 face-key hash가 맞는
  40 triangles/17 parts만 고른다. 나머지 792 triangles와 경계 미해결 면은 모두
  competitor로 둔다. 실제 convex face 내부가 raw concavity를 메우는 정도는 별도
  진단으로 보존하고, 접촉점의 기존 raw/live `0.5mm` gate는 그대로 적용한다.
- 모든 raw/live overlap contact에서 cylinder endpoint feature를 각각 분류하며,
  `colliding_part_paths`와 실제 contact-row path set도 exact 같아야 한다.
- 대칭적인 negative first-feature certificate가 없는 cap/rim·housing·outer endpoint는
  `REPAIR`로 단정하지 않고 `FAIL_STOP`한다. `REPAIR`는 positive inner+barrel 최초접촉
  certificate가 완전한 뒤 pinch-facing 또는 table 기하가 실패할 때만 허용한다.
- table clearance는 raw와 live 64-part proxy 중 작은 값을, 두 표현의 큰 회전반경으로
  연속 인증한다.

## 3. 동결 입력과 허용 경계

- Base Git HEAD:
  `cfd9e7501df89724c3cc2b1038fda05ce0d88e2f` (`D350`)
- 출력: `claudedocs/runtime_logs/grasp_track/g0a_d351/`
- q0-q5 OPEN Float32:
  `[0.03750238195061684,0.542945146560669,1.9687392711639404,
  0.18299327790737152,0.0,1.5413000583648682]rad`
- 원통 Float32 pose:
  position `[0.30000001192092896,0.0,0.03288299962878227]m`,
  quaternion `[1,0,0,0]`
- representation: D348/D349 callback-topology `link5 64 + gripper_link 64`
- inherited gates: clear `0.1mm`, raw/live spatial agreement `0.5mm`
- seed `33201`; q5 `0=CLOSED`, `1.5413=OPEN`

자산·분해·q0-q4·원통 pose·목표·기존 허용값·재질·질량·구동기·물리 설정은
변경하지 않는다. `CONTACT_Q5_WIDTH_RAD=1e-6`, runtime pivot/axis 재현 `1µm`,
D350 face tie/binding residual, D337 CLOSED anchor `0.05mm`는 각각 root-search 종료,
입력 무결성 또는 비권위 진단 수치다. 새 grasp/alignment 성공 허용값이 아니다.
table은 새 margin 없이 clearance가 엄격히 `>0`인지 인증한다. 원통 feature도 새
1µm gate 없이 cylinder witness z가 두 cap 평면 사이에 엄격히 들어오는지만 본다.

## 4. 등록된 실행 순서

1. D349/D350 및 authored USD 입력 hash, Git scope, Isaac 호환 pin, 실제 GUI launcher,
   prepare/validate fresh-process 계약을 확인한다.
2. authored points/counts/indices와 안쪽/바깥 patch digest를 exact 재현하고, runtime
   raw face order/points가 authored face ID에 정확히 대응하는지 확인한다.
3. exact D350 OPEN 상태를 쓰고 D349 raw/live 거리와 D350 fixed component digest를
   재현한다.
4. q5 anchor를 Float32 `linspace(1.5413,0,33)`으로 단조 감소시킨다. 각 sample은
   `write_joint_state_to_sim` + `sim.forward` + `scene.update(dt=0)`만 사용한다.
   q0-q4/object bits, custom counter, timeline time, Isaac simulation-context time/index가
   모두 불변이어야 한다.
5. 각 interval에서 runtime 검증된 q5 pivot/axis 기준 최대 회전반경 `R`과
   `2 R sin(|delta_q|/2)` 표면 이동 상한으로 미표본 구간을 재귀 인증한다. raw와
   live 각각 OPEN 쪽 최초 clear→overlap bracket을 `<=1e-6rad`까지 좁힌다.
   모든 clear/overlap endpoint는 `exact_consistent=true`, finite witness,
   EPA 비포화이고 overlap endpoint는 유효 EPA contact여야 한다.
6. raw/live 두 bracket의 네 common endpoint에서 signed distance 차와 접촉-q5
   surface-travel 차를 기존 `0.5mm` gate로 직접 비교한다. 한쪽이 이미 overlap인
   구간을 비교에서 빼지 않는다.
7. raw clear tied faces와 모든 raw overlap `Contact.b1` face가 같은 patch인지 확인한다.
   live clear 및 모든 live overlap contact는 같은 q5 raw mesh에 기존 `0.5mm`로
   공간 결합하고, raw/live 네 endpoint surface identity 합의를 요구한다.
8. raw/live 각 q_clear에서 non-inner complement와 full cylinder, inner surface와
   top/bottom zero-height analytic cylinder disk 거리를 재어, 각각 그 표현의 bracket
   Hausdorff 이동 상한보다 엄격히 큰지 확인한다. 모든 overlap contact의 cylinder
   feature와 colliding/contact-row path set도 합의해야 한다. 이로써 bracket 안의 순간
   non-inner 또는 cap/rim 선접촉을 두 표현 모두에서 배제한다.
9. moving/fixed surface chord, 두 inward normal, q5-decrease velocity, cylinder support
   witness의 반평면 부호, 중심축 XY miss와 높이 차를 기록한다. 부호/순서만 eligibility
   gate이며 각도·높이·axis miss 수치는 measurement-only다.
10. first-contact 전 raw/live union table clearance를 두 표현의 최대 회전 이동 상한으로
   연속 인증한다.
   link5↔gripper adjacent raw distance는 parent-child diagnostic-only이며 다른 로봇
   구조물 전체의 contact order를 증명했다고 주장하지 않는다.
11. 실제 `headless=False` Isaac Viewer에서 OPEN과 resolved decision pose 또는 명시적으로
    표시된 OPEN fallback의 actual PhysX collider,
    colored `64+64`, frozen inner patch, fixed/moving chord·normal·q5/cylinder axis를
    캡처하고 UI만 유지한다.
12. Rerun에는 `64+64` Mesh3D를 resolved decision pose 또는 명시적으로 표시된 OPEN
    fallback의 정적 문맥으로 두되, 매 q5 step마다 dense live gripper surface, frozen
    inner patch, runtime pivot/axis, raw/live witness를 실제로 움직여 기록한다.
    RRD/RBL footer/exact entity/component/screenshot 계약과 원본 해상도 수동 검사를
    완료한다.
    finalized RRD를 다시 읽어 각 필수 dynamic entity의 `step=0..N-1`가 정확히 한 번씩
    보존됐는지도 확인한다.
13. validate process 종료 뒤 Viewer 4장과 Rerun 1장을 RGBA full decode/load하고,
    세 번의 hash/size/mtime 안정성을 확인한 뒤에만 finalize한다.

## 5. 판정 어휘

- `D351_CURRENT_PREGRASP_BARREL_CLOSURE_ELIGIBLE`: raw/live 접촉 bracket과 기존
  `0.5mm` agreement가 모두 유효하고, raw first contact가 frozen inner patch의
  `barrel_interior`이며 live도 `barrel_interior`, exact pinch-facing 부호/순서와
  연속 table `>0`이 모두 성립한다. 이는 별도 single-close 물리 case 후보라는
  뜻뿐이다.
- `D351_CURRENT_POSE_CLOSURE_GEOMETRY_REPAIR_RECOMMENDED`: raw/live 모두에서 moving-inner
  `barrel_interior` 최초접촉과 competitor exclusion까지 완전하지만, 그 뒤
  pinch-facing 부호/순서 또는 table strict-clear가 실패한다. 다음 별도 target/IK
  geometry-repair를 추천하되 D351에서는 바꾸지 않는다.
- raw/live/EPA/patch identity/competitor exclusion/zero-step/관찰 계약이 불완전하거나
  clear/overlap surface identity가 애매하거나, 대칭 negative first-feature certificate
  없이 cap/rim·housing/nonpad·outer endpoint가 나오면 해당 `FAIL_STOP`으로 끝내며
  최초 feature 또는 repair 결론을 만들지 않는다.

`ELIGIBLE`도 실제 접촉력·마찰·원통 이동·양면 동시 접촉·force closure·grasp·hold/lift
또는 G0a PASS가 아니다. `g0a_pass=false`를 유지한다.

## 6. 불변 경계와 실패 가능성

- raw first, live callback-topology second; direct PhysX narrowphase 거리라고 부르지 않음
- source asset write/cook/property query/분해 변경 `0`
- controlled physics step/timeline play/dt>0 update `0`
- target/IK/path/기존 허용값/재질/구동기/물리 설정 변경 `0`
- settle/10-trial/G0b/RL/PPO/ladder `0`; `g0a_pass=false`
- D337-D350 산출물 덮어쓰기·silent rerun `0`
- commit/push `0`
- D351 prepare 전에 안정화된 사용자 소유 untracked
  `claudedocs/lab_meeting/20260715/d334_collision_table/{README.md,d334_collision_table_academic.html,d334_collision_table_academic.png}`는
  과학 입력이 아니며 수정하지 않는다. prepare/validate/finalize에서 최초 SHA-256과
  Markdown/HTML의 `??` untracked 상태 및 `.gitignore`가 적용된 PNG 상태가 exact
  불변인지 별도 확인한다.

이번 session은 RL update 대신 실제로 판정을 바꿀 수 있는 q5 perturbation 평가를
실행한다. positive inner+barrel certificate 뒤 pinch/table 기하가 실패하면 REPAIR이고,
cap/rim·다른 surface 또는 계약 불완전은 FAIL_STOP이므로 이 평가는 실패 가능하다.

## 7. 실행 결과

첫 `prepare` 시도는 D351 출력을 쓰기 전에 사용자 소유 `lab_meeting` 파일의 해시가
사전 관찰값과 달라져 의도대로 STOP했다. 확인 결과 기존 두 파일이 갱신되고 PNG 한
개가 추가된 상태였으며, 15초 재측정에서 세 파일의 hash/mtime이 안정됐다. 세 파일을
새 현재값으로 read-only 동결했으며 D351 output은 여전히 absent, Isaac 실행은 `0`이다.
이제 corrected `prepare`부터 진행한다.
