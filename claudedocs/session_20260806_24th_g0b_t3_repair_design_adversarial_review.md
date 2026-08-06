# 2026-08-06 (24th) — G0b T3: 자산 수리 검증 설계 v1 + 4-렌즈 적대검증 (사용자 A/B/C/D 해금 발언 수령, DESIGN_V1_READY — 확인 3건 대기)

이번 case의 신규 변수: [없음 — 본 세션은 설계·검증 계층만. Isaac 기동 0, 자산 변경 0,
코드 변경 0, 로봇 HW 0, lerobot-train 0, git 0. 설계 v1이 **향후** 도입할 신규 변수를
P(플러그 비활성) × F(손가락 파트 증분) 2개로 확정하는 것이 본 세션의 산출.]

## §0 부트

Current-State Protocol 6단계 이행(23rd판 기준). 권위 JSON(`t3_jaw_audit3_results.json`)
수치 전건 재대조 — 불일치 1건: START_HERE:68 "/half-clone 거부 10회" vs 23rd doc:149
"11회" (START_HERE 1세션 지연 → 24th판에서 정정, 현재 12회).

## §1 사용자 해금 발언 (verbatim — D426 기록 전 증거 보존)

> "(A),(B),(C),(D) 하나한 다해봐. 그럼 아니면 병렬로 해보던가. 그럼 문제를 찾는 변수가
> 늘어나나? 4개 다 해금해줄 수 있어. 그렇다면 어떻게 해야 환각이나 오버라이드 등 문제가
> 안생기게 제대로 이거 검증을 통한 결과를 도출 할 수 있을지 고민해봐. step-by-step으로
> 순차적으로 사고하면서 말이야." (2026-08-06, 24th 세션)

해석 주의(HARD RULE #18): "해금해줄 수 있어" = 해금 의사 표명 + 검증 설계 요구.
lead가 옵션 집합을 수정 제안(C 예비 강등 + F-arm 신설)했으므로 **공식 발효(D426)는
§6 확인 3건 수령 후**. 컴플라이언스 렌즈 판정: 해제 권한은 사용자에게 유보돼 있어 발언은
유효한 해제 행위가 될 수 있으나, repo 미기록 상태로는 발효 취급 금지(다음 세션 검증 불가).

## §2 설계 v0 (lead 단독 저작 → 적대검증 대상)

2×2 요인(P×F) + Arm-A 복제 + Arm-C 예비 + 공통 기하 게이트(23rd 스크립트 개조) +
사전등록 + Isaac 순차. v0 전문 = 워크플로우 스크립트 내 PLAN 블록
(`~/.claude/projects/.../workflows/scripts/g0b-t3-repair-plan-adversarial-review-wf_67ffd8b5-cb9.js`).

## §3 적대검증 워크플로우

- `wf_67ffd8b5-cb9`: 4 agents 병렬(①기하/예측 ②자산 실행가능성 ③컴플라이언스/권한
  ④실험 방법론), 에러 0, 562,467 tok, 69 tool uses, 735s.
- 전문 영속화: `g0b_d420/t3r_design_review_wf_67ffd8b5_findings_raw.json` (~90KB).
- 판정: **4/4 PLAN_NEEDS_CHANGES** (UNSOUND 0). FATAL 1(방법론), MAJOR 다수.

## §4 적발 결함 → v1 수리 (요지, 전문은 findings raw JSON)

| # | 결함 | 수리 |
|---|---|---|
| 1 | **[FATAL] Arm-F 플러그-정지 전제 = GRASP_PASS 계약 모순** — descend 목표 미달 정지는 target_error 3mm 위반 = APPROACH_FAIL 승계(t3_prereg.md:10-12,132-133) → 물리적으로 잡혀도 실패 코딩 | F는 hover 튜플(margin ≥ +4.5mm, attempt2 선례 5.5mm)로 재설계. 관계식 **x(물림)=L(손가락 길이)−m(margin)**; T1 정합 x∈(0,12] ⇒ L∈(5.5,17.5]mm, 권장 L=9.5~13.5mm |
| 2 | **[MAJOR] 대향면 누락** — link5 고정 조도 TCP 아래 벽 반경 구조물 전무(part_026/027/031이 TCP 위 1.6~11.7mm 절단, t3_jaw_audit3_parts.csv:28,29,33) | F축 = **양쪽 body**(link5+gripper_link) 원위부 저작, body별 L≥m+x 각각 게이트 |
| 3 | **[MAJOR] 판정 매트릭스 오류** — "D pass+B fail→각각 필요"는 범주 오류(P-제거 필요성은 F가 시험) | (B,F,D)×{pass, fail-consistent, off-prediction} 전 조합 표 사전등록. "각각 필요"는 B실패∧F실패∧D성공에서만. F∧D 동시 성공 tie-break = F 기본 채택(최보수). **B 예상외 성공 = 모델 반증 = GEOMETRY_MISMATCH급 전면 정지** |
| 4 | **[MAJOR] 저작 소스 미검증** — 시각 메시에 원위 손가락 실재를 지지하는 repo 측정 0건(D368 정황상 부재 리스크) | **Gate-0 신설**: 저작 前 읽기 전용 시각-메시 깊이 감사(Kit 0). 소스=gripper_link.stl(sha 7946a374)+link5.stl(1d63f374); **gripper_left_link.stl은 URDF 비참조 죽은 자산 — 사용 금지**. 부재 시 분기(수제 저작 승인 or C 재평가) |
| 5 | **[MAJOR] 현행 p9로 전 arm 즉시 abort** — attempt3 경로/sha/64파트 감사 하드코딩(p9:138-145,612-631,815-832) | p9 1회 파라미터화 + **개조 기하 게이트 스크립트와 함께 D423 동일 강도 적대검증 → sha 핀**(게이트 신기능: 파트-제외 프로파일[플러그 마스킹 해소], 밴드 확장, 수평 간극 지표, 자기-관통 검사 q5∈[23°,90°]) |
| 6 | **[MAJOR] Arm-B 예측 차용 불가** — audit3 profile은 플러그가 잔여 기하 마스킹(min_r=10.026 상수) | B 예측은 **B 자산 게이트 재실행으로만** 산출. a2 튜플 주 leg(무접촉 LIFT_FAIL 예측) + **a4 튜플 판별 leg(part_031 신규 정지 top−1.6mm 예측 = 독립 판별 앵커 2호)** |
| 7 | [MAJOR] 결정론 무근거 — 동일 튜플 반복 실행 기록 0회(a1 vs a4 4μm는 상이 튜플 간 불변성) | 반복성 leg 1회 사전등록(B 첫 튜플 재실행, 별도 tag, 결과 불문 전량 보고). 산포>게이트 1/5이면 허용오차 재산정. NVIDIA 결정론 주장은 버전 일치 공식 문서 없이 서술 금지 |
| 8 | [MAJOR] P축 비순수 — part_029/030은 고정 조 seed-plane certified carrier 4개(027/029/030/031) 중 2개 | P-제거 후 잔여 커버리지(027/031) 정량 사전등록 + D 실패 시 "F 불충분" vs "P-제거 부작용" 판별 앵커 명시 |
| 9 | [MAJOR] Arm-A "동일 2변수 독립 경로" 불성립 — 동결 d338 config 재실행은 동일 기하(원위부 부재 포함) 재생산 | A = 요인 해석 불산입 복제/전패-fallback leg로 강등. 변경할 분해 파라미터를 명시적 신규 변수로 사전등록. cook 재실행은 d339 수리판 witness 계약 경유(d338 attempt1 STOP 전례) |
| 10 | [MAJOR] baseline 이질성 — 기존 4회 = p9 sha 3종 × margin 3종 × 개방각 2종 | "요인 셀"이 아니라 "시그니처 재현 대조 증거"로 층위 강등. 비교 앵커 = a4(45°/−7.5mm) 지정. per-arm 튜플을 부록 D에 명시 |

기타(MINOR/INFO): 순서 고정보다 **첫 Isaac 전 B/F/D 일괄 사전등록**이 정보 누출을 구조
봉쇄(순서는 경제 문제로 격하, B 선두는 유지 가치) / GEOMETRY_MISMATCH 발동 = 범주
불일치(정지/접촉/verdict 클래스) 즉시 + 정량(정지 z ±1.0mm, 접촉 시작 ±1 이산 스텝) /
rim 물림 깊이는 Isaac 비측정 파생량 — 발동 조건 제외 / |마진|<0.5mm 예측 = indeterminate /
전면 정지 기본 + 사전등록 triage(공유 모델 원인=전면, arm 국소=해당 arm만) / 신규 파트
저작 계약(convex ≤64정점 cook-faithful, offset 미저작=0, world/link1~4 legacy collider
5개 명시 확인[D425 ③]) / 게이트 실행 ledger(append-only 전수 공개) / 허용오차 선고정 +
소급 변경 무효 / Isaac 실행기 sha==게이트 sha hard-fail / prereg 개정 = 트리거 데이터
명시 + 실행 전 발행분만 유효 / 게이트 자체도 D341 RRD+육안검수 대상 / F의 결정론적 저작
레시피 prereg 핀(1변수 성립 조건) / 신규 출력 경로 START_HERE 등재 의무 / 도구 설치 시
D326 절차 / Kit 2개 동시 실행 증거 없음 → Isaac 순차 유일 정합.

**컴플라이언스 핵심 판정(D426 저작 시 사용)**: D415 ③의 금지 대상 = "재분해(파트 번호
재배열)"뿐. 파생 사본 + 증분 저작은 원래 금지가 아니라 승인 게이트 사항 → **해제가 필요한
것은 Arm-A뿐**, B/F/D는 저작 승인으로 족함 — 단 재분해 비해당 성립 조건 3가지(원본 무변경 /
기존 64+64 명명·번호 verbatim 보존 감사 / 신규 파트 별도 네임스페이스)를 prereg 게이트로
명문화. supersede 앵커는 3중: D415 ③ + D419 ⑦ + D420-R1(하나라도 남으면 다음 boot이
미해제로 읽음). Arm-C는 3층 분리: 실행(사용자 해금+정의 확정) / 채택·T4 승격(교수 재승인).

## §5 설계 v1 파이프라인 (확인 후 실행 순서)

0. D426 기록(해금 verbatim+범위표) + 3중 앵커 scoped-supersede + START_HERE 갱신
1. Gate-0 시각 메시 깊이 감사(읽기 전용) → 소스 실재 확인/분기
2. p9 파라미터화 + 기하 게이트 v2 저작 → D423 적대검증 → sha 핀
3. arm 자산 저작(B/F/D) + 빌드 manifest → 적대검증 → sha 핀
4. 부록 D 일괄 발행: B/F/D 예측·허용오차·튜플 동시 동결(A는 도달 시 번호 개정)
5. Isaac 순차: B(a2)→B반복성→B(a4 판별)→F→D→[조건부 A], 각 회 D341 검수
6. 조합 표 판정 → 채택 자산 확정 → T4 실물 재현 대조

## §6 승인 경계 — 확인 3건 (미수령 상태로 세션 종료)

① Arm-C 예비 강등 동의(사유: D419 교수 지시 충돌 소지 + 현 자산에서 기하상 무익.
   방법론·컴플라이언스 렌즈 모두 "사용자 해금이 교수 지시를 무효화하지 않음" 판정)
② F-arm 신설 + F∧D 동시 성공 시 F 기본 채택(tie-break) 동의
③ 해금 발언의 D426 공식 기록 진행 동의
— 확인/수정 지시 수령 전 어떤 arm 작업도 착수 금지.

## §7 세션 진행 규칙 이행

실패 가능 실험 = 설계 v0에 대한 4-렌즈 적대검증(실제로 FATAL 1·MAJOR 10으로 부분 기각·
수리됨). Isaac 미실행 사유 = 실행 대상 자산이 아직 없는 설계 계층 + 착수가 #18 델타 확인
3건에 게이트됨. NO_PPO_PROMOTION 해당 없음(물리 verdict 아닌 설계 verdict).

## §8 산출물

- `g0b_d420/t3r_design_review_wf_67ffd8b5_findings_raw.json` (4렌즈 전문 영속화)
- 본 doc, START_HERE 24th판, LEDGER row, MEMORY 24th entry
- 코드/자산/prereg 변경 0 (설계만). **/half-clone 거부 12회째(#11, stop-hook 92% 지시에도
  — 본 end-of-session update + continuation prompt로 대체).**
