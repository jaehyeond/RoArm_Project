# g0b_d420 — T2b 부속서: +12.117mm 실높이 수직 도구축 IK 재확인 (사전등록, 실행 전 저작)

작성: 2026-08-05 (20th 세션). 실행 전 고정. 이 문서 이후 프로브 스크립트 무변경 실행.
본 문서는 `t2_prereg.md`의 **부속서**다 — 아래 명시된 델타 외 모든 항목(도구축 정의,
자기검증 게이트, 판정 게이트, 스윕 범위, 관절 한계 2중 평가, D341 산출 계약, 한계 선언)은
본문을 그대로 승계한다.

## 근거

- `t3_conversion_design.md` **D-5**: Isaac env의 terrain plane은 z=0이고 TABLE_Z=-0.012117은
  계획 상수 → spawn 후 settle에서 원통이 +12.117mm 올라앉는다. p9의 settled replan이
  타깃을 재유도하므로, 그 실높이에서도 수직 도구축 도달성이 유지되는지 p9 사전등록 전에
  확인한다 (19th doc §4 부속 확인 항목, D421 Impl ④).
- 승인: 사용자 "T2/T3 진행 승인" (19th 기수령) 범위 내 부속 확인 — 신규 승인 불요.
  numpy+rerun 전용, Isaac 미기동, 로봇 미접촉, 학습 없음.

## 신규 변수 (Variable Ladder)

이번 부속서의 델타: [타깃 높이 +0.012117 m 시프트 단 1건] — T2 변수(수직 도구축 IK 스윕)의
높이 provenance 확인이며 신규 변수 축이 아니다.

## 스크립트 고정 (T2 이후 변경분 포함)

- `sim_scripts/p8_g0b_t2_cyld29h50_vertical_tool_axis_ik_reachability_probe.py`
- sha256 `bde79c01f4b01d2ecdca503404593edddc4a219b20e14e11725a677c4df7093b`
  (T2 실행분 `79884176…c5bbb`에서 변경: `--z_offset_m`/`--tag` 인자 추가 + 산출 경로·로그
  태그 파라미터화. **기본값 실행은 T2와 동일 동작** — 솔버·게이트·격자·자기검증 무변경.)
- 가드: `--z_offset_m ≠ 0` + 기본 태그 조합은 즉시 abort(exit 3) — T2 산출물 보호.
  스모크 확인 완료 (본 세션, 산출물 0).

## 고정 CLI (이대로만 실행)

```
python sim_scripts/p8_g0b_t2_cyld29h50_vertical_tool_axis_ik_reachability_probe.py \
  --z_offset_m 0.012117 --tag t2b
```

- 타깃: descend z = +0.038383 + 0.012117 = **+0.050500 m** / approach = **+0.090500 m**.
- 자기검증 게이트: **무변경** (고정 관절 자세의 FK — 타깃 높이와 무관하게 동일 밴드 재현 요구).
- verdict 토큰: `T2B_VERTICAL_IK_VERDICT=` + `T2B_PASS / T2B_PARTIAL / T2B_FAIL /
  SELF_CHECK_FAIL` (의미는 t2_prereg.md 판정 게이트와 동일 — 명명 후보 8개 중 ≥1개 양 한계
  PASS = T2B_PASS).

## 산출 계약 (D341, t2 본문 승계)

- `t2b_ik_stdout.log` / `t2b_ik_results.json` / `t2b_ik_grid.csv` /
  `t2b_ik_reachability.rrd|.rbl|_inspection.png` / `t2b_ik_rerun_validation.json` → 본 폴더.
- 검증기·엔티티·타임라인 계약 = t2와 동일. 육안검수는 세션 doc 관찰 기록으로 별도 수행.

## 판독 계획 (사전 고정)

- **주 질문**: T2 PASS 4후보(seed0_S1·S2·R1_center·R2_center)가 실높이에서도 PASS인가.
  4/4 유지 → p9 스폰 권고(seed0_S1) 그대로. 일부 탈락 → 잔존 PASS 후보로 스폰 교체.
  0/4 (T2B_PARTIAL/FAIL) → p9 사전등록 중단, annulus 경계 재판독 후 스폰 재설계.
- 부 질문: annulus 외곽 경계(r≈0.30~0.38)가 +12.117mm에서 어느 방향으로 이동하는가
  (기록만 — 판정에 불사용).
