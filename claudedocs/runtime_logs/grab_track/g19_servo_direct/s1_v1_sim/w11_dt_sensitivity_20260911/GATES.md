# Gates: W11 dt sensitivity

Scope: 기존 W10 2μs 결과와 신규 1μs 한 셀의 비교, 원자료·Rerun 검증·육안 검수·종료 인계.

- [x] G1: 여섯 입력 해시와 설치 버전이 확인되고 설정 차이가 dt 및 새 출력 경로뿐이다.
  CHECK: /home/cgxr/miniconda3/envs/roarm/bin/python claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w11_dt_sensitivity_20260911/w11_workflow.py verify preflight
  EXPECT: W11_PREFLIGHT_VERIFIED
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/home/cgxr/Documents/Robotics/RoArm_Project; path=0295630a4757/19 entries; EXPECT=matched; output-sha256=cc45e415d4f139ef0891696fd97ca8afdad3a46288192011b3e47c912a5ba13f; output-bytes=23

- [x] G2: 신규 실제 물리 실행의 종료 상태와 비교 수치가 원자료로 검증된다. 과학 결과가 불리해도 숨기지 않는다.
  CHECK: /home/cgxr/miniconda3/envs/roarm/bin/python claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w11_dt_sensitivity_20260911/w11_workflow.py verify result
  EXPECT: W11_RESULT_VERIFIED
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/home/cgxr/Documents/Robotics/RoArm_Project; path=0295630a4757/19 entries; EXPECT=matched; output-sha256=fd84956b023ac7f55aabf267a9c302d6e70fb7061269a1790f456fc5d69c8960; output-bytes=20

- [x] G3: Rerun 0.34.1 파일 종료·footer·엔티티·타임라인·컴포넌트·원자료 대조와 RBL 검증이 완료된다.
  CHECK: /home/cgxr/miniconda3/envs/roarm/bin/python claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w11_dt_sensitivity_20260911/w11_workflow.py verify observability
  EXPECT: W11_OBSERVABILITY_VERIFIED
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/home/cgxr/Documents/Robotics/RoArm_Project; path=0295630a4757/19 entries; EXPECT=matched; output-sha256=dd9e053fcc751b130545868cf8922d644df67f1a4641cc733d85ded3664a2f9d; output-bytes=27

- [x] G4: 결정 시점 Rerun 화면, 목표/실제 프레임 및 높이맵 비교 그림을 직접 열어 관찰과 한계를 기록한다.
  EVIDENCE: 2026-09-12 view_image로 PNG4개 실제 확인. 같은 폴더 inspection.json에 절대경로/SHA256/개별 관찰 기록. Rerun 동적초기·최종정적·첫닫힘정적 시간구분, post고립높은셀/초기맵동일, 축라벨중첩/미세각수치판독 한계 명시. 재닫기별도정적검수라고 주장하지 않음.

- [x] G5: 상태·원장·세션·relay가 최종 결과와 일치하고 기존 입력/산출물·원장 prefix가 보존된다.
  CHECK: /home/cgxr/miniconda3/envs/roarm/bin/python claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w11_dt_sensitivity_20260911/w11_workflow.py verify closeout
  EXPECT: W11_CLOSEOUT_VERIFIED
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/home/cgxr/Documents/Robotics/RoArm_Project; path=0295630a4757/19 entries; EXPECT=matched; output-sha256=7777b9b5da278cf5e988b32810fa63f94823630d2785e902e32cef2106dcb701; output-bytes=22
