# 새 scoop 1회 · 잔류 배출 기울임 · HOME 복귀

**사용자 요청한 동작은 HOME 복귀까지 완료했다.** 초기 HOME에서 새로1회 퍼내고 컵으로 운반·개방한 뒤, 손목으로 잔류 배출 기울임을 실행하고 문을 닫아 HOME으로 돌아왔다. 앞서 중간 정지4건을 거쳐 경로를 수정했으므로 무중단 재현 성공은 아니다. 이번 잔류 개수와 무게는 아직 사용자 입력 전이다.

이번 case의 신규 변수: [기존790 scoop에 잔류배출·HOME 복귀 연결]. PID/토크/형상 변경 없음. 실패에 반응한 복귀 경로 수정과 시작 예외는 초기보고 및 세션에 명시했다.

## 실제로 한 순서

1. 직전 배출 자세에서 복귀 후 HOME [0,0,90,0,0,0] 명령. 새 plunge1회, 닫힘790, lift8. 리프트 중 문 최대 추가 개방0°.
2. 중간 어깨 추종정지를 기록하고 기존 P1로 복귀해 컵 위로 운반·문30° 개방.
3. 마지막 `execution_05`: 직전 통과raise1목표로 되돌림→어깨/팔꿈치 목표고정·롤90정렬→손목피치20° 감소→3초배출대기.
4. 역기울임→롤0→문목표0→직립P1→베이스0→HOME. 최종 `completed=true`, `home_reached=true`. 포트닫힘, 마지막 토크200 유지.

## 검증된 수치

| 항목 | 결과 | 정본 |
|---|---:|---|
| 새 scoop 수 | 1회 | combined_02/command_audit.json |
| 초기·최종 HOME 명령 | 각각1회, [0,0,90,0,0,0] | 같은 감사·raw |
| 최종 배출/복귀 구간 | 83.538273s · 4215행 | execution_05/analysis.json |
| 최종 구간 최대 settled 팔 오차 | 3.463352° (기준5° 이내) | execution_05/raw.jsonl settled |
| 배출 직전→기울인 후 출구면 경사 | 7.878947→22.529027° | execution_05/analysis.json |
| 기울임 후 실제 공구 기울기 | 15.029297° | 같은 파일, 관절FK |
| 배출 중 문 실제각 | 29.179688° 유지 | 같은 파일 |
| 최종 HOME 실제각 | [1.406250,2.988281,91.318359,1.054687,-0.175781,2.724609]° | execution_05/result.json |
| 전체 피드백 | 11029행 | combined_02/analysis.json |
| 전체 구간별 기록시간 합 | 217.040311s | combined_02/result.json |
| 이번 컵 포함 무게·잔류·영점 여부 | 미입력 | operator_measurement_pending_01.json |

손목20° 명령과 실제 공구15.029297°는 구분한다. 관절FK 기반 출구 경사/립변위는 외부 위치센서 실측이 아니다. 끝단은 약67mm 이동했고 컵은 사용자가 맞춰 받는 조건이다. 문 목표0은 장착물의 접촉으로 실제0°와 일치하지 않을 수 있으며 목표를실제값으로복사하지않았다.

## 중간 실패를 보존함

execution_01~04의 어깨 편차는 각각5.107378°,5.148438°,5.185547°,5.102327°로 정지했다. 원래 고정립점 상승/운반 경로가5°검사를 계속 통과한다는 주장은 성립하지 않는다. 마지막 구간은 같은 목표를 무작정 반복하지 않고, 이전통과자세에서 어깨/팔꿈치 목표를 고정한 손목기울임으로 완주했다. 실패원시와동결드라이버/초기보고 `REPORT_initial_stop.md` 및 `combined_01/`은그대로보존했다. 전체5실행구간을5회scoop로세지않는다.

첫~마지막 전체시각폭은 2202.242471s이며 포트닫힘공백4개가있다. 센서가관측한실행구간합217.040311s와같지않다. 소스5개원시바이트를순서대로연결한combined_02 SHA256 `fdcbe593270d1b6a292c053a601cec8bb3d1d5814217705da7d23a3d9720180b`. 개별해시/시작·끝시각은source_segments에보존했다. 원시수신은호스트시각이고서보내부시각이아니다.

## 자료와 판정

- 전체 `combined_02/feedback.csv`, `raw.jsonl`, `analysis.json`, `command_audit.json`, `result.json`.
- 재생 `combined_02/visual_01/release.rrd`와 `.rbl`; 최종구간은 `execution_05/visual_01/`.
- Rerun SDK/CLI0.34.1, footer·정확entity/timeline/component·전체4215/11029행읽기대조PASS, 스크린샷실제검수. `inspection.json`에보이는내용과한계를기록. 최종구간timeline.png에서기울임·대기·역동작·HOME까지확인했다. Rerun시작커서의차트는일부시간창만보일수있고공백보간은실측이아니다.
- 실행05 계획62단계는 **구동60+명시조회2**다. 기존감사JSON의`planned_actuator_commands=62`는단계수라필드이름이부정확하며`command_count_erratum.json`으로명시보완했다. 실제발행구동60개는계획의구동prefix와정확일치한다.
- 이전PID원시보존, 이번각도/부하/T105원시도수집. 새PID쓰기0; tG/전류/온도나레지스터직접읽기는없다.
- 이전사용자“알다떨어졌어”는직전outlet_tilt_01 관찰0알이다. 이번전량배출/질량으로복사하지않는다.
- **판정:** `ONE_SCOOP_DISCHARGE_MOTION_AND_HOME_COMPLETED_WITH_RECOVERIES__UNINTERRUPTED_REPLAY_NOT_PROVEN__NEW_MASS_RESIDUE_PENDING`.
- Git commit/push 없음. 사용자가push를직접한다. 다음입력은이번최종컵무게·고정jaw잔류알수·컵비움/영점여부다. 조건고정5회완료로세지않는다.
