# 코디네이터 인계 — P4 트랙 (Isaac PBD / Newton MPM / 그랩 포획)

작성 2026-09-02 · 워커 세션. **원장(`START_HERE.md`, `claudedocs/DECISIONS*.md`,
`EXPERIMENT_LEDGER.md`, `session_*.md`, `relay/`)은 코디네이터 배타 소유이므로 이 워커는
건드리지 않았다.** 아래는 그대로 옮겨 쓰라고 정리한 것이지 등재된 결정이 아니다.

브랜치 `jaehyeond/pellet-model`, 커밋 `6f16af3` ~ `adf9a46`.
보호 파일 무변경 확인: `git diff --exit-code -- sim_pellet_model.py sim_deme_pile.py
sim_deme_scoop.py scoop_grab_v1_design.py roarm_rl/heightmap.py GATES.md START_HERE.md
claudedocs/DECISIONS*.md claudedocs/EXPERIMENT_LEDGER.md claudedocs/relay` → 통과.

---

## 1. 결정번호가 필요한 항목 (제안)

### D-a. Isaac PBD 기각 — 사유를 D457 에서 교체할 것

D457 은 "GPU 파이프라인에서 입자 상태 readback 불가"를 기각 근거로 적었다. **그 전제는
틀렸다.** readback 은 되고(표준 standalone + `UsdGeom.PointInstancer`), 애초에 필요도 없다
(실물처럼 깊이 카메라로 관측하면 된다 — 깊이 경로가 해석 연산자와 rms 0.60~1.35 mm 일치).

진짜 사유로 교체할 것: **PBD 에서 안식각은 재료 성질이 아니라 타임스텝의 함수다.**
마찰 0.60·감쇠 0·점착 0 고정에서 서브스텝만 60→1920 Hz 로 바꾸면 **8.74° → 32.27°**,
마지막 반감에서도 11.51° 변해 수렴하지 않는다. 6개 타임스텝 **전부 정착 실패**.
960 Hz·감쇠 5.0(최선)에서도 크리프 −0.093 °/s.
근거: `claudedocs/runtime_logs/pbd_probe/REPORT_pbd_probe.md`, `gates_pbd_probe.json`
(G1~G5 전부 FAIL, 사전등록 임계값은 `GATES_pbd_probe.md`).

### D-b. 깊이 렌더 관측을 DEME 쪽에도 이식할 것 — 정량 근거 확보

P3 의 27개 DEME 더미(8.90~41.77°)를 같은 깊이 카메라로 재관측했다.
**카메라 자체의 높이 편향은 평균 −0.019 mm(|max| 0.038 mm), 각도 차이 최대 0.34°.**
겉보기 +1.44 mm / 최대 +3.26° 는 전부 **격자 샘플링 규약 차이**(P3 는 셀 중심값,
카메라는 셀 최댓값)였다. 지금 구조(DEME=중심샘플 / 실물=카메라)에서는 물리가 완벽해도
그 차이가 GP 잔차에 계통 오차로 남는다.
근거: `claudedocs/runtime_logs/pbd_probe/depth_on_deme/REPORT_depth_on_deme.md`.

### D-c. Newton MPM 은 재료로서 유망 — 단 채택 전 남은 관문 하나

별도 conda env(`newton` 1.5.1 / warp 1.17.0). **`isaaclab` 은 무변경, D326 핀 재확인 통과**
(numpy 1.26.0 / psutil 5.9.8 / warp 1.11.1 / isaacsim import OK).

* Q1 마찰 단조성 **PASS**: μ 0.20→1.40 에서 4.85→34.77°, DEME 실측 범위를 전부 덮는다.
* Q2 정착 **PASS(10셀 중 9)**: 전 셀 1.25~1.50 초 정착, 크리프 ±0.002 °/s 전형
  (PBD 최선의 1/47). 최악 셀 −0.0116 °/s.
* Q3 dt 무관성 **PASS**: dt 4배 변경에 0.62° (게이트 2.0°). PBD 는 dt 2배에 11.51°.
* 속도: sim 130 초를 벽시계 79 초(실시간 1.64배). **DEME 대비 단위 sim 시간당 약 38배.**
  VRAM 프로세스당 424 MiB → **로컬 RTX 4090 Laptop 으로 충분, RunPod 불필요.**
* **남은 관문**: 격자 의존성. 10/5/2.5 mm 에서 17.28/20.99/22.91° → **4배 범위에 5.63°**
  (dt 민감도의 9배). 격자를 잘게 하면 셀당 펠릿이 15.6→0.24 로 줄어 연속체 가정에서
  더 멀어지므로 정제로 풀리지 않는다.
근거: `claudedocs/runtime_logs/newton_mpm/REPORT_N1.md`.

⚠️ Newton 1.5.1 에 **DEM 솔버는 없다**. SIGGRAPH 2026 발표(15:46) 슬라이드가
"Particles: DEM, MPM" 이라 적지만 출하 클래스 9개에 DEM 은 없고 소스에 문자열 0건이다.
그리고 "Available now in Isaac Lab 3.0 beta.2" 라 적힌 커플링 경로는 PyPI 최신
`isaaclab` 이 2.3.2.post1 이라 **오늘 pip 로 열리지 않는다.**

### D-d. 🔴 그랩 v1 은 자유유동 재료를 담지 못한다 — 설계 사안

세 엔진(DEME·PBD·Newton MPM)이 모두 포획 0 을 냈고, **그 0 이 전부 맞는 답이었다.**
형상만 검사(삼각형당 435~861점 0.5 mm 격자 래스터화 + 바깥 flood fill)한 결과
**개폐 전 구간(0~44.5°)에서 밀폐 공동이 0.28~0.37 cm³** 이고 위치가 상단 플랜지다.

그랩 v1 은 힌지축 Z 둘레로 닫히는 **수직 관**이며 세 방향이 열려 있다:
1. **평면상 뒤쪽** — 두 호가 피벗 (±13, 0) 에서 시작하므로 **피벗 사이 26 mm 가 개방**
   (기어 맞물림 자리). 평면도가 말굽(C자).
2. 위 — 관의 입
3. 아래 — 바닥판 없음

바닥판·상단판을 붙여도 밀폐 공동이 안 는다(0.36→0.37 cm³). `design.json` 의
`inner_volume_cm3 = 70.34` 는 **단면 1,407 mm² × 폭 50 mm 각기둥 추정**이라 이 개방부를
반영하지 않는다. 담게 하려면 **(a) 피벗 간극을 가로지르는 뒷벽 + (b) 바닥 폐합**이 둘 다
필요하다. 시뮬 파라미터로는 해결되지 않는다.
근거: `b1_scoop/grab_cavity_scan.json`, `b1_scoop/endwall/endwall_cavity_scan.json`,
`b1_scoop/grab_xy_sections.png`, `b1_scoop/grab_xz_sections.png`.

---

## 2. 이 워커가 낸 **철회** (원장에 남길 것)

기술 주장을 성급히 단정했다가 뒤집은 것이 4건이다. 재발 방지를 위해 남긴다.

| # | 철회한 주장 | 실제 |
|---|---|---|
| R1 | "Newton MPM 은 이 도구를 표현할 수 없다" | `setup_collider(collider_margins=...)` 호출을 개수 오류로 실패시키고 안 고친 채 결론냈다. `ShapeConfig.margin` 기본 0.0 이라 셸이 두께 0 으로 들어갔다 |
| R2 | "충돌체가 복셀보다 훨씬 두꺼워야 한다" | 최소 컵 시험에서 **복셀 5 mm, 벽 1.5 mm → 98.8 %** 담긴다. NVIDIA 예제 비율에서 끌어낸 추론이 틀렸다 |
| R3 | "셸 벽이 1.03 mm 라 안 된다" | `2V/A` 는 기어이·허브가 섞인 부품의 벽 두께가 아니다. 설계는 `wall_mm = 2.0` |
| R4 | "`sim_deme_scoop.py` 가 힌지축을 90° 틀렸다" | **원 코드의 `rotz` 가 옳다.** 설계 규약(D461 §3) "호는 X–Y, 너비는 힌지축 Z" 그대로다. 내가 `shell_width_mm=50` 을 STL 의 Y extent 에 잘못 맞췄다 |

공통 원인: **소수의 실패한 테스트에서 엔진/타인 코드의 결함을 단정했다.** AGENTS.md 의
NVIDIA 스택 공식 소스 검증 규칙(주장 **전에** 버전 일치 문서 확인)을 지켰다면 R1 에서
걸렸을 것이다. 형상 검사(단면 자르기)를 먼저 했다면 R4 도 없었다.

`sim_deme_scoop.py` 에 대해 **유효하게 남는 지적**은 두 가지다: 셸을 완전히 열린 상태로
구워 놓고 폐합을 x 평행이동으로 근사한 것(파일 주석이 그렇게 적고 있다), 그리고 그 실행이
`SCOOP_NSUB=1500` 부분집합이었던 것. 다만 §D-d 때문에 이를 고쳐도 포획은 0 이다.

---

## 3. 막힌 것 / 다음

* **B1 게이트(포획 ≥ 100 펠릿, 반력 > 0, 5회 CoV ≤ 15%) 미통과.** 원인 규명은 끝났고
  다음 행동은 시뮬이 아니라 **그랩 재설계**다. 재설계 전에는 ③(B1 재도전)을 돌릴 이유가 없어
  돌리지 않았다.
* **B2(처리량)** 미착수. 참고 실측: DEME 스쿱 1회 170.47 초(1,500 입자) → 3,000 시행
  142 시간. P3 가 `cd_update_freq` 20→6 에서 6.3배 가속을 실측했으므로 ~22.5 시간 경로가
  보인다.
* **N1b(MPM↔DEME 퍼낸 양 대조)** 는 B1 이 풀린 뒤에야 의미가 있다.

## 4. 새로 추가된 파일 (전부 신규, 기존 경로 무수정)

```
sim_pbd_pellet_probe.py          Isaac PBD 게이트 하네스 (G1~G5)
sim_pbd_pellet_report.py         집계·그림·게이트 판정
sim_pbd_pellet_render.py         저장 좌표로 RGB 육안 검사 렌더
sim_depth_observation_probe.py   P3 DEME 더미의 깊이 재관측 + 편향 분해
sim_newton_mpm_probe.py          Newton MPM N1 (Q1~Q3)
sim_newton_scoop_probe.py        Newton 스쿱 dig 프로토콜
sim_newton_grab_capture.py       메시 실측 extent 기반 포획 시험
sim_newton_collider_cup_test.py  충돌체 최소 재현(벽 두께·경로·이동)
sim_newton_containment_test.py   닫힌 용기 누출 시험
sim_grab_endwall_variant.py      끝벽/바닥판 변형 + 밀폐 공동 검사
claudedocs/runtime_logs/pbd_probe/**      셀 35 + 그림 + 게이트
claudedocs/runtime_logs/newton_mpm/**     N1 셀 10 + B1 산출물
```
