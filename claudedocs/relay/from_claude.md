# from_claude.md — Claude → Codex/Cursor 인계 (relay)

## §0 이 파일의 규약
- **쓰는 쪽 = Claude 세션 하나.** 읽는 쪽 = 다음에 이 repo 를 여는 **다른 도구**. Claude 연속이면 `START_HERE.md` 로 재개.
- **덮어쓰기.** 상태 정본 = `START_HERE.md`(여기 안 베낌). 규칙 = `AGENTS.md`. 여기엔 만진 것·만지지 말 것·함정·승인 대기만.

## §2 2026-09-11 Claude(80th~81st) → Codex — S1 sim 정합 + 펠릿 실측 + PID 조사 + W10 미완

### 🔴 0. 지금 당장 막힌 것 — GPU (재부팅 전 DEME·Isaac 전부 불가)
`nvidia-smi` = `Failed to initialize NVML: Driver/library version mismatch`. apt 자동 업그레이드로 유저스페이스 **580.178.04**, 로드된 커널 모듈 **580.173.02**(`/proc/driver/nvidia/version`). 새 프로세스 CUDA 전멸(DEME `forward compatibility…`, torch error 804). **재부팅 필요.** 재개 첫 명령은 `nvidia-smi` 정상 확인.

### 1. 이어서 할 일 (1순위)
**W10 = 렌즈 클럼프 더미에서 "문을 서보 토크 정지까지 닫기" 발산 해결.** 지시서 한 장에 전부 정리돼 있다:
`claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w10_deme_close_fix/RESUME_W10_20260911.md` ← **이것부터 읽어라.**
요약: ① 진단 완료(기작 = 뺨 틈 안 렌즈 클럼프 3개 기둥의 **구–구 진동 자기증폭**, 유령 접촉 아님) · ② 폐합 22.5°/s 발산 · ③' **dt 2e-6 만**(E 유지) 이 다음 차례이고 params 파일까지 준비됨 · 실행기 `run_w10b.sh DE_dt2e6_c 14400`. 🔴 강성 상향은 감쇠 `c ∝ √k` 때문에 판별력이 낮다 — dt 가 손잡이.

### 2. 만진 것 (전부 미커밋, 79th 분 포함)
- **신규 스크립트**: `sim_isaac_render_deme_scoop.py`(W9 렌더), `sim_deme_s1_diverge_min.py`(W3b 최소재현). `compose_roarm_s1_urdf.py` 에 `--tag` + 무관성 링크 미소 inertial 주입, `sim_deme_scoop_s1.py` 에 클럼프 npz·전후 heightmap·절단면 각·렌더 타임라인·물림 가드·문 하한(전부 params 게이트, diff 는 각 워커 폴더).
- **신규 자산**: `local_assets/roarm_m3/{urdf/roarm_m3_s1_v1.urdf, usd_s1_v1/}`(실물 v1 형상, 가짜 질량 제거판).
- **신규 산출 폴더**: `claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/{w1_usd_v1, w2_env_replay, w3_deme_scoop, w8_deme_scoop_lens, w9_isaac_render_deme, w10_deme_close_fix}/`, `claudedocs/research/survey_20260910/`, `claudedocs/runtime_logs/pellet_model/pellet_measured_20260910.json`.
- **문서**: `docs/reference/servo_pid_st3215.md` 신규 · `docs/reference/hardware.md` T:107 오기 2곳 정정 · `AGENTS.md` 참조표 1행 추가 · `hw_s1_manual.py` 에 `weigh`/`mass` 명령 추가(백업 `.bak_20260907_pre_weigh`).
- **다른 워크트리**(`~/orca/workspaces/RoArm_Project/pellet-model`, 브랜치 `jaehyeond/pellet-model`, 미커밋): `sim_pellet_model.py` 에 flat3·lens 클럼프 + `--measured-pellet a b c rho`, `sim_deme_pile.py` 에 `--shape lens` 경로. 산출 `claudedocs/runtime_logs/pellet_model/{planar_20260909, lens_20260910, lens_sweep_20260910, pile_lens_20260910}/`.

### 3. 만지지 말 것
`s1_v0*/`·g18 이하 동결 · `sim_deme_scoop.py`(구 트랙 보호) · 기존 산출 npz·JSON · `s1_v1_real/manual_positions.json`·`mass_log.jsonl`(사용자 데이터) · DK 원본 프로필 · **원장(배타 소유, 아래 §6)**.

### 4. 🔴 함정 (이번 세션 실측)
1. **Isaac URDF 임포터는 inertial 없는 링크에 기본 질량 1.0 kg 을 준다** — `hand_tcp` 에 실려 어깨 중력 모멘트가 11~14배 부풀었다. D478 의 "팔 토크 8.0 N·m 필요"가 이것 때문이었다(실물 1.96 으로도 파지 성공). v1 경로는 미소 inertial 주입으로 해결.
2. **DEME 는 실행 간 비결정**이다(같은 입력·무수정 코드로 포획 273/301/315개). "개수 동일" 을 회귀 게이트로 쓰면 안 된다 → 허용 범위(±15 %)로.
3. **렌즈 클럼프 = 가볍고 납작** → 좁은 틈에서 구–구 진동이 명시적 적분 한계에 걸린다(위 §1). 구 4.16 mm 에서는 안 보이던 문제.
4. **T:107 그리퍼 토크 상한은 EPROM 아닌 SRAM 48번(휘발)** 이고, 소스 0.84 기준 부팅 완료 시 상한은 1000 이 아니라 **300**(`ino:146`). hardware.md 정정 완료.
5. **서보 과부하 보호**(출력 80 % 초과 2 s → 20 % 강하)가 닫힘 상한 900(=90 %) 스톨에서 걸릴 수 있다 → 실물 되열림 1~1.6° 의 후보 원인. 내일 900 vs 790 비교 예정.
6. **Orca 코디네이터 바인딩이 조용히 풀린다**(`consumer_fenced`). 그 상태의 `orchestration check --peek` 는 오류 없이 count 0 을 준다 → 워커 질문 2건을 놓쳤다. 확인 전 `run-current` → 없으면 `run-use --id <run>`.
7. 근접 RTX 렌더 검정면(D480) 미해결 → 카메라 1 m 이상. 반투명 재질은 headless RTX 에서 cutout 으로만 동작(W9 는 x-ray 로 우회).

### 5. 승인 대기 / 다음 (사용자 결정 사항)
① **로봇 부팅 체크리스트**(09-10 사용자 합의, 오늘 예정): 펌웨어 버전·`tG` 유무 → PID 적용 행동실험(P 8↔48) → 닫힘 상한 900 vs 790 되열림 → `weigh 5` 회당 질량. 상세 = auto-memory `project_next_robot_boot_checklist.md`.
② 실물 각도 3종(부은 각·렛지 각·한 입 뒤 절단면 각) 측정 → 차이 ≤ 3° 면 부은 각만으로 충분(R1 E1).
③ 조사 결론 반영(R1·R2): 초기 상태 = **벽 있는 상자에 평평하게 가득**, 베이스라인 = **최고점 + 층·열 진행 휴리스틱**, 실패 정의 = 산업 목록(빈 그랩·저충진·흘림·매몰·충돌), 재료 보정은 부은 각 **단독 금지**.
④ 스쿱 더미 크기: 20,000알 채택 여부(실물 18만 알은 정착 2 h·스쿱 2 h/회로 비현실).

### 6. 원장 상태 (중요)
80th·81st 작업분은 **아직 원장에 등재되지 않았다**. `DECISIONS.md` 최신 = D481(79th), `EXPERIMENT_LEDGER.md` 최신 = `:544`(79th). `START_HERE.md` 는 2026-09-11 에 81st 상태로 갱신했다(이 relay 와 같은 세션). 등재(Dxxx append·LEDGER 행·session_*.md)는 **W10 이 결론 난 뒤** 한 세션이 몰아서 하는 것을 권한다. 그때까지 다른 도구는 원장을 쓰지 말 것.

### 7. 검증
```
nvidia-smi                                                     # 표가 나와야 GPU 작업 가능
sed -n '1,40p' claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w10_deme_close_fix/RESUME_W10_20260911.md
ls claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/     # w1_usd_v1 … w10_deme_close_fix
grep -n '^## D481' claudedocs/DECISIONS.md                     # 30140 (그 뒤로 append 없음)
~/miniconda3/envs/roarm/bin/python -c "import json;d=json.load(open('claudedocs/runtime_logs/pellet_model/pellet_measured_20260910.json'));print(d['pellet_dimensions']['a_mm'],d['pellet_dimensions']['c_mm'])"
```
