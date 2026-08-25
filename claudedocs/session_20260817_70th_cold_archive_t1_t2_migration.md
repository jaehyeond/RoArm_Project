# 70th — 콜드 아카이브 T1/T2 이관 (14 폴더 ≈176GB, sha256 전량 검증 PASS) + resume 세션 완주 검증

날짜: 2026-08-17. 성격: **스토리지 인프라 세션 — 연구 실험 0, 물리 0, 로봇 0.**
세션 진행 규칙(AGENTS.md Session progress rule) 정당화: 본 세션의 목적은
디스크 압박 해소를 위한 대형 데이터 이관·검증이며 연구 가설이 없다.
실패-가능 실험은 해당 없음 — 대신 "이관 후 원본과 사본의 sha256 전량
일치"라는 실패-가능 검증 게이트를 두었고 전 폴더 PASS했다.

이번 case의 신규 변수: 없음 (yard_track 연구 변수 사다리와 무관한 인프라 작업).

## 0. 세션 경위 (두 개 컨텍스트에 걸친 단일 작업)

1. **08-17 이전 컨텍스트(중단됨)**: 폴더 정리 조사-전용 브리핑(수정 0,
   사용자 결정 5건 대기 — 69th doc §6-1 61회째 항목에 흔적) → **사용자
   "T1+T2 완전 이전" 승인** → 배치 스크립트 `migrate_one.sh` 백그라운드
   실행(task `bf6ge7ag2`) → 중간 보고(같은 §6-1 62회째 항목) 후 컨텍스트
   종료. ⚠️ 조사 브리핑 원문(결정 5건 목록)은 채팅에만 있었고 파일로
   미보존 — 본 doc은 남은 증거(ARCHIVE_INDEX.md·배치 로그·심링크·
   .gitignore diff)로만 기록하며, 미보존분은 복원 시도하지 않는다.
2. **08-17 본 resume 컨텍스트**: 배치 완주 여부 검증 + 상태 문서 마무리
   (본 doc + START_HERE + MEMORY). 신규 이관 실행 없음.

## 1. 이관 내용 (권위 = `ARCHIVE_INDEX.md` + `_manifests/`)

- 대상: **git 비추적 대형 데이터 14 폴더, 합계 ≈176GB** — `logs`,
  `sim_renders_v3/v4/v5`, `sim_renders_v4_dryrun/v5_dryrun`,
  `lerobot_dataset_v3/v4`, `collected_data`, `collected_data_v5`,
  `collected_data_v2_backup`, `collected_data_v6`,
  `collected_data_v6_phase0_singlearm_DISCARD`, `outputs`(79.1GB).
- 목적지: 외장 `/media/cgxr/ROBOT_DEV/RoArm_cold_archive/<폴더명>`.
  매니페스트 = 같은 볼륨 `_manifests/<폴더>_{src,dst}.sha256`.
- 절차(폴더당): src sha256 전량 매니페스트 → rsync 복사 → dst 매니페스트
  → diff 대조 PASS → **그때만 파일 단위 원본 제거(rm -rf 불사용)** →
  원경로에 심링크. 기존 문서의 증거 경로는 심링크로 계속 유효
  (AGENTS.md forward-only 규칙의 경로-보존 수단).
- `.gitignore`: 디렉토리 규칙(뒤 슬래시)은 심링크를 매칭하지 못하므로
  14개 정확명(`/logs`, `/outputs`, …)으로 재지정 (diff +17줄).

## 2. Resume 검증 결과 (본 컨텍스트, 전부 PASS)

| 검증 | 방법 | 결과 |
|---|---|---|
| 배치 완주 | task 로그 `bf6ge7ag2.output` tail | `=== batch finished ===` 존재, 14폴더 전부 `hash verify PASS` |
| 배치 에러 | 같은 로그 `grep -c "FAIL\|MISMATCH\|error"` | **0** |
| 심링크 | 프로젝트 루트 `ls -la` | 14개 전부 외장 새 경로로 연결, 원 디렉토리 잔존 없음 |
| 매니페스트 재대조 (스팟) | `outputs`·`lerobot_dataset_v3`의 src/dst sha256 파일을 sort 후 md5 비교 | 양쪽 모두 **MATCH** (outputs 467파일) |
| 프로세스 잔여 | `ps aux` rsync/sha256 | 없음 |
| 디스크 | `df -h` | 내장 `/` 62% (345G/590G, **215G 여유**) · 외장 19% (168G/916G) |

- 참고(비-결함): 로그의 폴더별 "bytes at dest" 합계가 src 대비 수십 KB
  수준으로 큰 폴더가 3곳(collected_data_v6 등). 권위 게이트는 파일 단위
  sha256 전량 일치이며 전부 PASS — 합계 차이는 심링크/파일시스템 회계
  차이로 추정(해석 가설, 검증 대상 아님). `outputs`는 내부 심링크 6개
  포함 상태로 이관됨.

## 3. 규율 준수

- **HARD RULE #28 / D232**: `collected_data*`·`outputs`는 삭제가 아니라
  **사용자 명시 승인(T1+T2) 하 검증-사본 후 move-only** — 조항의
  "archive/move-only, 명시 승인 필요"에 부합. `rm -rf` 불사용.
- **T3 미이관 (유일 사본 45G — 내장 유지)**: `b200_backup_20260521`(19G) +
  `b200_backup_20260522_final`(18G) + `openvla_oft_b200_pulls`(8G).
  이관분은 사본 1개(백업 아님)인 반면 T3는 유일 사본이므로 단순 이동은
  위험 프로필이 다름 — **2사본화(내장+외장 동시 보유 등) 사용자 결정 대기**.
- D341 Rerun 계약: 순수 파일/해시 감사로 공간·시간 판단 없음 → RRD 면제
  조항 해당 (본 절이 그 정당화 기재).
- 로봇 0, lerobot-train 0, git 커밋 0(사용자 전담), HANDOFF 0.

## 4. 운영 주의 (다음 세션 필독)

- **외장 `ROBOT_DEV` 미마운트 시 14개 심링크가 끊긴다** — 그 경우
  `ARCHIVE_INDEX.md`의 새 경로 열을 참조해 하드를 연결할 것. 이관분은
  사본 1개이므로 외장 분실 = 데이터 손실(T1/T2는 사용자가 이 조건으로 승인).
- git 미커밋 누적: 66th `sim_assets/`+o1, 67th~69th `yard_track/`+p26~p32,
  70th `ARCHIVE_INDEX.md`+`.gitignore`+본 doc. 커밋은 사용자 지시 시에만.

## 4-1. Stop-hook /half-clone 요구 → 거부 (63회째 [가정])

- 08-18: 명세서 v1 비판적 검토 브리핑 직후 "context 88% → /half-clone" 차단
  → **HARD RULE #11 거부**. harness 토큰 카운터 ≈14.96M/15M 잔여(≈0.3%
  사용)로 모순 — 55~62회째(67th·68th·69th doc 기록)와 동일 오탐 패턴
  (check-context.sh가 harness 카운터와 무관한 값을 읽는 문제, 수리는
  사용자 승인 사안).

## 5. 다음 (전부 사용자 결정 대기)

1. **T3 2사본화 방식 결정** (유일 사본 45G — 예: 외장 복사 후 내장 유지).
2. 연구 재개: y4 동작수 분화 채널 / Kinect depth 비교 / 실물 제작 /
   프로포절 v2 검토 (69th doc §7 그대로).
3. git 커밋 (whitelist: `sim_assets/` + `yard_track/` + p26~p32 + 70th분).
