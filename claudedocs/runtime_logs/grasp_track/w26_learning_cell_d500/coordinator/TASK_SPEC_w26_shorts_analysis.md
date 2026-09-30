# TASK_SPEC — 유튜브 쇼츠 2편 영상 분석 (읽기·분석 전용, 한국어 보고)

## 0. 역할·경계
- 당신은 **영상 분석 워커**다. 영상을 받아 자막·음성·화면 텍스트를 뽑고 내용을 구조화해 보고한다. 로봇·GPU·시뮬 실행 0, 코드 수정 0, 상태 문서(`START_HERE.md`, `claudedocs/DECISIONS*.md`, `EXPERIMENT_LEDGER.md`, `relay/`) 쓰기 금지(hook 이 막는다).
- 산출은 이 worktree 의 `claudedocs/research/w26_shorts_analysis_20260930/` 아래에만. 다운로드한 영상·오디오·프레임은 같은 폴더의 `media/`(repo 에 커밋할 대상이 아니다, 그대로 두되 목록만 보고).
- **들리지/보이지 않는 것은 지어내지 않는다.** 자막이 자동 생성이면 그렇게 표시하고, 오인식 의심 구간은 [?] 로 남긴다.

## 1. 대상
1. https://www.youtube.com/shorts/e1HtNprOOnE — "Fable까지 압도하는 새로운 Opus의 등장 ㄷㄷ" (조코딩 JoCoding)
2. https://www.youtube.com/shorts/5QSzXaPiRtg — "Opus 야무지게 활용하는 꿀팁" (조코딩 JoCoding)

## 2. 방법(도구는 이 PC 에 있는 것만)
- `yt-dlp`(`/home/cgxr/miniconda3/bin/yt-dlp`): 메타데이터(`--dump-json`), 자동 자막 `ko-orig`·`ko`(`--write-auto-sub --sub-lang ko-orig,ko --sub-format json3/srt --skip-download`), 그리고 영상 본체(최저 화질로 충분, ≤ 720p).
- `ffmpeg`(`/home/cgxr/.local/bin/ffmpeg`): 1 fps 또는 장면 전환 기준으로 프레임을 뽑아 PNG 로 저장한 뒤 **이미지를 직접 읽어** 화면 자막·코드·UI 텍스트를 옮겨 적는다(OCR 도구 없음 — 눈으로 읽고 시각을 적는다).
- 음성 인식 도구(whisper 등)는 없다. 자동 자막을 1차 전사로 쓰고, 화면 텍스트로 교차 확인한다.

## 3. 산출물 `claudedocs/research/w26_shorts_analysis_20260930/`
- `REPORT.md`
  ① 영상별 메타데이터(제목·채널·업로드일·길이·조회수·설명문·자막 종류)
  ② 전사(transcript): 시각(mm:ss) | 자막 문장 | 화면 텍스트/장면 설명. 자동 자막 오인식 의심 [?] 표시.
  ③ 내용 요약(영상별 5줄 이내) + **주장 목록**: 영상이 말하는 사실 주장(예: 모델 이름·성능·가격·기능·사용법)을 한 줄씩, 각 주장에 "영상 근거 시각" 을 붙인다.
  ④ **검증**: 각 주장을 Anthropic 공식 문서(docs.anthropic.com·anthropic.com 뉴스/모델 페이지)에서 확인 → 일치/불일치/미확인. 공식 출처 URL·접근 시각. 공식 출처가 없으면 "미확인"이라고만 쓴다.
  ⑤ 우리 프로젝트(RoArm 퍼내기 학습, Claude Code 메인 = Fable 5.1, Orca 워커 = Opus 5.5) 에 **적용 가능한 팁**을 "영상이 말한 그대로"와 "우리 환경에서 되는지(확인한 것/안 한 것)"로 나눠 적는다. 실행·설정 변경은 하지 않는다.
  ⑥ 방법 로그(명령어·프레임 수·자막 파일명)와 한계.
- `transcript_<id>.md` 영상별 전체 전사, `frames_<id>/` 대표 프레임 PNG 5~10장(장면 전환마다), `media/` 원본.

## 4. 완료 조건
- 두 영상 모두 ②③④⑤ 채움. 주장은 각 3개 이상, 검증은 출처 URL 포함.
- `worker_done` 로 REPORT.md 절대 경로·영상별 주장 수·검증 일치/불일치/미확인 개수를 한 줄로 보고.
