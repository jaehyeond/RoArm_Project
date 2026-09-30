# 유튜브 쇼츠 2편 분석 보고 (W26, 2026-09-30)

작성: Orca 워커(Claude Opus 5.5), task `task_e643464825d8`. 읽기·분석 전용 — 코드·설정·상태 문서 변경 0, 로봇·GPU·시뮬 0.
전체 전사는 `transcript_e1HtNprOOnE.md`, `transcript_5QSzXaPiRtg.md`. 대표 프레임은 `frames_<id>/tMMmSSs.png`.

**요약 한 줄**: 주장 19개(영상1 11 · 영상2 8)를 공식 출처와 대조했다. **일치 13 · 불일치 2 · 미확인 4.**
불일치 2건은 영상1 제목의 "Fable까지 압도"(공식 표현은 "대부분 작업에서 Fable 5.1 수준")와,
영상2가 Ultracode 를 "추론 정도" 단계처럼 보여 준 것(공식 문서상 노력 단계가 아니라 별도 설정)이다.
미확인 4건은 공식 발표에 없는 "종합 점수 1등" 그래프와 X 사용자 데모·본인 시연이다.

---

## ① 메타데이터 (`yt-dlp --dump-json`, 2026-09-30 19:10 KST 수집)

| 항목 | 영상 1 | 영상 2 |
|---|---|---|
| ID / URL | `e1HtNprOOnE` https://www.youtube.com/shorts/e1HtNprOOnE | `5QSzXaPiRtg` https://www.youtube.com/shorts/5QSzXaPiRtg |
| 제목 | Fable까지 압도하는 새로운 Opus의 등장 ㄷㄷ | Opus 야무지게 활용하는 꿀팁 |
| 채널 | 조코딩 JoCoding (@jocoding) | 조코딩 JoCoding (@jocoding) |
| 업로드일 | 2026-09-29 (timestamp 1790661625) | 2026-09-29 (timestamp 1790672435) |
| 길이 | 46 s | 29 s |
| 조회수 / 좋아요 / 댓글 (수집 시점) | 16,767 / 132 / 10 | 92,545 / 1,076 / 33 |
| 설명문 | 책·AX 컨설팅·조코헌트 링크, VVIP 후원자 명단, 멤버십 링크, `#ai #클로드 #opus #fable` (영상 내용 설명 없음) | 같은 링크·명단, `#ai #클로드 #opus #꿀팁` |
| 자막 종류 | 수동 자막 없음. **자동 생성** `ko`·`ko-orig`(둘이 바이트 동일) | 같음. 자동 자막은 00:22.9 에서 끝남(영상은 29 s) |
| 화면 번인 자막 | 있음(파란 말풍선, 제작자 삽입) — 교차 확인에 사용 | 있음 — 00:21 이후 내용은 번인 자막으로만 확인 |

## ② 전사 (요약본 — 전체는 transcript 파일)

자동 자막 오인식 의심은 `[?]`. 번인 자막이 자동 자막보다 정확해서 둘이 다르면 번인 자막을 따랐다.

### 영상 1 (e1HtNprOOnE)
| 시각 | 자막(번인 기준, 자동 자막 차이는 비고) | 화면 텍스트 / 장면 |
|---|---|---|
| 00:00–01 | Opus 5.5 새롭게 출시했습니다. (자동: "5% 5.5" [?]) | 타이틀 "Opus 5.5" → Claude 로고 |
| 00:02–03 | Fable 5.1 정도 (수준이라고 하고요) | 벤치마크 표 Opus 5.5 \| Fable 5.1: Terminal-Bench 4.0 66.4%\|55.8%, FrontierCode v1.1 54.4%\|50.3%, CursorBench 4.0 57.8%\|51.8%, GDPval-AA v2.1 1846\|1735, AutomationBench 40.0%\|31.4%, Multidisciplin…(잘림) …\|65.6% |
| 00:04–06 | Opus 5 대비 실행 비용이 적다고 합니다. | 가격표 Claude Opus 5.5 \| Claude Opus 5: $4\|$5, $20\|$25, $0.20\|$0.50, $5\|$6.25 (**행 이름 잘림**) |
| 00:07–08 | 종합 점수 기준으로 1등입니다. | 막대그래프 58(Claude Opus 5.5 max with fallback) · 53 · 53(GPT-6 Astra) · 51(Claude Opus 5) · 48(Muse Spark 1.3) · 48(GPT-6 Sol) · 47(GPT-5.6 Sol). 각주 "…Last Exam, GDP.pdf, CritPt, AA-Omniscience, AA-LCR v1.1". 제목 없음 |
| 00:09–12 | 3D 모델링 이런 거 엄청 잘해요. 직접 인터랙션도 하면서 굉장히 잘하고요. | 렌즈 광학 3D 장면 (출처 : X @RyanSael) |
| 00:13–15 | 오사카 성을 Three.js와 블렌더로 만든 거 (자동: "3JS" [?]) | 성 모델 Sketch→Massing→Detailed→Built Reality (출처 : X @onofumi_AI) |
| 00:16–19 | 바닐라 JS와 캔버스 2D 이용해서 도트 애니메이션 이런 것도 만들 수가 있고요. | 픽셀 말 달리기 (출처 : X @victormustar) |
| 00:20–28 | 비디오 만드는 거 되게 잘한다고 해요. AI 관련된 노래를 애니메이션 형식으로… 적절한 자막과 이미지를 잘 활용해 만들어준 걸 | 일러스트 뮤직비디오: "…a sudden drop in your training loss,", LOSS, SERVANT, "Claude is wor[king]", BOSS, "Done.", CHATGPT (출처 : X @donaldjewkes) |
| 00:29–31 | 레고 조립하는 애니메이션. 되게 잘 나옵니다. | 레고 조립 3D (출처 : X @victormustar) |
| 00:32–34 | 게임도 엄청 잘 만든다고 해요. 이런 게임을 만들 수가 있다고 합니다. | 보스 "SEVAROG, WARDEN OF THE FROZEN THRONE" (출처 : X @KanaWorks_AI) |
| 00:35–38 | 제가 최근에 야숨을 시작했거든요. 딱 프롬프트 한 줄만 쓰고 (시켜봤습니다) | 실제 게임 화면으로 보이는 장면 → 채팅 UI 프롬프트 "Make Three.js version of zelda breath of the wild". **모델명 화면 표시 없음** |
| 00:39–45 | 시작하면은 눈을 뜨세요 해서 시작하는 거… 칼 휘두르는 거, 화살 조준도 됩니다(자동: "화살도 됩니다"). 굉장히 잘 구현이 됐죠.(번인만) | "OPEN YOUR EYES…", localhost:8123, "GREAT PLATEAU", 칼·활·전투 |

### 영상 2 (5QSzXaPiRtg)
| 시각 | 자막(번인 기준) | 화면 텍스트 / 장면 |
|---|---|---|
| 00:00–01 | Opus 관련해서 팁들이 나오고 있는데 | 타이틀 "Opus 5.5", 예능 짤 |
| 00:02–05 | (5.5에) 일를[원문 오타] 맡길 때는 완료 기준과 멈춰야 할 조건을 명확하게 쓰고 (자동: "있는데이를" [?]) | 노란 제목 "*공식 블로그의 팁", 한국어 글 "Opus 5.5에 일을 맡길 때 완료 기준과 멈춰야 할 조건을 명확히 쓰고…", 출처 칩 "claude.dev", "결제 API를 이전하고, 기존 클라이언트를 삭제하고, 테스[트…]", "꼭 판단이 필요한 경우에만 질문하도록 지정할[…]" |
| 00:06–07 | 처음부터 전체 작업 전달해도 된다고 하고요. | "• 처음부터 전체 작업을 전달하기: …과하면 완료"처럼 끝나는 조건을…" |
| 00:08–14 | 깊이 생각해 같은 문구는 이제 빼도 된다고 합니다. AI가 충분히 똑똑하니까 알아서 결정하니까 이런 건 빼도 된다고 합니다. | ""깊이 생각해" 같은 문구는 빼기 … 간단한 답을 원하면 "바로 답해줘"라고 쓰[…]", 로봇 짤 |
| 00:15–17 | 추론 정도를 늘리면 늘릴수록 토큰이 많이 들잖아요. | 설정 UI "노력 높음" → 슬라이더 오른쪽 끝 "노력 Ultracode", 하단 "Opus 5.5 · 중간" → "Opus 5.5 · Ultracode" |
| 00:18 | 언제 이걸 늘리면 좋냐? | X 게시물 Thariq @trq212 "Using Claude Code: Spending Your Effort" (/effort medium), 오버레이 "터미널벤치 3.0으로 effort 단계별 실험한 결과" |
| 00:19–27 | Low로 먼저 구현하라고 합니다. 먼저 가볍게 구현한 다음에 내가 리뷰하고 수정 반복을 한 다음에 마지막 검증 테스트만 High로 써라라고 추천을 하고 있습니다. (자동 자막은 00:20.64 "합니다." 에서 끝) | "① Claude에게 먼저 나를 인터뷰하게 [해]서 스펙 작[성] → ② Low로 구현 → ③ 내가 리뷰하고 […]게 수정 반복 → ④ 마지막 검증·테스트만 High", "[무]조건 Max를 돌리는 것보다,", 툴팁 "ChatGPT에게 물어[보세요]" |

관찰: 영상2의 "*공식 블로그의 팁" 한국어 글 화면에는 00:20·00:26 에 "ChatGPT에게 물어보세요" 툴팁이 보인다. 그래서 그 글은 블로그 원문이 아니라
**ChatGPT 화면에 뜬 한국어 요약**으로 보인다(추정 — 앱 이름이 화면에 따로 나오지는 않는다). 출처 칩은 "claude.dev".

## ③ 내용 요약 + 주장 목록

### 영상 1 요약
1. Claude Opus 5.5 출시 소식. 성능은 Fable 5.1 정도, Opus 5보다 실행 비용이 적다고 소개.
2. 공식 벤치마크 표·가격표 캡처와 제목 없는 "종합 점수" 막대그래프(Opus 5.5 = 58, 1위)를 보여 줌.
3. X 사용자 데모(3D 렌즈, 오사카 성 Three.js+Blender, 픽셀 애니메이션, AI 노래 뮤직비디오, 레고, 보스 게임)로 "잘한다"고 전달.
4. 본인이 프롬프트 한 줄("Make Three.js version of zelda breath of the wild")로 야숨 비슷한 게임을 만들게 한 결과를 시연.

| # | 주장 | 영상 근거 시각 |
|---|---|---|
| C1-1 | Claude Opus 5.5 가 새로 출시됐다 | 00:00–00:01 |
| C1-2 | 성능이 Fable 5.1 정도 수준이다 | 00:02–00:04 |
| C1-3 | 벤치마크 표 수치(Opus 5.5 vs Fable 5.1): Terminal-Bench 4.0 66.4/55.8, FrontierCode v1.1 54.4/50.3, CursorBench 4.0 57.8/51.8, GDPval-AA v2.1 1846/1735, AutomationBench 40.0/31.4, Multidisciplin… Fable 65.6 | 00:02–00:03 (화면) |
| C1-4 | Opus 5 대비 실행 비용이 적다 | 00:04–00:06 |
| C1-5 | 가격 비교 Opus 5.5 vs Opus 5 = $4/$5, $20/$25, $0.20/$0.50, $5/$6.25 | 00:04–00:06 (화면, 행 이름 잘림) |
| C1-6 | 종합 점수 기준 1등(58점, 다음 53) | 00:07–00:08 |
| C1-7 | 3D 모델링을 엄청 잘한다(오사카 성 Three.js+Blender 등) | 00:09–00:15 |
| C1-8 | 비디오·애니메이션(AI 노래 뮤직비디오, 레고 조립)을 잘 만든다 | 00:20–00:31 |
| C1-9 | 게임을 엄청 잘 만든다 | 00:32–00:34 |
| C1-10 | (제목) Fable까지 압도한다 | 제목 |
| C1-11 | 프롬프트 한 줄로 야숨의 Three.js 버전(시작 연출·칼·활)이 구현됐다 | 00:35–00:45 |

### 영상 2 요약
1. Opus 5.5 에 일을 맡길 때는 완료 기준과 멈출 조건을 명확히 쓰고 전체 작업을 처음부터 넘기라고 소개("공식 블로그의 팁").
2. "깊이 생각해" 같은 문구는 모델이 알아서 판단하니 빼도 된다고 소개.
3. 노력(추론 정도)을 올릴수록 토큰이 많이 든다며 Claude 앱의 노력 슬라이더(높음 → Ultracode)를 보여 줌.
4. Anthropic Thariq 의 "Spending Your Effort" 글을 인용해 "Low 로 구현 → 리뷰·수정 반복 → 마지막 검증·테스트만 High" 순서를 추천.

| # | 주장 | 영상 근거 시각 |
|---|---|---|
| C2-1 | Opus 5.5 에 일을 맡길 때 완료 기준과 멈춰야 할 조건을 명확히 쓰라(공식 블로그 팁) | 00:02–00:06 |
| C2-2 | 처음부터 전체 작업을 전달해도 된다 | 00:06–00:08 |
| C2-3 | "깊이 생각해" 같은 문구는 빼도 된다 | 00:08–00:14 |
| C2-4 | 이유: AI 가 충분히 똑똑해 (생각할지) 알아서 결정한다 | 00:11–00:13 |
| C2-5 | 추론 정도(노력)를 늘릴수록 토큰이 많이 든다 | 00:15–00:17 |
| C2-6 | (화면) 노력 슬라이더가 높음 → Ultracode 로 올라가며 Ultracode 가 추론 정도의 최상단처럼 제시됨 | 00:15–00:17 |
| C2-7 | Thariq 가 터미널벤치 3.0 으로 effort 단계별 실험을 했다 | 00:18 |
| C2-8 | Low 로 먼저 구현 → 내가 리뷰·수정 반복 → 마지막 검증·테스트만 High 를 추천(무조건 Max 보다). 화면에는 ① "먼저 나를 인터뷰하게 해서 스펙 작성" 단계도 있음 | 00:19–00:27 |

## ④ 검증 (공식 출처, 접근 2026-09-30 19:11–19:15 KST)

출처 도메인 참고: `docs.anthropic.com` 은 `platform.claude.com/docs` 로 301 리다이렉트된다(curl 확인). 그래서 플랫폼 문서는 platform.claude.com URL로 인용한다.
`claude.dev` 블로그와 `code.claude.com` 문서는 페이지에 Anthropic 발행으로 표기되어 있지만 **과제가 지정한 도메인(anthropic.com·docs.anthropic.com) 밖**이다. 해당 판정에는 ‡ 를 붙였다.

사용한 출처:
- [A] Introducing Claude Opus 5.5 — https://www.anthropic.com/claude-opus-5-5 (발표일 2026-09-22)
- [B] Pricing — https://platform.claude.com/docs/en/about-claude/pricing
- [C] Prompting Claude Opus 5.5 — https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-opus-5-5
- [D] Effort — https://platform.claude.com/docs/en/build-with-claude/effort
- [E] Introducing Claude Opus 5 — https://www.anthropic.com/news/claude-opus-5
- [F]‡ Getting the most out of Opus 5.5 in Claude and Claude Code (claude.dev, Addy Osmani, 2026-09-22) — https://claude.dev/blog/getting-the-most-out-of-opus-5-5/
- [G]‡ Using Claude Code: Spending your effort (claude.dev, Thariq Shihipar, 2026-09-25) — https://claude.dev/blog/spending-your-effort/
- [H]‡ Claude Code Docs, Model configuration — https://code.claude.com/docs/en/model-config

### 영상 1
| # | 판정 | 근거 |
|---|---|---|
| C1-1 | **일치** | [A] 발표일 2026-09-22, "first model in the new Claude 5.5 family" |
| C1-2 | **일치** | [A] "It performs at the level of Claude Fable 5.1 on most work" |
| C1-3 | **일치** | [A] 표: Terminal-Bench 4.0 66.4%/55.8%, FrontierCode v1.1 (Main) 54.4%/50.3%, CursorBench 4.0 57.8%/51.8%, GDPval-AA v2.1 1846/1735, AutomationBench 40.0%/31.4%, Multidisciplinary reasoning(Humanity's Last Exam) 67.7%/65.6% with tools. 화면 수치 전부 일치 |
| C1-4 | **일치** | [A] "costs 40% less to run than Opus 5" (typical workloads) |
| C1-5 | **일치** | [B] Opus 5.5: 입력 $4 · 5분 캐시 쓰기 $5 · 캐시 히트 $0.20 · 출력 $20 / Opus 5: $5 · $6.25 · $0.50 · $25 (Opus 5 입력·출력은 [E] 에도 $5/$25). 네 쌍 값이 모두 맞는다. 화면에서 행 이름이 잘려 **어느 행이 입력/출력/캐시인지는 값으로 추정한 대응**이다 |
| C1-6 | **미확인** | [A] 에 종합 지수·순위 언급이 없다. 화면 각주("…Last Exam, GDP.pdf, CritPt, AA-Omniscience, AA-LCR v1.1")로 보아 외부 평가 기관의 종합 지수 그래프로 보이지만(기관명은 화면에 없음), 공식 출처로는 확인할 수 없다 |
| C1-7 | **미확인** | [A] 에 3D·Three.js·Blender 언급이 없다. 데모는 X 사용자 게시물이다 |
| C1-8 | **미확인** | [A] 에 비디오·애니메이션 언급이 없다. 데모는 X 사용자 게시물이다 |
| C1-9 | **일치** | [A] "One tester had models build a game from a single prompt; Opus 5.5 scored higher than any other model on the strength of its graphics and polish." (테스터 1명의 평가 인용) |
| C1-10 | **불일치(과장)** | [A] 표 9개 행 전부에서 Opus 5.5 수치가 Fable 5.1 보다 높다(% 항목 차이 0.6~10.6 %p, GDPval-AA 는 +111점; OSWorld 81.8 vs 80.7, Chartography 89.0 vs 88.4 는 근소). 그러나 공식 서술은 "at the level of Claude Fable 5.1 on most work"(동급)이고, 영상 본문도 "Fable 5.1 정도 수준"이라고 말한다. "압도"는 공식 표현과 다르다 |
| C1-11 | **미확인** | 제작자 개인 시연이다. 화면에 모델명·노력 설정이 보이지 않아 어떤 모델의 결과인지도 영상만으로는 알 수 없다 |

### 영상 2
| # | 판정 | 근거 |
|---|---|---|
| C2-1 | **일치** | [C] Unattended agentic runs: "state the completion condition up front", "It also helps to name the stops you do want, for example when no work can advance without the user's input." / [F]‡ "Name the finish line, like 'the tests pass'…", CLAUDE.md 에 keep going / stop and ask 규칙 추가. 화면의 결제 API 예시는 [F] "Migrate the payment endpoints … Done means: every endpoint uses the new client, the old client is deleted, and the test suite passes." 와 대응한다 |
| C2-2 | **일치**‡ | [F] "Give the whole task in one message." ([C] 에는 같은 문장이 없다) |
| C2-3 | **일치** | [C] "if your system prompt contains instructions that tell Claude to think carefully before answering, consider removing them for Claude Opus 5.5." / [F] "Delete 'think carefully' lines." 단 [C] 는 **채팅 앱 시스템 프롬프트** 맥락, [F] 는 일반 팁이다 |
| C2-4 | **일치** | [C] "The model decides for itself how much to think, and effort is the main control." / [F] "Opus 5.5 already thinks before every reply." |
| C2-5 | **일치** | [D] "The effort parameter affects all tokens in the response"; [C] 높은 노력에서 "expect longer turns and more output tokens"; [G]‡ Fable 5.1 에서 max 는 low 의 약 3배 토큰(중앙값 73k vs 222k) |
| C2-6 | **불일치** | [H]‡ "Ultracode is a Claude Code setting rather than a model effort level: with it on, Claude orchestrates dynamic workflows…"; `--effort ultracode` 는 설정을 켜면서 노력을 `xhigh` 로 맞춘다. 공식 노력 단계는 low·medium·high·xhigh·max 다섯 개([D]). 영상은 "추론 정도를 늘릴수록"이라는 말과 함께 Ultracode 를 슬라이더 최상단으로 보여 준다. 화면 UI 자체는 실제일 수 있으나 "추론 정도 = Ultracode" 로 읽히는 제시는 공식 정의와 다르다 |
| C2-7 | **일치**‡ | [G] 평가에 Terminal-Bench 3.0(70과제) 사용. 작성자 Thariq Shihipar, 2026-09-25 |
| C2-8 | **일치**‡ | [G] 루프: 스펙을 주고 인터뷰 요청 → low 로 구현 → low 로 리뷰·반복 → "Verify and test on high effort". max 는 완전 자율로 어려운 문제를 풀 때만 권장. [H] 도 low = "Quick exchanges where you review each result", high = "Work where verification matters" 로 같은 방향이다 |

**집계**: 영상1 11건 = 일치 6 · 불일치 1 · 미확인 4 / 영상2 8건 = 일치 7(그중 ‡ 도메인 밖 3건: C2-2·C2-7·C2-8) · 불일치 1 · 미확인 0. **합계 19건 = 일치 13 · 불일치 2 · 미확인 4.**
‡ 3건을 지정 도메인만으로 엄격히 보면 C2-2·C2-7·C2-8 은 "미확인"이 되고, 그 경우 합계는 일치 10 · 불일치 2 · 미확인 7 이다.

## ⑤ 우리 프로젝트에 적용 가능한 팁 (실행·설정 변경 0)

현재 구성: 메인 = Claude Code(Fable 5.1), Orca 워커 = Opus 5.5. 과제는 RoArm 퍼내기 학습(DEME/Isaac GPU 실행 + CPU 회계·감사).

| 팁 | 영상이 말한 그대로 | 우리 환경에서 되는지 — 확인한 것 / 안 한 것 |
|---|---|---|
| 완료 기준·멈춤 조건 명시 | "완료 기준과 멈춰야 할 조건을 명확하게 쓰고" (영상2 00:02–05) | **확인**: 이번 TASK_SPEC 에 §4 "완료 조건"과 §0 경계(안 하는 것)가 이미 들어 있다. **공식 문서 부수 확인**([C]): Opus 5.5 는 긴 무인 작업 중 진행 보고 텍스트만으로 턴을 끝낼 수 있어서, 하네스가 이를 "완료"로 취급하면 도중에 멈춘다. Orca 워커는 `worker_done` 으로 명시 종료하는 구조라 방향이 맞다. **안 한 것**: 워커가 실제로 도중에 멈춘 사례가 있는지 기록 조사, TASK_SPEC 템플릿 수정 |
| 처음부터 전체 작업 전달 | "처음부터 전체 작업 전달해도 된다고" (00:06–08) | **확인**: 이번 과제도 한 번에 전체 명세로 받았다. **안 한 것**: 다른 워커 과제와 비교 |
| "깊이 생각해" 문구 빼기 | "깊이 생각해 같은 문구는 이제 빼도 된다고 합니다" (00:08–14) | **확인**(읽기 전용 grep): `AGENTS.md`·`CLAUDE.md`·`.claude/agents/` 에 ultrathink / think hard / think carefully / 깊이 생각 류 문구 **0건**. 뺄 것이 없다. **안 한 것**: 전역 설정·Orca 디스패치 프리앰블 전수 검사 |
| Low 로 구현 → 리뷰·수정 반복 → 검증·테스트만 High | "Low로 먼저 … 마지막 검증 테스트만 High로 써라라고" (00:19–27) | **해당 범위**: 이 팁은 **LLM 토큰 비용**에 관한 것이다. GPU 물리 실행 비용(DEME ≈3.1 $/셀)과는 무관하다. 우리 역할 분리(구현 워커 vs `independent-auditor`·`raw-accountant` 검증)와 구조가 닮아서, 검증 역할만 높은 노력으로 두는 방식에 대응시킬 수 있다. **확인**: `.claude/agents/` 모델 고정 = 12개 `sonnet`, 실행 역할 4개 `claude-opus-5`(Opus 5.5 아님), 노력(effort) 설정 항목 0건. **공식 문서 부수 확인**([D][H]): Opus 5.5 기본 노력은 `medium`(Opus 5 는 `high`). 역할 파일을 Opus 5.5 로 바꾸면 노력을 적지 않은 경우 한 단계 낮게 돈다. **안 한 것**: 노력별 품질 비교(공식 권고 = 자기 과제로 effort sweep), 설정 변경 |
| Ultracode | 슬라이더 최상단으로 제시 (00:16–17) | **공식 정의**([H]): 노력 단계가 아니라 동적 워크플로(여러 에이전트로 작업 분산)를 켜는 설정이고 노력은 `xhigh` 로 맞춰진다. 우리 쪽은 워크플로(다중 에이전트)를 사용자 명시 요청 때만 쓰는 규칙이 있어서 **켜지 않는 것이 맞다**. **안 한 것**: 실제 토큰 비용 측정 |
| 게임·3D·영상 생성 능력(영상1) | 3D·애니메이션·게임 데모 | 연구 경로(퍼내기 학습·높이지도·GP 보정)와 직접 관련이 없다. 재생·시각화는 Rerun 계약(D341)으로 이미 정해져 있다. **적용 제안 없음** |

## ⑥ 방법 로그와 한계

### 명령(작업 폴더 `claudedocs/research/w26_shorts_analysis_20260930/`)
```sh
Y=/home/cgxr/miniconda3/bin/yt-dlp; FF=/home/cgxr/.local/bin/ffmpeg
$Y --dump-json URL > media/meta_<id>.json
$Y --write-auto-sub --write-sub --sub-lang "ko-orig,ko" --sub-format json3 --skip-download -o "media/sub_%(id)s.%(ext)s" URL
$Y -f "136+140/136+bestaudio" --ffmpeg-location $FF --merge-output-format mp4 -o "media/video_%(id)s.%(ext)s" URL   # 720x1280
$FF -i media/video_<id>.mp4 -vf fps=1 media/fps1_<id>/f_%03d.png                      # f_NNN = NNN-1 초
$FF -i media/fps1_<id>/f_%03d.png -vf "scale=480:-1,tile=3x2" <scratch>/sheet_<id>/s_%02d.png   # 읽기용 접촉 시트
$FF -ss <t> -i media/video_<id>.mp4 -frames:v 1 -vf "crop=720:560:0:40,scale=1080:-1" <scratch>/zoom_*.png  # 표·글 확대
```
- 처음 `-f "best[height<=720]"` 는 실패했다(세로 영상이라 720p 형식의 height 가 1280). 그래서 형식 136(720×1280)+오디오로 다시 받았다.
- 프레임: 영상1 46장, 영상2 28장(1 fps). 접촉 시트 8+5장과 확대 크롭 약 20장을 **눈으로** 읽었다(OCR 도구 없음).
- 자막 파일: `media/sub_e1HtNprOOnE.{ko,ko-orig}.json3`(14,808 B, 동일), `media/sub_5QSzXaPiRtg.{ko,ko-orig}.json3`(7,005 B, 동일).
- 공식 출처: WebSearch(anthropic.com·claude.com 계열 도메인 제한) → WebFetch 로 [A]–[H] 본문 확인. `curl -sI` 로 docs.anthropic.com → platform.claude.com 301 확인.

### media/ 목록 (커밋 대상 아님)
`video_e1HtNprOOnE.mp4`(6.5 MB) · `video_5QSzXaPiRtg.mp4`(2.9 MB) · `meta_*.json`(각 ≈1.7 MB) · `sub_*.json3` 4개 · `fps1_e1HtNprOOnE/`(46 PNG) · `fps1_5QSzXaPiRtg/`(28 PNG) · `log_sub_*.txt`·`log_vid_*.txt`·`err_vid_*.txt`.

### 한계
1. 음성 인식 도구가 없다. 음성 전사는 YouTube 자동 자막 + 번인 자막뿐이고, 영상2 00:23 이후는 자동 자막이 없어 번인 자막에만 의존했다.
2. 1 fps 추출이라 1초 안에 지나간 화면 글은 놓쳤을 수 있다(표·그래프 구간만 4 fps로 다시 봤다).
3. 가격표 행 이름과 그래프 제목·기관명은 영상 화면에서 잘려 있다. C1-5 의 행 대응과 C1-6 의 출처 기관은 추정이다.
4. WebFetch 는 페이지를 요약 모델을 거쳐 돌려준다. [A] 의 표 수치와 인용문은 요약 결과이고, [C]·[D]·[B] 는 원문 마크다운을 받았다.
5. 조회수·좋아요는 수집 시점 값이라 변한다.
6. 부수 사건: 처음에 하트비트를 맨 `orca` 로 보냈다가 `/usr/bin/orca`(스크린리더)가 실행돼 멈췄다. 해당 프로세스(내가 띄운 PID)를 종료하고 `orca-ide` 로 다시 보냈다.
