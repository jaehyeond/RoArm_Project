# 원인 분리용 진단 실행 — 하강 구간만

`SCOOP_ERRVEL=500` 으로 발산 가드를 풀어 폐합 끝의 힘 급등 형태를 보려던 실행.
GPU 를 다른 워커와 나눠 쓰게 되어 하강 60스텝까지만 진행한 뒤 중단했다.
`scoop_timeline_diag.json` 은 그 하강 구간 타임라인이고 **폐합 데이터는 없다.**

완전 폐합 힘 곡선은 `../s2_closure_rot/scoop_timeline_fullclose_diverged.json` 에 있다.
실제 결과는 `../s2_closure_rot/` 에 있다.
