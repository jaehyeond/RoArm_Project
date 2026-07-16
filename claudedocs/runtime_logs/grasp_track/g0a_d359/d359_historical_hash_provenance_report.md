# D359 historical hash provenance report

Verdict: `D359_D351_HASH_PROVENANCE_RECOVERED`

## 초보자용 핵심

과거 여섯 해시는 형상이 달라서 생긴 값이 아니었다. 같은 483개 patch 정점을
`원본 point ID 순서`로 나열한 one-off generator가 만든 값이었다. D351의 실제
validator와 D358 search는 좌표값을 기준으로 정점을 다시 나열했다. 정점 목록의
순서가 달라지면 같은 형상도 byte stream과 SHA-256이 달라진다.

- Historical original-point-ID replay: `8/8`
- Later coordinate-row replay: `2/8`
- Independent tuple/dict/struct replay: `True`
- First Git introduction: `c2cfa5f41d4c15fec15330cfad38b9b14e4c4f61`
- Actual generator transcript line/output: `1433` / `1434`

This closes the historical generator lineage. It does not rewrite D351/D354/D358,
does not run Isaac or physics, and does not decide contact or grasp.
