# 실측 반영 더미(4 cm 평평 층) — W26 refit (2026-09-30 17:00~17:32 KST, 로컬 RTX 4090 Laptop, 사용자 승인)
- 알 템플릿 4.6×3.8×3.2 mm 렌즈 7구, 26.506 mg, 905 kg/m³ (`../pellet_template_refit_20260930/`). 알 51,769개(1,372 g), 상자 310×220, 깊이 목표 40 mm, 시드 여백 4.7 mm, 물성·dt·CD·시드 = W25 와 동일. 명령 `generation_command.sh`, 생성기 sha `generator_sha256.txt`.
- 결과 NPZ `pile_lens6_a4p6_b3p8_c3p2_slab40_outer_310x220_n51769_rho0p503_seed460.npz` sha256 `68660882…`(`npz_sha256.txt`). 정착 0.600 s(12/80 검사, stable 5/5), 실제 경과 1,913 s(31.9 min; W25 67,737알 2,598 s).
- 검증: `--validate-output` VALID(N 51,769·rows 362,383·타이밍 有) · **FLAT40_PASS**(`settle_gate_flat40.json/.png`): 중앙값 39.90 mm, p5 36.86, p95 41.42, 최소 32.78, 최대 42.62, 벽 띠 중앙값 38.39(벽 4면 39.84/39.55/35.75/35.64), 벽 커버리지 1.00.
