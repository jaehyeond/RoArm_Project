# pod A (`zwmwq17s80vdfb`, RTX 4090 SECURE EU-RO-1, 드라이버 Open Kernel Module 570.211.01) 증거 — 2026-09-17 15:46~16:1x KST

DEME 2.4.0(로컬과 같은 빌드·같은 NVRTC 12.8)이 이 호스트에서 **초기화 직후 첫 접촉탐색 커널** `DEMCubContactDetection.cu:339` 에서 `illegal memory access` 로 확정적으로 죽었다(두 파이프라인·CUDA_LAUNCH_BLOCKING=1 동일). managed memory 기본 테스트는 통과. 물리 0. pod 는 증거 회수 후 Terminate.
- `w19_bootstrap_v1_failed_tos/` conda ToS 실패 로그, `w19_bootstrap/` v2 성공 로그 + 환경 영수증 2종 + 꾸러미 검증
- `w19_smokes/` smokes_attempt1/2(러너 경로·gymnasium)·시도 3 로그, `w19_out/` 러너 영수증·sim stdout/stderr(NPZ·_obj 제외), `w19_diag/` um_test·sphere_lb(CUDA_LAUNCH_BLOCKING)
