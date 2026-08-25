# posco_rocks_o1 — 비정형 convex 암석 유사체 (O-step)

- 52개 = 클래스 22/26/30/34 mm x 13. 파지 폭(min width) = 클래스값 정확 스케일.
- 프린트: PLA **무광 밝은 회색** (흑색/광택 절대 금지 — Azure Kinect ToF 판독성).
  권장 15% infill + 2 walls. STL 단위 = mm.
- 프린트 후 **개당 실측 질량을 manifest에 기록**한 뒤에만 sim 질량 사용
  (sim<->real 동일 메쉬 원칙: manifest.json vertices_m/faces가 정본).
- 게이트: 폭 정확 / 최장축 <=1.5x클래스 / 시드 결정론 bit-동일 / Euler 폐합.
