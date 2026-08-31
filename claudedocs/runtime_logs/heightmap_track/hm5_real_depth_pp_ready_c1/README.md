# HM5 — Kinect real-depth path, PP-frame ready

## 1. 결론

`capture.transformed_depth`(색상 카메라 1280×720 intrinsic에 정렬된 깊이) 한 장을
`roarm-heightmap-v1`로 바꾸는 경로를 완성했다. 격자 계약은 사용자 결정값 그대로다:
`frame=roarm_base`, `shape=(76,38)` (`row=y,col=x`), `cell=0.005 m`,
`origin=(0.125,-0.190) m`, `agg=max`, `empty_fill=0.0 m`.

실기는 현재 **연결 불가**다. 정확히는 `pyk4a==1.5.0`과
`libk4a/libk4arecord==1.4.2`는 설치됐지만, 호스트 `lsusb`에 Azure Kinect가 없고
`pyk4a.connected_device_count()==0`이었다. `k4a-tools`(`k4aviewer`, `k4arecorder`)도
없지만 pyk4a 단일 프레임 경로에는 필요하지 않다. 상세 증거는
`kinect_runtime_audit.json`이다.

그래서 카메라를 찾느라 더 진행하지 않고, 실제 캘리브 포즈와 1280×720 intrinsic을
사용한 저장형 합성 depth 양성대조로 전체 경로를 실행했다. 실제 프레임으로 아직
검증했다는 주장은 하지 않는다.

## 2. 실제 depth에서 추가한 처리

`roarm_rl.heightmap.filter_kinect_depth()`와
`heightmap_from_kinect_depth()`가 다음을 수행한다.

1. 0·음수·NaN·Inf·거리 범위 밖 픽셀을 `input_invalid_mask=True`로 제외한다.
2. 3×3 이웃의 깊이 범위가 30 mm를 넘는 불연속 가장자리를 1픽셀 모호성 띠로
   제외한다. ToF 다중경로/비행 픽셀을 `max` 집계 전에 제거하는 보수적 정책이다.
3. 이웃 중앙값에서 15 mm 넘게 단독 이탈한 값도 ToF spike로 제외한다.
4. **보간하지 않는다.** 살아남은 점이 없는 격자 셀은 `height=0.0` 패딩이지만
   `Heightmap.valid=False`다. 따라서 관측 사각을 실제 높이 0으로 읽지 않는다.

양성대조는 zero 25픽셀, NaN 25픽셀, 비행 픽셀 256개를 넣었다. zero/NaN은
전부 입력 무효로 남았고 비행 픽셀은 **256/256** 가장자리/이상치 마스크에 걸렸다.
격자 2,888셀 중 **2,765 valid / 123 unseen-or-occluded**였으며, valid인 바닥 근사
0-height 셀 2,129개와 invalid zero-padding 셀 123개가 같은 숫자 0을 가져도
`valid`로 완전히 구분된다 (`hm5_results.json`, `hm5_depth_masks.npz`).

## 3. 좌표 변환

`sim_scripts/kinect_calib.yaml`을 그대로 읽으며 재캘리브하지 않는다.

```text
p_base = R_cam_to_base @ p_cam + t_cam_to_base
```

캘리브 RMSE는 10.13 mm이고, FK 회전 규약은 과거 수정된 고정축
`Rz(yaw) @ Ry(pitch) @ Rx(roll)`로 기록했다. 카메라를 움직이면 이 외부파라미터는
무효가 되므로 카메라 위치를 바꾸면 안 된다.

## 4. PP 펠릿이 오면 할 일

카메라 전원/USB를 연결하고 위치는 건드리지 않은 채, 더미를 놓고 아래 한 줄만
실행한다. `<date>`는 새 폴더명으로 바꾼다.

```bash
/home/cgxr/miniconda3/envs/isaaclab/bin/python sim_scripts/p38_hm5_kinect_depth_pipeline.py \
  --live \
  --output-dir claudedocs/runtime_logs/heightmap_track/pp_pile_<date> \
  --rerun
```

이미 `capture.transformed_depth`를 mm 단위 `.npy`/`.npz`로 저장했다면 다음과 같다.

```bash
/home/cgxr/miniconda3/envs/isaaclab/bin/python sim_scripts/p38_hm5_kinect_depth_pipeline.py \
  --input-depth <transformed_depth.npy> --depth-unit mm \
  --output-dir claudedocs/runtime_logs/heightmap_track/pp_pile_<date> \
  --rerun
```

즉 **교체할 것은 1280×720 transformed-depth 프레임과 새 출력 폴더명뿐**이다.
코드·격자·캘리브·로봇 자세를 바꾸지 않으며, 이 스크립트는 로봇 SDK나
`/dev/ttyUSB*`를 사용하지 않는다. 실행 후 새 Rerun 스크린샷을 실제로 열어
사각과 더미 형상이 예상 위치에 있는지만 육안 확인한다.

이 교체 경로 자체도 `hm5_input_depth.npz`를 일반 저장 프레임 입력으로 다시 읽어
`../hm5_saved_frame_swap_smoke_c1/`에 별도 실행했다. 결과는 동일하게
`valid=2,765 / unseen=123`, `HM5_FRAME_TO_HEIGHTMAP_OK`와
`HM5_BUNDLE_CONTRACT_OK`였다. 즉 합성 생성 함수에만 우연히 결합된 경로가 아니다.

## 5. D341 Rerun

- SDK/CLI: `rerun 0.34.1`; Isaac 호환 핀 `numpy 1.26.0`, `psutil 5.9.8` 확인.
- RRD: `hm5_timeline_c1.rrd`, sha256
  `e4d5340f07c56e29e89f6771afab17d45706eaa4c04327a5240959b2937b364b`.
- RBL: `hm5_timeline_c1.rbl`, sha256
  `1770b63fdb14124769ae85484d0eaeef9d375fdf5bc1730ffb4dce33fbb2b05c`.
- `rrd verify --check-footers true` PASS, exact non-system entity 9개, exact timeline
  `blueprint/frame_idx/log_time`, required components, 고정 RBL, headless screenshot 모두 PASS.
- 별도 엔티티: raw depth image, pixel masks, roarm-base point cloud,
  valid heightmap cells, unseen heightmap cells, camera/grid origin+axes.
- 육안 검수: `hm5_inspection.json`. 깊이·마스크·점군·heightmap 4패널을 원본 해상도로
  열어 확인했고, unseen(red)이 경사 시점의 더미 뒤쪽에 집중됨을 관찰했다.

## 6. 산출물

- `hm5_input_depth.npz` — 입력 프레임 양성대조
- `hm5_depth_masks.npz` — pixel valid/invalid/edge/outlier 마스크
- `hm5_points_base.npz` — `roarm_base` 점군
- `hm5_heightmap.npz` + `.json` — 높이맵 배열 + 계약 헤더
- `hm5_results.json` — 수치 판정과 PP 실행 명령
- `hm5_diagnostic.png` — 입력/마스크/점군/heightmap/사각 6패널
- `hm5_timeline_c1.rrd`, `.rbl`, Rerun 검증/스크린샷
- `GATES.md` — 실행 가능한 CHECK/EXPECT 게이트

개발 중 첫 시도는 삭제·이름변경하지 않고 바로 앞 폴더
`hm5_real_depth_pp_ready/`에 보존했다. 최종 판정 권위는 이 `_c1` 폴더다.
