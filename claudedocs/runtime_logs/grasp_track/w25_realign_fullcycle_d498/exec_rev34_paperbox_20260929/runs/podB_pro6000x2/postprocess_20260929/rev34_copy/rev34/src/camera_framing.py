"""표시 카메라 프레이밍 — **경계 계산으로** 필요한 것이 전부 화면에 들어가게 한다. 순수 CPU.

왜 필요한가
----------
준비 시험 육안 검수에서 **트레이 벽이 안 보이고 윗화면이 더미를 잘라먹는** 문제가 나왔다
(코디네이터: "Missing actual S1 shell/door + cropped pile means scene readiness FAIL").
그때의 카메라는 `CAM_DIST = max(1.25, 1.55 * span)` 처럼 **두 지점 간격에 곱한 휴리스틱**이라
무엇이 들어오는지 계산으로 보장하지 않았다. 상수를 키우는 것은 증명이 아니다.

여기서는 **담아야 할 점 집합**(팔·더미·트레이 바닥/지지면·용기·도구 궤적)을 받아
핀홀 FOV 로부터 **필요한 최소 거리를 닫힌 형태로 풀고**, 그 거리에서 모든 점을 **되투영해**
화면 안에 있음을 확인한다. 확인 결과(최대 화면 점유율)를 영수증에 남긴다.

수식
----
카메라 기저: forward `f`(단위), right `r = normalize(f × up_hint)`, up `u = r × f`.
눈 위치 `eye = target − D·f` 로 두면 점 `p` 의 카메라 좌표는
  depth `d = (p − target)·f + D`,  가로 `a = (p − target)·r`,  세로 `b = (p − target)·u`
이고 `r ⊥ f`, `u ⊥ f` 이므로 **`a`·`b` 는 D 에 무관하다**. 화면 안 조건은
  `|a| ≤ k_h·d`,  `|b| ≤ k_v·d`   (`k_h = tan(hfov/2)·(1−margin)`, `k_v` 동일)
→ `D ≥ |a|/k_h − (p − target)·f` 와 `D ≥ |b|/k_v − (p − target)·f`.
두 식을 모든 점에 대해 최대화한 것이 **필요 최소 거리**다. 근접 클리핑도 같은 식으로 본다.

FOV 는 USD 핀홀 규약을 따른다: `hfov = 2·atan(aperture_h / (2·focal))`,
수직 구경은 `aperture_v = aperture_h · H / W`(Isaac/USD 가 가로 구경과 종횡비로 수직을 정한다).

주장하지 않는 것
--------------
프레이밍은 **표시 계약**이다. 배출 성공·구동 가능성의 증거가 아니다.
계산이 "들어온다"고 해도 **실제 렌더 이미지를 눈으로 본 것은 아니다** — 육안 검수는 별도이며
GPU HOLD 해제 후에만 가능하다. 이 모듈은 그 검수를 대체하지 않는다.
"""
import math

import numpy as np


def fov_rad(focal_mm, aperture_mm):
    """핀홀 FOV(라디안). USD `focalLength`/`horizontalAperture` 규약."""
    return 2.0 * math.atan(float(aperture_mm) / (2.0 * float(focal_mm)))


def camera_basis(forward, up_hint=(0.0, 0.0, 1.0)):
    f = np.asarray(forward, float)
    n = np.linalg.norm(f)
    if not np.isfinite(n) or n <= 0:
        raise ValueError(f"forward 가 0 또는 비유한값이다: {forward}")
    f = f / n
    up = np.asarray(up_hint, float)
    if abs(float(np.dot(f, up / np.linalg.norm(up)))) > 0.999:
        up = np.array([1.0, 0.0, 0.0])      # 거의 수직 시선이면 보조 up 으로 바꾼다
    r = np.cross(up, f)
    r /= np.linalg.norm(r)
    u = np.cross(f, r)
    return f, r, u


def aabb_corners(lo, hi):
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    return np.array([[lo[0] if i & 1 else hi[0],
                      lo[1] if i & 2 else hi[1],
                      lo[2] if i & 4 else hi[2]] for i in range(8)], float)


def solve_framing(points, *, forward, focal_mm, aperture_h_mm, width_px, height_px,
                  margin_frac=0.08, near_clip_m=0.05, up_hint=(0.0, 0.0, 1.0),
                  target=None, min_distance_m=0.0):
    """모든 `points` 가 화면 안에 들어오는 **최소 거리**를 풀고, 그 해를 되투영해 확인한다.

    `margin_frac` = 화면 각 변에서 남겨 둘 여백 비율(0.08 = 양쪽 8 %).
    `target` 을 주지 않으면 점집합 AABB 중심을 본다.
    반환 dict 의 `fits` 가 False 면 **프레이밍 실패**다 — 조용히 넘기지 말 것.
    """
    pts = np.asarray(points, float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points 는 (n, 3) 이어야 한다: {pts.shape}")
    if len(pts) == 0:
        raise ValueError("담을 점이 없다 — 빈 집합으로 프레이밍을 주장하지 않는다")
    if not np.isfinite(pts).all():
        raise ValueError("점집합에 비유한값이 있다")
    if not (0.0 <= float(margin_frac) < 0.5):
        raise ValueError(f"margin_frac 범위를 벗어났다: {margin_frac}")

    lo, hi = pts.min(0), pts.max(0)
    tgt = (lo + hi) / 2.0 if target is None else np.asarray(target, float)
    f, r, u = camera_basis(forward, up_hint)

    hfov = fov_rad(focal_mm, aperture_h_mm)
    aperture_v = float(aperture_h_mm) * float(height_px) / float(width_px)
    vfov = fov_rad(focal_mm, aperture_v)
    k_h = math.tan(hfov / 2.0) * (1.0 - float(margin_frac))
    k_v = math.tan(vfov / 2.0) * (1.0 - float(margin_frac))

    rel = pts - tgt
    depth0 = rel @ f                        # D 를 더하기 전의 깊이 성분
    a = np.abs(rel @ r)
    b = np.abs(rel @ u)
    need_h = a / k_h - depth0
    need_v = b / k_v - depth0
    need_near = float(near_clip_m) - depth0         # 모든 점이 근접면 뒤에 있어야 한다
    D = float(max(need_h.max(), need_v.max(), need_near.max(), float(min_distance_m)))
    i_h, i_v = int(np.argmax(need_h)), int(np.argmax(need_v))

    eye = tgt - D * f
    d = depth0 + D
    u_frac = np.where(d > 0, a / (d * math.tan(hfov / 2.0)), np.inf)
    v_frac = np.where(d > 0, b / (d * math.tan(vfov / 2.0)), np.inf)
    fits = bool((d > float(near_clip_m) - 1e-12).all()
                and (u_frac <= 1.0 + 1e-9).all() and (v_frac <= 1.0 + 1e-9).all())

    return {
        "eye_m": [float(v) for v in eye],
        "target_m": [float(v) for v in tgt],
        "forward_unit": [float(v) for v in f],
        "distance_m": D,
        "focal_mm": float(focal_mm),
        "aperture_h_mm": float(aperture_h_mm),
        "aperture_v_mm": aperture_v,
        "hfov_deg": math.degrees(hfov), "vfov_deg": math.degrees(vfov),
        "margin_frac": float(margin_frac),
        "near_clip_m": float(near_clip_m),
        "n_points": int(len(pts)),
        "aabb_lo_m": [float(v) for v in lo], "aabb_hi_m": [float(v) for v in hi],
        "aabb_extent_m": [float(v) for v in (hi - lo)],
        "binding_constraint": ("horizontal" if need_h.max() >= need_v.max() else "vertical"),
        "binding_point_h_m": [float(v) for v in pts[i_h]],
        "binding_point_v_m": [float(v) for v in pts[i_v]],
        "max_screen_frac_u": float(np.max(u_frac)),
        "max_screen_frac_v": float(np.max(v_frac)),
        "min_depth_m": float(np.min(d)),
        "fits": fits,
        "method": ("closed-form minimum distance from pinhole FOV, then re-projection check; "
                   "not a heuristic multiple of a span"),
    }
