"""Isaac `SimulationApp.close()` 차단 지점 **국소화** — durable receipt. 기준 완화·강제 종료 없음.

근거: root `msg_56ff30cd8eeb` (2), 정정 `msg_bcb387d139fc` (2)(3)(4),
재정정 `msg_2020e857ed59` ①②③.

⚠️ **철회된 주장** (root `msg_bcb387d139fc` (4))
    rev21 주석은 "설치본 `simulation_app.py` 803~805(Replicator 대기)에서 멈췄다" 고 적었다.
    **그 단정은 rev20 실측 반례로 철회한다** — rev20 은 `wait_for_replicator=False` 로 돌았고
    그 분기는 **실행되지 않았는데도** close 가 걸렸다. 따라서 차단 지점은 **미확정**이며
    후보는 `get_status` · `stop` · `set_capture_on_play` · 그 이후 프레임워크 종료 전부다.

⚠️ **이것은 순수 관측이 아니다** (root `msg_bcb387d139fc` (4))
    `set_capture_on_play(False)` 를 `stop()` **보다 먼저** 부르는 것은 **상태에 영향을 주는
    순서 개입**이다(설치본 순서는 get_status → stop → … → set_capture_on_play).
    따라서 이 모듈은 "순수 관측"이 아니라 **진단 절차 변경**이다. 영수증에도 그렇게 적는다.

계측 규칙
    · **모든 호출**이 전후로 디스크에 flush 된다 — 루프 안의 개별 `app_update`/`get_status` 도
      예외가 아니다(`msg_2020e857ed59` ①: bucket 일 때 flush 를 건너뛰면 바로 그 native hang
      지점이 영수증에 남지 않는다).
    · 상태 판정은 **실제 enum 동등성**으로 한다. `repr` 문자열이나 `"STOPPED" in ...` substring
      검사를 쓰지 않는다(`msg_2020e857ed59` ②). 비교는 호출 직후 `classify` 콜백 안에서 하고
      JSON 가능한 bool 만 저장한다. **계측 밖 `get_status` 재호출은 하지 않는다.**
    · 경과는 **실측 `t_after - t_before`** 다. `min(budget, actual)` 로 자르지 않는다.
    · **진단 총시간** 과 **실제 close 시간** 을 분리해 남기고, **정리 합계**도 따로 남긴다.
    · 프로브 예외 · 미완료 · STOPPED 미관측 · **per_call 오류/미반환** 은 조용히 PASS 시키지
      않는다 — `measurement_satisfied=False` 로 표시하고 호출자가 **비성공**으로 다뤄야 한다.
      루프에서는 **오류를 STOPPED 판정보다 먼저** 본다(`msg_2020e857ed59` ③: 그러지 않으면
      `app_update` 오류 + status STOPPED 조합이 거짓 PASS 가 된다).
    · `SystemExit`/`KeyboardInterrupt` 는 **삼키지 않는다** — 기록 후 **재전파**한다.

하지 않는 것
    기준 완화·강제 `rc 0`·`skip_cleanup=True`·private `__del__` 금지.
    `wait_until_complete()` 는 부르지 않는다(무한 대기 후보라 국소화 대상 밖).
    실패를 성공으로 바꾸지 않는다.
"""
import json
import time
import weakref
from pathlib import Path

ARTIFACT = "W13R_CLOSE_LOCALIZATION_RECEIPT"
# 재전파해야 하는 제어 예외 — 절대 삼키지 않는다.
_PROPAGATE = (SystemExit, KeyboardInterrupt)


class CloseProbe:
    """각 단계를 재현하고 **모든 호출 전후**로 영수증을 디스크에 원자적 flush 한다."""

    def __init__(self, receipt_path, wall_clock_utc=None):
        self.path = Path(receipt_path)
        self.rec = {
            "artifact": ARTIFACT,
            "started_utc": wall_clock_utc or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "t0_monotonic": time.monotonic(),
            "installed_sequence_reference": (
                "get_status() -> (if not STOPPED/STOPPING) stop() -> "
                "(if wait_for_replicator) wait_until_complete()+sleep(1) -> set_capture_on_play(False)"),
            "this_is_not_pure_observation": True,
            "procedure_change_note": (
                "set_capture_on_play(False) 를 stop() 보다 **먼저** 부른다. 설치본 순서와 다르고 "
                "상태에 영향을 주는 **순서 개입**이므로 순수 관측이 아니라 진단 절차 변경이다."),
            "withdrawn_claim": (
                "'설치본 803~805 Replicator 대기에서 멈췄다' 는 단정은 rev20 실측 반례로 철회됐다 "
                "(wait_for_replicator=False 라 그 분기 미실행인데도 걸렸다). 차단 지점은 미확정이다."),
            "status_compared_by": "실제 enum 동등성(classify 콜백) — repr/substring 검사 아님",
            "every_call_flushed_before_and_after": True,
            "wait_until_complete_called": False,
            "steps": [],
            "completed_all_steps": False,
            "measurement_satisfied": False,
            "measurement_unmet_reasons": [],
        }
        self._flush()

    def _flush(self):
        now = time.monotonic()
        self.rec["last_flush_monotonic"] = now
        self.rec["diagnostic_elapsed_s"] = round(now - self.rec["t0_monotonic"], 4)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.rec, ensure_ascii=False, indent=2, default=str) + "\n")
        tmp.replace(self.path)          # 원자적 교체 — 부분 기록 파일을 남기지 않는다

    def unmet(self, reason):
        self.rec["measurement_unmet_reasons"].append(reason)

    def step(self, name, fn, classify=None, bucket=None):
        """`fn` 을 부르고 before/after 를 **항상 디스크에** 남긴다.

        `classify(value) -> dict`: 반환값 판정을 **호출 직후 실제 객체로** 수행해 JSON 가능한
        bool 만 저장한다. 문자열 `repr` 재파싱이나 substring 검사를 쓰지 않기 위한 장치다.

        · 반환하지 않으면 영수증에 `returned=False` 인 채로 남는다(= 여기서 막혔다는 증거).
        · `SystemExit`/`KeyboardInterrupt` 는 **기록 후 재전파**한다(삼키지 않는다).
        · 그 외 예외는 기록하고 전파하지 않되 `raised=True` 로 **반환과 구별**한다.
        """
        ent = {"step": name, "entered_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "entered_monotonic": time.monotonic(), "returned": False, "raised": False,
               "propagated": False, "returned_monotonic": None, "elapsed_s": None,
               "value": None, "classified": None, "classify_error": None, "error": None}
        (self.rec["steps"] if bucket is None else bucket).append(ent)
        self._flush()                   # ← 진입 사실을 **언제나** 먼저 디스크에 남긴다
        try:
            v = fn()
            ent["value"] = repr(v)[:300]            # 사람이 읽는 용도. 판정에는 쓰지 않는다.
            if classify is not None:
                try:
                    ent["classified"] = classify(v)
                except _PROPAGATE:
                    raise
                except BaseException as cexc:       # noqa: BLE001
                    ent["classify_error"] = f"{type(cexc).__name__}: {cexc}"[:300]
            ent["returned"] = True
        except _PROPAGATE as exc:       # 제어 예외는 기록 후 **재전파**
            ent["raised"] = True
            ent["propagated"] = True
            ent["error"] = f"{type(exc).__name__}: {exc}"[:300]
            ent["returned_monotonic"] = time.monotonic()
            ent["elapsed_s"] = round(ent["returned_monotonic"] - ent["entered_monotonic"], 4)
            self.unmet(f"{name}: {type(exc).__name__} 재전파")
            self._flush()
            raise
        except BaseException as exc:    # noqa: BLE001 — 기록하고 계속(반환과 구별)
            ent["raised"] = True
            ent["error"] = f"{type(exc).__name__}: {exc}"[:300]
        ent["returned_monotonic"] = time.monotonic()
        # 실측 경과. 어떤 예산으로도 자르지 않는다.
        ent["elapsed_s"] = round(ent["returned_monotonic"] - ent["entered_monotonic"], 4)
        self._flush()                   # ← 반환 사실도 **언제나** 디스크에 남긴다
        return ent


def record_single_call(receipt_path, name, fn, note=None):
    """공개 호출 **하나**를 durable receipt 로 감싼다 (root `msg_d1e4f0b148ed`).

    rev24 의 유일한 수명주기 개입인 `SimulationContext.clear_instance()` 를 위해 쓴다.
    `CloseProbe.step` 을 그대로 재사용하므로 전후 원자 flush · `SystemExit`/`KeyboardInterrupt`
    재전파 · 반환/예외 구별 계약이 동일하게 적용된다.

    반환: `(rec, ent)` — `ent["returned"]`/`ent["raised"]`/`ent["elapsed_s"]` 가 호출 결과다.
    **예외는 성공으로 바뀌지 않는다**: `raised` 면 `measurement_satisfied=False` 다.
    """
    p = CloseProbe(receipt_path)
    p.rec["artifact"] = "W13R_SINGLE_CALL_RECEIPT"
    p.rec["call"] = name
    if note:
        p.rec["note"] = note
    p._flush()
    ent = p.step(name, fn)                      # 전후 flush · 제어예외 재전파는 step 계약 그대로
    p.rec["completed_all_steps"] = True
    if ent["raised"]:
        p.unmet(f"{name}: 예외 발생 — 성공으로 취급하지 않는다")
    if not ent["returned"]:
        p.unmet(f"{name}: 반환하지 않았다")
    unmet = list(dict.fromkeys(p.rec["measurement_unmet_reasons"]))
    p.rec["measurement_unmet_reasons"] = unmet
    p.rec["measurement_satisfied"] = bool(ent["returned"] and not ent["raised"] and not unmet)
    p.rec["elapsed_s"] = ent["elapsed_s"]
    p._flush()
    return p.rec, ent


#: 계약 2 가 요구하는 **정확한 키 집합**. 누락·추가 모두 실패다.
RELEASE_MAPPING_KEYS = ("cams_is_none", "scene_is_none", "robot_is_none", "sim_is_none",
                        "public_instance_is_none")


def _classify_release_mapping(value, expected_keys=RELEASE_MAPPING_KEYS):
    """`release()` 가 돌려준 **실제 mapping** 을 엄격 분류한다 (REFERENCE_RELEASE_CONTRACT_REV2 §2).

    · exact keys: 누락도 추가도 실패.
    · exact bool: `type(v) is bool` 만 통과 — truthy(1, "x", np.bool_)는 **타입 오류**로 실패.
    · 모든 값이 `True` 여야 `all_true`.
    · repr 문자열·상수 True 로 관측을 대신하지 않는다(이 함수는 호출 직후 **그 객체**를 본다).
    """
    exp = tuple(expected_keys)
    res = {"expected_keys": list(exp), "is_mapping": isinstance(value, dict),
           "keys_ok": False, "types_ok": False, "all_true": False,
           "missing": [], "extra": [], "bad_type": [], "false_keys": [], "values": {}}
    if not isinstance(value, dict):
        res["actual_type"] = type(value).__name__
        return res
    keys = set(value)
    res["missing"] = sorted(set(exp) - keys)
    res["extra"] = sorted(keys - set(exp))
    res["keys_ok"] = not res["missing"] and not res["extra"]
    for k in exp:
        if k not in value:
            continue
        v = value[k]
        if type(v) is bool:                                  # noqa: E721 — 엄격 타입 검사
            res["values"][k] = v
            if v is not True:
                res["false_keys"].append(k)
        else:
            res["bad_type"].append(k)
            res["values"][k] = f"<non-bool {type(v).__name__}: {v!r}>"[:80]
    res["types_ok"] = not res["bad_type"]
    res["all_true"] = bool(res["keys_ok"] and res["types_ok"] and not res["false_keys"]
                           and all(value[k] is True for k in exp))
    return res


def record_reference_release(receipt_path, targets, release, note=None,
                             required=("camera_side", "camera_top", "scene", "robot"),
                             informational=("sim",), expected_keys=RELEASE_MAPPING_KEYS):
    """렌더러가 보유한 참조를 놓는 과정을 durable receipt 로 남긴다 (root `msg_2b0a2544945f`).

    `targets`: `{이름: 객체}` — 해제 **전에** 관측만 한다.
    `release`: 실제로 이름들을 None 으로 만드는 콜러블(여기서 `__del__` 을 직접 부르지 않는다).

    관측 규칙
        · weakref 는 **강한 참조를 남기지 않는다**. 객체 자체를 영수증에 담지 않고
          타입명·id·생존여부만 적는다. `dict` 처럼 weakref 불가 타입은 그 사실을 적는다.
        · `gc.collect()` 같은 전역 수집을 부르지 않는다 — 살아남은 것은 **관측 그대로** 남긴다.
        · 해제 호출 자체는 `CloseProbe.step` 계약(전후 flush · 제어예외 재전파 · 반환/예외 구별).

    ⚠️ **native 효과는 미입증**이다. 이 영수증은 파이썬 참조가 언제 풀렸는지만 말한다.
    """
    p = CloseProbe(receipt_path)
    p.rec["artifact"] = "W13R_REFERENCE_RELEASE_RECEIPT"
    if note:
        p.rec["note"] = note
    p.rec["native_effect_unproven"] = True
    # 관측 단계에서 만든 **임시/컨테이너 강한 참조를 전부** 해제 전에 버린다
    # (root `msg_d23c68b2ac6e`). 남는 것은 weakref 와 JSON 가능한 메타뿐이다.
    obs, wrefs = [], {}
    for name, obj in targets.items():
        ent = {"name": name, "type": type(obj).__name__, "id": (None if obj is None else id(obj)),
               "was_none": obj is None, "weakref_supported": None, "alive_after_release": None}
        if obj is not None:
            try:
                wrefs[name] = weakref.ref(obj)
                ent["weakref_supported"] = True
            except TypeError as exc:                       # 예: dict 은 weakref 불가
                ent["weakref_supported"] = False
                ent["weakref_error"] = str(exc)[:200]
        obs.append(ent)
    # ⚠️ `for` 의 순회 변수는 루프가 끝나도 **바인딩이 남는다** — `obj` 가 마지막 대상을 계속
    #    붙들면 해제해도 살아남아 거짓 생존으로 보고된다. 이름과 함께 명시적으로 버린다.
    name = obj = None
    del name, obj
    del targets                                            # 우리 쪽 강한 참조를 먼저 버린다
    p.rec["targets"] = obs
    p._flush()

    # 계약 2: `release()` 가 **실제 셀 값**에서 계산해 돌려준 mapping 을 호출 직후 그 객체로
    # 분류한다. `CloseProbe.step` 의 `classify` 를 쓰므로 repr 재파싱이 아니다.
    ent = p.step("release_owned_references", release,
                 classify=lambda v: _classify_release_mapping(v, expected_keys))
    p.rec["release_mapping_classification"] = ent["classified"]
    if ent["classify_error"]:
        p.unmet(f"release 반환 mapping 분류 실패: {ent['classify_error']}")

    for o in obs:
        wr = wrefs.get(o["name"])
        if wr is not None:
            o["alive_after_release"] = wr() is not None     # 임시 참조는 이 문장에서 끝난다
    wrefs.clear()
    p.rec["completed_all_steps"] = True
    if ent["raised"]:
        p.unmet("참조 해제 호출이 예외로 끝났다 — 성공으로 취급하지 않는다")
    if not ent["returned"]:
        p.unmet("참조 해제 호출이 반환하지 않았다")

    # 계약 3·4: **필수**(카메라 2 / scene / robot)와 **정보성**(sim)을 혼동하지 않는다.
    by_name = {o["name"]: o for o in obs}
    req_missing = [n for n in required if n not in by_name]
    req_unobservable = [n for n in required
                        if n in by_name and not by_name[n]["weakref_supported"]]
    required_survivors = [n for n in required
                          if n in by_name and by_name[n]["weakref_supported"]
                          and by_name[n]["alive_after_release"]]
    informational_survivors = [n for n in informational
                               if n in by_name and by_name[n]["weakref_supported"]
                               and by_name[n]["alive_after_release"]]
    for o in obs:
        o["role"] = ("required" if o["name"] in required
                     else "informational" if o["name"] in informational else "declared_only")
    p.rec["required_targets"] = list(required)
    p.rec["informational_targets"] = list(informational)
    p.rec["required_survivors"] = required_survivors
    p.rec["informational_survivors"] = informational_survivors
    p.rec["n_weakref_observed"] = sum(1 for o in obs if o["weakref_supported"])
    p.rec["informational_note"] = (
        "informational_survivors 는 실패가 아니다. sim 생존만으로 외부 엔진 소유·누수·안전·"
        "정상 해제를 단정하지 않는다. 다만 공개 instance() 가 None 이 아니면 분류에서 실패다.")
    if req_missing:
        p.unmet(f"필수 관측 대상 누락: {req_missing} — 누락으로 통과시키지 않는다")
    if req_unobservable:
        p.unmet(f"필수 대상이 weakref 로 관측 불가: {req_unobservable} — 대체로 통과시키지 않는다")
    if required_survivors:
        p.unmet(f"필수 대상이 해제 후에도 살아 있다: {required_survivors}")
    cls = ent["classified"]
    if not (isinstance(cls, dict) and cls.get("all_true") is True):
        p.unmet(f"release 반환 mapping 이 exact keys/bool True 를 만족하지 않는다: {cls}")

    unmet = list(dict.fromkeys(p.rec["measurement_unmet_reasons"]))
    p.rec["measurement_unmet_reasons"] = unmet
    p.rec["measurement_satisfied"] = bool(ent["returned"] and not ent["raised"] and not unmet)
    p.rec["elapsed_s"] = ent["elapsed_s"]
    p._flush()
    return p.rec, ent


def _status_classifier(orch):
    """실제 enum 동등성으로 판정하는 콜백. 문자열/substring 비교를 쓰지 않는다."""
    def classify(v):
        st = orch.Status
        return {"is_stopped": bool(v == st.STOPPED), "is_stopping": bool(v == st.STOPPING)}
    return classify


def localize_replicator_close(receipt_path, app_update=None, bounded_update_budget_s=5.0,
                              max_update_iters=200):
    """공식 호출을 하나씩 재현하며 차단 지점을 국소화한다. **close 는 여기서 부르지 않는다.**

    반환 dict 의 `measurement_satisfied` 가 False 면 호출자는 **비성공**으로 다뤄야 한다.
    """
    p = CloseProbe(receipt_path)
    try:
        import omni.replicator.core as rep                              # noqa: PLC0415
    except _PROPAGATE:
        raise
    except BaseException as exc:                                        # noqa: BLE001
        p.rec["import_error"] = f"{type(exc).__name__}: {exc}"[:300]
        p.unmet("omni.replicator.core import 실패")
        p._flush()
        return p.rec

    orch = rep.orchestrator
    cls = _status_classifier(orch)
    p.rec["status_enum_names"] = [n for n in dir(orch.Status) if not n.startswith("_")]
    p._flush()

    p.step("get_status_1", lambda: orch.get_status(), classify=cls)
    s2 = p.step("set_capture_on_play_False", lambda: orch.set_capture_on_play(False))
    # 두 번째 get_status 도 **정식 step** 이며, 판정은 그 호출의 `classified` 만 쓴다.
    # 계측 밖에서 get_status 를 다시 부르지 않는다.
    s2b = p.step("get_status_2_before_stop", lambda: orch.get_status(), classify=cls)
    c2 = s2b["classified"]
    already = (bool(c2["is_stopped"] or c2["is_stopping"])
               if (s2b["returned"] and isinstance(c2, dict)) else None)
    p.rec["already_stopped_or_stopping_before_stop"] = already
    p.rec["already_decided_from"] = "get_status_2_before_stop 의 classified(실제 enum 동등성)"
    if s2b["classify_error"]:
        p.unmet(f"get_status_2 enum 비교 실패: {s2b['classify_error']}")
    if s2["raised"]:
        p.unmet("set_capture_on_play(False) 가 예외였다")
    p._flush()

    s3 = None
    if already is False:
        s3 = p.step("stop", lambda: orch.stop())
    elif already is True:
        p.rec["stop_skipped_reason"] = "이미 STOPPED/STOPPING — 설치본과 같은 조건으로 건너뛴다"
        p._flush()
    else:
        p.rec["stop_skipped_reason"] = "두 번째 get_status 가 값을 주지 못해 판단 불가 — stop 미호출"
        p.unmet("get_status_2 실패로 stop 조건 판단 불가")
        p._flush()

    # `stop()` 이 **반환한 경우에만** bounded update 로 STOPPED 관측.
    stop_ok = already is True or (s3 is not None and s3["returned"] and not s3["raised"])
    if app_update is not None and stop_ok:
        t_begin = time.monotonic()
        t_end = t_begin + float(bounded_update_budget_s)
        obs = {"budget_s": float(bounded_update_budget_s), "max_iters": int(max_update_iters),
               "iters": 0, "reached_stopped": False, "final_status_classified": None,
               "error": None, "per_call": [], "per_call_error": None,
               "note": ("stop() 이 반환한 경우에만 수행. 예산·횟수 이중 상한. 모든 호출 전후 flush. "
                        "무한 native 호출은 이 파이썬 상한으로 막지 못하며 외부 실행기가 최종 경계다.")}
        p.rec["bounded_update_observation"] = obs
        p._flush()
        try:
            while time.monotonic() < t_end and obs["iters"] < int(max_update_iters):
                eu = p.step(f"app_update[{obs['iters']}]", app_update, bucket=obs["per_call"])
                eg = p.step(f"get_status[{obs['iters']}]", lambda: orch.get_status(),
                            classify=cls, bucket=obs["per_call"])
                obs["iters"] += 1
                obs["final_status_classified"] = eg["classified"]
                # ① 오류를 **먼저** 본다. STOPPED 를 먼저 break 하면 app_update 오류 +
                #    status STOPPED 조합이 거짓 PASS 가 된다(msg_2020e857ed59 ③).
                if (eu["raised"] or not eu["returned"] or eg["raised"] or not eg["returned"]
                        or eg["classify_error"]):
                    obs["per_call_error"] = (
                        f"app_update(raised={eu['raised']},returned={eu['returned']}) / "
                        f"get_status(raised={eg['raised']},returned={eg['returned']},"
                        f"classify_error={eg['classify_error']})")
                    p.unmet(f"bounded update per_call 오류/미반환: {obs['per_call_error']}")
                    p._flush()
                    break
                if isinstance(eg["classified"], dict) and eg["classified"]["is_stopped"]:
                    obs["reached_stopped"] = True
                    p._flush()
                    break
        except _PROPAGATE:
            raise
        except BaseException as exc:                                    # noqa: BLE001
            obs["error"] = f"{type(exc).__name__}: {exc}"[:300]
            p.unmet(f"bounded update 예외: {obs['error']}")
        # 실측 경과. min(budget, actual) 로 자르지 않는다 — 실제 초과가 보여야 한다.
        obs["elapsed_s"] = round(time.monotonic() - t_begin, 4)
        obs["exceeded_budget"] = bool(obs["elapsed_s"] > float(bounded_update_budget_s))
        if not obs["reached_stopped"]:
            p.unmet("bounded update 에서 STOPPED 미관측")
        p._flush()
    elif app_update is not None:
        p.rec["bounded_update_skipped_reason"] = (
            "stop() 이 반환하지 않았거나 예외였다 — 관측을 시도하지 않는다")
        p.unmet("stop 미반환/예외로 STOPPED 관측 불가")
        p._flush()

    p.rec["completed_all_steps"] = True
    top = p.rec["steps"]
    per_call = (p.rec.get("bounded_update_observation") or {}).get("per_call") or []
    allc = list(top) + list(per_call)                  # 판정은 **per_call 까지** 본다
    p.rec["summary"] = {
        "first_call_that_did_not_return": next((e["step"] for e in allc if not e["returned"]), None),
        "calls_that_raised": [e["step"] for e in allc if e["raised"]],
        "per_step_elapsed_s": {e["step"]: e["elapsed_s"] for e in top},
        "n_per_call_records": len(per_call),
        # 진단 총시간과 정리 합계를 **분리**해 남긴다. close_s 하나로 누락시키지 않는다.
        "diagnostic_total_s": round(time.monotonic() - p.rec["t0_monotonic"], 4),
        "probe_steps_sum_s": round(sum(e["elapsed_s"] or 0.0 for e in allc), 4),
        "bounded_update_elapsed_s": (p.rec.get("bounded_update_observation") or {}).get("elapsed_s"),
        "note_actual_close_time_is_separate": (
            "실제 close() 소요는 렌더러의 close 계측(close_s)이며 이 진단 시간과 **별개**다. "
            "정리 합계는 두 구간을 감싼 **실측 t_end-t_start** 로 보고해야 한다(부분합 반올림 금지)."),
    }
    unmet = list(dict.fromkeys(p.rec["measurement_unmet_reasons"]))
    p.rec["measurement_unmet_reasons"] = unmet
    p.rec["measurement_satisfied"] = bool(
        p.rec["completed_all_steps"] and not unmet
        and p.rec["summary"]["first_call_that_did_not_return"] is None
        and not p.rec["summary"]["calls_that_raised"])
    p.rec["caller_contract"] = (
        "measurement_satisfied=False 면 호출자는 **비성공**으로 다뤄야 한다. "
        "프로브 예외·미완료·STOPPED 미관측·per_call 오류를 조용히 PASS 시키지 않는다.")
    p._flush()
    return p.rec
