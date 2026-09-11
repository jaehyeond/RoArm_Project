"""보고서용 마크다운 표 생성(손으로 적는 값 없음). usage: python report_tables_w10.py <cell_dir> [n_last_rows=12]
① 타임라인 diag 행(마지막 n): sync 시각·q·최대속도 owner·메시별 최대 접촉(그룹·h·안·기하 관입·유령·힘)
② diverge_event*.json: culprit 직전 sync 표 + GetContactDetailedInfo 상위 pair
③ gates_w10.json G4 비교표(있으면)"""
import json, sys
from pathlib import Path
cell = Path(sys.argv[1]); n = int(sys.argv[2]) if len(sys.argv) > 2 else 12
tl = json.load(open(cell / "timeline_seed460.json")); rows = [r for r in tl["rows"] if "diag" in r]
print(f"### 타임라인 diag 행 (state={tl['state']}, diag 행 {len(rows)}개, 마지막 {n}개)\n")
print("| i | sim_t s | q ° | v_max m/s (owner) | 단일최대 N | M_hinge N·m | top_fixed 그룹·h mm·안·관입 mm·유령·F N·owner | top_door 그룹·h·안·관입·유령·F·owner |")
print("|---|---|---|---|---|---|---|---|")
f = lambda g: "—" if not g else f"{g['group']} · {g['h_mm']:.3f} · {'안' if g['inside'] else '밖'} · {g['pen_geo_mm']:.3f} · {'👻' if g['ghost'] else '-'} · {g['F_N']:.3f} · {g['owner']}"
for r in rows[-n:]:
    d = r["diag"]
    print(f"| {r['i']} | {r['sim_t']:.5f} | {r['q_deg']:.3f} | {r['v_particle_max']:.3f} ({d['v_max_owner']}) | {r['max_single_contact_N']:.3f} | {r['M_hinge_res_Nm']:.4f} | {f(d.get('top_fixed'))} | {f(d.get('top_door'))} |")
for evp in sorted(cell.glob("diverge_event*.json")):
    e = json.load(open(evp)); t = e["trigger"]
    print(f"\n### 이벤트 `{evp.name}` — kind={e.get('kind', 'v')} · 트리거 [{t['phase']}] i={t['i']} t={t['sim_t']} q={t['q_deg']}° v_max={t['v_max']} 단일={t['max_single_contact_N']} N · culprit {e['culprit']} · 기작 **{e['mechanism']}** (ghost_prepop={e['ghost_any_prepop']}, F_max 열={e['squeeze_seq_F_max_N']})\n")
    print("| sync | sim_t | q ° | pop | 근방집합 | culprit v m/s | 중심(립 기준) mm | fixed: 그룹·구k·h·안·관입(DEME)·유령·접촉힘 max/n | door: 그룹·구k·h·안·관입(DEME)·유령·접촉힘 max/n |")
    print("|---|---|---|---|---|---|---|---|---|")
    g2 = lambda g: "—" if not g else f"{g['group']} · k{g['k']} · {g['h_mm']:.3f} · {'안' if g['inside'] else '밖'} · {g['pen_geo_mm']:.3f}({g['deme_pen_mm']:.3f}) · {'👻' if g['ghost'] else '-'} · {g['contacts_on_culprit']['F_max_N']:.3f}/{g['contacts_on_culprit']['n']}"
    for k, s in enumerate(e["culprit_last4"]):
        vm = s.get("vs_mesh", {})
        print(f"| {k - len(e['culprit_last4']) + 1} | {s['sim_t']:.5f} | {s['q_deg']:.4f} | {'pop' if s['pop_sync'] else '직전'} | {'예' if s['in_ring_near_set'] else '아니오'} | {s.get('v_m_s', '—')} | {s.get('centre_rel_lip_mm', '—')} | {g2(vm.get('fixed'))} | {g2(vm.get('door'))} |")
    now = e["culprit_now"]; print(f"\nculprit 지금: pos {now['pos_mm']} mm · v {now['v_m_s']} m/s · 립선까지 {now['dist_to_lip_lines_mm']} mm · vs_mesh {json.dumps({k: {kk: v[kk] for kk in ('group', 'k', 'h_mm', 'inside', 'pen_geo_mm', 'ghost')} for k, v in now.get('vs_mesh', {}).items()}, ensure_ascii=False)}")
    cd = e["contact_detail"]
    if "error" in cd:
        print(f"\nGetContactDetailedInfo: 실패 `{cd['error']}`")
    else:
        print(f"\nGetContactDetailedInfo({cd['wall_s']} s): 전체 잠재 pair {cd['n_pairs_total']} · culprit 관련 {cd['n_pairs_culprit']} · 최대힘 pair SS? {cd.get('top_is_sphere_sphere')}\n")
        print("| type | A | B | AGeo | BGeo | B그룹(geo) | F N | point mm | normal |"); print("|---|---|---|---|---|---|---|---|---|")
        for p in cd["pairs"][:8]:
            print(f"| {p['type']} | {p['A']} | {p['B']} | {p['AGeo']} | {p['BGeo']} | {p['B_group_by_geo'] or p['A_group_by_geo']} | {p['F_N']} | {p['point_mm']} | {p['normal']} |")
res = cell / "scoop_s1_seed460.json"
if res.exists():
    r = json.load(open(res)); print(f"\n결과: diverged={r['diverged']} · stops={[(s['phase'], s['reason'], s['q_deg']) for s in r['door']['stops']]} · 포획 {r['capture']['n_in_cavity']} 개 {r['capture']['mass_g']} g · 립 틈 {r['door']['lip_gap_final_mm']} mm · 물림 {r['door']['n_pinched_at_lip']} · 립등가 피크 {r['forces']['close_peak_lipF_N']} N · v_max {r['pops']['v_particle_max_m_s']} · 벽시계 {r['wall_seconds']} s")
