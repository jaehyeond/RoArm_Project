"""자 눈금 사진에서 px/mm(눈금 주기)와 스캔선의 어두운 구간 폭을 자동 검출. 결과 오버레이 PNG 저장."""
import numpy as np, json, sys
from PIL import Image, ImageOps, ImageDraw
D='/home/cgxr/Downloads/'
# 각 사진: 원본 크롭박스(비율) → 1600px 다운스케일, 그 좌표계에서 tick_y(자 눈금 줄), scan_y(부품 스캔 줄), x범위
CFG={
 'm5_06':dict(f='KakaoTalk_20260903_154920827_06.jpg',box=(0.30,0.35,0.80,0.75),tick_y=905,scan_y=700,x=(455,760)),
 'm6_05':dict(f='KakaoTalk_20260903_154920827_05.jpg',box=(0.30,0.30,0.75,0.65),tick_y=1185,scan_y=1128,x=(540,820)),
 'm7_04':dict(f='KakaoTalk_20260903_154920827_04.jpg',box=(0.25,0.30,0.75,0.65),tick_y=915,scan_y=650,x=(660,900)),
 'm4_base':dict(f='KakaoTalk_20260903_154920827.jpg',box=(0.05,0.25,0.62,0.75),tick_y=560,scan_y=528,x=(340,1360)),
 'm1_03':dict(f='KakaoTalk_20260903_154920827_03.jpg',box=(0.20,0.15,0.75,0.55),tick_y=1430,scan_y=1385,x=(440,1560)),
 'm3_01':dict(f='KakaoTalk_20260903_154920827_01.jpg',box=(0.15,0.20,0.70,0.60),tick_y=815,scan_y=770,x=(520,1580)),
}
def load(c):
    im=ImageOps.exif_transpose(Image.open(D+c['f'])); w,h=im.size; a,b,cc,d=c['box']
    cr=im.crop((int(w*a),int(h*b),int(w*cc),int(h*d))); cr.thumbnail((1600,1600)); return cr
def period_px(gray,y,x0,x1):
    row=gray[y-2:y+3,x0:x1].mean(axis=0).astype(float); row=row-row.mean()
    ac=np.correlate(row,row,'full')[len(row)-1:]; ac/=ac[0]
    # 첫 극대 (8~60 px 범위)
    cand=[(ac[i],i) for i in range(8,60) if ac[i]>ac[i-1] and ac[i]>=ac[i+1]]
    return max(cand)[1] if cand else None, ac
def dark_segments(gray,y,x0,x1,thr=None):
    row=gray[y-2:y+3,x0:x1].mean(axis=0)
    if thr is None: thr=(row.min()+row.max())/2
    dark=row<thr; segs=[]; s=None
    for i,v in enumerate(dark):
        if v and s is None: s=i
        if not v and s is not None: segs.append((s+x0,i-1+x0)); s=None
    if s is not None: segs.append((s+x0,x1-1))
    return segs,thr
out={}
for k,c in CFG.items():
    cr=load(c); cr.save(k+'_wide.png')
    if c['tick_y'] is None: print(k,'wide crop only',cr.size); continue
    g=np.array(cr.convert('L')).astype(float); x0,x1=c['x']
    p,ac=period_px(g,c['tick_y'],x0,x1+300 if x1+300<g.shape[1] else g.shape[1]-1)
    segs,thr=dark_segments(g,c['scan_y'],x0,x1)
    segs=[s for s in segs if s[1]-s[0]>=3]
    mm=[((b-a+1)/p) for a,b in segs]; gaps=[((segs[i+1][0]-segs[i][1]-1)/p) for i in range(len(segs)-1)]
    out[k]={'px_per_mm':p,'segments_px':segs,'seg_mm':[round(v,2) for v in mm],'gap_mm':[round(v,2) for v in gaps],'span_mm':round((segs[-1][1]-segs[0][0]+1)/p,2) if segs else None}
    print(k,out[k])
    dr=ImageDraw.Draw(cr); dr.line([(x0,c['tick_y']),(x1,c['tick_y'])],fill='lime',width=2); dr.line([(x0,c['scan_y']),(x1,c['scan_y'])],fill='red',width=2)
    for (a,b),v in zip(segs,mm): dr.rectangle([a,c['scan_y']-25,b,c['scan_y']+25],outline='yellow',width=2); dr.text((a,c['scan_y']-45),f'{v:.2f}',fill='yellow')
    dr.text((x0,c['tick_y']+15),f'{p} px/mm',fill='lime'); cr.save(k+'_meas.png')
json.dump(out,open('ruler_measure.json','w'),indent=1)
