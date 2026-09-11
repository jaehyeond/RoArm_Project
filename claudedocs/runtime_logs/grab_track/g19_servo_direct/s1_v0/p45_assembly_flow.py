"""p45: S1 실물 조립 순서도. 근거 = STEP 조립 관계(vendor_step_parts/stock_jaw_interface.json), 실물 대조(real_arm_confirmation_20260903.json), hardware.md 그리퍼 서보 규약 5조."""
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, matplotlib.font_manager as fm, os
from matplotlib.patches import FancyBboxPatch
_fp='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'; fm.fontManager.addfont(_fp); plt.rcParams['font.family']=fm.FontProperties(fname=_fp).get_name()
D=os.path.dirname(os.path.abspath(__file__))+'/'
steps=[
 ('0','준비','인쇄 2개(문·고정부) + M3×6 ×9 + M3×8 ×3 + M3 너트 ×3, 육각 2.5 렌치, 캘리퍼스. 팔 주변 30 cm 비우기','부품 전부 있음','white'),
 ('1','부팅해서 닫힘 위치 만들기','⚠ 순정 조 붙은 채로 전원 ON → moveInit 이 팔 HOME + 그리퍼 π(닫힘). 끝나면 전원 OFF (어댑터 뽑기)','두 블레이드가 나란히 닫혀 있음 = 디스크가 π 위치','lightyellow'),
 ('2','순정 가동 조 떼기','구동측(ST3215-HS 라벨 쪽) 뺨의 M3×4 4개 → 반대측 4개. 중앙 나사(디스크↔스플라인)는 그대로 둠. 조를 디스크 위로 빼서 보관','나사 8개 + 조 1개 보관. 디스크 두 장은 서보에 남아 있음','lightyellow'),
 ('3','고정부 달기','판을 고정 조 바깥면(LED 쪽)에 대고 3구멍 맞춤 → M3×8 을 바깥에서 넣고 안쪽(빈 간극)에서 너트. 스파인이 블레이드 끝 아래로, 반쪽 보울이 그 밑','3구멍 일치 · 볼트 머리와 LED 간섭 없음 · 판이 플랜지 안쪽에 있음','lightgreen'),
 ('4','문 끼워 보기 (나사 없이)','허브 창(지름 11)을 디스크 보스에 씌우고 4구멍을 디스크 탭 구멍에 맞춤. 4방향 중 반쪽 보울이 고정 반쪽과 만나는 방향으로','양쪽 4구멍 맞음 · 문 립과 고정 립이 한 선에서 만남 · 뺨이 서보 케이스에 안 닿음','lightblue'),
 ('5','문 나사 체결','M3×6 을 구동측 4개 → 반대측 4개. PLA 라 살짝만 조임(디스크 탭 2.5 mm 물림)','디스크 뒤로 나사 돌출 ≤ 1 mm(케이스 면까지 2.3 여유)','lightblue'),
 ('6','손으로 동작 확인 (무전원)','문을 손으로 30° 정도 열었다 닫기. 서보 기어가 역구동됨','열림에서 link5·LED·고정부와 안 닿음 · 닫힘에서 립 틈 ≤ 0.5 mm · 걸림 없음','lightblue'),
 ('7','전원 ON 절차','문을 손으로 닫아 둠 → 주변 비움 → 전원 ON(팔 HOME 이동, 그리퍼 π=닫힘) → 먼저 {"T":107,"tor":200} → SDK 각도 0→10→20→30 순으로','[금지] 맨 {"T":106} 금지 · 각도 상한 30° (30° 넘는 구간은 스윕 미검증) · 개구>44 mm 면 롤 |r|≤14°','mistyrose'),
 ('8','치수 검수','닫힘: 립 틈. 30°: 립 사이 입 폭(설계 58). 캘리퍼스','입 55~60 mm · 립 틈 ≤ 0.5 · 처짐 육안 없음','lightgreen'),
 ('9','첫 파지 시험','지름 30 공을 파팅면에서 문 쪽으로 ≥ 22 mm 떨어뜨려 놓고 시뮬과 같은 순서(접근→열기→하강→닫기→들기)','공이 보울 안에 들려 올라옴. 시뮬 결과(z 0.163, 3.2 mm)와 대조','lightgreen')]
fig,ax=plt.subplots(figsize=(19,15.5)); ax.axis('off'); ax.set_xlim(0,100); ax.set_ylim(0,100)
y=97
for n,title,body,check,col in steps:
    h=8.3
    ax.add_patch(FancyBboxPatch((1,y-h),64,h,boxstyle='round,pad=0.3',fc=col,ec='gray'))
    ax.text(2.5,y-1.6,f'{n}. {title}',fontsize=13,fontweight='bold',va='top'); ax.text(2.5,y-4.0,body,fontsize=10.5,va='top',wrap=True)
    ax.add_patch(FancyBboxPatch((67,y-h),32,h,boxstyle='round,pad=0.3',fc='white',ec='gray',ls='--'))
    ax.text(68.5,y-1.6,'확인',fontsize=11,fontweight='bold',va='top',color='dimgray'); ax.text(68.5,y-4.0,check,fontsize=10,va='top',wrap=True)
    if n!='9': ax.annotate('',(33,y-h-0.3),(33,y-h-1.5),arrowprops=dict(arrowstyle='<-',color='k',lw=1.2))
    y-=h+1.5
ax.set_title('S1 실물 조립 순서 — 노랑 = 순정 해체 · 초록 = 고정부/검수 · 파랑 = 문 · 분홍 = 전원 규약 (hardware.md 5조)',fontsize=14)
fig.text(0.01,0.005,'근거: 벤더 STEP 조립 관계(문 = 디스크 두 장에 4+4 볼트, 중앙 나사는 디스크↔스플라인) · 실물 대조 09-03(고정 조 3구멍 M3 관통·너트 OK, 라벨 ST3215-HS) · 시뮬 게이트(0~30° 스윕 간섭 1.3 mm, 립 정합 0.0)',fontsize=9.5,color='dimgray')
plt.tight_layout(); plt.savefig(D+'p45_assembly_flow.png',dpi=105); print('saved')
