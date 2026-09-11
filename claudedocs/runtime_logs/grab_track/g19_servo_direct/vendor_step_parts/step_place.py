"""STEP 텍스트에서 각 occurrence 의 배치(원점·z·x 방향)를 뽑아 루트 프레임으로 합성한다."""
import re, collections, json, numpy as np
S=open('step/RoArm-M3_STEP/RoArm-M3.step',encoding='utf-8',errors='ignore').read()
def dec(n):
    n=re.sub(r'\\X2\\([0-9A-F]+)\\X0\\',lambda m:''.join(chr(int(m.group(1)[i:i+4],16)) for i in range(0,len(m.group(1)),4)),n)
    return re.sub(r'\\X\\([0-9A-F]{2})',lambda m:chr(int(m.group(1),16)),n).replace('\n','')
ent={}
for m in re.finditer(r'#(\d+)\s*=\s*(.*?)\s*;\s*\n',S,re.S):
    ent[int(m.group(1))]=m.group(2)
def typ(b): 
    b=b.strip(); return 'COMPLEX' if b.startswith('(') else b.split('(',1)[0].strip()
def refs(b): return [int(x) for x in re.findall(r'#(\d+)',b)]
def strs(b): return re.findall(r"'((?:[^']|'')*)'",b)
def nums(b): return [float(x) for x in re.findall(r'\(([^()]*)\)',b)[-1].split(',')]
prod={i:dec(strs(b)[0]) for i,b in ent.items() if typ(b)=='PRODUCT'}
pdf={i:refs(b)[0] for i,b in ent.items() if typ(b).startswith('PRODUCT_DEFINITION_FORMATION')}
pd={i:refs(b)[0] for i,b in ent.items() if typ(b)=='PRODUCT_DEFINITION'}
pdname=lambda i: prod.get(pdf.get(pd.get(i,-1),-1),'?')
nauo={i:(refs(b)[0],refs(b)[1]) for i,b in ent.items() if typ(b)=='NEXT_ASSEMBLY_USAGE_OCCURRENCE'}
# PRODUCT_DEFINITION_SHAPE -> nauo ; CDSR(#rr, #pds)
pds_to_nauo={i:refs(b)[0] for i,b in ent.items() if typ(b)=='PRODUCT_DEFINITION_SHAPE' and refs(b) and refs(b)[0] in nauo}
def axis(i):
    b=ent[i]; r=refs(b); p=np.array(nums(ent[r[0]])); z=np.array(nums(ent[r[1]])) if len(r)>1 else np.array([0,0,1.]); x=np.array(nums(ent[r[2]])) if len(r)>2 else np.array([1.,0,0]); return p,z,x
def T_of(ax_from, ax_to):
    # 변환: from 프레임 -> to 프레임 (둘 다 부모 좌표계 표현). 보통 from = 단위, to = 자식 배치
    def M(p,z,x):
        z=z/np.linalg.norm(z); x=x-np.dot(x,z)*z; x=x/np.linalg.norm(x); y=np.cross(z,x)
        T=np.eye(4); T[:3,0]=x; T[:3,1]=y; T[:3,2]=z; T[:3,3]=p; return T
    return M(*ax_to) @ np.linalg.inv(M(*ax_from))
placement={}  # nauo id -> 4x4 (child frame in parent frame)
for i,b in ent.items():
    if typ(b)!='CONTEXT_DEPENDENT_SHAPE_REPRESENTATION': continue
    rr,pds=refs(b)[:2]
    if pds not in pds_to_nauo: continue
    rb=ent[rr]; idt=[r for r in refs(rb) if typ(ent[r])=='ITEM_DEFINED_TRANSFORMATION']
    if not idt: continue
    a=refs(ent[idt[0]]); placement[pds_to_nauo[pds]]=T_of(axis(a[0]),axis(a[1]))
print('placements', len(placement), 'of', len(nauo))
children=collections.defaultdict(list)
for i,(par,ch) in nauo.items(): children[par].append((ch,i))
root=[p for p in children if p not in {c for _,c in nauo.values()}][0]
world={}  # (path) -> T
def walk(pdid,T,path):
    for ch,ni in children.get(pdid,[]):
        Tc=T@placement.get(ni,np.eye(4)); nm=pdname(ch); p=path+[nm]; world['/'.join(p)]=Tc; walk(ch,Tc,p)
walk(root,np.eye(4),[])
kw=['夹头','手腕','ST3215','SCS215','舵盘','轴承','BEARING','M3x4','M3*6','KM2.5','PA2']
rows=[]
for k,T in world.items():
    if any(w in k for w in kw):
        o=T[:3,3]; z=T[:3,2]; x=T[:3,0]; rows.append((k,o,z,x))
rows.sort(key=lambda r:-r[1][2])
for k,o,z,x in rows: print(f'{k:60s} o=({o[0]:8.2f},{o[1]:8.2f},{o[2]:8.2f}) z=({z[0]:5.2f},{z[1]:5.2f},{z[2]:5.2f}) x=({x[0]:5.2f},{x[1]:5.2f},{x[2]:5.2f})')
json.dump({k:T.tolist() for k,T in world.items()}, open('world_placements.json','w'))
