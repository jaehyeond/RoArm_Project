"""STEP AP214 텍스트에서 조립 트리(부모→자식 occurrence)를 뽑는다. 기하는 안 읽는다."""
import re, sys, collections, json
S=open('step/RoArm-M3_STEP/RoArm-M3.step',encoding='utf-8',errors='ignore').read()
def dec(n):
    def r(m): 
        h=m.group(1); return ''.join(chr(int(h[i:i+4],16)) for i in range(0,len(h),4))
    n=re.sub(r'\\X2\\([0-9A-F]+)\\X0\\',r,n)
    n=re.sub(r'\\X\\([0-9A-F]{2})',lambda m:chr(int(m.group(1),16)),n)
    return n.replace('\n','')
ent={}
for m in re.finditer(r'#(\d+)\s*=\s*([A-Z_0-9]+)\s*\((.*?)\)\s*;',S,re.S):
    ent[int(m.group(1))]=(m.group(2),m.group(3))
def args(a): return re.findall(r"#(\d+)",a)
def strs(a): return re.findall(r"'((?:[^']|'')*)'",a)
prod={i:dec(strs(a)[0]) for i,(t,a) in ent.items() if t=='PRODUCT'}
pdf={i:int(args(a)[0]) for i,(t,a) in ent.items() if t.startswith('PRODUCT_DEFINITION_FORMATION')}
pd={i:int(args(a)[0]) for i,(t,a) in ent.items() if t=='PRODUCT_DEFINITION'}
def pdname(i): return prod.get(pdf.get(pd.get(i,-1),-1),'?')
children=collections.defaultdict(list); haspar=set()
for i,(t,a) in ent.items():
    if t=='NEXT_ASSEMBLY_USAGE_OCCURRENCE':
        ids=args(a); par,ch=int(ids[0]),int(ids[1]); nm=dec(strs(a)[1]) if len(strs(a))>1 else ''
        children[par].append((ch,nm)); haspar.add(ch)
roots=[i for i in pd if i not in haspar and (i in children)]
print('roots:',[(i,pdname(i)) for i in roots])
def walk(i,depth,out,maxd=9):
    kids=children.get(i,[])
    out.append(('  '*depth)+f'{pdname(i)}'+(f'  [{len(kids)} children]' if kids else ''))
    if depth<maxd:
        for ch,nm in kids: walk(ch,depth+1,out)
out=[]; 
for r in roots: walk(r,0,out)
open('tree_full.txt','w').write('\n'.join(out))
print('tree lines', len(out))
# subtrees of interest
kw=sys.argv[1:] or ['夹头','手腕','舵盘']
def find(i,path,res):
    n=pdname(i)
    if any(k in n for k in kw): res.append((i,path+[n]))
    for ch,_ in children.get(i,[]): find(ch,path+[n],res)
res=[]; 
for r in roots: find(r,[],res)
seen=set()
for i,p in res:
    if pdname(i) in seen: continue
    seen.add(pdname(i)); print('\n>>> PATH:', ' / '.join(p)); o=[]; walk(i,0,o,maxd=6); print('\n'.join(o[:80]))
