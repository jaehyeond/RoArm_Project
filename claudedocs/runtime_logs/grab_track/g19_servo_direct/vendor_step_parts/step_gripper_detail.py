"""가동 조·서보 디스크·나사의 원형 에지(구멍·보스)를 루트 프레임에서 뽑는다. 축은 디스크 형상에서 자동 추정."""
import json, numpy as np
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TDocStd import TDocStd_Document
from OCP.TCollection import TCollection_ExtendedString
from OCP.XCAFDoc import XCAFDoc_DocumentTool
from OCP.TDF import TDF_LabelSequence, TDF_Label
from OCP.TDataStd import TDataStd_Name
from OCP.TopLoc import TopLoc_Location
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
from OCP.BRep import BRep_Tool
from OCP.TopoDS import TopoDS
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_Circle, GeomAbs_Cylinder
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp
doc=TDocStd_Document(TCollection_ExtendedString('doc')); rd=STEPCAFControl_Reader(); rd.SetNameMode(True)
rd.ReadFile('step/RoArm-M3_STEP/RoArm-M3.step'); rd.Transfer(doc); st=XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
def name(lab):
    n=TDataStd_Name(); return n.Get().ToExtString() if lab.FindAttribute(TDataStd_Name.GetID_s(), n) else ''
shapes={}
def walk(lab, loc, path):
    if st.IsAssembly_s(lab):
        comps=TDF_LabelSequence(); st.GetComponents_s(lab, comps, False)
        for i in range(1, comps.Length()+1):
            c=comps.Value(i); ref=TDF_Label(); st.GetReferredShape_s(c, ref); walk(ref, loc.Multiplied(st.GetLocation_s(c)), path+[name(c) or name(ref)])
    else: shapes['/'.join(path)]=st.GetShape_s(lab).Moved(loc)
free=TDF_LabelSequence(); st.GetFreeShapes(free)
for i in range(1, free.Length()+1): walk(free.Value(i), TopLoc_Location(), [name(free.Value(i))])
def bbox(s):
    b=Bnd_Box(); BRepBndLib.AddOptimal_s(s,b,True,False); return np.array(b.Get())
def com(s):
    g=GProp_GProps(); BRepGProp.VolumeProperties_s(s,g); c=g.CentreOfMass(); return np.array([c.X(),c.Y(),c.Z()]), g.Mass()
def circles(s):
    out=[]; ex=TopExp_Explorer(s, TopAbs_EDGE); seen=set()
    while ex.More():
        e=ex.Current(); ex.Next()
        try: ad=BRepAdaptor_Curve(TopoDS.Edge_s(e))
        except Exception: continue
        if ad.GetType()==GeomAbs_Circle:
            c=ad.Circle(); l=c.Location(); ax=c.Axis().Direction(); r=c.Radius()
            k=(round(l.X(),2),round(l.Y(),2),round(l.Z(),2),round(r,2))
            if k in seen: continue
            seen.add(k); out.append({'c':[k[0],k[1],k[2]],'r':k[3],'n':[round(ax.X(),3),round(ax.Y(),3),round(ax.Z(),3)]})
    return out
def cylinders(s):
    out=[]; ex=TopExp_Explorer(s, TopAbs_FACE); seen=set()
    while ex.More():
        f=TopoDS.Face_s(ex.Current()); ex.Next(); ad=BRepAdaptor_Surface(f)
        if ad.GetType()==GeomAbs_Cylinder:
            cy=ad.Cylinder(); l=cy.Location(); ax=cy.Axis().Direction(); r=cy.Radius()
            k=(round(r,2),round(ax.X(),2),round(ax.Y(),2),round(ax.Z(),2))
            out.append({'loc':[round(l.X(),2),round(l.Y(),2),round(l.Z(),2)],'r':round(r,3),'n':[round(ax.X(),3),round(ax.Y(),3),round(ax.Z(),3)]})
    return out
mov=[k for k in shapes if '活动侧' in k][0]; S=shapes[mov]; B=bbox(S)
def inter(a,b,m): return all(a[i]-m<=b[i+3] and b[i]-m<=a[i+3] for i in range(3))
near={k:v for k,v in shapes.items() if k!=mov and inter(bbox(v),B,4.0)}
res={'movable_jaw':{'bbox':B.round(2).tolist()}, 'neighbors':{}}
for k,v in near.items():
    bb=bbox(v); c,m=com(v); d={'bbox':bb.round(2).tolist(),'vol':round(m,2),'com':c.round(2).tolist()}
    if '舵盘' in k or '螺丝' in k or 'ST3215' in k or 'SCS215' in k or '手腕' in k or '夹头' in k:
        d['circles']=circles(v)[:40]
    res['neighbors'][k]=d
# movable jaw: cylinders (holes/bosses) + circles
res['movable_jaw']['cylinders']=cylinders(S); res['movable_jaw']['circles']=circles(S)
fix=[k for k in shapes if '固定侧' in k][0]; Bf=bbox(shapes[fix]); res['fixed_jaw']={'bbox':Bf.round(2).tolist(),'neighbors':{}}
for k,v in shapes.items():
    if k in (fix,mov): continue
    if inter(bbox(v),Bf,1.0) and ('螺丝' in k or '底座' in k or '螺母' in k or '铜柱' in k):
        c,m=com(v); res['fixed_jaw']['neighbors'][k]={'bbox':bbox(v).round(2).tolist(),'com':c.round(2).tolist()}
res['fixed_jaw']['cylinders']=cylinders(shapes[fix])
json.dump(res, open('gripper_detail.json','w'), ensure_ascii=False, indent=1)
print('neighbors:', list(near)); print('jaw cylinders', len(res['movable_jaw']['cylinders']), 'circles', len(res['movable_jaw']['circles']))
