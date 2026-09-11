"""OCP/XDE 로 STEP 조립체를 읽어 leaf 부품마다 루트 프레임 bbox·부피·무게중심을 JSON 으로."""
import json, sys, time
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TDocStd import TDocStd_Document
from OCP.TCollection import TCollection_ExtendedString
from OCP.XCAFDoc import XCAFDoc_DocumentTool, XCAFDoc_ShapeTool
from OCP.TDF import TDF_LabelSequence, TDF_Label
from OCP.TDataStd import TDataStd_Name
from OCP.IFSelect import IFSelect_RetDone
from OCP.TopLoc import TopLoc_Location
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp
from OCP.BRep import BRep_Tool
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_SOLID
t0=time.time()
path='step/RoArm-M3_STEP/RoArm-M3.step'
doc=TDocStd_Document(TCollection_ExtendedString('doc'))
rd=STEPCAFControl_Reader(); rd.SetNameMode(True)
assert rd.ReadFile(path)==IFSelect_RetDone; rd.Transfer(doc)
st=XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
def name(lab):
    n=TDataStd_Name()
    return n.Get().ToExtString() if lab.FindAttribute(TDataStd_Name.GetID_s(), n) else ''
rows=[]
def leaf(lab, loc, path):
    shp=st.GetShape_s(lab).Moved(loc)
    bb=Bnd_Box(); BRepBndLib.AddOptimal_s(shp, bb, True, False)
    if bb.IsVoid(): return
    xmin,ymin,zmin,xmax,ymax,zmax=bb.Get()
    gp=GProp_GProps(); BRepGProp.VolumeProperties_s(shp, gp); c=gp.CentreOfMass()
    ns=0; ex=TopExp_Explorer(shp, TopAbs_SOLID)
    while ex.More(): ns+=1; ex.Next()
    rows.append({'path':'/'.join(path),'name':path[-1],'bbox':[round(v,3) for v in (xmin,ymin,zmin,xmax,ymax,zmax)],'vol':round(gp.Mass(),2),'com':[round(c.X(),3),round(c.Y(),3),round(c.Z(),3)],'nsolid':ns})
def walk(lab, loc, path):
    if st.IsAssembly_s(lab):
        comps=TDF_LabelSequence(); st.GetComponents_s(lab, comps, False)
        for i in range(1, comps.Length()+1):
            c=comps.Value(i); ref=TDF_Label(); st.GetReferredShape_s(c, ref)
            cl=st.GetLocation_s(c)
            walk(ref, loc.Multiplied(cl), path+[name(c) or name(ref)])
    else:
        leaf(lab, loc, path)
free=TDF_LabelSequence(); st.GetFreeShapes(free)
for i in range(1, free.Length()+1):
    lab=free.Value(i); walk(lab, TopLoc_Location(), [name(lab)])
json.dump(rows, open('parts_bbox.json','w'), ensure_ascii=False, indent=0)
print('leaves', len(rows), 'time', round(time.time()-t0,1),'s')
