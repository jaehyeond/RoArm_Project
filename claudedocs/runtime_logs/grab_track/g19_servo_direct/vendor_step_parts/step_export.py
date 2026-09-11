"""벤더 STEP 에서 그리퍼 주변 핵심 부품을 루트 프레임 STL 로 내보낸다 (출처·sha 매니페스트 포함)."""
import json, hashlib, os
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TDocStd import TDocStd_Document
from OCP.TCollection import TCollection_ExtendedString
from OCP.XCAFDoc import XCAFDoc_DocumentTool
from OCP.TDF import TDF_LabelSequence, TDF_Label
from OCP.TDataStd import TDataStd_Name
from OCP.TopLoc import TopLoc_Location
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.StlAPI import StlAPI_Writer
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
from OCP.TopoDS import TopoDS_Compound
from OCP.BRep import BRep_Builder
ZIP='RoArm-M3_STEP_260310.zip'; OUT='/home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/runtime_logs/grab_track/g19_servo_direct/vendor_step_parts/'
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
want={'movable_jaw':'活动侧','fixed_jaw':'固定侧','gripper_base':'夹头-底座','wrist_b':'手腕b','wrist_a':'手腕a',
      'gripper_servo_drive_disc':'ST3215 v1 (4):1/金属舵盘（驱动）','gripper_servo_driven_disc':'ST3215 v1 (4):1/金属舵盘（从动）',
      'gripper_servo_gear_GE27':'ST3215 v1 (4):1/GE_27','gripper_servo_case_SG':'ST3215 v1 (4):1/SG-ZIJI_15','gripper_servo_case_ZK':'ST3215 v1 (4):1/ZK_122','gripper_servo_case_XG':'ST3215 v1 (4):1/XG-ZIJI_16'}
man={'source_zip':ZIP,'source_zip_sha256':hashlib.sha256(open(ZIP,'rb').read()).hexdigest(),'source_url':'https://files.waveshare.com/wiki/RoArm-M3/RoArm-M3_STEP_260310.zip','step_file':'RoArm-M3_STEP/RoArm-M3.step','frame':'STEP root frame, mm. X=arm axis(link5 +Z), Y=blade normal(link5 -X), Z=gripper hinge axis(link5 -Y). Gripper servo axis at x=289.002,y=-0.880.','parts':{}}
w=StlAPI_Writer(); w.ASCIIMode=False
for tag,key in want.items():
    ks=[k for k in shapes if key in k]
    if not ks: print('MISSING',tag,key); continue
    s=shapes[ks[0]]; BRepMesh_IncrementalMesh(s,0.05,False,0.2,True)
    fn=OUT+f'{tag}.stl'; w.Write(s,fn)
    b=Bnd_Box(); BRepBndLib.AddOptimal_s(s,b,True,False)
    man['parts'][tag]={'step_path':ks[0],'file':os.path.basename(fn),'sha256_16':hashlib.sha256(open(fn,'rb').read()).hexdigest()[:16],'bbox':[round(v,3) for v in b.Get()]}
    print(tag, os.path.getsize(fn), 'bytes')
# 나사 묶음: 가동 조 bbox(+4) 에 닿는 M3x4 (디스크 체결 9개) + PA2*5 (서보 케이스 4개)
mov=shapes[[k for k in shapes if '活动侧' in k][0]]; bm=Bnd_Box(); BRepBndLib.AddOptimal_s(mov,bm,True,False); B=bm.Get()
def inter(b,m=4.0): return all(b[i]-m<=B[i+3] and B[i]-m<=b[i+3] for i in range(3))
for tag,key in (('screws_M3x4_disc','M3x4'),('screws_PA2x5_servo_case','PA2*5')):
    comp=TopoDS_Compound(); bld=BRep_Builder(); bld.MakeCompound(comp); n=0; paths=[]
    for k,v in shapes.items():
        if key in k:
            b=Bnd_Box(); BRepBndLib.AddOptimal_s(v,b,True,False)
            if inter(b.Get()): bld.Add(comp,v); n+=1; paths.append(k)
    BRepMesh_IncrementalMesh(comp,0.05,False,0.2,True); fn=OUT+f'{tag}.stl'; w.Write(comp,fn)
    man['parts'][tag]={'step_paths':paths,'count':n,'file':os.path.basename(fn),'sha256_16':hashlib.sha256(open(fn,'rb').read()).hexdigest()[:16]}
    print(tag,n)
json.dump(man,open(OUT+'manifest.json','w'),ensure_ascii=False,indent=1)
