import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {execFileSync} from 'node:child_process';
const main='/home/cgxr/Documents/Robotics/RoArm_Project';
const here=path.join(main,'claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/coordinator');
const previous=path.join(main,'claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w12_isaac_w10_w11_compare_20260912/coordinator/input_baseline.json');
const dest=path.join(here,'input_baseline.json');
function digest(p){const b=fs.readFileSync(p);return {sha256:crypto.createHash('sha256').update(b).digest('hex'),bytes:b.length};}
const head=()=>execFileSync('git',['rev-parse','HEAD'],{cwd:main,encoding:'utf8'}).trim();
if(process.argv[2]==='init'){
  const old=JSON.parse(fs.readFileSync(previous,'utf8'));
  const files={};
  for(const [p,want] of Object.entries(old.files)){
    const got=digest(p);if(got.sha256!==want.sha256||got.bytes!==want.bytes)throw Error('Old input mismatch: '+p);
    files[p]=got;
  }
  for(const p of ['sim_deme_scoop_s1.py','sim_isaac_render_deme_scoop.py','AGENTS.md','claudedocs/DECISIONS.md','claudedocs/DECISIONS_ACTIVE.md'])files[path.join(main,p)]=digest(path.join(main,p));
  fs.writeFileSync(dest,JSON.stringify({head:head(),files},null,2)+'\n',{flag:'wx'});
  console.log('W13_BASELINE_CREATED',Object.keys(files).length);
}else if(process.argv[2]==='check'){
  const b=JSON.parse(fs.readFileSync(dest,'utf8'));
  if(head()!==b.head)throw Error('Main HEAD changed');
  for(const [p,want] of Object.entries(b.files)){const got=digest(p);if(got.sha256!==want.sha256||got.bytes!==want.bytes)throw Error('Input changed: '+p);}
  console.log('W13_PRESERVATION_OK',Object.keys(b.files).length);
}else throw Error('Use init or check');
