import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {execFileSync} from 'node:child_process';
const main='/home/cgxr/Documents/Robotics/RoArm_Project';
const rel='claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484';
const here=path.join(main,rel,'resume_20260913/coordinator');
const manifest=path.join(here,'preservation_baseline.json');
const old=path.join(main,rel,'coordinator/input_baseline.json');
function digest(p){
  const st=fs.lstatSync(p);
  if(st.isSymbolicLink()) return {symlink:fs.readlinkSync(p)};
  if(!st.isFile()) throw Error('Not file: '+p);
  const fd=fs.openSync(p,'r'),h=crypto.createHash('sha256'),b=Buffer.alloc(1048576);
  let size=0,n;
  try{while((n=fs.readSync(fd,b,0,b.length,null))>0){h.update(b.subarray(0,n));size+=n;}}
  finally{fs.closeSync(fd);}
  return {sha256:h.digest('hex'),bytes:size};
}
const head=()=>execFileSync('git',['rev-parse','HEAD'],{cwd:main,encoding:'utf8'}).trim();
function walk(root,files){
  for(const de of fs.readdirSync(root,{withFileTypes:true})){
    if(de.name==='resume_20260913')continue;
    const p=path.join(root,de.name);
    if(de.isDirectory())walk(p,files);else files[p]=digest(p);
  }
}
function equal(a,b){return JSON.stringify(a)===JSON.stringify(b);}
if(process.argv[2]==='init'){
  if(fs.existsSync(manifest))throw Error('Refuse existing baseline');
  const prior=JSON.parse(fs.readFileSync(old,'utf8')),files={};
  for(const [p,want] of Object.entries(prior.files)){
    const got=digest(p);if(!equal(got,want))throw Error('Prior evidence changed: '+p);
    files[p]=got;
  }
  if(head()!==prior.head)throw Error('Prior HEAD changed');
  for(const root of [main,'/home/cgxr/orca/workspaces/RoArm_Project/w13-full-cycle','/home/cgxr/orca/workspaces/RoArm_Project/w13-cycle-audit'])walk(path.join(root,rel),files);
  fs.writeFileSync(manifest,JSON.stringify({head:head(),created_utc:new Date().toISOString(),files},null,2)+'\n',{flag:'wx'});
  console.log('W13R_BASELINE_CREATED',Object.keys(files).length);
}else if(process.argv[2]==='check'){
  const b=JSON.parse(fs.readFileSync(manifest,'utf8'));
  if(head()!==b.head)throw Error('Main HEAD changed; distinguish user commit from unauthorized mutation');
  for(const [p,want]of Object.entries(b.files))if(!equal(digest(p),want))throw Error('Evidence changed: '+p);
  console.log('W13R_PRESERVATION_VERIFIED',Object.keys(b.files).length);
}else throw Error('Use init or check');
