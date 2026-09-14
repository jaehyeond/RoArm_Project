// Read-only acceptance checks. Does not invoke physics, renderers, hardware or training.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
const root='/home/cgxr/Documents/Robotics/RoArm_Project';
const dir=root+'/claudedocs/research/closeout_20260914';
const name='20260915_W13결과_실행시간_최적화와학습전략_출력용.md';
const cont=root+'/claudedocs/CONTINUE_20260914_W13_REPAIR_PERFORMANCE.md';
const prod='/home/cgxr/orca/workspaces/RoArm_Project/w13-full-cycle/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/implementation';
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
const pinned=[
 [prod+'/run_01/w13_cycle_seed460.npz','529f422e962730b17b520c3c9af2261aae7c431ec5e078cd2c368dfab0c46b0f'],
 [prod+'/run_01/w13_cycle_seed460.json','e482b939a4cf42e5284b9b15e36a9f3613e989161ec538475e2c05f4260d9340'],
 [prod+'/partial_post_03/isaac/w13_full_cycle.mp4','1471127134fc06a88d546c891de0d2515f4911fcd1832f0ae841258ac1a2bc34'],
 ['/home/cgxr/Downloads/랩미팅 9월15일_초안.pptx','59d08134b7fac0617b2b3d5495bc9439ac7d54b7e74ab65a337a0c7f3f8b61c9'],
];
function links(p){
 let n=0;const s=fs.readFileSync(p,'utf8');
 for(const m of s.matchAll(/\]\(([^)]+)\)/g)){
  let target=m[1];if(/^https?:/.test(target))continue;
  target=target.replace(/^<|>$/g,'').replace(/:\d+$/,'').split('#')[0];
  const abs=path.resolve(path.dirname(p),target);assert(fs.existsSync(abs),'Missing link '+abs);n++;
 }
 return n;
}
const mode=process.argv[2];
if(mode==='delivery'){
 const a=fs.readFileSync(dir+'/'+name),b=fs.readFileSync('/home/cgxr/Downloads/'+name);
 assert(a.equals(b));assert(a.length>15000);console.log(JSON.stringify({bytes:a.length,sha256:sha(a),local_links:links(dir+'/'+name)}));
 assert.equal(sha(fs.readFileSync(pinned[3][0])),pinned[3][1]);
 console.log('DELIVERY_OK');
}else if(mode==='continuation'){
 const s=fs.readFileSync(cont,'utf8');assert.equal((s.match(/```/g)||[]).length,2);
 for(const id of ['원자료 FAIL2','재생 FAIL3','CPU','새 명시 요청 전','동결 rev28','기존버그 FAIL→수정본 PASS'])assert(s.includes(id),'Missing boundary '+id);
 for(const [p,h] of pinned){assert.equal(sha(fs.readFileSync(p)),h);if(p!==pinned[3][0])assert(s.includes(h));}
 console.log(JSON.stringify({local_links:links(cont),preserved_files:pinned.length}));
 console.log('CONTINUATION_OK');
}else if(mode==='numbers'){
 const rec=JSON.parse(fs.readFileSync(prod+'/run_01/EXECUTION_RECEIPT.json'));const step=rec.steps[0];
 assert.equal(step.stage_total_s_including_cleanup,31218.753855);assert.equal(step.timed_out,true);assert.equal(step.exit_class_exit_code,124);
 const code=`import json,sys,numpy as np\np=sys.argv[1]\nz=np.load(p,allow_pickle=False)\nt=z['sync_t_s']; pt=z['particle_frame_t_s']; idx=z['particle_frame_sync_index']; wall=z['sync_wall_elapsed_s']\nassert len(t)==16304 and len(pt)==283\nassert abs(float(t[-1])-24.486802938176766)<1e-12\nassert np.max(np.abs(pt-t[idx]))==0\nassert np.all(np.diff(t)>0)\nassert int(np.sum(z['sync_derived_internal_steps']))==24486803\nprint(json.dumps(dict(sync_rows=len(t),particle_frames=len(pt),sim_s=float(t[-1]),derived_internal_steps=int(np.sum(z['sync_derived_internal_steps'])),max_adjacent_wall_gap_s=float(np.max(np.diff(wall))),wall_to_sim=31218.753855/float(t[-1]),repeat_10_days=31218.753855*10/86400,repeat_40_days=31218.753855*40/86400,repeat_100_days=31218.753855*100/86400)))`;
 const r=execFileSync('/home/cgxr/miniconda3/envs/roarm/bin/python',['-B','-c',code,prod+'/run_01/w13_cycle_seed460.npz'],{encoding:'utf8'});console.log(r.trim());console.log('NUMBERS_OK');
}else if(mode==='publication'){
 const report=JSON.parse(fs.readFileSync(dir+'/PUBLICATION_MANIFEST.json'));
 const remote=execFileSync('git',['-C',root,'ls-remote','--heads','origin'],{encoding:'utf8'});
 for(const w of report.worktrees){
  const head=execFileSync('git',['-C',w.cwd,'rev-parse','HEAD'],{encoding:'utf8'}).trim();
  assert(remote.includes(head+'\trefs/heads/'+w.branch+'\n'),'Remote mismatch '+w.branch);
  if(w.name!=='master')assert.equal(head,w.commit);
  const changed=execFileSync('git',['-C',w.cwd,'diff','--name-only','HEAD'],{encoding:'utf8'}).trim();assert.equal(changed,'','Tracked changes '+w.name);
  for(const f of w.files){if(w.name==='master'&&['START_HERE.md','claudedocs/relay/from_codex.md','claudedocs/session_20260914_research_closeout_git.md','claudedocs/research/closeout_20260914/GIT_PUBLICATION.md'].includes(f.path))continue;assert.equal(sha(fs.readFileSync(w.cwd+'/'+f.path)),f.sha256,'Changed evidence '+f.path);}
  console.log(w.branch+' '+head+' REMOTE_MATCH');
 }
 for(const [p,h] of pinned)assert.equal(sha(fs.readFileSync(p)),h);
 console.log('PUBLICATION_OK');
}else throw Error('Expected delivery | continuation | numbers | publication');
