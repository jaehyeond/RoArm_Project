// Explicit, reviewable snapshot publication. Never merges, resets, deletes or force-pushes.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {execFileSync} from 'node:child_process';
const ROOT='/home/cgxr/Documents/Robotics/RoArm_Project';
const OUT=ROOT+'/.unlazy/research_closeout_20260914';
const case13='claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484';
const case12='claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w12_isaac_w10_w11_compare_20260912';
const case11='claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w11_dt_sensitivity_20260911';
const config={
 master:{cwd:ROOT,scopes:['claudedocs/research/labmeeting_20260915',case11,case12,case13,'claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_real/boot_check_20260911/scoop_tilt_cycle_01/closeout_01']},
 'pellet-model':{scopes:['claudedocs/research','claudedocs/runtime_logs/pellet_model','claudedocs/runtime_logs/repose_survey','claudedocs/runtime_logs/scoop_track/s3_cdfreq']},
 'research-survey':{scopes:['claudedocs/research']},
 'w12-input-audit':{scopes:[case12]},
 'w12-isaac-replay':{scopes:['claudedocs/research/labmeeting_20260915',case12]},
 'w13-cycle-audit':{scopes:[case13]},
 'w13-full-cycle':{scopes:[case13]},
};
for(const [name,c] of Object.entries(config)){c.cwd??='/home/cgxr/orca/workspaces/RoArm_Project/'+name;c.branch=name==='master'?'master':'jaehyeond/'+name;}
const git=(c,args,opts={})=>execFileSync('git',['-C',c.cwd,...args],{encoding:'utf8',maxBuffer:100e6,...opts});
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
const list=s=>s.split('\0').filter(Boolean);
const read=p=>JSON.parse(fs.readFileSync(p,'utf8'));
const put=(p,v)=>fs.writeFileSync(p,JSON.stringify(v,null,2)+'\n',{flag:'wx'});
const excluded=p=>/(__pycache__\/|\.pyc$|\.bak(?:_|$)|\.bak_pre_|\.env(?:\.|$)|\.pem$|\.key$)/.test(p);
const binary=p=>/\.(npz|npy|rrd|png|jpg|jpeg|mp4|rbl|usd|usda|usdc)$/i.test(p);
const artifact=p=>/\.(npz|npy|png|jpg|jpeg|mp4|csv|log|usd|usda|usdc)$/i.test(p);
function scanSecrets(p,b){
 if(binary(p)||b.includes(0))return [];
 const t=b.toString('utf8'),hits=[];
 const patterns={private_key:/-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----/,provider_token:/\b(?:sk-(?:proj-)?[A-Za-z0-9_-]{24,}|gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{25,})\b/,bearer:/\bBearer\s+[A-Za-z0-9_.-]{25,}/,credential_url:/https?:\/\/[^\s/:]+:[^\s/@]+@/};
 for(const [kind,re] of Object.entries(patterns))if(re.test(t))hits.push(kind);
 // Only report key/path, never credential values. Human review distinguishes fixtures from credentials.
 if(/(?:api[_-]?key|access[_-]?token|refresh[_-]?token|password|secret|dispatchCapability|capability[_-]?token|capability|token)["']?\s*[:=]\s*["'][^"'\s]{20,}["']/i.test(t))hits.push('credential_field_candidate');
 return hits;
}
const [mode,name,tag='01']=process.argv.slice(2);
if(mode==='scanner-selftest'){
 const fixtures=[['key','sk-'+'X'.repeat(30)],['field','{"dispatchCapability":"'+'X'.repeat(30)+'"}'],['bearer','Bearer '+'X'.repeat(30)],['url','https://example:fixture@example.invalid']];
 for(const [name,t] of fixtures)if(!scanSecrets(name,Buffer.from(t)).length)throw Error('Missed fixture '+name);
 if(scanSecrets('safe.json',Buffer.from('{"sha256":"'+'a'.repeat(64)+'","token":null}')).length)throw Error('False positive control');
 console.log('SCANNER_CONTROLS_OK');
}else if(mode==='inventory'){
 const result={created_utc:new Date().toISOString(),remote:git(config.master,['remote','get-url','origin']).trim(),worktrees:[],scope_note:'New/modified normal files plus scoped ignored evidence. Historical backups and caches excluded; no merges.'};
 if(result.remote!=='git@github.com:jaehyeond/RoArm_Project.git')throw Error('Unexpected remote');
 for(const [name,c] of Object.entries(config)){
  if(git(c,['branch','--show-current']).trim()!==c.branch)throw Error('Unexpected branch');
  if(git(c,['diff','--cached','--name-only']).trim())throw Error('Existing index changes: '+name);
  const changed=list(git(c,['diff','--name-only','-z']));
  const normal=list(git(c,['ls-files','--others','--exclude-standard','-z']));
  const ignored=list(git(c,['ls-files','--others','--ignored','--exclude-standard','-z','--',...c.scopes]));
  const candidates=[...new Set([...changed,...normal,...ignored.filter(artifact)])].sort();
  const w={name,...c,head:git(c,['rev-parse','HEAD']).trim(),files:[],excluded:[],secret_candidates:[]};
  for(const p of candidates){
   if(excluded(p)){w.excluded.push({path:p,reason:'backup/cache/sensitive filename'});continue;}
   const abs=c.cwd+'/'+p,s=fs.lstatSync(abs);if(!s.isFile())throw Error('Not regular file: '+abs);
   const b=fs.readFileSync(abs),hits=scanSecrets(p,b);
   if(hits.length)w.secret_candidates.push({path:p,kinds:hits});
   w.files.push({path:p,bytes:b.length,sha256:sha(b),ignored:ignored.includes(p),lfs:binary(p)||b.length>50*1024*1024});
  }
  w.bytes=w.files.reduce((s,f)=>s+f.bytes,0);w.lfs_bytes=w.files.filter(f=>f.lfs).reduce((s,f)=>s+f.bytes,0);
  result.worktrees.push(w);
 }
 put(OUT+'/inventory_'+tag+'.json',result);
 console.log(JSON.stringify(result.worktrees.map(w=>({name:w.name,files:w.files.length,bytes:w.bytes,lfs_bytes:w.lfs_bytes,excluded:w.excluded,secret_candidates:w.secret_candidates})),null,2));
}else if(mode==='attributes'){
 const inv=read(OUT+'/inventory_'+tag+'.json'),w=inv.worktrees.find(w=>w.name===name);if(!w)throw Error('Unknown worktree');
 const c=config[name],p=c.cwd+'/.gitattributes',old=fs.existsSync(p)?fs.readFileSync(p,'utf8'):'';
 const lines=['# 2026-09-14: explicit research evidence snapshot; preserve bytes and use LFS.'];
 for(const f of w.files)if(f.lfs)lines.push(f.path+' filter=lfs diff=lfs merge=lfs -text');
 if(lines.length===1){console.log('No new LFS attributes needed');process.exit(0);}
 if(old.includes(lines[0]))throw Error('Publication attribute block already exists');
 const patch=old?`*** Begin Patch\n*** Update File: ${p}\n@@\n ${old.trimEnd().split('\n').at(-1)}\n+\n+${lines.join('\n+')}\n*** End Patch\n`:`*** Begin Patch\n*** Add File: ${p}\n+${lines.join('\n+')}\n*** End Patch\n`;
 execFileSync('apply_patch',[],{input:patch,encoding:'utf8'});
 console.log('ATTRIBUTES_PREPARED '+name+' '+(lines.length-1));
}else if(mode==='stage'){
 const inv=read(OUT+'/inventory_'+tag+'.json'),w=inv.worktrees.find(w=>w.name===name);if(!w)throw Error('Unknown worktree');
 const c=config[name];if(git(c,['rev-parse','HEAD']).trim()!==w.head)throw Error('HEAD changed');
 for(const f of w.files)if(sha(fs.readFileSync(c.cwd+'/'+f.path))!==f.sha256)throw Error('Concurrent file change: '+f.path);
 const findings=read(OUT+'/secret_review_'+tag+'.json');
 if(findings.inventory_sha256!==sha(fs.readFileSync(OUT+'/inventory_'+tag+'.json'))||findings.publication_approved!==true)throw Error('Missing reviewed inventory approval');
 for(let i=0;i<w.files.length;i+=60)git(c,['add','-f','--',...w.files.slice(i,i+60).map(f=>f.path)]);
 console.log('STAGED '+name+' '+w.files.length);
}else if(mode==='audit-index'){
 const c=config[name];if(!c)throw Error('Unknown worktree');
 const inv=read(OUT+'/inventory_'+tag+'.json'),w=inv.worktrees.find(w=>w.name===name);
 const staged=list(git(c,['diff','--cached','--name-only','-z']));
 const expected=new Set(w.files.map(f=>f.path));
 const unexpected=staged.filter(p=>!expected.has(p));if(unexpected.length)throw Error('Unexpected staged paths '+unexpected.join(','));
 const report={name,head:git(c,['rev-parse','HEAD']).trim(),count:staged.length,lfs:0,max_git_blob_bytes:0,byte_mismatch:[],secret_candidates:[],syntax_errors:[],files:[]};
 for(const p of staged){
  const b=git(c,['show',':'+p],{encoding:'buffer'}),disk=fs.readFileSync(c.cwd+'/'+p);
  report.max_git_blob_bytes=Math.max(report.max_git_blob_bytes,b.length);
  if(b.length>=100*1024*1024)throw Error('Git blob too large '+p);
  const t=b.toString('utf8');let lfs=false;
  if(t.startsWith('version https://git-lfs.github.com/spec/v1\n')){lfs=true;report.lfs++;if(!t.includes('oid sha256:'+sha(disk)+'\n')||!t.includes('size '+disk.length+'\n'))report.byte_mismatch.push(p);}
  else if(sha(b)!==sha(disk))report.byte_mismatch.push(p);
  const hits=scanSecrets(p,disk);if(hits.length)report.secret_candidates.push({path:p,kinds:hits});
  if(p.endsWith('.py'))try{execFileSync('/home/cgxr/miniconda3/envs/roarm/bin/python',['-B','-c','import ast,sys; ast.parse(sys.stdin.read(), filename=sys.argv[1])',p],{input:disk,encoding:'utf8',stdio:['pipe','pipe','pipe']});}catch(e){report.syntax_errors.push({path:p,error:String(e.stderr).slice(-600)});}
  report.files.push({path:p,sha256:sha(disk),bytes:disk.length,lfs});
 }
 put(OUT+'/index_'+name+'_'+tag+'.json',report);
 console.log(JSON.stringify({...report,files:undefined},null,2));
 if(report.byte_mismatch.length||report.syntax_errors.length)process.exitCode=1;
}else throw Error('Use inventory | attributes NAME | stage NAME | audit-index NAME; optional snapshot tag');
