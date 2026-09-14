// Emit a new audit artifact from reviewed snapshots; never edits evidence or Git refs.
import fs from 'node:fs';
import crypto from 'node:crypto';
import {execFileSync} from 'node:child_process';
const root='/home/cgxr/Documents/Robotics/RoArm_Project';
const local=root+'/.unlazy/research_closeout_20260914';
const out=root+'/claudedocs/research/closeout_20260914';
const read=p=>JSON.parse(fs.readFileSync(p));
const git=(cwd,args)=>execFileSync('git',['-C',cwd,...args],{encoding:'utf8'}).trim();
const inv=read(local+'/inventory_03.json');
const main=read(local+'/inventory_04.json').worktrees.find(w=>w.name==='master');
const result={artifact:'RESEARCH_PUBLICATION_MANIFEST_20260914',created_utc:new Date().toISOString(),remote:inv.remote,note:'Evidence snapshot. This manifest does not hash itself. Closing state/relay/report may be updated in a subsequent documentation commit; their final authority is Git HEAD and the remote ref. Scientific evidence files must remain unchanged.',worktrees:[],excluded:[]};
for(const w of inv.worktrees){
 const r=w.name==='master'?main:read(local+'/index_'+w.name+'_03.json');
 result.worktrees.push({name:w.name,cwd:w.cwd,branch:w.branch,commit:w.name==='master'?null:git(w.cwd,['rev-parse','HEAD']),snapshot_parent:w.head,files:r.files.map(f=>({path:f.path,bytes:f.bytes,sha256:f.sha256,lfs:f.lfs})),index_checks:w.name==='master'?{status:'audited after manifest export'}:{files:r.count,lfs:r.lfs,max_blob_bytes:r.max_git_blob_bytes,byte_mismatch:r.byte_mismatch.length,syntax_errors:r.syntax_errors.length}});
 result.excluded.push(...w.excluded.map(f=>({worktree:w.name,...f})));
}
const objects=new Map();for(const w of result.worktrees)for(const f of w.files)if(f.lfs)objects.set(f.sha256,f.bytes);
result.unique_lfs_objects=objects.size;result.unique_lfs_bytes=[...objects.values()].reduce((a,b)=>a+b,0);
const bytes=JSON.stringify(result,null,2)+'\n';fs.writeFileSync(out+'/PUBLICATION_MANIFEST.json',bytes,{flag:'wx'});
console.log(JSON.stringify({worktrees:result.worktrees.map(w=>({name:w.name,commit:w.commit,files:w.files.length})),unique_lfs_objects:objects.size,unique_lfs_bytes:result.unique_lfs_bytes,manifest_sha256:crypto.createHash('sha256').update(bytes).digest('hex')}));
