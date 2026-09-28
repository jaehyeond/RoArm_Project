// Read-only checks except for refreshing generated verification receipts in this report folder.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {execFileSync} from 'node:child_process';
const ROOT='/home/cgxr/Documents/Robotics/RoArm_Project';
const OUT=path.join(ROOT,'claudedocs/research/professor_review_20260916');
const mode=process.argv[2];
const read=p=>JSON.parse(fs.readFileSync(p));
const assert=(b,m)=>{if(!b)throw Error(m);};
function run(bin,args){return execFileSync(bin,args,{cwd:ROOT,encoding:'utf8',maxBuffer:8*1024*1024,timeout:55000});}
if(mode==='physics'){
 process.stdout.write(run('/home/cgxr/miniconda3/envs/roarm/bin/python',[path.join(OUT,'audit_physics.py')]));
 const a=read(path.join(OUT,'PARAMETER_AUDIT.json')),s=read(path.join(OUT,'OFFICIAL_SOURCE_CROSSCHECK.json'));
 assert(s.rows.length===8&&s.rows.every(x=>x.byte_identical),'official comparison');
 for(const r of s.rows){const h=crypto.createHash('sha256').update(fs.readFileSync(r.local)).digest('hex');assert(h===r.local_sha256&&h===r.official_sha256,'source drift');}
 assert(a.recomputed.n_independent_pellets===20000&&a.recomputed.contact_spheres===140000,'clump count');
 assert(a.render_prototype_reconstruction.vertices===294&&a.render_prototype_reconstruction.triangles===560,'visual prototype');
 assert(a.render.w13.instancer.n_instances===20000&&a.render.w13.instancer.n_prototypes===6,'w13 instancer');
 assert(a.render.w10.frame_contract.n_clumps===20000,'w12 instancer total');
 assert(a.historic_10us.engine_error_reported_speed_m_s===22291.97&&!a.historic_10us.complete_result_json_present,'10us history');
 assert(a.source_derived_sync_durations.at(-1).source_derived_actual_duration_s>1e-3,'1ms vs sync');
 assert(a.render.w10.camera_intrinsics_sensor.side[0][0]===1954.6650390625,'pixel intrinsics');
 console.log('PHYSICS_PROVENANCE_OK (source/array/render independent assertions)');
}else if(mode==='git'){
 process.stdout.write(run(process.execPath,[path.join(OUT,'defer_lfs.mjs'),'verify']));
 const m=read(path.join(OUT,'LFS_BEFORE.json'));const md=fs.readFileSync(path.join(ROOT,'claudedocs/LFS_DEFERRED_20260916.md'),'utf8');
 for(const w of m.worktrees)for(const f of w.files)assert(md.includes(`| ${f.path} | ${f.bytes} | ${f.sha256} |`),'missing manifest row');
 console.log('DEFERRED_LFS_LOCAL_PRESERVATION_OK (all Markdown rows)');
}else if(mode==='docs'){
 const files=['claudedocs/research/professor_review_20260916/REPORT.md','claudedocs/research/professor_review_20260916/PARAMETERS_ALL.md','claudedocs/LFS_DEFERRED_20260916.md','claudedocs/CONTINUE_20260916_PHYSICS_AUDIT_DT.md','claudedocs/session_20260916_professor_physics_lfs_defer.md'];
 let links=0;for(const rel of files){const p=path.join(ROOT,rel),txt=fs.readFileSync(p,'utf8');assert(txt.length>100,rel);
   for(const m of txt.matchAll(/\[[^\]\n]+\]\(([^)\n]+)\)/g)){
     if(/^(https?:|#)/.test(m[1]))continue;
     const dest=m[1].replace(/:\d+$/,'').split('#')[0];
     assert(fs.existsSync(path.resolve(path.dirname(p),dest)),'broken local link '+rel+': '+m[1]);links++;
   }
 }
 const dash=fs.readFileSync(path.join(ROOT,'START_HERE.md'),'utf8'),relay=fs.readFileSync(path.join(ROOT,'claudedocs/relay/from_codex.md'),'utf8');
 assert(dash.includes('2026-09-16')&&dash.includes('LFS_DEFERRED_20260916.md'),'dashboard');
 assert(relay.includes('§2 2026-09-16')&&relay.includes('force-add'),'relay updated');
 const report=fs.readFileSync(path.join(OUT,'REPORT.md'),'utf8');
 for(const term of ['22,291.97','EXTENDED_TAYLOR','5.229911','1954.6650390625','원래 첫 작업','과거 커밋','torque-only','실물 보정'])assert(report.includes(term),'missing distinction '+term);
 const status={artifact:'PROFESSOR_REVIEW_HANDOVER_CHECK',local_links_checked:links,files,scoped_documents_present:true,original_next_step_preserved:true,lfs_history_warning_present:true};
 fs.writeFileSync(path.join(OUT,'DOCS_VERIFICATION.json'),JSON.stringify(status,null,2)+'\n');
 console.log('REVIEW_HANDOVER_OK',links,'local links');
}else throw Error('physics | git | docs');
