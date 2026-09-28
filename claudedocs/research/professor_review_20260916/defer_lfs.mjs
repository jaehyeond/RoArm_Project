// Bounded, local-only deferral of the exact previously approved publication payload.
// Does not delete working files, rewrite history, commit, push, or modify LFS objects.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {execFileSync, spawnSync} from 'node:child_process';
const ROOT='/home/cgxr/Documents/Robotics/RoArm_Project';
const OUT=path.join(ROOT,'claudedocs/research/professor_review_20260916');
const source=path.join(ROOT,'claudedocs/research/closeout_20260914/PUBLICATION_MANIFEST.json');
const manifest=JSON.parse(fs.readFileSync(source));
const MARK='# ---- 2026-09-16 LOCAL-ONLY deferred LFS publication ----';
const mode=process.argv[2];
if(!['prepare','apply','verify'].includes(mode)) throw Error('prepare | apply | verify');
function git(cwd,args,input){return execFileSync('git',['-C',cwd,...args],{encoding:'utf8',input,maxBuffer:32*1024*1024});}
async function sha(p){const h=crypto.createHash('sha256');for await(const b of fs.createReadStream(p))h.update(b);return h.digest('hex');}
const ws=manifest.worktrees.filter(w=>w.files.some(f=>f.lfs));
const rows=[];const uniques=new Map();
for(const w of ws){
  if(![ROOT, ...['pellet-model','w12-isaac-replay','w13-cycle-audit','w13-full-cycle'].map(x=>'/home/cgxr/orca/workspaces/RoArm_Project/'+x)].includes(w.cwd))throw Error('unexpected worktree');
  if(git(w.cwd,['branch','--show-current']).trim()!==w.branch)throw Error('branch changed '+w.name);
  const tracked=new Set(git(w.cwd,['ls-files','-z']).split('\0').filter(Boolean));
  const selected=w.files.filter(f=>f.lfs);
  for(const f of selected){
    if(path.isAbsolute(f.path)||f.path.split('/').includes('..')||/[\n\r\t*?\[\]\\]/.test(f.path))throw Error('unsafe literal path '+f.path);
    const p=path.join(w.cwd,f.path);
    if(fs.lstatSync(p).isSymbolicLink()||fs.statSync(p).size!==f.bytes||await sha(p)!==f.sha256)throw Error('original changed '+p);
    if((mode==='prepare'||mode==='apply')&&!tracked.has(f.path))throw Error('not tracked '+p);
    if(mode==='verify'&&tracked.has(f.path))throw Error('still tracked '+p);
    uniques.set(f.sha256,f.bytes);
  }
  if(mode==='verify'){
    const ignored=new Set(git(w.cwd,['check-ignore','-z','--stdin'],selected.map(f=>f.path).join('\0')+'\0').split('\0').filter(Boolean));
    if(selected.some(f=>!ignored.has(f.path)))throw Error('not ignored '+w.name);
    const deleted=git(w.cwd,['diff','--cached','--name-only','--diff-filter=D','-z']).split('\0').filter(Boolean).sort();
    if(JSON.stringify(deleted)!==JSON.stringify(selected.map(f=>f.path).sort()))throw Error('unexpected index removal '+w.name);
  }else if(git(w.cwd,['diff','--cached','--name-only']).trim())throw Error('index not clean '+w.name);
  rows.push({name:w.name,cwd:w.cwd,branch:w.branch,head:git(w.cwd,['rev-parse','HEAD']).trim(),files:selected,
    gitignore_before:mode==='verify'?null:fs.readFileSync(path.join(w.cwd,'.gitignore'),'utf8')});
}
const data={artifact:'LFS_DEFERRED_20260916',manifest:source,manifest_sha256:await sha(source),path_count:rows.reduce((s,w)=>s+w.files.length,0),unique_objects:uniques.size,unique_bytes:[...uniques.values()].reduce((a,b)=>a+b,0),worktrees:rows};
if(data.path_count!==1016||data.unique_objects!==982||data.unique_bytes!==3316053800)throw Error('payload differs from reviewed total');
if(mode==='prepare'){
  if(fs.existsSync(path.join(OUT,'LFS_BEFORE.json')))throw Error('before snapshot already exists');
  fs.writeFileSync(path.join(OUT,'LFS_BEFORE.json'),JSON.stringify(data,null,2)+'\n');
  let md='# LFS 나중 게시 명단 — 2026-09-16\n\n';
  md+='이 명단은 9/14 PUBLICATION_MANIFEST.json의 lfs:true 항목만 보류한다. 코드·JSON·설명 문서를 포괄 제외하지 않는다.\n\n';
  md+='## 현재 정책\n\n- 원본 파일을 이동·삭제하지 않는다. 아래 파일은 현재 index에서만 제외하고 각 worktree .gitignore 끝에 정확한 경로를 추가한다.\n- 1,016개 경로 / SHA256 중복 제거 982개 객체 / 3,316,053,800 bytes (3.0883 GiB). 작업복사본들의 중복 용량을 LFS 원격 용량에 더하지 않는다.\n- 과거 커밋과 .gitattributes의 LFS 규칙은 보존한다. **과거 커밋에는 포인터가 남으므로 ignore만으로 과거 이력을 push할 때의 LFS 업로드를 막을 수 없다. push 보류.**\n- 이 명단은 로컬 보관 목록이지 원격 백업 완료 증거가 아니다. LFS 계정 잔여 용량·과거 일부 업로드 여부는 미확인이다.\n- 기존 git_publication.mjs의 stage/push 모드를 재사용하지 않는다. force-add가 보류를 되돌릴 수 있다.\n- 임의 git clean, worktree 삭제, git lfs prune, reset/rebase/history rewrite 금지. 실제 파일이 있는 worktree를 보존한다.\n\n';
  md+='## 나중에 LFS를 할 때\n\n1. 이 문서 + LFS_BEFORE.json + LFS_AFTER.json을 읽고 worktree/branch/경로/SHA256으로 원본 존재를 대조한다. 다른 PC에서는 절대경로 prefix만 다를 수 있다.\n2. 계정 실제 잔여 storage/bandwidth와 공개 저장소의 자료 공개 범위를 확인하고 사용자에게 게시 승인을 받는다.\n3. 원자료 NPZ/필수 RRD·RBL/검증 PNG/최종 MP4를 우선 검토하고 중간 프레임까지 모두 필요한지 재선정한다. 본 명단은 삭제 명단이 아니다.\n4. 기존 unpublished 커밋의 LFS 포인터를 포함해 게시할지, 별도 승인받아 비-LFS 게시 이력을 준비할지 먼저 결정한다. 이번 작업은 과거 커밋을 수정하지 않았다.\n5. 승인한 파일만 ignore 예외를 되돌리고 .gitattributes 적용을 검사한 뒤 명시적으로 stage한다. staged deletion은 파일 삭제가 아니라 이번 추적 제외임을 확인한다.\n6. 게시 이후 원격 SHA와 새 clone/LFS 수신을 별도로 검증한다. 로컬 커밋만으로 백업 성공이라고 하지 않는다.\n\n';
  md+='## 전체 명단\n\n각 경로는 해당 worktree 기준이며 SHA256은 실제 파일 본문(=LFS OID)이다.\n\n';
  for(const w of rows){md+=`### ${w.name}\n\n- Worktree: \`${w.cwd}\`\n- Branch: \`${w.branch}\`\n- 보류 전 HEAD: \`${w.head}\`\n- 파일: ${w.files.length}개\n\n| 파일 (repo 상대경로) | bytes | SHA256 |\n|---|---:|---|\n`;for(const f of w.files)md+=`| ${f.path} | ${f.bytes} | ${f.sha256} |\n`;md+='\n';}
  fs.writeFileSync(path.join(ROOT,'claudedocs/LFS_DEFERRED_20260916.md'),md);
  console.log('PREPARED',data.path_count,data.unique_objects,data.unique_bytes);
}
if(mode==='apply'){
  const before=JSON.parse(fs.readFileSync(path.join(OUT,'LFS_BEFORE.json')));
  for(const w of rows){const b=before.worktrees.find(x=>x.name===w.name);if(b.head!==w.head||b.gitignore_before!==w.gitignore_before)throw Error('preflight changed '+w.name);}
  // Exact last-line context; apply_patch is the only text editor used here.
  for(const w of rows){
    if(w.gitignore_before.includes(MARK))throw Error('already patched');
    const lines=w.gitignore_before.trimEnd().split('\n');const tail=lines.slice(-3).join('\n');
    const block=[MARK,'# Local originals retained; history still contains LFS pointers. NO PUSH.', '# Manifest: main claudedocs/LFS_DEFERRED_20260916.md',...w.files.map(f=>'/'+f.path)].join('\n');
    const patch='*** Begin Patch\n*** Update File: '+path.join(w.cwd,'.gitignore')+'\n@@\n'+tail.split('\n').map(l=>' '+l).join('\n')+'\n+\n'+block.split('\n').map(l=>'+'+l).join('\n')+'\n*** End Patch\n';
    const r=spawnSync('apply_patch',[patch],{encoding:'utf8'});if(r.status!==0)throw Error('apply_patch failed '+r.stdout+r.stderr);
    for(let i=0;i<w.files.length;i+=40)git(w.cwd,['--literal-pathspecs','rm','--cached','--',...w.files.slice(i,i+40).map(f=>f.path)]);
    console.log('INDEX_ONLY_REMOVED',w.name,w.files.length);
  }
}
if(mode==='verify'){
  const before=JSON.parse(fs.readFileSync(path.join(OUT,'LFS_BEFORE.json')));
  for(const w of rows)if(before.worktrees.find(b=>b.name===w.name).head!==w.head)throw Error('HEAD changed');
  const report={artifact:'LFS_DEFERRED_VERIFICATION_20260916',path_count:data.path_count,unique_objects:data.unique_objects,unique_bytes:data.unique_bytes,all_original_hashes_match:true,all_selected_untracked:true,all_selected_ignored:true,only_expected_index_deletions:true,heads_unchanged:true,history_still_has_lfs:true,push_performed:false,checked_utc:new Date().toISOString(),worktrees:rows.map(w=>({name:w.name,cwd:w.cwd,branch:w.branch,head:w.head,paths:w.files.length}))};
  fs.writeFileSync(path.join(OUT,'LFS_AFTER.json'),JSON.stringify(report,null,2)+'\n');
  console.log('DEFERRED_LFS_LOCAL_PRESERVATION_OK',data.path_count);
}
