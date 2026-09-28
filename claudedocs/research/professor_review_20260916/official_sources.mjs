import fs from 'node:fs';
import crypto from 'node:crypto';
const base='/home/cgxr/miniconda3/envs/roarm/lib/python3.11/site-packages/';
const commit='12f13cb15805d891eddc7b5b545d1f6823f523d7';
const maps=[['include/DEM/API.h','src/DEM/API.h'],['include/DEM/Models.h','src/DEM/Models.h'],
 ...['DEMHelperKernels.cuh','DEMCollectForceKernels_Compact.cu','DEMIntegrationKernels.cu','DEMCalcForceKernels.cu',
 'DEMCustomizablePolicies/FullHertzianForceModel.cu','DEMCustomizablePolicies/IntegrationVelPassOnExtendedTaylor.cu'].map(p=>['share/DEME/kernel/'+p,'src/kernel/'+p])];
const hash=b=>crypto.createHash('sha256').update(b).digest('hex');
const rows=await Promise.all(maps.map(async([local,remote])=>{
 const url=`https://raw.githubusercontent.com/projectchrono/DEM-Engine/${commit}/${remote}`;
 const r=await fetch(url,{signal:AbortSignal.timeout(30000)});if(!r.ok)throw Error(url+' '+r.status);
 const bytes=Buffer.from(await r.arrayBuffer());const a=hash(fs.readFileSync(base+local)),b=hash(bytes);
 return {local:base+local,url,local_sha256:a,official_sha256:b,byte_identical:a===b};
}));
fs.writeFileSync('claudedocs/research/professor_review_20260916/OFFICIAL_SOURCE_CROSSCHECK.json',JSON.stringify({commit,installed_distribution:'deme 2.4.0',not_a_release_tag:true,rows},null,2)+'\n');
if(rows.some(r=>!r.byte_identical))throw Error('official/installed mismatch');
console.log('OFFICIAL_SOURCE_IDENTICAL',rows.length);
