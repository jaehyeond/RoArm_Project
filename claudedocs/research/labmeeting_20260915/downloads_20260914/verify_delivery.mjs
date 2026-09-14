// Read-only verification. No simulator, renderer, network, or filesystem writes.
import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const dir = path.dirname(fileURLToPath(import.meta.url));
const root = '/home/cgxr/Documents/Robotics/RoArm_Project';
const names = [
  '20260915_입자물리_DEME_Isaac_PBD_설명_출력용.md',
  '20260915_랩미팅PPT_연구실PC_제작인계.md',
];
const manifest = JSON.parse(fs.readFileSync(path.join(dir, 'ASSET_MANIFEST.json'), 'utf8'));
const digest = data => createHash('sha256').update(data).digest('hex');
const texts = names.map(name => fs.readFileSync(path.join(dir, name), 'utf8'));
const [explanation, brief] = texts;
const mode = process.argv[2];
assert.ok(['assets', 'delivery'].includes(mode), 'choose assets or delivery');

function links(text) {
  return [...text.matchAll(/\]\((?:<([^>]+)>|([^\s)]+))\)/g)]
    .map(m => m[1] ?? m[2]).filter(p => p.startsWith('/'));
}
function verifyLinks(text) {
  for (const target of links(text)) {
    const p = target.replace(/:\d+$/, '');
    assert.ok(fs.statSync(p).isFile(), `missing local file: ${target}`);
  }
}
function verifyCards(text) {
  for (const a of manifest.assets) {
    const start = text.indexOf(`### ${a.id} —`);
    assert.ok(start >= 0, `missing asset card ${a.id}`);
    const next = text.indexOf('\n### ', start + 1);
    const card = text.slice(start, next < 0 ? text.length : next);
    for (const value of [a.source, a.destination, a.sha256, a.bytes.toLocaleString('en-US')]) {
      assert.ok(card.includes(value), `${a.id} card missing ${value}`);
    }
  }
}

assert.equal(digest(fs.readFileSync(manifest.source_ppt)), manifest.source_ppt_sha256);
assert.ok(explanation.includes('<!-- ORIGINAL_ANSWER_BEGIN -->'));
assert.ok(explanation.includes('<!-- ORIGINAL_ANSWER_END -->'));
assert.equal((explanation.match(/<!-- ORIGINAL_ANSWER_BEGIN -->/g) ?? []).length, 1);
const body = explanation.split('<!-- ORIGINAL_ANSWER_BEGIN -->\n')[1].split('\n<!-- ORIGINAL_ANSWER_END -->')[0];
assert.ok(body.startsWith('네. **“PBD에서 더미를 안정적으로 유지하기 어려워'));
assert.ok(body.endsWith('새 시뮬레이션이나 하이브리드 구현은 시작하지 않았습니다.'));

if (mode === 'delivery') {
  const documents = names.map(name => {
    const src = fs.readFileSync(path.join(dir, name));
    const destination = path.join('/home/cgxr/Downloads', name);
    const dst = fs.readFileSync(destination);
    assert.ok(src.equals(dst), `delivery differs: ${name}`);
    return { name, destination, bytes: dst.length, sha256: digest(dst) };
  });
  console.log(JSON.stringify({ marker: 'DOWNLOAD_DELIVERY_VERIFICATION_PASSED', documents,
    original_ppt_sha256: manifest.source_ppt_sha256, explanation_body_sha256: digest(body) }, null, 2));
} else {
  assert.equal(manifest.assets.length, 12);
  assert.equal(manifest.assets.filter(a => a.required).length, 6);
  assert.equal(new Set(manifest.assets.map(a => a.id)).size, manifest.assets.length);
  assert.equal(new Set(manifest.assets.map(a => a.destination)).size, manifest.assets.length);
  for (const a of manifest.assets) {
    const data = fs.readFileSync(a.source);
    assert.equal(data.length, a.bytes, `${a.id} byte count`);
    assert.equal(digest(data), a.sha256, `${a.id} hash`);
    if (a.probe) {
      const actual = JSON.parse(execFileSync('ffprobe', ['-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=codec_name,width,height,r_frame_rate,nb_frames:format=duration',
        '-of', 'json', a.source], { encoding: 'utf8', timeout: 10000 }));
      assert.deepEqual(actual.streams, a.probe.streams, `${a.id} stream metadata`);
      assert.equal(actual.format.duration, a.probe.format.duration, `${a.id} video duration`);
    }
  }
  assert.equal(manifest.assets.filter(a => a.required).reduce((sum, a) => sum + a.bytes, 0), manifest.required_media_bytes);
  assert.equal(manifest.assets.reduce((sum, a) => sum + a.bytes, 0), manifest.all_media_bytes);
  assert.ok(brief.includes(manifest.required_media_bytes.toLocaleString('en-US')));
  assert.ok(brief.includes(manifest.all_media_bytes.toLocaleString('en-US')));
  texts.forEach(verifyLinks);
  verifyCards(brief);
  assert.throws(() => verifyCards(brief.replace(manifest.assets[0].sha256, '0'.repeat(64))), /card missing/);
  assert.throws(() => verifyLinks('[negative](/tmp/nonexistent_labmeeting_link_20260914_control_xyz.md)'), /ENOENT/);
  assert.equal((brief.match(/^### 새 [1-8]장 —/gm) ?? []).length, 8);
  for (let i = 1; i <= 6; i++) assert.ok(brief.includes(`| 기존 ${i}장 |`));
  assert.ok(!brief.includes('<!-- ASSET_CARDS -->'));
  const numerical = JSON.parse(execFileSync('/home/cgxr/miniconda3/envs/roarm/bin/python',
    [path.join(root, 'claudedocs/research/labmeeting_20260915/coordinator/verify_basics.py')],
    { encoding: 'utf8', timeout: 60000, maxBuffer: 1024 * 1024 }));
  assert.equal(numerical.ppt_slides, 6);
  for (const name of ['W10', 'W11']) {
    const row = numerical.cells[name];
    assert.ok(brief.includes(`${row.capture_mask_count}알`));
    assert.ok(brief.includes(`${row.capture_mass_recomputed_g.toFixed(4)} g`));
  }
  console.log(JSON.stringify({ marker: 'ASSET_AND_LINK_VERIFICATION_PASSED', assets: manifest.assets.length,
    required: manifest.assets.filter(a => a.required).length, videos_reprobed: manifest.assets.filter(a => a.probe).length,
    local_links: texts.reduce((n, t) => n + links(t).length, 0), original_ppt_slides: numerical.ppt_slides,
    target_body_slides: 8, mutated_hash_rejected: true, missing_link_rejected: true,
    required_media_bytes: manifest.required_media_bytes, all_media_bytes: manifest.all_media_bytes,
    numerical_crosscheck: numerical, explanation_body_sha256: digest(body) }, null, 2));
}
