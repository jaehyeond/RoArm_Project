// Read-only pre/post replay preservation check. Never modifies raw files or reports.
// Usage: node verify_partial_raw_preservation.mjs MANIFEST.json EXPECTED_MANIFEST_SHA256
import fs from 'node:fs';
import crypto from 'node:crypto';
import path from 'node:path';

const ROOT = '/home/cgxr/orca/workspaces/RoArm_Project/w13-full-cycle/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/implementation/run_01';
const [manifestPath, expectedManifestSha, ...extra] = process.argv.slice(2);
const report = {artifact: 'W13R_PARTIAL_RAW_PRESERVATION_CHECK_V1', checked_utc: new Date().toISOString(),
  root: ROOT, manifest_path: manifestPath, manifest_sha256: null, expected_manifest_sha256: expectedManifestSha,
  files_checked: 0, issues: [], success: false};

async function shaFile(p) {
  const hash = crypto.createHash('sha256');
  for await (const chunk of fs.createReadStream(p)) hash.update(chunk);
  return hash.digest('hex');
}

function inventory(dir, base = '') {
  const files = [], dirs = [];
  for (const e of fs.readdirSync(dir, {withFileTypes: true})) {
    const rel = base ? `${base}/${e.name}` : e.name;
    if (e.isSymbolicLink()) throw new Error(`Unexpected symlink: ${rel}`);
    if (e.isDirectory()) {
      dirs.push(rel);
      const sub = inventory(path.join(dir, e.name), rel);
      files.push(...sub.files); dirs.push(...sub.dirs);
    } else if (e.isFile()) files.push(rel);
    else throw new Error(`Unexpected non-regular entry: ${rel}`);
  }
  return {files: files.sort(), dirs: dirs.sort()};
}

try {
  if (!manifestPath || !/^[a-f0-9]{64}$/.test(expectedManifestSha ?? '') || extra.length) {
    throw new Error('Expected exactly manifest path and its independently pinned SHA256');
  }
  report.manifest_sha256 = await shaFile(manifestPath);
  if (report.manifest_sha256 !== expectedManifestSha) throw new Error('Manifest SHA256 mismatch');
  const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
  if (manifest.artifact !== 'W13R_PARTIAL_RAW_PRESERVATION_MANIFEST_V1' || manifest.root !== ROOT) {
    throw new Error('Unexpected manifest identity or source root');
  }
  const names = Object.keys(manifest.files).sort();
  if (names.length !== 14 || names.some(n => path.isAbsolute(n) || n.split('/').includes('..'))) {
    throw new Error('Expected exactly fourteen relative source file paths');
  }
  const actual = inventory(ROOT);
  if (JSON.stringify(actual.files) !== JSON.stringify(names)) report.issues.push('Source file membership changed');
  if (JSON.stringify(actual.dirs) !== JSON.stringify([...manifest.expected_directories].sort())) {
    report.issues.push('Source directory membership changed');
  }
  for (const name of names) {
    const want = manifest.files[name], p = path.join(ROOT, name);
    try {
      const st = fs.lstatSync(p);
      if (!st.isFile() || st.isSymbolicLink()) throw new Error('not a regular file');
      if (!/^[a-f0-9]{64}$/.test(want.sha256) || !Number.isSafeInteger(want.bytes) || want.bytes < 0) {
        throw new Error('invalid expected hash or size');
      }
      const got = await shaFile(p);
      report.files_checked += 1;
      if (st.size !== want.bytes || got !== want.sha256) report.issues.push(`${name}: size/hash mismatch`);
    } catch (e) { report.issues.push(`${name}: ${e.message}`); }
  }
  report.success = report.issues.length === 0 && report.files_checked === 14;
} catch (e) { report.issues.push(e.message); }

console.log(JSON.stringify(report, null, 2));
if (report.success) console.log('W13R_PARTIAL_RAW_PRESERVATION_VERIFIED 14');
process.exitCode = report.success ? 0 : 1;
