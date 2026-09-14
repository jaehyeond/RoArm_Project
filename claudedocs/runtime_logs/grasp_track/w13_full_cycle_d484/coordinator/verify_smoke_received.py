"""Read-only parent recheck of the completed smoke, not full-cycle acceptance."""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import runpy

AUDIT = Path('/home/cgxr/orca/workspaces/RoArm_Project/w13-cycle-audit/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/audit/wall_smoke_result_01/verify_wall_smoke.py')
EXPECTED = 'abb0bd001981e35380bccea8cc0b391d3f50ec83f8c78ae8921c9337726982a0'


class MemoryResult:
    """Intercept the auditor's RESULT write; never rewrite its evidence."""
    result = None

    def write_text(self, value, encoding=None):
        self.result = json.loads(value)
        return len(value)


def main():
    assert hashlib.sha256(AUDIT.read_bytes()).hexdigest() == EXPECTED, 'Auditor source changed'
    module = runpy.run_path(str(AUDIT), run_name='w13_parent_readonly_audit')
    sink = MemoryResult()
    module['main'].__globals__['OUT'] = sink
    rc = None
    with redirect_stdout(io.StringIO()):
        try:
            module['main']()
        except SystemExit as exc:
            rc = exc.code
    result = sink.result
    assert rc == 0 and result and result['all_pass'], 'Independent smoke check failed'
    assert len(result['execution']['returncodes']) == 3
    assert result['hash_receipt']['n'] == 44
    print(json.dumps({
        'scope': 'completed stationary smoke only; full cycle NOT executed',
        'auditor_source_sha256': EXPECTED,
        'hash_receipt': result['hash_receipt'],
        'finalized_outputs': result['finalized_outputs'],
        'raw': result['raw'],
        'negative_control_qualification': 'Seven array mutations are rejected; the eighth hash comparison is only a nonzero-hash sanity check, not an end-to-end corruption-injection test.',
        'writes': 0,
    }, ensure_ascii=False, indent=2))
    print('W13_PARTIAL_SMOKE_VERIFIED')


if __name__ == '__main__':
    main()
