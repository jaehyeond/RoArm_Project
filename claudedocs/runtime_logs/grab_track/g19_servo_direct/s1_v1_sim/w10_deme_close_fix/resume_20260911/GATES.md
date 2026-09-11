# Gates: W10 reboot continuation

Scope: Resume the authorized dt-only cell, follow the documented conditional stiffness cell only upon divergence, and report the scientific outcome with replay evidence and current state. Original GATES_w10.md remains the scientific acceptance contract.

- [x] G1: NVIDIA kernel and user driver agree and GPU memory meets the launch requirement
  CHECK: python3 -c "import subprocess,pathlib; r=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version,memory.free','--format=csv,noheader,nounits'],text=True).strip().splitlines()[0].split(','); assert r[0].strip() in pathlib.Path('/proc/driver/nvidia/version').read_text(); assert int(r[1])>=3072; print('GPU_RESUME_READY')"
  EXPECT: GPU_RESUME_READY
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/home/cgxr/Documents/Robotics/RoArm_Project; path=d8c1fc7bfaf5/19 entries; EXPECT=matched; output-sha256=6ae06649870cc8fa5d5e9d2c0729be408bbcb2b1c0e2b83827b6b2afb7ef1511; output-bytes=17

- [x] G2: Authorized W10 cells have an evidence-backed final disposition and Korean report
  CHECK: python3 -c "import json,pathlib; p=pathlib.Path('claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w10_deme_close_fix'); d=json.loads((p/'resume_20260911/verification.json').read_text()); assert d['scientific_disposition_verified'] is True; assert (p/'REPORT_w10.md').stat().st_size>0; print('W10_DISPOSITION_VERIFIED')"
  EXPECT: W10_DISPOSITION_VERIFIED
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/home/cgxr/Documents/Robotics/RoArm_Project; path=d8c1fc7bfaf5/19 entries; EXPECT=matched; output-sha256=610fbf979d4371d0aefffdfc3cf74275418489e05213b401440a4146d139d478; output-bytes=25

- [x] G3: Replay artifacts satisfy the D341 machine-verifiable contract
  CHECK: python3 -c "import json,pathlib; d=json.loads(pathlib.Path('claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w10_deme_close_fix/resume_20260911/verification.json').read_text()); assert d['d341_machine_contract_pass'] is True; print('W10_REPLAY_CONTRACT_VERIFIED')"
  EXPECT: W10_REPLAY_CONTRACT_VERIFIED
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/home/cgxr/Documents/Robotics/RoArm_Project; path=d8c1fc7bfaf5/19 entries; EXPECT=matched; output-sha256=bc20b6f31d4fe44952e41bedc44c729592adbe26037cab098a5bac2b365c121c; output-bytes=29

- [x] G4: Decision screenshot was visually inspected with observations recorded
  EVIDENCE: view_image inspected cell_DE_dt2e6_c/scoop_s1_seed460_w10_inspection.png and decision_frames.png; observations, limitations and screenshot SHA256 recorded in cell_DE_dt2e6_c/scoop_s1_seed460_w10_inspection.json.

- [x] G5: Current dashboard and Codex relay link the new session evidence
  CHECK: python3 -c "from pathlib import Path; name='session_20260911_w10_reboot_resume.md'; assert Path('claudedocs',name).is_file(); assert name in Path('START_HERE.md').read_text(); assert name in Path('claudedocs/relay/from_codex.md').read_text(); print('W10_STATE_LINKED')"
  EXPECT: W10_STATE_LINKED
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/home/cgxr/Documents/Robotics/RoArm_Project; path=d8c1fc7bfaf5/19 entries; EXPECT=matched; output-sha256=7a73ed7e40aa0deb8ab441b8e7d42942022d97a47809645127a0d96a9d3c43ec; output-bytes=17
