# 55th — g0b/t3u side-midpoint P13 physics, meeting evidence, RunPod RTX A/B

Date: 2026-08-13 KST  
Active case: `g0b_d420`  
이번 case의 신규 변수: `[파지점=side midpoint, SDG candidate pose]`  
Robot hardware: 0; local Isaac physics: 1 completed P13; cloud physics rerun: 0; cloud RTX attempts: 1 bounded A/B.

## 1. What and why

The approved `t3u` case tested whether a fixed-base RoArm-M3 could grasp a D29×H50 mm,
24.83 g cylinder at its side midpoint using the selected SDG candidate and five radial
offsets. The physics trace was completed first. Because the native RTX attachment stalled,
we generated a truthful CPU trace video for the 15:00 lab meeting and separately tested
whether an RTX PRO 6000 RunPod isolated the renderer failure.

## 2. Observable procedure

1. Ran the frozen P13 physics profile: two diagnostic callbacks, full rebaseline, then
   2,340 task callbacks. The physics child exited raw 0 and all 30 result-semantic checks
   passed. No cloud physics rerun occurred.
2. Verified the five active rows and recomputed the phase metrics from the durable NPZ/JSON.
3. Tried the local native RTX replay. The original renderer and the forward-only render1
   repair both reached capture but produced zero PNGs before their bounded timeouts.
4. Produced a CPU posthoc MP4 from the exact trace: 234 frames, 20 fps, 11.7 s, H.264
   1280×720; full ffmpeg decode returned 0. This is a schematic trace replay, not an Isaac
   viewport or scientific authority.
5. Deleted the prior A100 Pod and created RunPod Pod `aoyagwoz7blwiv`, RTX PRO 6000
   Blackwell Server Edition 96 GB, secure US-NE-1, `$2.09/h`.
6. Verified driver 580.126.16, `cuInit=0`, PyTorch 2.7.0+cu128 CUDA true, Isaac Lab 2.3.0,
   Isaac Sim 5.1.0.0, numpy 1.26.0, and psutil 5.9.8.
7. Transferred 248 frozen files. The first tar lost nanosecond mtimes on `/workspace`; a PAX
   archive plus the container disk and the original absolute repo path restored the exact
   historical validators. The final cloud input gate then passed.
8. The exact render1 still produced zero frames after capture start. A minimal
   `simulation_app.update()` A/B exposed the lower-level failure: Kit logged no suitable
   CUDA graphics GPU/foundation interface, and `vulkaninfo` could not create an instance.
9. Installed missing GLU/X11 packages and supplied matching 580.126.16 NVIDIA userspace GL
   libraries separately. Vulkan still failed because the Pod exposed `/dev/nvidia3` and
   `/dev/nvidiactl` but no `/dev/nvidia-modeset`.
10. Recovered the raw cloud evidence archive, stopped and permanently deleted the Pod, then
    verified the RunPod account contained zero Pods.

## 3. Quantified result

### Physics/grasp

- Success: **0/5**.
- Population verdict: **`NO_BILATERAL_SIDE_CONTACT`**.
- Classifications: `c05_o00=premature_jaw_contact`; `c05_o01..o04=no_bilateral_close`.
- Arrival error: 1.525311–1.525837 mm, all five PASS.
- Close fixed/moving/bilateral force: 0/0/0 N for all five.
- Corrected object lift: −0.0002366 to +0.0007916 mm vs required >6 mm.
- TCP rise: 24.0486–24.1584 mm. The arm rose; the cylinder did not.
- Numeric and measurement validity: 5/5 true. The failure is observed grasp behavior, not
  corrupt instrumentation.

Authority files:

- `g0b_d420/t3u_side_preflight13_results.json`, SHA
  `8324ed7a9682ccb297985dd733c9e91c480bed9ce65bb02672d5b40226eea6d5`.
- `g0b_d420/t3u_side_preflight13_trace.npz`, SHA
  `ee67d3516a1c7871e5f48d455b420c3f5985ae889bceb097536904548e8134ee`.

### Meeting attachments

- `t3u_side_meeting1_trace_video.mp4`, SHA
  `14a9b6d9ef6dee9fae0210c7f7eda524692548d3d62e3a3608972f10b51f8414`.
- `t3u_side_meeting1_lab_bundle_v3.zip`, 4,172,534 bytes, 9/9 ZIP members verified,
  SHA `2bcdc926e60b1848026cf4c6bcd62610e04f1047bedabe355f92cb953ce67ac1`.
- `t3u_side_rendercloud1_runpod_evidence.tar.gz`, 59 entries, 73,679 bytes, SHA
  `5469c2fcf6eaed522c6a670ebc1731e8b0da360b15972237cf288d80e91e0610`.

### Cloud renderer conclusion

The RTX PRO 6000 hardware was not the blocker. CUDA compute worked, but the selected RunPod
PyTorch Pod runtime was compute-only from Vulkan's perspective. `NVIDIA_DRIVER_CAPABILITIES=all`
did not cause `/dev/nvidia-modeset` to be exposed. Therefore the cloud run was a valid
infrastructure A/B but did not produce an Isaac RTX frame.

## 4. Official NVIDIA basis

- NVIDIA Isaac Sim 5.1, **System Requirements**:
  https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html
  — RTX PRO 6000 Blackwell is an intended high-end GPU; A100/H100 are not suitable for RTX
  rendering because they lack RT cores.
- NVIDIA Isaac Sim 5.1, **Container Installation**:
  https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_container.html
- Isaac Lab 2.3, **Cloud Installation**:
  https://isaac-sim.github.io/IsaacLab/v2.3.0/source/setup/installation/cloud_installation.html

Local/runtime evidence: the recovered `gpu_vulkan_diagnosis.txt` in the cloud evidence
archive records the actual device files and Vulkan error; `rendercloud1_launcher.log`
records the Isaac 5.1 Kit errors.

## 5. Verdict and next authorization boundary

Everyday-language verdict: **the arm reached the commanded side-midpoint motions, but the
jaws never pinched the cylinder during close, so the arm lifted without the cylinder.**
The correct meeting result is a failed grasp (0/5), not a successful-grasp video.

Use the CPU trace MP4 and label it “exact-trace schematic, non-RTX, posthoc.” Do not retry
the deleted RunPod configuration. Any future RTX attempt must first pass a two-command
infrastructure gate (`test -e /dev/nvidia-modeset` and `vulkaninfo --summary`) using an
official/prepared Isaac container. Changing the grasp target, depth, trajectory, gripper,
or candidate begins a new case and requires user approval.
