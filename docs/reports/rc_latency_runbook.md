# A100 latency run on the RIT cluster: runbook

Everything lives in `~/grasp_work` on the cluster. The only deletions are the jobs' private temp directories and the temporary Hugging Face token file. Nothing computes on the login node; the only things run there are the clone, copies and read-only checks below. Every file that arrives (downloaded or copied) is checked against a known SHA-256 (`configs/rc_expected_sha256.txt`), and the latency job refuses to start if anything is missing or different.

## What is measured (one A100-PCIE-40GB, job `slurm/rc_latency.sbatch`, about 30 to 40 minutes)

Components: the segmenters SAM2.1 tiny/small/large and the four-member classifier; each classifier member; YOLO26s-seg; SAM2-large and EdgeTAM propagation per instrument, causal (past 20 frames) and non-causal (+-10).

Whole pipelines, decoded frame to final label, 32 test keyframes with full 20-past and 10-future windows (30 timed, 2 warm-up), the paper's fine-tuned segmenters (tiny, SAM2-large and SAM3, all trained on the 8 training cases) and the live gate S1 >= 2.85e-4:
`p1_rt_none` (tiny + flip, classifier, gate), `p2_rt_yolo`, `p3_rt_edgetam`, `p4_causal_sam2` (tiny masks, SAM2-large causal), `p5_noncausal_final` (SAM2-large + SAM3 masks, SAM2-large over +-10), `p6_causal_final` (the same masks, past 20 frames). `summarize_pipeline_latency.py` gives per-instrument (time until the instrument's own final label) and per-keyframe (queue) latencies.

## Steps (you run these; I never see the passphrase or the token)

1. **Check the batch files without running anything** (after step 2): `sbatch --test-only code/slurm/rc_setup.sbatch` and the same for `rc_latency.sbatch`. This validates account, partition and resources only. Also `sinfo -s` to confirm the CPU partition name used in `rc_setup.sbatch` (`sporc-cpu`); edit the `#SBATCH --partition` line if it differs.
2. Login from PowerShell: `ssh ab1340@sporcsubmit.rc.rit.edu`, then
   `mkdir -p ~/grasp_work/logs ~/grasp_work/weights_extra ~/grasp_work/weights_ft && git clone --depth 1 --branch realtime-gtbox https://github.com/volcanictv/grasp-lightweight-instrument-recognition ~/grasp_work/code`
3. From the laptop (PowerShell), through titanxp, 118 MB bundle, 23 MB YOLO weights, 2.8 GB fine-tuned segmenters:
   `scp -r titanxp:~/rc_bundle/subset3 .\rc_bundle`, then `scp -r .\rc_bundle ab1340@sporcsubmit.rc.rit.edu:~/grasp_work/bundle`
   `scp titanxp:~/rc_dry/weights_extra/yolo26s_official_last.pt .`, then `scp yolo26s_official_last.pt ab1340@sporcsubmit.rc.rit.edu:~/grasp_work/weights_extra/`
   `scp titanxp:~/grasp_yolo26/experiments/sam2_gtbox/tiny_all8/weights.pt tiny_all8.pt`, `scp titanxp:~/grasp_yolo26/experiments/sam2_gtbox/final_all8/weights.pt large_all8.pt`, `scp titanxp:~/grasp_yolo26/experiments/sam3/final_all8/weights.pt sam3_all8.pt`, then `scp tiny_all8.pt large_all8.pt sam3_all8.pt ab1340@sporcsubmit.rc.rit.edu:~/grasp_work/weights_ft/`
4. SAM3 token (gated model). On the cluster, a read-only fine-grained token from huggingface.co/settings/tokens:
   `umask 077; read -rsp "token: " t; printf %s "$t" > ~/grasp_work/.hf_token; unset t`
   The setup job reads it once, downloads `facebook/sam3` into `~/grasp_work/hf_cache`, and deletes the file. Revoke the token on Hugging Face afterwards.
5. `cd ~/grasp_work && sbatch code/slurm/rc_setup.sbatch` (CPU only, about 20 to 40 minutes). Slack pings at start and end. Then `tail logs/grasp-setup_*.out`: it must end with "setup done" and list `ok:` for every file. Anything `missing` or `DIFFERENT` is fixed before step 6.
6. `cd ~/grasp_work && sbatch code/slurm/rc_latency.sbatch` (one A100, onboard partition). Slack pings. Results in `~/grasp_work/results/run_<jobid>/`, headline file `pipeline_latency.json`.
7. Copy the results off the cluster (no backups there): `scp -r ab1340@sporcsubmit.rc.rit.edu:~/grasp_work/results/run_<jobid> docs\reports\latency_a100\`

## Caveats stated in the paper
A100-PCIE-40GB (not SXM), PyTorch 2.8.0 (CUDA 12.6 build, as on the Titan Xp), no custom CUDA extension for SAM2 (as on the Titan Xp), frame copying into the video-predictor folder excluded, bundle frames read from local disk. EdgeTAM's pipeline is assembled from two processes (its masks and gate come from `p1`).
