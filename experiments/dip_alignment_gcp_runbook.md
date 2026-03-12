# DIP Alignment GCP Runbook

Use this runbook to execute the DIP alignment sanity sweep on GCP with full log visibility and automatic teardown.

## 1) Canary (required)
Run a cheap end-to-end canary first.

```bash
./run_on_gcp_live.sh \
  -e submission_staging/experiments/dip_alignment_validation.py \
  -r results/dip_alignment_gcp_canary \
  -c 4 \
  -z us-central1-a \
  -t 1800 \
  -k 10 \
  --monitor-interval 60 \
  --monitor-stale-minutes 8 \
  --monitor-grace-minutes 15 \
  --monitor-max-failures 2 \
  --args "--targets 1 --seeds 1 --noise-levels 0.05 --steps 5 --max-jobs 2"
```

Success criteria:
- Console shows `STEP 5: Downloading results...`
- `results/dip_alignment_gcp_canary/status.txt` ends with `FINISHED`
- Instance is deleted automatically at the end

## 2) Full run
After canary passes, launch the full sweep.

```bash
./run_on_gcp_live.sh \
  -e submission_staging/experiments/dip_alignment_validation.py \
  -r results/dip_alignment_gcp_full_$(date +%Y%m%d_%H%M%S) \
  -c 32 \
  -z us-central1-a \
  -t 21600 \
  -k 20 \
  --monitor-interval 120 \
  --monitor-stale-minutes 10 \
  --monitor-grace-minutes 20 \
  --monitor-max-failures 2
```

## 3) Safety checks
Check active workers:

```bash
gcloud compute instances list --format="table(name,zone,status)"
```

Check local recovered artifacts:

```bash
ls -lah results/dip_alignment_gcp_full_*/
```

Expected key files:
- `run_results.jsonl`
- `validation_results.json`
- `status.json`
- `status.txt`
- `bootstrap.log`
- `experiment.log`
- `alignment_validation.png`
- `alignment_validation.pdf`

## 4) Emergency stop
If needed, stop a worker immediately:

```bash
gcloud compute instances delete <instance-name> --zone <zone> --quiet
```

The local heartbeat monitor is enabled by default and will stop stale workers if the parent runner dies.
