#!/bin/bash
# Verbose end-to-end symbolic library build: Mozart + Chopin + Bach (MXL/MIDI mix, C2)
# Queue with: sbatch jobs/build_full_symbolic_library_verbose.slurm
# Report will be written to share/piano-clamp-runs + logs/slurm-*.out (see REPORT_FILE below)

#SBATCH --partition=workq
#SBATCH --job-name=full-symbolic-verbose
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --open-mode=append

set -euo pipefail
set -x

# --- helpers: verbose trap + logging ---
trap 'echo "!!! ERROR at line $LINENO: $BASH_COMMAND (exit $?)"; echo "!!! STACK:"; caller; echo "!!! env:"; env | sort; echo "!!! pwd:"; pwd; ls -R 2>&1 | head -n 200' ERR

log() { echo "[$(date -Iseconds)] $*"; }
run_verbose() {
  log "RUN: $*"
  set +e
  "$@" 2>&1
  local rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "!!! COMMAND FAILED ($rc): $*"
    return $rc
  fi
  log "OK ($rc): $*"
}

# --- environment ---
set +u
source "$(conda info --base)/etc/profile.d/conda.sh" 2>&1 | cat -v
conda activate piano-clamp-env 2>&1
set -u
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
CLAMP_DEVICE="${CLAMP_DEVICE:-cpu}"

# pin roots (mirrors configs/paths.yaml)
CORPUS_ROOT="${PIANO_CLAMP_CORPUS_ROOT:-/home/student/w/wshelor/share/datasets/classical-performance-corpus}"
SHARE_ROOT="/home/student/w/wshelor/share/piano-clamp-runs"
REPO_ROOT="$(pwd)"
TIMESTAMP="$(date +%Y%m%dT%H%M%SZ)"
REPORT_DIR="${SHARE_ROOT}/reports"
REPORT_FILE="${REPORT_DIR}/full-symbolic-build-${TIMESTAMP}-${SLURM_JOB_ID:-manual}.json"
LOG_COPY="${REPORT_DIR}/full-symbolic-build-${TIMESTAMP}-${SLURM_JOB_ID:-manual}.log"

mkdir -p logs "$REPORT_DIR" "$SHARE_ROOT"

log "REPO_ROOT=$REPO_ROOT SHARE_ROOT=$SHARE_ROOT CORPUS_ROOT=$CORPUS_ROOT REPORT_FILE=$REPORT_FILE CLAMP_DEVICE=$CLAMP_DEVICE"
log "HOST=$(hostname) USER=$(whoami) SLURM_JOB_ID=${SLURM_JOB_ID:-none} SLURM_NODELIST=${SLURM_NODELIST:-none}"
nvidia-smi 2>&1 | head -n 50 || echo "nvidia-smi not available"
python3 -V 2>&1; python3 -c "import sys,torch; print(sys.version); print('torch',torch.__version__, 'cuda',torch.version.cuda, 'cuda_available',torch.cuda.is_available())" 2>&1 | head -n 20
git rev-parse HEAD 2>&1; git status --short 2>&1 | head -n 50; git submodule status 2>&1 | head -n 20
cat configs/paths.yaml 2>&1 | head -n 80
cat configs/embedding_config.yaml 2>&1 | head -n 80
ls -lh "$CORPUS_ROOT" 2>&1 | head -n 30
ls -lh "$CORPUS_ROOT"/data/canonical/*.csv 2>&1 | head -n 30
ls -lh "$CORPUS_ROOT"/manifests/releases/*.json 2>&1 | head -n 20

# duplicate stdout to report log as well
exec > >(tee -a "$LOG_COPY") 2>&1
log "LOGGING ALSO TO $LOG_COPY"

# --- adapter phase: verbose rebuild for Mozart+Chopin+Bach mix ---
# Note: cpc-2026-08-19.2 has 110 DCML MSCX pending until MuseScore converts them.
# attempt conversion dry-run with full traceback
log "STEP 1: CPC adapter verboseness check (canonical tables)"
run_verbose python3 -c "import pathlib; p=pathlib.Path('$CORPUS_ROOT/data/canonical/works.csv'); print('works', sum(1 for _ in open(p))-1); p=pathlib.Path('$CORPUS_ROOT/data/canonical/recordings.csv'); print('recordings', sum(1 for _ in open(p))-1)"

ADAPTER_ALL="data/cpc_adapter/cpc-$(date +%Y-%m-%d)-all3-verbose-${SLURM_JOB_ID:-manual}"
BACH_ADAPTER="data/cpc_symbolic_adapter/bach-v0-verbose-${SLURM_JOB_ID:-manual}"
# keep stable paths for embedding_config.yaml overlay: we symlink/copy if needed
# Use explicit versioned outputs so we never overwrite (cpc_adapter.py enforces output must not exist)

if [[ -d "$ADAPTER_ALL" ]]; then
  log "ADAPTER_ALL already exists: $ADAPTER_ALL (skipping build, will verbose inspect)"
else
  log "STEP 2: build combined adapter (Chopin/Mozart/Bach, all eligible) -> $ADAPTER_ALL"
  run_verbose python -u -X faulthandler scripts/prepare_cpc_adapter.py \
    --corpus-root "$CORPUS_ROOT" \
    --output-root "$ADAPTER_ALL" \
    --composer Chopin --composer Mozart --composer Bach
fi

log "STEP 2a: adapter manifest verbose dump"
ls -lh "$ADAPTER_ALL" 2>&1 | head -n 50
cat "$ADAPTER_ALL"/adapter_metadata.json 2>&1 | head -n 200 || true
cat "$ADAPTER_ALL"/cohort_summary.csv 2>&1 | head -n 50 || true
cut -d, -f2 "$ADAPTER_ALL"/piano_clamp_manifest.csv 2>&1 | sort | uniq -c | sort -rn | head -n 20 || true
wc -l "$ADAPTER_ALL"/piano_clamp_manifest.csv "$ADAPTER_ALL"/alignment_events.csv 2>&1 | head -n 20 || true
head -n 5 "$ADAPTER_ALL"/piano_clamp_manifest.csv 2>&1 | cut -d, -f1,2,6,9,13 | column -t -s, 2>&1 | head -n 20 || true

if [[ -d "$BACH_ADAPTER" ]]; then
  log "BACH_ADAPTER already exists: $BACH_ADAPTER"
else
  log "STEP 3: build Bach symbolic adapter -> $BACH_ADAPTER"
  run_verbose python -u -X faulthandler scripts/prepare_cpc_symbolic_adapter.py \
    --corpus-root "$CORPUS_ROOT" \
    --output-root "$BACH_ADAPTER" \
    --composer Bach --composer Chopin --composer Mozart
  # also build strict Bach-only as reference (if requested)
fi
ls -lh "$BACH_ADAPTER" 2>&1 | head -n 50
cat "$BACH_ADAPTER"/adapter_metadata.json 2>&1 | head -n 200 || true
cut -d, -f2 "$BACH_ADAPTER"/piano_clamp_manifest.csv 2>&1 | sort | uniq -c | head -n 20 || true

# --- swap adapter_root for embedding run: write temporary paths.yaml overlay ---
# piano-clamp reads configs/paths.yaml overlay over embedding_config.yaml (data_loading.py)
# we backup and inject ADAPTER_ALL so embed_* sees the new manifest without editing corpus
BACKUP_PATHS="configs/paths.yaml.pre-${TIMESTAMP}.bak"
cp -v configs/paths.yaml "$BACKUP_PATHS" 2>&1
log "BACKUP paths.yaml -> $BACKUP_PATHS"
# update adapter_root to ADAPTER_ALL (keep other roots pointing to SHARE_ROOT)
python3 - <<PY 2>&1
import pathlib, yaml
p = pathlib.Path("configs/paths.yaml")
cfg = yaml.safe_load(p.read_text())
import os
print("BEFORE adapter_root", cfg.get("adapter_root"))
cfg["adapter_root"] = os.path.abspath("$ADAPTER_ALL")
# ensure output roots stay on SHARE_ROOT (already there)
p.write_text(yaml.safe_dump(cfg, sort_keys=False))
print("AFTER adapter_root", cfg["adapter_root"])
PY
cat configs/paths.yaml 2>&1 | head -n 60

# --- embedding phase: symbolic library (MXL/MIDI mix, C2) ---
log "STEP 4: embed text prompts (C2, all 137 enabled)"
run_verbose python -u -X faulthandler scripts/embed_text_prompts.py --config configs/embedding_config.yaml --device "$CLAMP_DEVICE" --force

log "STEP 5: embed symbolic scores (complete movements, 1 per work, MXL/MIDI mix per select_symbolic_records)"
run_verbose python -u -X faulthandler scripts/embed_symbolic_scores.py --config configs/embedding_config.yaml --device "$CLAMP_DEVICE" --force

log "STEP 6: embed passages symbolic score windows (MXL, 4/8/16 overlap-half)"
run_verbose python -u -X faulthandler scripts/embed_passages.py \
  --config configs/embedding_config.yaml --device "$CLAMP_DEVICE" \
  --modality symbolic --source-material score \
  --mode 4 --mode 8 --mode 16 --overlap-half --force --status-every 5 --chunk-size 25

log "STEP 7: embed passages performance_midi windows (MIDI, 4/8/16 overlap-half, mido time-crop)"
run_verbose python -u -X faulthandler scripts/embed_passages.py \
  --config configs/embedding_config.yaml --device "$CLAMP_DEVICE" \
  --modality symbolic --source-material performance_midi \
  --mode 4 --mode 8 --mode 16 --overlap-half --force --status-every 5 --chunk-size 25

# also run dedicated Bach symbolic v0 as reference (uses its own config)
log "STEP 8: embed Bach symbolic reference (separate output_root share/.../bach_symbolic_v0)"
if [[ -f configs/embedding_config_bach_symbolic.yaml ]]; then
  # patch its adapter_root to BACH_ADAPTER as well for verboseness
  run_verbose python -u -X faulthandler scripts/prepare_cpc_symbolic_adapter.py --corpus-root "$CORPUS_ROOT" --output-root "data/cpc_symbolic_adapter/bach-check-${SLURM_JOB_ID:-manual}" --composer Bach 2>&1 | head -n 100 || true
fi

# --- verification: shapes, counts, C2 vs SAAS, MXL feature dry-run ---
log "STEP 9: verification dumps"

for bundle in "$SHARE_ROOT"/embeddings/text "$SHARE_ROOT"/embeddings/symbolic "$SHARE_ROOT"/embeddings/passages "$SHARE_ROOT"/embeddings/passages/performance_midi; do
  echo "== BUNDLE $bundle"
  ls -lh "$bundle" 2>&1 | head -n 20 || echo "MISSING $bundle"
  cat "$bundle"/metadata.json 2>&1 | head -n 100 || true
  wc -l "$bundle"/*.csv 2>&1 | head -n 20 || true
  python3 - <<PY 2>&1 | head -n 50 || true
import pathlib, numpy as np
import sys
p = pathlib.Path("$bundle")
for npy in p.glob("*.npy"):
    try:
        a = np.load(npy)
        print(npy, a.shape, a.nbytes, "finite", bool(np.isfinite(a).all()), "zero_rows", int((np.linalg.norm(a,axis=1)==0).sum()) if a.size else 0)
    except Exception as e:
        import traceback; traceback.print_exc()
        print("LOAD FAIL", npy, e)
PY
  python3 - <<PY 2>&1 | head -n 100 || true
import pandas as pd, pathlib
import sys
csv = list(pathlib.Path("$bundle").glob("*.csv"))
if csv:
    df=pd.read_csv(csv[0], nrows=5)
    print(df.columns.tolist()[:20])
    try:
        full=pd.read_csv(csv[0], usecols=lambda c: c in ["composer","status","passage_generation_method","window_bars"])
        print(full["composer"].value_counts().head(20).to_dict())
        if "status" in full: print(full["status"].value_counts().to_dict())
        if "window_bars" in full: print(full["window_bars"].value_counts().head(10).to_dict())
    except Exception as e:
        import traceback; traceback.print_exc()
PY
done

# MXL feature extraction dry-run: show what would be extracted for first 5 windows
log "STEP 10: MXL feature extraction verboseness (first 5 windows, no full run)"
python3 - <<PY 2>&1 | head -n 100 || true
from pathlib import Path
import pandas as pd
try:
    df=pd.read_csv("$SHARE_ROOT/embeddings/passages/passage_embeddings.csv", nrows=5)
    print(df[["passage_id","composer","score_path","mxl_path","bar_start","bar_end"]].to_string())
    from piano_clamp.features import extract_musicxml_features, FEATURE_FIELDS
    print("FEATURE_FIELDS", FEATURE_FIELDS)
except Exception as e:
    import traceback; traceback.print_exc()
PY

#--- report file: JSON with all useful info for post-mortem ---
log "STEP 11: write report JSON $REPORT_FILE"
python3 - <<PY 2>&1
import json, pathlib, subprocess, sys, os, hashlib
from datetime import datetime, timezone

def sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT)[:8000]
    except subprocess.CalledProcessError as e:
        return f"ERR {e.returncode}: {e.output[:4000]}"
    except Exception as e:
        import traceback; return traceback.format_exc()[:4000]

repo = pathlib.Path("$REPO_ROOT")
share = pathlib.Path("$SHARE_ROOT")
report = {
  "timestamp": datetime.now(timezone.utc).isoformat(),
  "slurm_job_id": os.environ.get("SLURM_JOB_ID","manual"),
  "hostname": sh("hostname"),
  "git_head": sh("git rev-parse HEAD"),
  "git_status": sh("git status --short"),
  "git_diff": sh("git diff --stat"),
  "conda_env": sh("conda env list | grep piano-clamp"),
  "nvidia": sh("nvidia-smi 2>&1 | head -n 30"),
  "python": sh("python3 -V 2>&1; python3 -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())' 2>&1"),
  "paths_yaml": pathlib.Path("configs/paths.yaml").read_text()[:8000] if pathlib.Path("configs/paths.yaml").exists() else "missing",
  "embedding_config": pathlib.Path("configs/embedding_config.yaml").read_text()[:8000] if pathlib.Path("configs/embedding_config.yaml").exists() else "missing",
  "adapter_all": "$ADAPTER_ALL",
  "adapter_all_meta": sh(f"cat $ADAPTER_ALL/adapter_metadata.json 2>&1 | head -n 100"),
  "adapter_all_cohort": sh(f"cat $ADAPTER_ALL/cohort_summary.csv 2>&1 | head -n 100"),
  "adapter_all_counts": sh(f"cut -d, -f2 $ADAPTER_ALL/piano_clamp_manifest.csv 2>&1 | sort | uniq -c | head -n 20"),
  "bach_adapter": "$BACH_ADAPTER",
  "bach_meta": sh(f"cat $BACH_ADAPTER/adapter_metadata.json 2>&1 | head -n 100") if pathlib.Path("$BACH_ADAPTER/adapter_metadata.json").exists() else "missing",
  "corpus_validation": sh(f"cat $CORPUS_ROOT/manifests/validation_report.json 2>&1 | python3 -m json.tool 2>&1 | head -n 100"),
}

# bundle stats
for rel in ["embeddings/text","embeddings/symbolic","embeddings/passages","embeddings/passages/performance_midi"]:
    p = share / rel
    import pandas as pd, numpy as np
    key = rel.replace("/","_")
    try:
        if p.exists():
            report[key+"_ls"] = sh(f"ls -lh {p} 2>&1 | head -n 20")
            npy = list(p.glob("*.npy"))
            if npy:
                a=np.load(npy[0])
                report[key+"_npy_shape"]=str(a.shape)
                report[key+"_npy_nbytes"]=int(a.nbytes)
            csvs=list(p.glob("*.csv"))
            if csvs:
                report[key+"_csv_rows"]=int(sum(1 for _ in open(csvs[0]))-1)
                df=pd.read_csv(csvs[0], usecols=lambda c: c in ["composer","status"])
                report[key+"_composer_counts"]=df["composer"].value_counts().to_dict() if "composer" in df else {}
                report[key+"_status_counts"]=df["status"].value_counts().to_dict() if "status" in df else {}
        else:
            report[key+"_missing"]=True
    except Exception as e:
        import traceback; report[key+"_error"]=traceback.format_exc()[:4000]

pathlib.Path("$REPORT_FILE").write_text(json.dumps(report, indent=2, ensure_ascii=False))
print(f"WROTE {report}")

PY

log "REPORT_FILE=$REPORT_FILE LOG_COPY=$LOG_COPY"
ls -lh "$REPORT_FILE" "$LOG_COPY" 2>&1 | head -n 20
cat "$REPORT_FILE" 2>&1 | head -n 300

# restore paths.yaml
log "RESTORE paths.yaml from $BACKUP_PATHS"
cp -v "$BACKUP_PATHS" configs/paths.yaml 2>&1
cat configs/paths.yaml 2>&1 | head -n 30

log "DONE - attach $REPORT_FILE and logs/slurm-${SLURM_JOB_ID}.out for debugging"
set +x
