#!/usr/bin/env bash
# Witches' Brew -- one-shot remote setup + piste 1 run.
#
# Paste this whole file into a fresh SSH session, OR run it:  bash bootstrap_remote.sh
# It installs everything on the /Data mount (never the shared cloud home) and runs the frequency
# probe (E0 + E1). No `set -e`: if a line fails you see it instead of your login shell dying.
#
# Edit these two if your layout differs:
DATA_ROOT="/Data"                 # the big local mount ("dossier mère" -> Data)
WORK="khl"                        # your folder under it
REPO_URL="https://github.com/khalilzzz/AdversarialML"

# --- 0. go to the Data mount and make your workspace ---------------------------------------------
# (equivalent to climbing cd .. up to / then into Data, but deterministic)
cd "$DATA_ROOT" || { echo "no $DATA_ROOT on this machine -- edit DATA_ROOT"; return 2>/dev/null || exit 1; }
mkdir -p "$DATA_ROOT/$WORK"
cd "$DATA_ROOT/$WORK" || exit 1

# --- 1. clone the repo ---------------------------------------------------------------------------
git clone "$REPO_URL" || echo "(clone skipped -- folder already there)"
cd "$(basename "$REPO_URL")" || exit 1
# the package lives in witches_brew/ inside the repo; cd in if present
[ -d witches_brew ] && cd witches_brew
echo "working in: $(pwd)"

# --- 2. keep every cache on /Data, not on the shared cloud home ----------------------------------
export PIP_CACHE_DIR="$DATA_ROOT/$WORK/.cache/pip"   # pip wheels cache
export XDG_CACHE_HOME="$DATA_ROOT/$WORK/.cache"      # torch / torchvision caches too
export TORCH_HOME="$DATA_ROOT/$WORK/.cache/torch"
mkdir -p "$PIP_CACHE_DIR" "$TORCH_HOME"

# --- 3. virtualenv (created under /Data/khl, survives a re-clone) --------------------------------
python3 -m venv "$DATA_ROOT/$WORK/.venv" && source "$DATA_ROOT/$WORK/.venv/bin/activate"
python -m pip install --upgrade pip
# GPU machine: the default Linux torch wheel ships CUDA. Pin an index here only if CUDA mismatches,
# e.g.  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .                                     # makes `import witchesbrew` work (src layout)

# --- 4. sanity: tests + GPU visible --------------------------------------------------------------
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
python -m pytest -q

# --- 5. PISTE 1 (E0 diagnostic + E1 WB-LF) -------------------------------------------------------
# CIFAR-10 downloads itself into data/raw (on /Data) the first brew. The first brew also trains and
# caches the clean surrogate (theta); the second reuses it.
#
# brew + export the two perturbations (GPU)
python scripts/brew_case.py --config cifar10/baseline.yaml --case 0 --export packed \
       --set name=wb_full
python scripts/brew_case.py --config cifar10/baseline.yaml --case 0 --export packed \
       --set name=wb_lf --set delta_param=lowfreq --set lowfreq_cutoff=12

# diagnose each (CPU-cheap): spectrum + per-Psi budget survival + alignment
python scripts/diagnose_frequency.py --config cifar10/baseline.yaml \
       --set name=wb_full --poisons poisons/wb_full_case00
python scripts/diagnose_frequency.py --config cifar10/baseline.yaml \
       --set name=wb_lf --set delta_param=lowfreq --poisons poisons/wb_lf_case00

# figure: WB-full vs WB-LF (writes results/figures/frequency.png)
python scripts/plot.py frequency --file results/wb_full/frequency_probe.json \
       --lowfreq results/wb_lf/frequency_probe.json

echo
echo "DONE. Figure: $(pwd)/results/figures/frequency.png"
echo "JSON : results/wb_full/frequency_probe.json  results/wb_lf/frequency_probe.json"

# --- 6. (optional) E2 WB-Psi + E3 trade-off ------------------------------------------------------
# Uncomment to also brew Psi-aware and evaluate under a JPEG defense, then sweep the trade-off curve.
# python scripts/run_experiment.py --config cifar10/baseline.yaml \
#        --set name=wb_psi --set brew_purify=jpeg --set brew_purify_eot=2 --set purify=jpeg
# python scripts/tradeoff.py --config cifar10/baseline.yaml \
#        --eps 16,8 --cutoff 8,12,16 --include-full --set purify=jpeg --set brew_purify=jpeg
# python scripts/plot.py tradeoff --file results/cifar10_baseline/tradeoff.json
