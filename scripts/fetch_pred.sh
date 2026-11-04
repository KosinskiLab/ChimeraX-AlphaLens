#!/usr/bin/env bash
# Copy only the files DistoPAE reads from one prediction directory, usually on a remote
# machine, so ChimeraX can open it locally.
#
#   scripts/fetch_pred.sh [--no-pickles] [user@]host:/path/to/prediction_dir local_dir
#
# The prediction ends up in local_dir/<prediction_dir name>/. The source can also be a local
# path. --no-pickles skips the AlphaFold2 / ColabFold result pickles: they are large and hold
# the AF2 distogram, so without them the plot shows PAE only.
set -euo pipefail

usage() {
    sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//' >&2
    exit 1
}

pickles=1
if [[ ${1:-} == --no-pickles ]]; then
    pickles=0
    shift
fi
[[ $# -eq 2 ]] || usage

src=${1%/}
dest=$2
name=$(basename "${src#*:}")

filters=(
    --include='*/'
    # AlphaFold2 / AlphaPulldown
    --include='ranked_*.pdb' --include='ranked_*.cif'
    --include='ranking_debug.json'
    --include='pae_*.json' --include='pae_*.json.gz' --include='pae_*.json.xz'
    # AlphaFold3
    --include='*_model.cif' --include='model.cif'
    --include='*confidences.json' --include='*confidences.json.gz' --include='*confidences.json.xz'
    --include='ranking_scores.csv'
    --include='seed-*_distogram/*.npz'
    # ColabFold
    --include='*_rank_[0-9][0-9][0-9]_*.pdb' --include='*_scores_rank_*.json'
)
if (( pickles )); then
    filters+=(--include='result_*.pkl*' --include='*_all_rank_*.pickle*')
fi
filters+=(--exclude='*')

mkdir -p "$dest/$name"
rsync -a --info=progress2 --prune-empty-dirs "${filters[@]}" "$src/" "$dest/$name/"
echo "In ChimeraX:  distopae open $dest/$name"
