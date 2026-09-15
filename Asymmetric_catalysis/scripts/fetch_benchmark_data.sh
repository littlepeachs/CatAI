#!/bin/bash
set -euo pipefail

destination=${1:?"usage: fetch_benchmark_data.sh DESTINATION"}
commit=d65bc0efe877ac9a2485f72704ac1c032daaac84

if [[ -f "$destination/9478_file.csv" && -f "$destination/data/output.csv" ]]; then
    exit 0
fi
if [[ -e "$destination" ]]; then
    echo "Incomplete data directory already exists: $destination" >&2
    exit 1
fi

mkdir -p "$(dirname "$destination")"
git clone --filter=blob:none --no-checkout https://github.com/CHENGLi-96/ChemAHNet.git "$destination"
git -C "$destination" sparse-checkout init --no-cone
git -C "$destination" sparse-checkout set /9478_file.csv /data/
git -C "$destination" checkout --detach "$commit"
