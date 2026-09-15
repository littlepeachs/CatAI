#!/bin/bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$ROOT/run_common.sh"

bash "$ROOT/scripts/fetch_benchmark_data.sh" "$CATAI_ASY_DATA_WORKSPACE/ChemAHNet"

required_paths=(
    "$CATAI_ASY_MODULE_ROOT/jarvis_bandgap/best_checkpoint.pt"
    "$CATAI_ASY_MODULE_ROOT/qmof_bg/best_checkpoint.pt"
    "$CATAI_ASY_MODULE_ROOT/mp_gvrh/best_checkpoint.pt"
    "$CATAI_ASY_MODULE_ROOT/mp_energy_above_hull/best_checkpoint.pt"
    "$CATAI_ASY_MODULE_ROOT/surface_energy/best_checkpoint.pt"
    "$CATAI_ASY_MODULE_ROOT/adsorption_energy/best_checkpoint.pt"
    "$CATAI_ASY_FREEZE_ROOT/checkpoints/aho_ddg_j185/model.pt"
    "$CATAI_ASY_FREEZE_ROOT/checkpoints/cpa/cpa_main.pt"
    "$CATAI_ASY_FREEZE_ROOT/checkpoints/cpa/cpa_transformation.pt"
    "$CATAI_ASY_FREEZE_ROOT/checkpoints/cpa/cpa_imine.pt"
    "$CATAI_ASY_FREEZE_ROOT/checkpoints/cpa/cpa_thiol.pt"
    "$CATAI_ASY_AHO_CONFORMER_CACHE"
    "$CATAI_ASY_CPA_CONFORMER_ROOT/cpa_main.joblib"
    "$CATAI_ASY_CPA_CONFORMER_ROOT/cpa_transformation.joblib"
    "$CATAI_ASY_CPA_CONFORMER_ROOT/cpa_imine.joblib"
    "$CATAI_ASY_CPA_CONFORMER_ROOT/cpa_thiol.joblib"
    "$CATAI_ASY_CPA_CONFORMER_ROOT/cpa_catalyst.joblib"
    "$CATAI_ASY_CATALYST_TOKEN_SOURCE/component_atom_tokens.npz"
)

missing=0
for path in "${required_paths[@]}"; do
    if [[ ! -e "$path" ]]; then
        echo "Missing required asset: $path" >&2
        missing=1
    fi
done
if [[ "$missing" -ne 0 ]]; then
    echo "Place the separately distributed asymmetric-catalysis assets under catai_data/asymmetric_catalysis or override CATAI_ASY_* paths." >&2
    exit 1
fi

if [[ "${1:-}" == "--check" ]]; then
    "$CATAI_ASY_PYTHON_BIN" "$ROOT/scripts/run_historical.py" --help >/dev/null
    "$CATAI_ASY_PYTHON_BIN" "$ROOT/scripts/prepare_runtime_assets.py" --help >/dev/null
    echo "Asymmetric-catalysis setup: OK"
    exit 0
fi

if [[ ! -f "$CATAI_ASY_RUNTIME_ROOT/runtime_manifest.json" ]]; then
    "$CATAI_ASY_PYTHON_BIN" "$ROOT/scripts/prepare_runtime_assets.py" \
        --workspace-root "$CATAI_ASY_DATA_WORKSPACE" \
        --runtime-root "$CATAI_ASY_RUNTIME_ROOT" \
        --picfm-root "$CATAI_ASY_MODULE_ROOT" \
        --moma-root "$CATAI_ASY_MOMA_ROOT" \
        --freeze-root "$CATAI_ASY_FREEZE_ROOT" \
        --cpa-conformer-root "$CATAI_ASY_CPA_CONFORMER_ROOT" \
        --aho-conformer-cache "$CATAI_ASY_AHO_CONFORMER_CACHE" \
        --catalyst-token-source "$CATAI_ASY_CATALYST_TOKEN_SOURCE" \
        --device "$CATAI_ASY_DEVICE"
fi

"$CATAI_ASY_PYTHON_BIN" "$ROOT/scripts/run_historical.py" \
    --workspace-root "$CATAI_ASY_DATA_WORKSPACE" \
    --runtime-root "$CATAI_ASY_RUNTIME_ROOT" \
    --output-root "$CATAI_ASY_OUTPUT_ROOT" \
    --freeze-root "$CATAI_ASY_FREEZE_ROOT" \
    --cpa-conformer-root "$CATAI_ASY_CPA_CONFORMER_ROOT" \
    --aho-conformer-cache "$CATAI_ASY_AHO_CONFORMER_CACHE" \
    --moma-root "$CATAI_ASY_MOMA_ROOT" \
    --picfm-root "$CATAI_ASY_MODULE_ROOT" \
    --seed "$CATAI_ASY_SEED" \
    --gpu "$CATAI_ASY_GPU"
