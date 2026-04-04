#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${CATAI_PYTHON_BIN:-python}

# $PYTHON_BIN "$SCRIPT_DIR/catai_weight_optimize.py" --dataset transfer-oc20
# $PYTHON_BIN "$SCRIPT_DIR/catai_weight_optimize.py" --dataset dissociation-oc20
# $PYTHON_BIN "$SCRIPT_DIR/catai_weight_optimize.py" --dataset desorption-oc20
# $PYTHON_BIN "$SCRIPT_DIR/catai_weight_optimize.py" --dataset Alloy-GMAE
# $PYTHON_BIN "$SCRIPT_DIR/catai_weight_optimize.py" --dataset FG-GMAE
# $PYTHON_BIN "$SCRIPT_DIR/catai_weight_optimize.py" --dataset OCD-GMAE

# python weight-optimize.py --dataset transfer-oc20 --scaling 10
# python weight-optimize.py --dataset dissociation-oc20 --scaling 10
# python weight-optimize.py --dataset desorption-oc20 --scaling 10
# python weight-optimize.py --dataset Alloy-GMAE --scaling 10
# python weight-optimize.py --dataset FG-GMAE --scaling 10
# python weight-optimize.py --dataset OCD-GMAE --scaling 10

$PYTHON_BIN "$SCRIPT_DIR/catai_weight_optimize.py" --dataset transfer-oc20 --scaling 15
$PYTHON_BIN "$SCRIPT_DIR/catai_weight_optimize.py" --dataset dissociation-oc20 --scaling 15
$PYTHON_BIN "$SCRIPT_DIR/catai_weight_optimize.py" --dataset desorption-oc20 --scaling 15
$PYTHON_BIN "$SCRIPT_DIR/catai_weight_optimize.py" --dataset Alloy-GMAE --scaling 15
$PYTHON_BIN "$SCRIPT_DIR/catai_weight_optimize.py" --dataset FG-GMAE --scaling 15
$PYTHON_BIN "$SCRIPT_DIR/catai_weight_optimize.py" --dataset OCD-GMAE --scaling 15

$PYTHON_BIN "$SCRIPT_DIR/catai_weight_optimize.py" --dataset transfer-oc20 --scaling 20
$PYTHON_BIN "$SCRIPT_DIR/catai_weight_optimize.py" --dataset dissociation-oc20 --scaling 20
$PYTHON_BIN "$SCRIPT_DIR/catai_weight_optimize.py" --dataset desorption-oc20 --scaling 20
$PYTHON_BIN "$SCRIPT_DIR/catai_weight_optimize.py" --dataset Alloy-GMAE --scaling 20
$PYTHON_BIN "$SCRIPT_DIR/catai_weight_optimize.py" --dataset FG-GMAE --scaling 20
$PYTHON_BIN "$SCRIPT_DIR/catai_weight_optimize.py" --dataset OCD-GMAE --scaling 20

$PYTHON_BIN "$SCRIPT_DIR/catai_weight_optimize.py" --dataset transfer-oc20 --scaling 25
$PYTHON_BIN "$SCRIPT_DIR/catai_weight_optimize.py" --dataset dissociation-oc20 --scaling 25
$PYTHON_BIN "$SCRIPT_DIR/catai_weight_optimize.py" --dataset desorption-oc20 --scaling 25
$PYTHON_BIN "$SCRIPT_DIR/catai_weight_optimize.py" --dataset Alloy-GMAE --scaling 25
$PYTHON_BIN "$SCRIPT_DIR/catai_weight_optimize.py" --dataset FG-GMAE --scaling 25
$PYTHON_BIN "$SCRIPT_DIR/catai_weight_optimize.py" --dataset OCD-GMAE --scaling 25

