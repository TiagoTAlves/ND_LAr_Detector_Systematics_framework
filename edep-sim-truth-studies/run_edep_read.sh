#!/bin/bash
CHUNK_INDEX=$1
CHUNK_SIZE=$2

cd /vols/dune/tta20/mach3/ND_LAr_Detector_Systematics_framework/edep-sim-truth-studies

if [ ! -d "logs" ]; then
    mkdir logs
fi

echo "Running chunk $CHUNK_INDEX with size $CHUNK_SIZE"
python3 edep_read.py --chunk $CHUNK_INDEX --chunksize $CHUNK_SIZE