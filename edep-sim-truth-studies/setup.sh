#!/bin/bash

EDEP_SIM_DIR="edep-sim"
EDEP_SIM_REPO="https://github.com/DUNE/edep-sim.git"
EDEP_SIM_BUILD="$EDEP_SIM_DIR/edep-gcc-11-x86_64-redhat-linux/"

if [ ! -d "$EDEP_SIM_DIR" ]; then
    echo "edep-sim not found. Cloning from $EDEP_SIM_REPO..."
    git clone "$EDEP_SIM_REPO" "$EDEP_SIM_DIR"
fi

if [ -d "$EDEP_SIM_BUILD" ]; then
    echo "edep-sim already built"
else
    echo "Building edep-sim..."
    cd "$EDEP_SIM_DIR/build"
    source edep-build.sh
    cd ../..
fi