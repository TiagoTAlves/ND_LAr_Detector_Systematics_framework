#!/bin/bash

DUNEANAOBJ_DIR="duneanaobj"
DUNEANAOBJ_REPO="https://github.com/DUNE/duneanaobj.git"

DUNEANAOBJ_BRANCH="main"
while [[ $# -gt 0 ]]; do
    case $1 in
        --duneanaobj-branch)
            DUNEANAOBJ_BRANCH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ ! -d "DUNEANAOBJ_DIR" ]; then
    echo "duneanaobj not found. Cloning from $DUNEANAOBJ_REPO..."
    git clone "$DUNEANAOBJ_REPO" "$DUNEANAOBJ_DIR"
fi

cd "$DUNEANAOBJ_DIR"
git checkout ${DUNEANAOBJ_BRANCH}
# Compile all .cxx source files:
g++ -fPIC -c `root-config --cflags` -I./ duneanaobj/StandardRecord/*.cxx

# Compile dictionary:
rootcling -f StandardRecordDict.cxx -c duneanaobj/StandardRecord/StandardRecord.h duneanaobj/StandardRecord/classes_def.xml -I./

# Link all .o and dictionary together:
g++ -shared -fPIC `root-config --cflags` -I./ StandardRecordDict.cxx *.o -o libduneanaobj_StandardRecord.so
cd ..



