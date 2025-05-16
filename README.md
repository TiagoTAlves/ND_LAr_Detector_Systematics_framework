# ND LAr Detector Systematics Framework

This repository contains code, documentation, and resources related to the study and implementation of detector systematic uncertainties for the Near Detector Liquid Argon (ND LAr) component of DUNE.

## Overview

The goal of this project is to develop, organize, and document tools and workflows for handling systematics in ND LAr analyses.

## Getting Started

1. **Clone the repository:**
    ```bash
    git clone https://github.com/TiagoTAlves/ND_LAr_Detector_Systematics_framework.git
    ```
2. **Install dependencies:**  
    See the [requirements](#requirements) section below.


## Requirements

- Python 3.8+

Install dependencies with:
```bash
pip install -r requirements.txt
```

## Repository Structure

```
.
├──📂 edep-sim-truth-studies/
|  ├─📂 outputs/
|  ├─📂 plots/
|  └─ edep_read.py
├──📂 input-root-files/
|  ├─📂 CAF
|  └─📂 EDEP-SIM
├── requirements.txt
├── .gitignore
└── README.md
```

## EDep Sim Truth Studies

### For analysis

This repository was not nessecerily made to analyse and make plots. Instead it is used to store data in an output root file so plotting can be made easier and much quicker for finding correct binnings for various particles. For now this is still a work in progress. 

To make these root files you must clone the [EDep-Sim](https://github.com/DUNE/edep-sim/tree/master) repository and then build it. This can be done very easily using the setup.sh bash script.

```bash
$ cd edep-sim-truth-studies
$ source setup.sh
```

This both clones the EDep-Sim repo and builds it so that reading EDep-Sim files can be done using the python script. Now to create the output root file you can simply run:

```bash
$ python3 edep_read.py
```

For now, if you run it as is, it will create 2 directories within edep-sim-truth-studies; plots and outputs.

The `plots/` directory will eventually be discontinued but for now gives us a tool to make changes whilst these tools are still in development. The `outputs/` diretory will deposit the output directory file. For now the output files will be written over each other so if there is a file you would like, please rename it in the python script or after creating it.

### Using edep-sim-tools

There is currently only one tool within `edep-sim-tools` which is the `point_walk.C`. This allows you to see the different materials that compose the current ND materials in use through sending a point through the whole complex and see the different materials that are interacted with. This is how we managed to get the exact measurements to find distance to wall in `edep_read.py`.

To run it you must go into the file 

```C
void point_walk(double  x=0, double  y=0, double  z=100000,
                double px=0, double py=0, double pz=-1,
```

Here you can change the position and direction you start from to take the point walk.

Once you have made these changes you then run it.

```bash
root point_walk.C
```

Which should give you an output that looks like this:

```bash
------------------------------------------------------------------------------------------------------------------------------------------
start[  0] {   0.00000,10000.00000, 11403.00000} ds=    0.00000 enter volWorld_PV_1          of Rock            [in no-mom                ]
point[  1] {   0.00000,10000.00000, 11403.00000} ds=    0.00000 enter volDetEnclosure_PV_0   of Air             [in rockBox_lv_PV_0       ]
point[  2] {   0.00000,1247.77000, 11403.00000} ds= 8752.23000 enter volTMS_PV_0            of Air             [in volDetEnclosure_PV_0  ]
point[  3] {   0.00000,1158.77000, 11403.00000} ds=   89.00000 enter modulelayervol2_PV_0   of Air             [in volTMS_PV_0           ]
point[  4] {   0.00000,-3863.23000, 11403.00000} ds= 5022.00000 enter volTMS_PV_0            of Air             [in volDetEnclosure_PV_0  ]
point[  5] {   0.00000,-5652.23000, 11403.00000} ds= 1789.00000 enter volDetEnclosure_PV_0   of Air             [in rockBox_lv_PV_0       ]
point[  6] {   0.00000,-6614.73000, 11403.00000} ds=  962.50000 enter rockBox_lv_PV_0        of Rock            [in volWorld_PV_1         ]
point[  7] {   0.00000,-11614.73000, 11403.00000} ds= 5000.00000 enter volWorld_PV_1          of Rock            [in no-mom                ]
point[  8] {   0.00000,-300000.00000, 11403.00000} ds=288385.27000 enter volWorld_PV_1          of Rock            [in no-mom                ]
------------------------------------------------------------------------------------------------------------------------------------------
```