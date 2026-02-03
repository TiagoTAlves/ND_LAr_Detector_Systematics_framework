# CAF Selection YAML Configuration Guide

This guide explains how to configure YAML files for selecting events or particles from CAF (Common Analysis Framework) ROOT files using `caf_read_yaml.py`. The script supports two modes: `event` and `particle`, each with specific cuts and output options.

## Prerequisites

- Python 3.8+

Install dependencies with:
```bash
pip install -r ../requirements.txt
```
To make selection files you must clone the [duneanaobj](https://github.com/DUNE/duneanaobj) repository and then build it. This can be done very easily using the setup.sh bash script.

```bash
$ source setup.sh
```

## General Structure

All YAML files must include the following top-level keys:

- `mode`: Select `"event"` or `"particle"` to specify the selection type.
- `detector`: Currently supports `"lar"` (LAr detector). Required for particle mode.
- `caf`: Configuration for CAF files, including:
  - `StandardRecordLibs`: Path to the directory containing StandardRecord libraries (required).
  - Either `root_dir`: Path to a directory containing multiple ROOT files, or `root_file`: Path to a single ROOT file.
- `output`: Output configuration:
  - `out_dir`: Directory to save the output ROOT file.
  - `out_filename`: Base name for the output file (do not include `.root` extension; the script appends chunk info).

## Event Mode (`mode: "event"`)

In event mode, selections are applied at the neutrino event level. Define cuts under `event_cuts` as a dictionary where keys are variable names and values are either:
- A list of allowed values (e.g., for categorical variables like PDG codes).
- A dictionary with `min` and/or `max` keys for numerical ranges.

### Supported Event Variables

The following variables can be used in `event_cuts` (based on the CAF StandardRecord structure):

- `mc_nupdg`: Neutrino PDG code (e.g., [-12, 12] for electron neutrinos).
- `E_tru`: True neutrino energy (GeV).
- `reco_vtx_x`, `reco_vtx_y`, `reco_vtx_z`: Reconstructed vertex positions (cm).
- `npip`: Number of charged pions.
- `nproton`: Number of protons.
- `nneutron`: Number of neutrons.
- `npim`: Number of negative pions.
- `npi0`: Number of neutral pions.
- `nprim`: Number of primary particles.
- `nsec`: Number of secondary particles.
- `nprefsi`: Number of pre-FSI particles.
- `mode`: Interaction mode (integer).
- `q0`, `q3`: Q0 and Q3 values.
- `bjorkenX`: Bjorken x.
- `inelasticty`: Inelasticity.
- `nu_vtx_x`, `nu_vtx_y`, `nu_vtx_z`: True neutrino vertex positions.
- `nu_mom_x`, `nu_mom_y`, `nu_mom_z`: True neutrino momentum components.
- `E_reco_calo`, `E_reco_lep_calo`, `E_reco_mu_range`, `E_reco_mu_mcs`, `E_reco_e_calo`, `E_reco_regcnn`: Reconstructed energies from various methods.
- `truth_overlap`: Truth overlap value.

Example:
```yaml
event_cuts:
  mc_nupdg: [-12, 12]
  E_tru:
    min: 0.0
    max: 10.0
  reco_vtx_z:
    min: 450.0
    max: 900.0
  npip:
    min: 0
    max: 2
```

## Particle Mode (`mode: "particle"`)

In particle mode, selections are applied at the individual particle level within events. Define cuts under `particle_cuts` similarly to event cuts.

### Supported Particle Variables

The following variables can be used in `particle_cuts`:

- `pdg`: Particle PDG code (e.g., [-211, 211] for pions).
- `E`: True energy (GeV).
- `E_kin`: True kinetic energy (GeV).
- `px`, `py`, `pz`: True momentum components (GeV/c).
- `start_x`, `start_y`, `start_z`: True start positions (cm).
- `end_x`, `end_y`, `end_z`: True end positions (cm).
- `is_contained`: Boolean indicating if the particle is contained.
- `reco_contained`: Reconstructed containment.
- `reco_pdg`: Reconstructed PDG.
- `reco_E`: Reconstructed energy.
- `reco_px`, `reco_py`, `reco_pz`: Reconstructed momentum components.
- `reco_theta`, `reco_phi`: Reconstructed angles.
- `reco_start_x`, `reco_start_y`, `reco_start_z`: Reconstructed start positions.
- `reco_end_x`, `reco_end_y`, `reco_end_z`: Reconstructed end positions.
- `reco_length`: Reconstructed track length.
- `E_true_ratio_common`: Ratio of true energy to reconstructed.
- `E_kin_ratio_common`: Ratio of true kinetic energy to reconstructed.
- `reco_truth_overlap`: Truth overlap for reconstruction.

Example:
```yaml
particle_cuts:
  pdg: [-211, 211]
  E:
    min: 3.0
    max: 5.0
  px:
    min: 0.2
    max: 10.0
```

## Running the Script

To run `caf_read_yaml.py`, use command-line arguments for chunk processing (if using `root_dir`):
- `--chunk_index` or `--chunk`: Index of the chunk to process (0-based).
- `--chunk_size` or `--chunksize`: Number of files per chunk.
- `--selections_path` or `--selections`: Path to the YAML file.
- `--interactive`: Flag for processing all files in interactive mode (default False).

Example command:
```
python3 caf_read_yaml.py --chunk=0 --chunksize=10 --selections=selections_events.yaml
```

The script applies the specified cuts, filters the data, and saves a ROOT file with the selected events or particles. For event mode, output is saved to `outputs/cafs/neutrino/neutrino_chunk_{chunk_index}.root`. For particle mode, uses the `output` config.

## Notes

- Ensure paths in `caf.StandardRecordLibs` and file paths are absolute or relative to the working directory.
- For `root_dir`, the script processes files in sorted order and supports chunking for parallel processing.
- Output includes a filtered DataFrame saved as a ROOT tree and a text dump for verification (for particle mode).
- Refer to `caf_read_yaml.py` and `caf_funcs.py` for implementation details on supported variables and cut application.
- The script uses uproot for ROOT file handling and pandas for data manipulation.