import ROOT
import os
import pandas as pd
import numpy as np
import uproot
import glob
from edep_funcs import (
    parse_args,
    is_contained,
    ND_WALLS,
    stop_position
)

def is_muon(pdg):
    """Check if the particle is a muon (mu- or mu+)."""
    return abs(pdg) == 13

stop_pos = {}

chunk_index, chunk_size, interactive = parse_args()

ROOT.gSystem.Load("./edep-sim/edep-gcc-11-x86_64-redhat-linux/io/libedepsim_io.so")
ROOT.gInterpreter.ProcessLine('#include "./edep-sim/edep-gcc-11-x86_64-redhat-linux/include/EDepSim/TG4Event.h"')

root_dir = '/vols/dune/tta20/mach3/ND_LAr_Detector_Systematics_framework/input-root-files/EDEP-SIM/'
all_root_files = sorted(glob.glob(os.path.join(root_dir, '*EDEPSIM.root')))
start_idx = chunk_index * chunk_size
end_idx = start_idx + chunk_size
root_files = all_root_files[start_idx:end_idx]

if interactive:
    print("Interactive mode: processing all ROOT files")
    root_files = all_root_files
else:
    if not root_files:
        print(f"No ROOT files found for chunk {chunk_index}. Exiting.")
        exit()
    print(f"Processing chunk {chunk_index}, files {start_idx} to {end_idx-1}")

data_particles = {
    "run_id": [],
    "interaction_id": [],
    "track_id": [],
    "px_mu": [],
    "py_mu": [],
    "pz_mu": [],
    "px_nu": [],
    "py_nu": [],
    "pz_nu": [],    
    "E_mu": [],
    "E_nu": [],
    "is_contained": []
}

for root_file in root_files:
    tFile = ROOT.TFile.Open(root_file)
    events = tFile.Get("EDepSimEvents")
    event = ROOT.TG4Event()
    events.SetBranchAddress("Event", ROOT.AddressOf(event))
    print(f"Reading {root_file}")

    for i in range(events.GetEntries()):
        print(f"Processing event {i+1}/{events.GetEntries()}")
        events.GetEntry(i)
        px_nu, py_nu, pz_nu, E_nu = None, None, None, None

        for traj in event.Trajectories:
            stop_pos[i,traj.GetTrackId()] = stop_position(traj, traj.GetTrackId())

            if traj.GetTrackId() == -2:
                print(f"PDG code of neutrino: {traj.GetPDGCode()}")      
                px_nu = traj.GetInitialMomentum().Px()
                py_nu = traj.GetInitialMomentum().Py()
                pz_nu = traj.GetInitialMomentum().Pz()
                E_nu = traj.GetInitialMomentum().Energy()
                print(f"Neutrino momentum: px={px_nu}, py={py_nu}, pz={pz_nu}, E={E_nu}")

        for prim in event.Primaries:
            x = prim.GetPosition().X()
            y = prim.GetPosition().Y()
            z = prim.GetPosition().Z()

            print(prim.info)0

            # Process only the primary muon
            primary_muon = None
            for particle in prim.Particles:
                pdg = particle.GetPDGCode()
                track_id = particle.GetTrackId()
                if is_muon(pdg):
                    primary_muon = particle
                    break  # Only process the first muon

            if primary_muon is None:
                continue  # Skip the event if no primary muon is found

            px_mu = particle.GetMomentum().Px()
            py_mu = particle.GetMomentum().Py()
            pz_mu = particle.GetMomentum().Pz()
            E_mu = particle.GetMomentum().Energy()
            print(f"Primary muon momentum: px={px_mu}, py={py_mu}, pz={pz_mu}, E={E_mu}")
            # Calculate containment
            is_contained_muon = is_contained(
                x, y, z,
                stop_pos, i, track_id
            )
            # Append muon data
            data_particles["run_id"].append(event.RunId)
            data_particles["interaction_id"].append(prim.GetInteractionNumber() + 1)
            data_particles["px_mu"].append(px_mu)
            data_particles["py_mu"].append(py_mu)
            data_particles["pz_mu"].append(pz_mu)
            data_particles["px_nu"].append(px_nu)
            data_particles["py_nu"].append(py_nu)
            data_particles["pz_nu"].append(pz_nu)
            data_particles["E_mu"].append(E_nu)
            data_particles["E_nu"].append(E_nu)
            data_particles["is_contained"].append(is_contained_muon)

df = pd.DataFrame(data_particles)

# Group by event and keep only events with at least one muon
muon_event_df = (
    df.groupby(['run_id', 'interaction_id'])
    .agg(
        muon_px=('px_mu', 'first'),  # Take the first muon in the event
        muon_py=('py_mu', 'first'),
        muon_pz=('pz_mu', 'first'),
        muon_E=('E_mu', 'first'),
        is_contained=('is_contained', 'max')  # If any muon is contained, mark the event as contained
    )
    .reset_index()
)

# Write the filtered DataFrame to the CAF
print(muon_event_df.head())
print(f"Created muon-only event-by-event DataFrame with {len(muon_event_df)} events.")

os.makedirs("plots", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
if not interactive:
    os.makedirs("logs", exist_ok=True)

output_file = f"outputs/q0q3/edep_sim_q0q3_{chunk_index}.CAF.root" if not interactive else "outputs/q0q3/edep_sim_q0q3.CAF.root"
with uproot.recreate(output_file) as f:
    f["caf"] = muon_event_df
