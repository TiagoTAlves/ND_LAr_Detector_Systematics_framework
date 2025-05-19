import ROOT
import os
import pandas as pd
import numpy as np
import uproot
from edep_funcs import (
    calc_distance_to_wall,
    pdg_to_particle_mass,
    update_parent_to_tracks,
    is_contained,
    ND_WALLS,
    sum_by_keys,
    accumulate_energy_deposit,
    accumulate_track_length,
    stop_position,
    is_contained_TMS_matching
)

ROOT.gSystem.Load("./edep-sim/edep-gcc-11-x86_64-redhat-linux/io/libedepsim_io.so")
ROOT.gInterpreter.ProcessLine('#include "./edep-sim/edep-gcc-11-x86_64-redhat-linux/include/EDepSim/TG4Event.h"')

root_dir = '../input-root-files/EDEP-SIM/'
root_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.root')]

detectors = [
    (b'TPCActive_shape', "E_vis_TPC", "track_length_TPC"),
    (b'volTMS', "E_vis_TMS", "track_length_TMS"),
    (b'muTag', "E_vis_muTag", "track_length_muTag")
]

data_particles = {
    "run_id": [],
    "interaction_id": [],
    "track_id": [],
    "pdg": [],
    "E_vis_TPC": [],
    "E_vis_TMS": [],
    "E_vis_muTag": [],
    "E_vis": [],
    "E": [],
    "E_kin": [],
    "start_x": [],
    "start_y": [],
    "start_z": [],
    "stop_x": [],
    "stop_y": [],
    "stop_z": [],
    "px": [],
    "py": [],
    "pz": [],
    "p": [],
    "theta": [],
    "phi": [],
    "track_length_TPC": [],
    "track_length_TMS": [],
    "track_length_muTag": [],
    "track_length": [],
    "d_wall_TPC": [],
    "is_contained_TPC": [],
    "is_contained_TMS_matching": [],
    "is_contained": []
}

energy_deposit_by_key = {}
track_length_by_key = {}
parent_to_tracks = {}
stop_pos = {}
is_all_contained_TPC = {}
is_contained_TMS_matched = {}

for root_file in root_files:

    tFile = ROOT.TFile.Open(root_file)
    events = tFile.Get("EDepSimEvents")
    event = ROOT.TG4Event()
    events.SetBranchAddress("Event", ROOT.AddressOf(event))
    print(f"Reading {root_file}")

    for i in range(events.GetEntries()):
        print(f"Processing event {i+1}/{events.GetEntries()}")
        events.GetEntry(i)
        
        for traj in event.Trajectories:
            update_parent_to_tracks(traj, parent_to_tracks)
            stop_pos[i,traj.GetTrackId()] = stop_position(traj, traj.GetTrackId())

        for segments in event.SegmentDetectors:
            if not hasattr(segments, 'second'):
                continue
            accumulate_energy_deposit(segments, i, energy_deposit_by_key)
            accumulate_track_length(segments, i, track_length_by_key)
                
        for prim in event.Primaries:
            x=prim.GetPosition().X()
            y=prim.GetPosition().Y()
            z=prim.GetPosition().Z()

            for particle in prim.Particles:
                E_minus_rest_mass = particle.GetMomentum().Energy() - pdg_to_particle_mass(particle.GetPDGCode())
                all_tracks = parent_to_tracks.get(particle.GetTrackId(), [particle.GetTrackId()])

                run_id = event.RunId
                interaction_id = prim.GetInteractionNumber() + 1
                track_id = particle.GetTrackId()
                pdg = particle.GetPDGCode()
                px = particle.GetMomentum().Px()
                py = particle.GetMomentum().Py()
                pz = particle.GetMomentum().Pz()
                p = particle.GetMomentum().P()
                theta = particle.GetMomentum().Theta()
                phi = np.arcsin(py/p) if p != 0 else 0
                
                is_all_contained_TPC[track_id] = 1
                is_contained_TMS_matched[track_id] = 0
                for tid in all_tracks:
                    if not is_contained(x, y, z, stop_pos, i, tid, detector="TPC"):
                        is_all_contained_TPC[track_id] = 0
                        if is_contained_TMS_matching(x, y, z, stop_pos, i, tid, detector="TMS") or energy_deposit_by_key.get((i, tid, b'volTMS'), 0) > 0:
                            is_contained_TMS_matched[track_id] = 1
                            break
                        else:
                            continue
                    

                energy_sums = {}
                track_length_sums = {}
                for det, ekey, tkey in detectors:
                    energy_sums[ekey] = sum(energy_deposit_by_key.get((i, tid, det), 0) for tid in all_tracks)
                    track_length_sums[tkey] = sum(track_length_by_key.get((i, tid, det), 0) for tid in all_tracks)
                
                data_particles["run_id"].append(run_id)
                data_particles["interaction_id"].append(interaction_id)
                data_particles["track_id"].append(track_id)
                data_particles["pdg"].append(pdg)
                data_particles["start_x"].append(x)
                data_particles["start_y"].append(y)
                data_particles["start_z"].append(z)
                data_particles["stop_x"].append(stop_pos[i, track_id][0])
                data_particles["stop_y"].append(stop_pos[i, track_id][1])
                data_particles["stop_z"].append(stop_pos[i, track_id][2])
                data_particles["px"].append(px)
                data_particles["py"].append(py)
                data_particles["pz"].append(pz)
                data_particles["p"].append(p)
                data_particles["theta"].append(theta)
                data_particles['phi'].append(phi)
                data_particles["d_wall_TPC"].append(calc_distance_to_wall(x, y, z, theta, phi, detector="TPC"))
                data_particles["E"].append(particle.GetMomentum().Energy())
                data_particles["E_kin"].append(E_minus_rest_mass)
                data_particles["E_vis_TPC"].append(energy_sums["E_vis_TPC"])
                data_particles["E_vis_TMS"].append(energy_sums["E_vis_TMS"])
                data_particles["E_vis_muTag"].append(energy_sums["E_vis_muTag"])
                data_particles["E_vis"].append(
                    sum(energy_sums[k] for k in energy_sums)
                )
                data_particles["track_length_TPC"].append(track_length_sums["track_length_TPC"])
                data_particles["track_length_TMS"].append(track_length_sums["track_length_TMS"])
                data_particles["track_length_muTag"].append(track_length_sums["track_length_muTag"])
                data_particles["track_length"].append(
                    sum(track_length_sums[k] for k in track_length_sums)
                )
                data_particles["is_contained_TPC"].append(is_all_contained_TPC[track_id])
                data_particles["is_contained_TMS_matching"].append(is_contained_TMS_matched[track_id])
                data_particles["is_contained"].append(1 if is_all_contained_TPC[track_id] == 1 or is_contained_TMS_matched[track_id] == 1 else 0)

    parent_to_tracks.clear()
    energy_deposit_by_key.clear()
    track_length_by_key.clear()
    stop_pos.clear()
    is_all_contained_TPC.clear()
    is_contained_TMS_matched.clear()

df = pd.DataFrame(data_particles)

df.set_index(["run_id", "interaction_id", "track_id"], inplace=True, drop=True)

os.makedirs("plots", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
with uproot.recreate("outputs/edep_sim_output.root") as f:
    f["events"] = df.reset_index()

print(df.head(15))

