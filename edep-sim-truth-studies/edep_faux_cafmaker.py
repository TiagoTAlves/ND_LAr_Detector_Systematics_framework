import ROOT
import os
import pandas as pd
import numpy as np
import uproot
import glob
from particle import Particle
from edep_funcs import (
    parse_args,
    calc_distance_to_wall,
    pdg_to_particle_mass,
    update_parent_to_tracks,
    is_contained,
    ND_WALLS,
    sum_by_keys,
    accumulate_energy_deposit,
    accumulate_track_length,
    stop_position,
    is_contained_TMS_matching,
    containment
)

def is_not_neutrino_or_neutron(pdg):
    return abs(pdg) not in [12, 14, 16, 2112]

def is_lepton(pdg):
    return abs(pdg) in {11, 13, 15}

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
    start_idx = chunk_index * chunk_size
    end_idx = start_idx + chunk_size
    root_files = all_root_files[start_idx:end_idx]
    if not root_files:
        print(f"No ROOT files found for chunk {chunk_index}. Exiting.")
        exit()
    print(f"Processing chunk {chunk_index}, files {start_idx} to {end_idx-1}")


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
    "is_contained": [],
    "E_vis/E_true": []
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
                phi = np.arccos(py/p) if p != 0 else 0 #CHECK PHI!!!!
                
                is_all_contained_TPC[track_id] = 1
                is_contained_TMS_matched[track_id] = 0
                containment(all_tracks, track_id, energy_deposit_by_key, stop_pos, i, x, y, z, is_all_contained_TPC, is_contained_TMS_matched)

                energy_sums = {}
                track_length_sums = {}
                for det, ekey, tkey in detectors:
                    energy_sums[ekey] = sum(energy_deposit_by_key.get((i, tid, det), 0) for tid in all_tracks)
                    track_length_sums[tkey] = sum(track_length_by_key.get((i, track_id, det), 0) for tid in all_tracks)
                
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
                data_particles["E_vis/E_true"].append(data_particles["E_vis"][-1] / data_particles["E"][-1] if data_particles["E"][-1] != 0 else 0)
    parent_to_tracks.clear()
    energy_deposit_by_key.clear()
    track_length_by_key.clear()
    stop_pos.clear()
    is_all_contained_TPC.clear()
    is_contained_TMS_matched.clear()

df = pd.DataFrame(data_particles)

df['is_lepton'] = df['pdg'].apply(is_lepton)

event_df_simple = (
    df.assign(
        is_contained_masked=df.apply(
            lambda row: row["is_contained"] if is_not_neutrino_or_neutron(row["pdg"]) else np.nan, axis=1
        )
    )
    .groupby(['run_id', 'interaction_id'])
    .agg(
        erec=("E_vis", "sum"),
        E=("E", lambda x: (
            df.loc[x.index].apply(
                lambda row: row["p"] if abs(row["pdg"]) > 1000 else row["E"],
                axis=1
            ).sum()
        )),        is_contained_masked=("is_contained_masked", lambda x: 1 if (x.dropna() == 1).all() else 0),
        LepE=('E', lambda x: x[df.loc[x.index, 'is_lepton']].sum()),
        HadE=('E', lambda x: x[~df.loc[x.index, 'is_lepton']].sum()),
        ePip=('E', lambda x: x[df.loc[x.index, 'pdg'] == 211].sum()),
        ePim=('E', lambda x: x[df.loc[x.index, 'pdg'] == -211].sum()),
        ePi0=('E', lambda x: x[df.loc[x.index, 'pdg'] == 111].sum()),
        eN=('E', lambda x: x[df.loc[x.index, 'pdg'] == 2112].sum()),
        eP=('E', lambda x: x[df.loc[x.index, 'pdg'] == 2212].sum()),
        erec_lep=('E_vis', lambda x: x[df.loc[x.index, 'is_lepton']].sum()),
        erec_had=('E_vis', lambda x: x[~df.loc[x.index, 'is_lepton']].sum()),
        eRecoPip=('E_vis', lambda x: x[df.loc[x.index, 'pdg'] == 211].sum()),
        eRecoPim=('E_vis', lambda x: x[df.loc[x.index, 'pdg'] == -211].sum()),
        eRecoPi0=('E_vis', lambda x: x[df.loc[x.index, 'pdg'] == 111].sum()),
        eRecoN=('E_vis', lambda x: x[df.loc[x.index, 'pdg'] == 2112].sum()),
        eRecoP=('E_vis', lambda x: x[df.loc[x.index, 'pdg'] == 2212].sum()),
    ).rename(columns={"is_contained_masked": "is_contained"})
)
event_df_simple["E_minus_erec"] = event_df_simple["E"] - event_df_simple["erec"]

sigma_tables = {}
for pdg_code in df['pdg'].unique():
    try:
        # Get the particle name using the Particle package
        particle = Particle.from_pdgid(pdg_code)
        # Use the name as in your CSVs, e.g. 'mu+', 'pi-', etc.
        particle_name = particle.name.replace(" ", "")
    except Exception:
        # If Particle can't resolve, fallback to pdg_code as string
        particle_name = str(pdg_code)
    # Try to find the CSV for this particle
    if pdg_code > 1000:
        csv_path = os.path.join(
            "/vols/dune/tta20/mach3/ND_LAr_Detector_Systematics_framework/edep-sim-truth-studies/plots/fit_vs_vars_2d/d_wall_TPC_vs_p_fit_not_contained",
            f"{particle_name}_fit_mu_sigma_d_wall_TPC_vs_p_E_kin_not_contained_fiducial.csv"
        )
    else:
        csv_path = os.path.join(
            "/vols/dune/tta20/mach3/ND_LAr_Detector_Systematics_framework/edep-sim-truth-studies/plots/fit_vs_vars_2d/d_wall_TPC_vs_p_fit_not_contained",
            f"{particle_name}_fit_sigma_d_wall_TPC_vs_p_E_not_contained_fiducial.csv"
        )
    if os.path.exists(csv_path):
        sigma_tables[pdg_code] = pd.read_csv(csv_path)
    else:
        sigma_tables[pdg_code] = 1.0

for pdg_code in df['pdg'].unique():
    df_part = df[df['pdg'] == pdg_code]
    df_part = df_part[['run_id', 'interaction_id', 'd_wall_TPC', 'p']]
    if sigma_tables.get(pdg_code) is None:
        event_df_simple[f'sigma_{particle_name}'] = 1
        continue
    sigma_table = sigma_tables[pdg_code]
    sigmas = []
    for idx, row in df_part.iterrows():
        if sigma_table.empty:
            continue
        bin_row = sigma_table.iloc[((sigma_table['d_wall_TPC_bin'] - row['d_wall_TPC']).abs().argsort()[:1])]
        if bin_row.empty:
            continue
        bin_row = bin_row.iloc[((bin_row['p_bin'] - row['p']).abs().argsort()[:1])]
        if bin_row.empty or 'sigma' not in bin_row:
            continue
        sigmas.append((row['run_id'], row['interaction_id'], float(bin_row['sigma'].iloc[0])))
    if sigmas:
        sigmas_df = pd.DataFrame(sigmas, columns=['run_id', 'interaction_id', f'sigma_{Particle.from_pdgid(pdg_code)}'])
        max_sigmas = sigmas_df.groupby(['run_id', 'interaction_id'])[f'sigma_{Particle.from_pdgid(pdg_code)}'].max().reset_index()
        event_df_simple = event_df_simple.merge(max_sigmas, on=['run_id', 'interaction_id'], how='left')
    else:
        event_df_simple[f'sigma_{Particle.from_pdgid(pdg_code)}'] = 1
event_df_simple = event_df_simple.fillna(1.0)

print(event_df_simple.head())
print(f"Created event-by-event DataFrame with {len(event_df_simple)} events.")
os.makedirs("plots", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
if not interactive:
    os.makedirs("logs", exist_ok=True)
output_file = f"outputs/cafs/edep_sim_{chunk_index}.CAF.root" if not interactive else "outputs/cafs/edep_sim_output_all.root"
with uproot.recreate(output_file) as f:
    f["caf"] = event_df_simple.reset_index()
