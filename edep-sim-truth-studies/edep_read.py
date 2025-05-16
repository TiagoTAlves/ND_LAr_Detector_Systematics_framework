import ROOT
import os
import pandas as pd
import matplotlib.pyplot as plt
from particle import Particle
import numpy as np
import uproot

ROOT.gSystem.Load("./edep-sim/edep-gcc-11-x86_64-redhat-linux/io/libedepsim_io.so")
ROOT.gInterpreter.ProcessLine('#include "./edep-sim/edep-gcc-11-x86_64-redhat-linux/include/EDepSim/TG4Event.h"')

def pdg_to_particle_mass(pdg_code):
    try:
        particle = Particle.from_pdgid(pdg_code)
        if particle.mass == None:
            return 0
        else:
            return particle.mass  
    except Exception as e:
        return 0

def calc_distance_to_wall(x, y, z, theta, phi):
    nd_wall_x_min = -3500
    nd_wall_x_max = 4000
    nd_wall_y_min = -2182
    nd_wall_y_max = 1242
    nd_wall_z_min = 4158 
    nd_wall_z_max = 9182

    epsilon = 1e-8
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    distances = [
        (nd_wall_x_max - x) / (cos_phi * sin_theta) if abs(cos_phi * sin_theta) > epsilon else float('inf'),
        (nd_wall_x_min - x) / (cos_phi * sin_theta) if abs(cos_phi * sin_theta) > epsilon else float('inf'),
        (nd_wall_y_max - y) / (sin_phi * sin_theta) if abs(sin_phi * sin_theta) > epsilon else float('inf'),
        (nd_wall_y_min - y) / (sin_phi * sin_theta) if abs(sin_phi * sin_theta) > epsilon else float('inf'),
        (nd_wall_z_max - z) / cos_theta if abs(cos_theta) > epsilon else float('inf'),
        (nd_wall_z_min - z) / cos_theta if abs(cos_theta) > epsilon else float('inf'),
    ]
    positive_distances = [d for d in distances if d > 0]
    return min(positive_distances) if positive_distances else float('inf')

def calc_distance_to_wall_TMS(x, y, z, theta, phi):
    nd_wall_x_min = -3518
    nd_wall_x_max = 3518
    nd_wall_y_min = -3863
    nd_wall_y_max = 1158
    nd_wall_z_min = 11348 
    nd_wall_z_max = 18318

def update_parent_to_tracks(traj, parent_to_tracks):
    parent_id = traj.GetParentId()
    track_id = traj.GetTrackId()
    if parent_id == -1:
        parent_to_tracks[track_id] = [track_id]
        if track_id not in parent_to_tracks:
            parent_to_tracks[track_id] = []
    else:
        found_parent = False
        for orig_parent in parent_to_tracks:
            descendants = [orig_parent] + parent_to_tracks.get(orig_parent, [])
            if parent_id in descendants:
                parent_to_tracks[orig_parent].append(track_id)
                found_parent = True
                break
        if not found_parent:
            parent_to_tracks[parent_id] = [track_id]

def sum_by_keys(keys, data_dict):
    return sum(data_dict.get((i, tid, det), 0) for tid in all_tracks for det in keys)

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
    "x": [],
    "y": [],
    "z": [],
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
    "is_contained": [],
}

root_dir = '../input-root-files/EDEP-SIM/'
root_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.root')]
# root_files = [os.path.join(root_dir, "MicroProdN3p4_NDLAr_2E18_FHC.edep.nu.0000001.EDEPSIM.root")]
energy_deposit_by_key = {}
track_length_by_key = {}
parent_to_tracks = {}


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

        for segments in event.SegmentDetectors:
            if not hasattr(segments, 'second'):
                continue
            detector_id = segments.first

            for seg in segments.second:
                energy_deposit = seg.GetEnergyDeposit()
                track_length = seg.GetTrackLength()

                key = (i, seg.GetPrimaryId(), detector_id)
                energy_deposit_by_key[key] = energy_deposit_by_key.get(key, 0) + energy_deposit
                track_length_by_key[key] = track_length_by_key.get(key, 0) + track_length

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

                detectors = [
                    (b'TPCActive_shape', "E_vis_TPC", "track_length_TPC"),
                    (b'volTMS', "E_vis_TMS", "track_length_TMS"),
                    (b'muTag', "E_vis_muTag", "track_length_muTag")
                ]

                # Precompute energy and track length sums for each detector
                energy_sums = {}
                track_length_sums = {}
                for det, ekey, tkey in detectors:
                    energy_sums[ekey] = sum(energy_deposit_by_key.get((i, tid, det), 0) for tid in all_tracks)
                    track_length_sums[tkey] = sum(track_length_by_key.get((i, tid, det), 0) for tid in all_tracks)
                
                data_particles["run_id"].append(run_id)
                data_particles["interaction_id"].append(interaction_id)
                data_particles["track_id"].append(track_id)
                data_particles["pdg"].append(pdg)
                data_particles["x"].append(x)
                data_particles["y"].append(y)
                data_particles["z"].append(z)
                data_particles["px"].append(px)
                data_particles["py"].append(py)
                data_particles["pz"].append(pz)
                data_particles["p"].append(p)
                data_particles["theta"].append(theta)
                data_particles['phi'].append(phi)
                data_particles["d_wall_TPC"].append(calc_distance_to_wall(x, y, z, theta, phi))
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
                data_particles["is_contained"].append(1 if calc_distance_to_wall(x, y, z, theta, phi) > track_length_sums["track_length_TPC"] else 0)

    parent_to_tracks.clear()
    energy_deposit_by_key.clear()
    track_length_by_key.clear()




df = pd.DataFrame(data_particles)


df.set_index(["run_id", "interaction_id", "track_id"], inplace=True, drop=True)

os.makedirs("plots", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
with uproot.recreate("outputs/edep_sim_output.root") as f:
    f["events"] = df.reset_index()


theta_bins = [(0, 3.2)]
x_bins = [(-3500, 3500)]
y_bins = [(-2500, 1400)]
z_bins = [(4000, 9200)]

particle_species = df['pdg'].unique()

print(df.head(15))

for pdg in particle_species:
    df_pdg = df[df['pdg'] == pdg]
    for tmin, tmax in theta_bins:
        for xmin, xmax in x_bins:
            for ymin, ymax in y_bins:
                for zmin, zmax in z_bins:
                    cut = (
                        (df_pdg['theta'] >= tmin) & (df_pdg['theta'] < tmax) &
                        (df_pdg['x'] >= xmin) & (df_pdg['x'] < xmax) &
                        (df_pdg['y'] >= ymin) & (df_pdg['y'] < ymax) &
                        (df_pdg['z'] >= zmin) & (df_pdg['z'] < zmax)
                    )
                    df_cut = df_pdg[cut]
                    if len(df_cut) == 0:
                        continue
                    ratio = df_cut['E_vis'] / df_cut['E']
                    plt.figure(figsize=(6,4))
                    plt.hist(ratio, bins=50, range=(0,1.2), histtype='step', color='blue')
                    plt.xlabel('E_vis / E')
                    plt.ylabel('Counts')
                    plt.title(f'PDG={pdg}, θ=[{tmin},{tmax}), x=[{xmin},{xmax}), y=[{ymin},{ymax}), z=[{zmin},{zmax})')
                    plt.tight_layout()
                    fname = f"plots/hist_pdg{pdg}_theta{tmin}-{tmax}_x{xmin}-{xmax}_y{ymin}-{ymax}_z{zmin}-{zmax}.png"
                    plt.savefig(fname)
                    plt.close()

