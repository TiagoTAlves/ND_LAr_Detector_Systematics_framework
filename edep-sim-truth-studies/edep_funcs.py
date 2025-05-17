import numpy as np
from particle import Particle

ND_WALLS = {
    "TPC": {
        "x_min": -3500,
        "x_max": 4000,
        "y_min": -2182,
        "y_max": 1242,
        "z_min": 4158,
        "z_max": 9182,
    },
    "TMS": {
        "x_min": -3518,
        "x_max": 3518,
        "y_min": -3863,
        "y_max": 1158,
        "z_min": 11348,
        "z_max": 18318,
    }
}

def pdg_to_particle_mass(pdg_code):
    try:
        particle = Particle.from_pdgid(pdg_code)
        if particle.mass is None:
            return 0
        else:
            return particle.mass
    except Exception:
        return 0

def calc_distance_to_wall(x, y, z, theta, phi, detector="TPC"):
    walls = ND_WALLS[detector]
    epsilon = 1e-8
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    distances = [
        (walls["x_max"] - x) / (cos_phi * sin_theta) if abs(cos_phi * sin_theta) > epsilon else float('inf'),
        (walls["x_min"] - x) / (cos_phi * sin_theta) if abs(cos_phi * sin_theta) > epsilon else float('inf'),
        (walls["y_max"] - y) / (sin_phi * sin_theta) if abs(sin_phi * sin_theta) > epsilon else float('inf'),
        (walls["y_min"] - y) / (sin_phi * sin_theta) if abs(sin_phi * sin_theta) > epsilon else float('inf'),
        (walls["z_max"] - z) / cos_theta if abs(cos_theta) > epsilon else float('inf'),
        (walls["z_min"] - z) / cos_theta if abs(cos_theta) > epsilon else float('inf'),
    ]
    positive_distances = [d for d in distances if d > 0]
    return min(positive_distances) if positive_distances else float('inf')

def inner(x, y, z, detector="TPC"):
    walls = ND_WALLS[detector]
    return (walls["x_min"] <= x <= walls["x_max"] and
            walls["y_min"] <= y <= walls["y_max"] and
            walls["z_min"] <= z <= walls["z_max"])

def is_contained(x, y, z, theta, phi, track_length, detector="TPC"):
    if not inner(x, y, z, detector):
        return False
    d_wall = calc_distance_to_wall(x, y, z, theta, phi, detector)
    return track_length <= d_wall

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

def sum_by_keys(keys, data_dict, all_tracks):
    return sum(data_dict.get((i, tid, det), 0) for tid in all_tracks for det in keys)

def accumulate_energy_deposit(segments, event_idx, energy_deposit_by_key):
    detector_id = segments.first
    for seg in segments.second:
        energy_deposit = seg.GetEnergyDeposit()
        key = (event_idx, seg.GetPrimaryId(), detector_id)
        energy_deposit_by_key[key] = energy_deposit_by_key.get(key, 0) + energy_deposit

def accumulate_track_length(segments, event_idx, track_length_by_key):
    detector_id = segments.first
    for seg in segments.second:
        track_length = seg.GetTrackLength()
        key = (event_idx, seg.GetPrimaryId(), detector_id)
        track_length_by_key[key] = track_length_by_key.get(key, 0) + track_length