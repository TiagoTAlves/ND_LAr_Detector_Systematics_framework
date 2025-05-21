import numpy as np
from particle import Particle

ND_WALLS = { # numbers from: https://github.com/DUNE/dune-tms/blob/main/config/TMS_Default_Config.toml 
    "TPC": {
        "x_min": -3478.48,
        "x_max": 3478.48,
        "y_min": -2166.71,
        "y_max": 829.282,
        "z_min": 4179.24,
        "z_max": 9135.88,
    },
    "TMS": {
        "x_min": -3300,
        "x_max": 3300,
        "y_min": -2850,
        "y_max": 160,
        "z_min": 11362,
        "z_max": 18314,
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

def stop_position(traj, track_id):
    if traj.GetTrackId() != track_id:
        return None
    stop_pos = None
    for point in traj.Points:
        pos = point.GetPosition()
        stop_pos = (pos.X(), pos.Y(), pos.Z())
    return stop_pos

def is_contained(x, y, z, stop_pos, i, track_id, detector="TPC"):
    walls = ND_WALLS[detector]
    stop = stop_pos.get((i, track_id))
    if stop is None:
        return 0
    if (walls["x_min"] <= x <= walls["x_max"] and
        walls["y_min"] <= y <= walls["y_max"] and
        walls["z_min"] <= z <= walls["z_max"] and
        walls["x_min"] <= stop[0] <= walls["x_max"] and
        walls["y_min"] <= stop[1] <= walls["y_max"] and
        walls["z_min"] <= stop[2] <= walls["z_max"]):
        return 1
    else:
        return 0

def is_contained_TMS_matching(stop_pos, i, track_id, detector="TMS"):
    walls = ND_WALLS[detector]
    stop = stop_pos.get((i, track_id))
    if stop is None:
        return 0
    if (walls["x_min"] <= stop[0] <= walls["x_max"] and
        walls["y_min"] <= stop[1] <= walls["y_max"] and
        walls["z_min"] <= stop[2] <= walls["z_max"]):
        return 1
    else:
        return 0

def update_parent_to_tracks(traj, parent_to_tracks):
    parent_id = traj.GetParentId()
    track_id = traj.GetTrackId()
    pdg = traj.GetPDGCode()
    if pdg in {12, 14, 16, -12, -14, -16, 2112, -2112}: # remove invisible(ish) particles
        return
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

def accumulate_energy_deposit(segments, tid, energy_deposit_by_key):
    detector_id = segments.first
    for seg in segments.second:
        energy_deposit = seg.GetEnergyDeposit()
        key = (tid, seg.GetPrimaryId(), detector_id)
        energy_deposit_by_key[key] = energy_deposit_by_key.get(key, 0) + energy_deposit

def accumulate_track_length(segments, event_idx, track_length_by_key):
    detector_id = segments.first
    for seg in segments.second:
        track_length = seg.GetTrackLength()
        key = (event_idx, seg.GetPrimaryId(), detector_id)
        track_length_by_key[key] = track_length_by_key.get(key, 0) + track_length

def containment(all_tracks, track_id, energy_deposit_by_key, stop_pos, i, x, y, z, is_all_contained_TPC, is_contained_TMS_matched):
    for tid in all_tracks:
        if not is_contained(x, y, z, stop_pos, i, tid, detector="TPC"):
            is_all_contained_TPC[track_id] = 0
            if is_contained_TMS_matching(stop_pos, i, tid, detector="TMS") or energy_deposit_by_key.get((i, tid, b'volTMS'), 0) > 250: #After conversation with Asa, Track reconstruction requires at least 14 steel hits for reconstruction to work
                is_contained_TMS_matched[track_id] = 1
                break
            else:
                continue
    return is_all_contained_TPC[track_id], is_contained_TMS_matched[track_id]