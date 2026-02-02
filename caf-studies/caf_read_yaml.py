import sys
import yaml
import os
import ROOT
import glob
import pandas as pd
import uproot
import numpy as np
from caf_funcs import (
    parse_args_yaml, 
    MaCh3Modes_lookup, 
    is_contained, 
    pdg_to_particle_mass, 
    E_method_lookup,
    compute_E_true_ratio, 
    compute_E_kin_ratio, 
    load_config, 
    load_standard_record_libs,
    apply_particle_cuts,
    apply_event_cuts,
)

chunk_index, chunk_size, selections_path, interactive = parse_args_yaml()
config = load_config(selections_path)
caf_cfg = config.get("caf", {})
StandardRecordLibs = caf_cfg.get("StandardRecordLibs")
mode = config.get("mode")
detector = config.get("detector")


if StandardRecordLibs is None:
    raise ValueError("Location of StandardRecordLibs must be set in selections.yaml")

load_standard_record_libs(StandardRecordLibs)


if "root_file" in caf_cfg:
    root_file = caf_cfg.get("root_file")
    root_files = [root_file]
elif "root_dir" in caf_cfg:
    root_dir = caf_cfg.get("root_dir")
    all_root_files = sorted([
        os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.root')
    ])
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
else:
    raise ValueError("caf.root_dir or caf.root_file must be set in selections.yaml")


if mode == "event":
    data = {
        "ID": [],
        "spill_ID": [],
        "nu_ID": [],
        "mc_nupdg": [],
        "mc_nupdg_unosc": [],
        "E_tru": [],
        "E_reco_calo": [],
        "E_reco_lep_calo": [],
        "E_reco_mu_range": [],
        "E_reco_mu_mcs": [],
        "E_reco_e_calo": [],
        "E_reco_regcnn": [],
        "mode": [],
        "mode_name": [],
        "q0": [],
        "q3": [],
        "bjorkenX": [],
        "inelasticty": [],
        "nu_vtx_x": [],
        "nu_vtx_y": [],
        "nu_vtx_z": [],
        "nu_mom_x": [],
        "nu_mom_y": [],
        "nu_mom_z": [],
        "nproton": [],
        "nneutron": [],
        "npip": [],
        "npim": [],
        "npi0": [],
        "nprim": [],
        "nsec": [],
        "nprefsi": [],
        "reco_vtx_x": [],
        "reco_vtx_y": [],
        "reco_vtx_z": [],
        "truth_overlap": [],
    }

elif mode == "particle":
    data = {
        "ID": [],
        "idx": [],
        "part_type": [],
        "part_idx": [],
        "pdg": [],
        "E": [],
        "E_kin": [],
        "start_x": [],
        "start_y": [],
        "start_z": [],
        "end_x": [],
        "end_y": [],
        "end_z": [],
        "px": [],
        "py": [],
        "pz": [],
        "is_contained": [],
        "E_true_ratio_common": [],
        "E_kin_ratio_common": [],
        "reco_contained": [],
        "reco_pdg": [],
        "reco_E_method": [],
        "reco_px": [],
        "reco_py": [],
        "reco_pz": [],
        "reco_contained": [],    
        "reco_theta": [],
        "reco_phi": [],
        "reco_E": [],
        "reco_start_x": [],
        "reco_start_y": [],
        "reco_start_z": [],
        "reco_end_x": [],
        "reco_end_y": [],
        "reco_end_z": [],
        "reco_length": [],
        "common_dlp_truth_overlap": [],
        "nd_lar_dlp_truth_overlap": [],
    }

else:
    raise ValueError(f"Invalid mode '{mode}' in selections.yaml. Use 'neutrino' or 'particle' only.")

if mode == "event":
    for root_file in root_files:
        tFile = ROOT.TFile.Open(root_file)
        tree = tFile.Get("cafTree")
        record = ROOT.caf.StandardRecord()
        tree.SetBranchAddress("rec", record)
        
        nspills = tree.GetEntries()
        print(f"Processing file: {root_file}, Total entries: {nspills}")

        for i in range(nspills):
                    tree.GetEntry(i)
                    mc = record.mc
                    common = record.common
                    spill_index_to_data_row = {}
                    for j in range(mc.nnu):
                        current_row = len(data["ID"])
                        spill_index_to_data_row[j] = current_row
                        data["ID"].append(mc.nu[j].id)
                        data["spill_ID"].append(i)
                        data["nu_ID"].append(j)
                        data["mc_nupdg"].append(mc.nu[j].pdg)
                        data["mc_nupdg_unosc"].append(mc.nu[j].pdgorig)
                        data["E_tru"].append(mc.nu[j].E)
                        data["mode"].append(mc.nu[j].mode)
                        data["mode_name"].append(MaCh3Modes_lookup(mc.nu[j].mode))
                        data["q0"].append(mc.nu[j].q0)
                        data["q3"].append(mc.nu[j].Q2)
                        data["bjorkenX"].append(mc.nu[j].bjorkenX)
                        data["inelasticty"].append(mc.nu[j].inelasticity)
                        data["nu_vtx_x"].append(mc.nu[j].vtx.x)
                        data["nu_vtx_y"].append(mc.nu[j].vtx.y)
                        data["nu_vtx_z"].append(mc.nu[j].vtx.z)
                        data["nu_mom_x"].append(mc.nu[j].momentum.x)
                        data["nu_mom_y"].append(mc.nu[j].momentum.y)
                        data["nu_mom_z"].append(mc.nu[j].momentum.z)
                        data["nproton"].append(mc.nu[j].nproton)
                        data["nneutron"].append(mc.nu[j].nneutron)
                        data["npip"].append(mc.nu[j].npip)
                        data["npim"].append(mc.nu[j].npim)
                        data["npi0"].append(mc.nu[j].npi0)
                        data["nprim"].append(mc.nu[j].nprim)
                        data["nsec"].append(mc.nu[j].nsec)
                        data["nprefsi"].append(mc.nu[j].nprefsi)
                        data["reco_vtx_x"].append(np.nan)
                        data["reco_vtx_y"].append(np.nan)
                        data["reco_vtx_z"].append(np.nan)
                        data["truth_overlap"].append(-1.0) # Use -1.0 as "no data" baseline
                        data["E_reco_calo"].append(np.nan)
                        data["E_reco_lep_calo"].append(np.nan)
                        data["E_reco_mu_range"].append(np.nan)
                        data["E_reco_mu_mcs"].append(np.nan)
                        data["E_reco_e_calo"].append(np.nan)
                        data["E_reco_regcnn"].append(np.nan)

                    for j in range(common.ixn.ndlp):
                        dlp_obj = common.ixn.dlp[j]
                        for k in range(len(dlp_obj.truth)):
                            matched_nu_id = int(dlp_obj.truth[k])
                            overlap = dlp_obj.truthOverlap[k]
                            if matched_nu_id in spill_index_to_data_row:
                                target_row = spill_index_to_data_row[matched_nu_id]
                                if overlap > data["truth_overlap"][target_row]:
                                    data["E_reco_calo"][target_row] = dlp_obj.Enu.calo
                                    data["E_reco_lep_calo"][target_row] = dlp_obj.Enu.lep_calo
                                    data["E_reco_mu_range"][target_row] = dlp_obj.Enu.mu_range
                                    data["E_reco_mu_mcs"][target_row] = dlp_obj.Enu.mu_mcs
                                    data["E_reco_e_calo"][target_row] = dlp_obj.Enu.e_calo
                                    data["E_reco_regcnn"][target_row] = dlp_obj.Enu.regcnn
                                    data["reco_vtx_x"][target_row] = dlp_obj.vtx.x
                                    data["reco_vtx_y"][target_row] = dlp_obj.vtx.y
                                    data["reco_vtx_z"][target_row] = dlp_obj.vtx.z
                                    data["truth_overlap"][target_row] = overlap
        tFile.Close()

    df = pd.DataFrame(data)
    df["truth_overlap"] = df["truth_overlap"].replace(-1.0, np.nan)
    event_cuts = config.get("event_cuts")

    df_filtered = apply_event_cuts(df, event_cuts)
    
    output_dir = "outputs/cafs/neutrino"
    os.makedirs(output_dir, exist_ok=True)
    out_file = f"{output_dir}/neutrino_chunk_{chunk_index}.root"
    with uproot.recreate(out_file) as f:
        f["neutrino_tree"] = df_filtered
    print(f"Saved to {out_file}")

elif mode == "particle":
    for root_file in root_files:
        tFile = ROOT.TFile.Open(root_file)
        tree = tFile.Get("cafTree")
        record = ROOT.caf.StandardRecord()
        tree.SetBranchAddress("rec", record)

        nspills = tree.GetEntries()
        print(f"Processing file: {root_file}, Total entries: {nspills}")
        
        for i in range(nspills):
            tree.GetEntry(i)
            mc = record.mc
            common = record.common
            nd = record.nd
            truth_map = {}

            for j in range(mc.nnu):
                def add_truth_particle(p_obj, p_type, p_idx):
                    row_idx = len(data["ID"])
                    truth_map[(j, p_type, p_idx)] = row_idx
                    
                    mass = pdg_to_particle_mass(p_obj.pdg)/1000
                    data["ID"].append(mc.nu[j].id)
                    data["idx"].append(j)
                    data["part_type"].append(p_type)
                    data["part_idx"].append(p_idx)
                    data["pdg"].append(p_obj.pdg)
                    data["E"].append(p_obj.p.E)
                    data["E_kin"].append(p_obj.p.E - mass)
                    data["start_x"].append(p_obj.start_pos.x)
                    data["start_y"].append(p_obj.start_pos.y)
                    data["start_z"].append(p_obj.start_pos.z)
                    data["end_x"].append(p_obj.end_pos.x)
                    data["end_y"].append(p_obj.end_pos.y)
                    data["end_z"].append(p_obj.end_pos.z)
                    data["px"].append(p_obj.p.px)
                    data["py"].append(p_obj.p.py)
                    data["pz"].append(p_obj.p.pz)
                    data["is_contained"].append(is_contained(p_obj.start_pos.x, p_obj.start_pos.y, p_obj.start_pos.z,
                                                           p_obj.end_pos.x, p_obj.end_pos.y, p_obj.end_pos.z, detector="TPC"))
                    data["common_dlp_truth_overlap"].append(-1.0)
                    data["reco_contained"].append(np.nan)
                    data["reco_pdg"].append(np.nan)
                    data["reco_px"].append(np.nan)
                    data["reco_py"].append(np.nan)
                    data["reco_pz"].append(np.nan)
                    data["reco_E_method"].append("None")
                    data["reco_theta"].append(np.nan)
                    data["reco_phi"].append(np.nan)
                    data["reco_E"].append(np.nan)
                    data["reco_start_x"].append(np.nan)
                    data["reco_start_y"].append(np.nan)
                    data["reco_start_z"].append(np.nan)
                    data["reco_end_x"].append(np.nan)
                    data["reco_end_y"].append(np.nan)
                    data["reco_end_z"].append(np.nan)
                    data["reco_length"].append(np.nan)
                    data["nd_lar_dlp_truth_overlap"].append(-1.0)


                for k in range(mc.nu[j].nprim): add_truth_particle(mc.nu[j].prim[k], 1, k)
                for k in range(mc.nu[j].nprefsi): add_truth_particle(mc.nu[j].prefsi[k], 2, k)
                for k in range(mc.nu[j].nsec): add_truth_particle(mc.nu[j].sec[k], 3, k)

            for inter_idx in range(common.ixn.ndlp):
                interaction = common.ixn.dlp[inter_idx]
                for p_idx in range(interaction.part.ndlp):
                    reco_part = interaction.part.dlp[p_idx]
                    for l in range(len(reco_part.truth)):
                        t0 = reco_part.truth[l]
                        overlap = float(reco_part.truthOverlap[l])
                        key = (int(t0.ixn), int(t0.type), int(t0.part))
                        if key in truth_map:
                            target_row = truth_map[key]
                            if overlap > data["common_dlp_truth_overlap"][target_row]:
                                data["common_dlp_truth_overlap"][target_row] = overlap
                                data["reco_contained"][target_row] = int(bool(reco_part.contained))
                                data["reco_pdg"][target_row] = reco_part.pdg
                                data["reco_px"][target_row] = reco_part.p.x
                                data["reco_py"][target_row] = reco_part.p.y
                                data["reco_pz"][target_row] = reco_part.p.z
                                data["reco_E_method"][target_row] = E_method_lookup(reco_part.E_method)
                                data["reco_theta"][target_row] = np.arctan(reco_part.p.y / reco_part.p.x) if reco_part.p.x != 0 else np.nan
                                data["reco_phi"][target_row] = np.arccos(reco_part.p.z / np.sqrt(reco_part.p.x**2 + reco_part.p.y**2 + reco_part.p.z**2)) if (reco_part.p.x != 0 or reco_part.p.y !=0 or reco_part.p.z !=0) else np.nan

            if hasattr(nd, detector):
                detector_obj = getattr(nd, detector)
                for inter_idx in range(detector_obj.ndlp):
                    interaction = detector_obj.dlp[inter_idx]
                    for track_idx in range(interaction.ntracks):
                        tracks = interaction.tracks[track_idx]
                        for l in range(len(tracks.truth)):
                            t0 = tracks.truth[l]
                            overlap = float(tracks.truthOverlap[l])
                            key = (int(t0.ixn), int(t0.type), int(t0.part))
                            if key in truth_map:
                                target_row = truth_map[key]
                            if overlap > data["nd_lar_dlp_truth_overlap"][target_row]:
                                data["nd_lar_dlp_truth_overlap"][target_row] = overlap
                                data["reco_E"][target_row] = tracks.E
                                data["reco_start_x"][target_row] = tracks.start.x
                                data["reco_start_y"][target_row] = tracks.start.y
                                data["reco_start_z"][target_row] = tracks.start.z
                                data["reco_end_x"][target_row] = tracks.end.x
                                data["reco_end_y"][target_row] = tracks.end.y
                                data["reco_end_z"][target_row] = tracks.end.z
                                data["reco_length"][target_row] = tracks.len_cm


        tFile.Close()

    data['E_true_ratio_common'] = compute_E_true_ratio(data, 'reco_E')
    data['E_kin_ratio_common'] = compute_E_kin_ratio(data, 'reco_E')

    df = pd.DataFrame(data)
    df["common_dlp_truth_overlap"] = df["common_dlp_truth_overlap"].replace(-1.0, np.nan)
    df["nd_lar_dlp_truth_overlap"] = df["nd_lar_dlp_truth_overlap"].replace(-1.0, np.nan)
    df["reco_truth_overlap"] = df["common_dlp_truth_overlap"].copy()
    df = df.drop(columns=['common_dlp_truth_overlap', 'nd_lar_dlp_truth_overlap'])
    

    particle_cuts = config.get("particle_cuts")
    df_filtered = apply_particle_cuts(df, particle_cuts)

    output_dir = "outputs/cafs"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/particle/caf_{config['mode']}_{particle_cuts['pdg']}_output{'_chunk_'+str(chunk_index) if not interactive else ''}.root"

    index_cols = ["ID", "idx", "part_idx"]
    if all(c in df_filtered.columns for c in index_cols):
        df_filtered = df_filtered.set_index(index_cols)

    with uproot.recreate(output_file) as f:
        f["particle_tree"] = df_filtered.reset_index()

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    output_path = "df_part_filtered_final_part_full_output.txt"
    with open(output_path, "w") as f:
        f.write(df_filtered.to_string(index=False))
    print(f"Full DataFrame written to {output_path}")
