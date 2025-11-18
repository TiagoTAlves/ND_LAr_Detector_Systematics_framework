import ROOT
import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np
import uproot
from caf_funcs import (
    MaCh3Modes_lookup,
    parse_args,
    update_part,
    pad_dict_lists_to_same_length,
    E_method_lookup,
    update_df,
    is_contained,
    compute_E_true_ratio,
    compute_E_kin_ratio,
    filter_reco_lists,
    pdg_to_particle_mass,
)


chunk_index, chunk_size, interactive = parse_args()

ROOT.gErrorIgnoreLevel = ROOT.kError
ROOT.gSystem.Load("/vols/dune/tta20/mach3/ND_LAr_Detector_Systematics_framework/caf-studies/duneanaobj/libduneanaobj_StandardRecord.so")
header_dir = "/vols/dune/tta20/mach3/ND_LAr_Detector_Systematics_framework/caf-studies/duneanaobj/duneanaobj/StandardRecord/"
header_files = []
header_files = glob.glob(os.path.join(header_dir, "*.h"))
for header in header_files:
    ROOT.gInterpreter.ProcessLine(f'#include "{header}"')


root_dir = '/vols/dune/tta20/mach3/ND_LAr_Detector_Systematics_framework/input-root-files/CAF/run-cafmaker/MicroProdN4p1_NDComplex_FHC.caf.full.light.spineonly/CAF/0002000/'
all_root_files = sorted([os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.root')])


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



data = {
    "ID": [],
    "spill_ID": [],
    "nu_ID": [],
    "mc_nupdg": [],
    "mc_nupdg_unosc": [],
    "E_tru": [],
    "mode": [],
    "mode_name": [],
    "q0": [],
    "q3": [],
    "bjorkenX": [],
    "inelasticty": [],
    "nu_vtx_x": [],
    "nu_vtx_y": [],
    "nu_vtx_z": [],
    "common_dlp_reco_vtx_x": [],
    "common_dlp_reco_vtx_y": [],
    "common_dlp_reco_vtx_z": [],
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
}

part = {
    "ID": [],
    "idx": [],
    "part_type": [],
    "part_idx": [],
    "pdg":[],
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
    "common_dlp_contained": [],
    "common_dlp_pdg_reco": [],
    "common_dlp_E_method": [],
    "common_dlp_E": [],
    "common_dlp_start_x": [],
    "common_dlp_start_y": [],
    "common_dlp_start_z": [],
    "common_dlp_end_x": [],
    "common_dlp_end_y": [],
    "common_dlp_end_z": [],
    "common_dlp_px_reco": [],
    "common_dlp_py_reco": [],
    "common_dlp_pz_reco": [],
    "common_dlp_contained": [],    
    "common_dlp_truth_overlap": [],
    # "nd_lar_dlp_track_start_x": [],
    # "nd_lar_dlp_track_start_y": [],
    # "nd_lar_dlp_track_start_z": [],
    # "nd_lar_dlp_track_end_x": [],
    # "nd_lar_dlp_track_end_y": [],
    # "nd_lar_dlp_track_end_z": [],
    # "nd_lar_dlp_track_dir_x": [],
    # "nd_lar_dlp_track_dir_y": [],
    # "nd_lar_dlp_track_dir_z": [],
    # "nd_lar_dlp_track_len_cm": [],
    # "nd_lar_dlp_track_E_vis": [],
    # "nd_lar_dlp_track_E_reco": [],
    # "nd_lar_dlp_track_truth_overlap_E": [],
    # "nd_lar_dlp_shower_start_x": [],
    # "nd_lar_dlp_shower_start_y": [],
    # "nd_lar_dlp_shower_start_z": [],
    # "nd_lar_dlp_shower_dir_x": [],
    # "nd_lar_dlp_shower_dir_y": [],
    # "nd_lar_dlp_shower_dir_z": [],
    # "nd_lar_dlp_shower_E_vis": [],
    # "nd_lar_dlp_shower_truth_overlap_E": [],
}

for root_file in root_files:
    tFile = ROOT.TFile.Open(root_file)
    tree = tFile.Get("cafTree")
    genie = tFile.Get("genieEvt")

    record = ROOT.caf.StandardRecord()
    tree.SetBranchAddress("rec", record)
    tree.SetBranchAddress()

    tree

    nspills = tree.GetEntries()
    print(f"Processing file:{root_file}, Total entries: {nspills}")

    for i in range(nspills):
        tree.GetEntry(i)
        mc = record.mc
        for j in range(mc.nnu):
            if hasattr(mc, "nu") and len(mc.nu) > 0:
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

                for k in range(mc.nu[j].nprim):
                    mass = pdg_to_particle_mass(mc.nu[j].prim[k].pdg)/1000
                    part["ID"].append(mc.nu[j].id)
                    part["idx"].append(j)
                    part["part_type"].append(1)
                    part["part_idx"].append(k)
                    part["pdg"].append(mc.nu[j].prim[k].pdg)
                    part["start_x"].append(mc.nu[j].prim[k].start_pos.x)
                    part["start_y"].append(mc.nu[j].prim[k].start_pos.y)
                    part["start_z"].append(mc.nu[j].prim[k].start_pos.z)
                    part["end_x"].append(mc.nu[j].prim[k].end_pos.x)
                    part["end_y"].append(mc.nu[j].prim[k].end_pos.y)
                    part["end_z"].append(mc.nu[j].prim[k].end_pos.z)
                    part["E"].append(mc.nu[j].prim[k].p.E)
                    part["E_kin"].append(mc.nu[j].prim[k].p.E - mass)
                    part["px"].append(mc.nu[j].prim[k].p.px)
                    part["py"].append(mc.nu[j].prim[k].p.py)
                    part["pz"].append(mc.nu[j].prim[k].p.pz)
                    part["is_contained"].append(is_contained(mc.nu[j].prim[k].start_pos.x,
                                                            mc.nu[j].prim[k].start_pos.y,
                                                            mc.nu[j].prim[k].start_pos.z,
                                                            mc.nu[j].prim[k].end_pos.x,
                                                            mc.nu[j].prim[k].end_pos.y,
                                                            mc.nu[j].prim[k].end_pos.z,
                                                            detector="TPC"))
                for k in range(mc.nu[j].nprefsi):
                    mass = pdg_to_particle_mass(mc.nu[j].prefsi[k].pdg)/1000
                    part["ID"].append(mc.nu[j].id)
                    part["idx"].append(j)
                    part["part_type"].append(2)
                    part["part_idx"].append(k)
                    part["pdg"].append(mc.nu[j].prefsi[k].pdg)
                    part["start_x"].append(mc.nu[j].prefsi[k].start_pos.x)
                    part["start_y"].append(mc.nu[j].prefsi[k].start_pos.y)
                    part["start_z"].append(mc.nu[j].prefsi[k].start_pos.z)
                    part["end_x"].append(mc.nu[j].prefsi[k].end_pos.x)
                    part["end_y"].append(mc.nu[j].prefsi[k].end_pos.y)
                    part["end_z"].append(mc.nu[j].prefsi[k].end_pos.z)
                    part["E"].append(mc.nu[j].prefsi[k].p.E)
                    part["E_kin"].append(mc.nu[j].prefsi[k].p.E - mass)
                    part["px"].append(mc.nu[j].prefsi[k].p.px)
                    part["py"].append(mc.nu[j].prefsi[k].p.py)
                    part["pz"].append(mc.nu[j].prefsi[k].p.pz)
                    part["is_contained"].append(is_contained(mc.nu[j].prefsi[k].start_pos.x,
                                                            mc.nu[j].prefsi[k].start_pos.y,
                                                            mc.nu[j].prefsi[k].start_pos.z,
                                                            mc.nu[j].prefsi[k].end_pos.x,
                                                            mc.nu[j].prefsi[k].end_pos.y,
                                                            mc.nu[j].prefsi[k].end_pos.z,
                                                            detector="TPC"))
                for k in range(mc.nu[j].nsec):
                    mass = pdg_to_particle_mass(mc.nu[j].sec[k].pdg)/1000
                    part["ID"].append(mc.nu[j].id)
                    part["idx"].append(j)
                    part["part_type"].append(3)
                    part["part_idx"].append(k)
                    part["pdg"].append(mc.nu[j].sec[k].pdg)
                    part["start_x"].append(mc.nu[j].sec[k].start_pos.x)
                    part["start_y"].append(mc.nu[j].sec[k].start_pos.y)
                    part["start_z"].append(mc.nu[j].sec[k].start_pos.z)
                    part["end_x"].append(mc.nu[j].sec[k].end_pos.x)
                    part["end_y"].append(mc.nu[j].sec[k].end_pos.y)
                    part["end_z"].append(mc.nu[j].sec[k].end_pos.z)
                    part["E"].append(mc.nu[j].sec[k].p.E)
                    part["E_kin"].append(mc.nu[j].sec[k].p.E - mass)
                    part["px"].append(mc.nu[j].sec[k].p.px)
                    part["py"].append(mc.nu[j].sec[k].p.py)
                    part["pz"].append(mc.nu[j].sec[k].p.pz)
                    part["is_contained"].append(is_contained(mc.nu[j].sec[k].start_pos.x,
                                                            mc.nu[j].sec[k].start_pos.y,
                                                            mc.nu[j].sec[k].start_pos.z,
                                                            mc.nu[j].sec[k].end_pos.x,
                                                            mc.nu[j].sec[k].end_pos.y,
                                                            mc.nu[j].sec[k].end_pos.z,
                                                            detector="TPC"))
        
        common = record.common
        for j in range(common.ixn.ndlp):
            truth_vec_df = common.ixn.dlp[j].truth
            if truth_vec_df and len(truth_vec_df) > 0:
                for k in range(len(truth_vec_df)):
                    if common.ixn.dlp[j].truthOverlap[k] > 0.9:
                        update_df(data, j, k, {
                            "common_dlp_reco_vtx_x": common.ixn.dlp[j].vtx.x,
                            "common_dlp_reco_vtx_y": common.ixn.dlp[j].vtx.y,
                            "common_dlp_reco_vtx_z": common.ixn.dlp[j].vtx.z,
                            "common_dlp_truth_overlap": common.ixn.dlp[j].truthOverlap[k]
                        })

            for k in range(common.ixn.dlp[j].part.ndlp):
                truth_vec = common.ixn.dlp[j].part.dlp[k].truth
                if truth_vec and len(truth_vec) > 0:
                    for l in range(len(truth_vec)): 
                        if common.ixn.dlp[j].part.dlp[k].truthOverlap[l] > 0.9:
                            t0 = truth_vec[l]
                            update_part(part, t0.ixn, t0.type, t0.part, {
                                "common_dlp_contained": int(bool(common.ixn.dlp[j].part.dlp[k].contained)),
                                "common_dlp_start_x": common.ixn.dlp[j].part.dlp[k].start.x,
                                "common_dlp_start_y": common.ixn.dlp[j].part.dlp[k].start.y,
                                "common_dlp_start_z": common.ixn.dlp[j].part.dlp[k].start.z,
                                "common_dlp_end_x": common.ixn.dlp[j].part.dlp[k].end.x,
                                "common_dlp_end_y": common.ixn.dlp[j].part.dlp[k].end.y,
                                "common_dlp_end_z": common.ixn.dlp[j].part.dlp[k].end.z,
                                "common_dlp_E": common.ixn.dlp[j].part.dlp[k].E,
                                "common_dlp_px_reco": common.ixn.dlp[j].part.dlp[k].p.x,
                                "common_dlp_py_reco": common.ixn.dlp[j].part.dlp[k].p.y,
                                "common_dlp_pz_reco": common.ixn.dlp[j].part.dlp[k].p.z,
                                "common_dlp_E_method": E_method_lookup(common.ixn.dlp[j].part.dlp[k].E_method),
                                "common_dlp_pdg_reco": common.ixn.dlp[j].part.dlp[k].pdg,
                                "common_dlp_truth_overlap": common.ixn.dlp[j].part.dlp[k].truthOverlap[l]
                            })
                else:
                    pass

        # nd = record.nd
        # for j in range(nd.lar.ndlp):
        #     for k in range(nd.lar.dlp[j].ntracks):
        #         truth_vec = nd.lar.dlp[j].tracks[k].truth
        #         if truth_vec and len(truth_vec) > 0:
        #             for l in range(len(truth_vec)): 
        #                 if nd.lar.dlp[j].tracks[k].truthOverlap[l] > 0.9:
        #                     t0 = truth_vec[l]
        #                     update_part(part, t0.ixn, t0.type, t0.part, {
        #                         "nd_lar_dlp_track_start_x": nd.lar.dlp[j].tracks[k].start.x,
        #                         "nd_lar_dlp_track_start_y": nd.lar.dlp[j].tracks[k].start.y,
        #                         "nd_lar_dlp_track_start_z": nd.lar.dlp[j].tracks[k].start.z,
        #                         "nd_lar_dlp_track_end_x": nd.lar.dlp[j].tracks[k].end.x,
        #                         "nd_lar_dlp_track_end_y": nd.lar.dlp[j].tracks[k].end.y,
        #                         "nd_lar_dlp_track_end_z": nd.lar.dlp[j].tracks[k].end.z,
        #                         "nd_lar_dlp_track_dir_x": nd.lar.dlp[j].tracks[k].dir.x,
        #                         "nd_lar_dlp_track_dir_y": nd.lar.dlp[j].tracks[k].dir.y,
        #                         "nd_lar_dlp_track_dir_z": nd.lar.dlp[j].tracks[k].dir.z,
        #                         "nd_lar_dlp_track_len_cm": nd.lar.dlp[j].tracks[k].len_cm,
        #                         "nd_lar_dlp_track_E_vis": nd.lar.dlp[j].tracks[k].Evis,
        #                         "nd_lar_dlp_track_E_reco": nd.lar.dlp[j].tracks[k].E,
        #                         "nd_lar_dlp_track_truth_overlap_E": nd.lar.dlp[j].tracks[k].truthOverlap[l]
        #                     })
        #         else:
        #             pass

            # for k in range(nd.lar.dlp[j].nshowers):
            #     truth_vec = nd.lar.dlp[j].showers[k].truth
            #     if truth_vec and len(truth_vec) > 0:
            #         for l in range(len(truth_vec)): 
            #             if nd.lar.dlp[j].showers[k].truthOverlap[l] > 0.9:
            #                 t0 = truth_vec[l]
            #                 update_part(part, t0.ixn, t0.type, t0.part, {
            #                     "nd_lar_dlp_showers_start_x": nd.lar.dlp[j].showers[k].start.x,
            #                     "nd_lar_dlp_showers_start_y": nd.lar.dlp[j].showers[k].start.y,
            #                     "nd_lar_dlp_showers_start_z": nd.lar.dlp[j].showers[k].start.z,
            #                     "nd_lar_dlp_shower_dir_x": nd.lar.dlp[j].showers[k].direction.x,
            #                     "nd_lar_dlp_shower_dir_y": nd.lar.dlp[j].showers[k].direction.y,
            #                     "nd_lar_dlp_shower_dir_z": nd.lar.dlp[j].showers[k].direction.z,
            #                     "nd_lar_dlp_shower_E_vis": nd.lar.dlp[j].showers[k].Evis,
            #                     "nd_lar_dlp_shower_truth_overlap_E": nd.lar.dlp[j].showers[k].truthOverlap[l]
            #                 })
            #     else:
            #         pass                

part['E_true_ratio_common'] = compute_E_true_ratio(part, 'common_dlp_E')
part['E_kin_ratio_common'] = compute_E_kin_ratio(part, 'common_dlp_E')


pad_dict_lists_to_same_length(data)
pad_dict_lists_to_same_length(part)
df = pd.DataFrame(data)
df_part = pd.DataFrame(part)

filtered_rows = []
for _, row in df_part.iterrows():
    result = filter_reco_lists(row)
    if isinstance(result, list):
        filtered_rows.extend(result)  
    else:
        filtered_rows.append(result) 
df_part_filtered = pd.DataFrame(filtered_rows)

mask = (
    (df_part_filtered['pdg'].isin([-11, 11])) &
    (df_part_filtered['is_contained'] == 1)
)
df_part_filtered_final = df_part_filtered[mask].dropna().copy()

output_dir = "outputs/cafs"
os.makedirs(output_dir, exist_ok=True)
output_file = f"{output_dir}/electron/caf_electron_output{'_chunk_'+str(chunk_index) if not interactive else ''}.root"

index_cols = ["ID", "idx", "part_idx"]
if all(c in df_part_filtered_final.columns for c in index_cols):
    df_part_filtered_final = df_part_filtered_final.set_index(index_cols)

with uproot.recreate(output_file) as f:
    f["electron_tree"] = df_part_filtered_final.reset_index()


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
output_path = "df_part_filtered_final_part_full_output.txt"
with open(output_path, "w") as f:
    f.write(df_part_filtered_final.to_string(index=False))
# with open("df_full_part_output.txt", "w") as f:
#     f.write(df_part.to_string(index=False))
print(f"Full DataFrame written to {output_path}")
# print(df_part.head(50))



