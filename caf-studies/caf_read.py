import ROOT
import os
import pandas as pd
import numpy as np
import argparse
import sys
import glob

def MaCh3Modes_lookup(value):
    MaCh3Modes = [
        "CCQE",
        "CC1Kaon",
        "CCDIS",
        "CCRES",
        "CCCOH",
        "CCDiff",
        "CCNuEl",
        "CCIMD",
        "CCAnuGam",
        "CCMEC",
        "CCCOHEL",
        "CCIBD",
        "CCGlRES",
        "CCIMDAnn",
        "NCQE",
        "NCDIS",
        "NCRES",
        "NCCOH",
        "NCDiff",
        "NCNuEl",
        "NCIMD",
        "NCAnuGam",
        "NCMEC",
        "NCCOHEL",
        "NCIBD",
        "NCGlRES",
        "NCIMDAnn"
    ]
    MaCh3Modes_dict = {mode: idx for idx, mode in enumerate(MaCh3Modes)}
    if isinstance(value, int):
        if 0 <= value < len(MaCh3Modes):
            return MaCh3Modes[value]
        else:
            raise ValueError("Index out of range")
    elif isinstance(value, str):
        if value in MaCh3Modes_dict:
            return MaCh3Modes_dict[value]
        else:
            raise ValueError("Mode not found")
    else:
        raise TypeError("Input must be an integer or string")

def parse_args():
    parser = argparse.ArgumentParser(description="Process CAF ROOT files in chunks.")
    parser.add_argument('--chunk', type=int, help='Chunk index to process')
    parser.add_argument('--chunksize', type=int, default=10, help='Number of files per chunk')
    args = parser.parse_args()

    if len(sys.argv) == 1:
        # Interactive mode
        return None, None, True
    else:
        return args.chunk, args.chunksize, False

def update_part(part_dict, j, k, l, updates):
    indices = [i for i, (idx_val, pt_val, pidx_val) in enumerate(zip(part_dict["idx"], part_dict["part_type"], part_dict["part_idx"]))
               if idx_val == j and pt_val == k and pidx_val == l]

    for i in indices:
        for col, val in updates.items():
            if col not in part_dict:
                part_dict[col] = [None] * len(part_dict["ID"])
            while len(part_dict[col]) <= i:
                part_dict[col].append(None)
            if isinstance(part_dict[col][i], list):
                part_dict[col][i].append(val)
            else:
                part_dict[col][i] = [part_dict[col][i], val] if part_dict[col][i] is not None else [val]

def pad_dict_lists_to_same_length(d):
    max_len = max(len(v) for v in d.values())
    for key, lst in d.items():
        if len(lst) < max_len:
            # Choose pad value based on content type or key
            pad_val = np.nan if all(isinstance(x, (float, int, type(np.nan))) or x is None for x in lst) else None
            lst.extend([pad_val] * (max_len - len(lst)))

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
    "mc_nupdg": [],
    "mc_nupdg_unosc": [],
    "E_tru": [],
    "mode": [],
    "mode_name": [],
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
}

part = {
    "ID": [],
    "idx": [],
    "part_type": [],
    "part_idx": [],
    "pdg":[],
    "E": [],
    "px": [],
    "py": [],
    "pz": [],
    "px_reco": [],
    "py_reco": [],
    "pz_reco": [],
    "pdg_reco": [],
    "pdg_truth_overlap": [],
    "E_reco_track": [],
    "E_vis_track": [],
    "truth_overlap_E_track": [],
    "E_vis_shower": [],
    "truth_overlap_E_shower": [],
}

for root_file in root_files:
    tFile = ROOT.TFile.Open(root_file)
    tree = tFile.Get("cafTree")

    record = ROOT.caf.StandardRecord()
    tree.SetBranchAddress("rec", record)

    nspills = tree.GetEntries()
    print(f"Processing file:{root_file}, Total entries: {nspills}")

    for i in range(nspills):
        # print(f"Processing event {i+1}/{tree.GetEntries()}")
        tree.GetEntry(i)
        mc = record.mc
        # print(f"{mc.nnu}")
        # print(f"Number of True Neutrino entries per slice: {mc.nnu}")
        for j in range(mc.nnu):
            if hasattr(mc, "nu") and len(mc.nu) > 0:
                data["ID"].append(mc.nu[j].id)
                data["mc_nupdg"].append(mc.nu[j].pdg)
                data["mc_nupdg_unosc"].append(mc.nu[j].pdgorig)
                data["E_tru"].append(mc.nu[j].E)
                data["mode"].append(mc.nu[j].mode)
                data["mode_name"].append(MaCh3Modes_lookup(mc.nu[j].mode))
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
                    part["ID"].append(mc.nu[j].id)
                    part["idx"].append(j)
                    part["part_type"].append(1)
                    part["part_idx"].append(k)
                    part["pdg"].append(mc.nu[j].prim[k].pdg)
                    part["E"].append(mc.nu[j].prim[k].p.E)
                    part["px"].append(mc.nu[j].prim[k].p.px)
                    part["py"].append(mc.nu[j].prim[k].p.py)
                    part["pz"].append(mc.nu[j].prim[k].p.pz)
                for k in range(mc.nu[j].nprefsi):
                    part["ID"].append(mc.nu[j].id)
                    part["idx"].append(j)
                    part["part_type"].append(2)
                    part["part_idx"].append(k)
                    part["pdg"].append(mc.nu[j].prefsi[k].pdg)
                    part["E"].append(mc.nu[j].prefsi[k].p.E)
                    part["px"].append(mc.nu[j].prefsi[k].p.px)
                    part["py"].append(mc.nu[j].prefsi[k].p.py)
                    part["pz"].append(mc.nu[j].prefsi[k].p.pz)
                for k in range(mc.nu[j].nsec):
                    part["ID"].append(mc.nu[j].id)
                    part["idx"].append(j)
                    part["part_type"].append(3)
                    part["part_idx"].append(k)
                    part["pdg"].append(mc.nu[j].sec[k].pdg)
                    part["E"].append(mc.nu[j].sec[k].p.E)
                    part["px"].append(mc.nu[j].sec[k].p.px)
                    part["py"].append(mc.nu[j].sec[k].p.py)
                    part["pz"].append(mc.nu[j].sec[k].p.pz)
        
        common = record.common
        for j in range(common.ixn.ndlp):
            for k in range(common.ixn.dlp[j].part.ndlp):
                truth_vec = common.ixn.dlp[j].part.dlp[k].truth
                if truth_vec and len(truth_vec) > 0:
                    # Then safe to access truth[0]
                    for l in range(len(truth_vec)): 
                        # print(f"{i}.{j}.{k}.{l} Number:{common.ixn.dlp[j].part.dlp[k].truthOverlap[l]}")
                        if common.ixn.dlp[j].part.dlp[k].truthOverlap[l] > 0.9:
                            t0 = truth_vec[l]
                            update_part(part, t0.ixn, t0.type, t0.part, {
                                # "px_reco": common.ixn.dlp[j].part.dlp[k].p.x,
                                # "py_reco": common.ixn.dlp[j].part.dlp[k].p.y,
                                # "pz_reco": common.ixn.dlp[j].part.dlp[k].p.z,
                                "pdg_reco": common.ixn.dlp[j].part.dlp[k].pdg,
                                "pdg_truth_overlap": common.ixn.dlp[j].part.dlp[k].truthOverlap[l]
                            })
                else:
                    # Handle or skip empty truth vector
                    # Optionally log or count how often this happens
                    pass

        nd = record.nd
        for j in range(nd.lar.ndlp):
            # print(f"{nd.lar.dlp[j].ntracks}")
            for k in range(nd.lar.dlp[j].ntracks):
                truth_vec = nd.lar.dlp[j].tracks[k].truth
                if truth_vec and len(truth_vec) > 0:
                    for l in range(len(truth_vec)): 
                        # print(f"{i}.{j}.{k}.{l} Number:{common.ixn.dlp[j].part.dlp[k].E}")
                        if nd.lar.dlp[j].tracks[k].truthOverlap[l] > 0.9:
                            t0 = truth_vec[l]
                            update_part(part, t0.ixn, t0.type, t0.part, {
                                "E_reco_track": nd.lar.dlp[j].tracks[k].E,
                                "truth_overlap_E_track": nd.lar.dlp[j].tracks[k].truthOverlap[l]
                            })
                else:
                    pass

            for k in range(nd.lar.dlp[j].nshowers):
                truth_vec = nd.lar.dlp[j].showers[k].truth
                if truth_vec and len(truth_vec) > 0:
                    for l in range(len(truth_vec)): 
                        if nd.lar.dlp[j].showers[k].truthOverlap[l] > 0.9:
                            t0 = truth_vec[l]
                            update_part(part, t0.ixn, t0.type, t0.part, {
                                "E_vis_shower": nd.lar.dlp[j].showers[k].Evis,
                                "truth_overlap_E_shower": nd.lar.dlp[j].showers[k].truthOverlap[l]
                            })
                else:
                    pass                







                
# df = pd.DataFrame(data)
pad_dict_lists_to_same_length(part)
df_part = pd.DataFrame(part)

# output_dir = "outputs/cafs"
# os.makedirs(output_dir, exist_ok=True)
# if interactive:
#     output_file = f"{output_dir}/caf_output_all.root"
# else:
#     output_file = f"{output_dir}/caf_output_chunk{chunk_index}.root"

# print(df.head(30))
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
output_path = "df_part_full_output.txt"
with open(output_path, "w") as f:
    f.write(df_part.to_string(index=False))

print(f"Full DataFrame written to {output_path}")
# print(df_part.head(50))

