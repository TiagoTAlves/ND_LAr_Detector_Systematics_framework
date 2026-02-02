import ROOT
import os
import glob
import time
# from caf_funcs import parse_args
import argparse
import sys


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

# Start timing
start_time = time.time()

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

events = {
  "Vtx_x": [],
  "Vtx_y": [],
  "Vtx_z": [],
  "Part_pdg": [],
  "Part_start_x": [],
  "Part_start_y": [],
  "Part_start_z": [],
  "Part_end_x": [],
  "Part_end_y": [],
  "Part_end_z": [],
  "Part_contained": [],
  "Part_primary": []
}

for root_file in root_files:
    tFile = ROOT.TFile.Open(root_file)
    tree = tFile.Get("cafTree")
    genie = tFile.Get("genieEvt")

    record = ROOT.caf.StandardRecord()
    tree.SetBranchAddress("rec", record)
    nspills = tree.GetEntries()

    for i in range(nspills):
        tree.GetEntry(i)
        common = record.common
        for j in range(common.ixn.ndlp):
            events["Vtx_x"].append(common.ixn.dlp[j].vtx.x)
            events["Vtx_y"].append(common.ixn.dlp[j].vtx.y)
            events["Vtx_z"].append(common.ixn.dlp[j].vtx.z)
            for k in range(common.ixn.dlp[j].part.ndlp):
                part = common.ixn.dlp[j].part.dlp[k]
                events["Part_pdg"].append(part.pdg)
                events["Part_start_x"].append(part.start.x)
                events["Part_start_y"].append(part.start.y)
                events["Part_start_z"].append(part.start.z)
                events["Part_end_x"].append(part.end.x)
                events["Part_end_y"].append(part.end.y)
                events["Part_end_z"].append(part.end.z)
                events["Part_contained"].append(part.contained)
                events["Part_primary"].append(part.primary)


end_time = time.time()
elapsed_time = end_time - start_time

print(f"\n{'='*50}")
print(f"Execution completed in {elapsed_time:.2f} seconds")
# print(f"Total rows processed: {len(df_part_filtered_final)}")
if len(root_files) > 0:
    print(f"Average time per file: {elapsed_time/len(root_files):.2f} seconds")
print(f"{'='*50}\n")

print("Data extraction complete. Sample data:")
for key in events:
    print(f"{key}: {events[key][:5]}")  




