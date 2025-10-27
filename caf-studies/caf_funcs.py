import argparse
import sys
import numpy as np

def E_method_lookup(value):
    E_method_modes = [
        "Unknown",
        "Range",
        "Multiple Scattering",
        "Calorimetry"
    ]
    E_method_dict = {mode: idx for idx, mode in enumerate(E_method_modes)}
    if isinstance(value, int):
        if 0 <= value < len(E_method_modes):
            return E_method_modes[value]
        else:
            raise ValueError(f"Index out of range {value}")
    elif isinstance(value, str):
        if value in E_method_dict:
            return E_method_dict[value]
        else:
            raise ValueError("Mode not found")
    else:
        raise TypeError("Input must be an integer or string")

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

def update_df(df_dict, j, k, updates):
    indices = [i for i, (idx_val, pt_val) in enumerate(zip(df_dict["spill_ID"], df_dict["nu_ID"]))
               if idx_val == j and pt_val == k]

    for i in indices:
        for col, val in updates.items():
            if col not in df_dict:
                df_dict[col] = [None] * len(df_dict["ID"])
            while len(df_dict[col]) <= i:
                df_dict[col].append(None)
            if isinstance(df_dict[col][i], list):
                df_dict[col][i].append(val)
            else:
                df_dict[col][i] = [df_dict[col][i], val] if df_dict[col][i] is not None else [val]

def pad_dict_lists_to_same_length(d):
    max_len = max(len(v) for v in d.values())
    for key, lst in d.items():
        if len(lst) < max_len:
            # Choose pad value based on content type or key
            pad_val = np.nan if all(isinstance(x, (float, int, type(np.nan))) or x is None for x in lst) else None
            lst.extend([pad_val] * (max_len - len(lst)))