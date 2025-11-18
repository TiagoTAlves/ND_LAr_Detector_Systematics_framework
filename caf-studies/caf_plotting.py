import uproot
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os 
from particle import Particle

bins_ratio = np.linspace(0, 2, 101)
bins_pos_diff = np.linspace(-30, 30, 121)
plots_dir = "plots"
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# for particle in ['muon', 'pion', 'electron', 'kaon', 'proton']:
# # for particle in ['kaon']:
#     file_path = f"outputs/cafs/caf_{particle}_output_full.root"
#     root_file = uproot.open(file_path)
#     tree = root_file[f'{particle}_tree']

#     arrays = tree.arrays()

#     data_dict = {key: arrays[key].to_numpy() for key in arrays.fields}

#     df = pd.DataFrame(data_dict)

#     df.set_index('index', inplace=True)

#     plt.hist(df['E_true_ratio_common'], bins=bins_ratio, color='blue', alpha=0.7)
#     plt.xlabel('E_true Ratio')
#     plt.ylabel('Counts')
#     plt.xlim(0, 2)
#     plt.title('Histogram of E_true Ratio for ' + particle.capitalize())
#     plt.grid(True)
#     plt.savefig(f"{plots_dir}/E_true_ratio_common_{particle}_histogram.png")
#     plt.show()

# for particle in ['proton','muon', 'pion', 'electron', 'kaon']:
#     file_path = f"outputs/cafs/caf_{particle}_output_full.root"
#     root_file = uproot.open(file_path)
#     tree = root_file[f'{particle}_tree']

#     arrays = tree.arrays()

#     data_dict = {key: arrays[key].to_numpy() for key in arrays.fields}

#     df = pd.DataFrame(data_dict)

#     df.set_index('index', inplace=True)

#     plt.hist(df['E_kin_ratio_common'], bins=bins_ratio, color='blue', alpha=0.7)
#     plt.xlabel('E_kin Ratio')
#     plt.ylabel('Counts')
#     plt.xlim(0, 2)
#     plt.title('Histogram of E_kin Ratio for ' + particle.capitalize())
#     plt.grid(True)
#     plt.savefig(f"{plots_dir}/E_kin_ratio_common_{particle}_histogram.png")
#     plt.show()

# for particle in ['muon', 'pion', 'electron', 'kaon', 'proton']:
#     file_path = f"outputs/cafs/caf_{particle}_output_full.root"
#     root_file = uproot.open(file_path)
#     tree = root_file[f'{particle}_tree']

#     arrays = tree.arrays()

#     data_dict = {key: arrays[key].to_numpy() for key in arrays.fields}

#     df = pd.DataFrame(data_dict)

#     df.set_index('index', inplace=True)
#     for dir in ['x', 'y', 'z']:
#         for pos in ['start', 'end']:
#             diff = df[f'{pos}_{dir}'] - df[f'common_dlp_{pos}_{dir}']
#             plt.hist(diff, bins=bins_pos_diff, color='blue', alpha=0.7)
#             plt.xlabel(f'Difference in {dir.upper()} {pos.capitalize()} Position [cm]')
#             plt.ylabel('Counts')
#             plt.xlim(-30, 30)
#             plt.title(f'Histogram of {pos.capitalize()} {dir.upper()} Position Difference for {particle.capitalize()}s')
#             # plt.grid(True)
#             plt.savefig(f"{plots_dir}/{pos}_{dir}_diff_{particle}_histogram.png")
#             plt.show()

# for particle in ['proton','muon', 'pion', 'electron', 'kaon']:
#     file_path = f"outputs/cafs/caf_{particle}_output_full.root"
#     root_file = uproot.open(file_path)
#     tree = root_file[f'{particle}_tree']

#     arrays = tree.arrays()

#     data_dict = {key: arrays[key].to_numpy() for key in arrays.fields}

#     df = pd.DataFrame(data_dict)

#     df.set_index('index', inplace=True) 
    
#     plt.hist(df['common_dlp_E'], bins=100, color='blue', alpha=0.7)
#     plt.xlabel(f'Energy Distribution of {particle.capitalize()}s [GeV]')
#     plt.ylabel('Counts')
#     # plt.xlim(-30, 30)
#     plt.title(f'Energy Distribution of {particle.capitalize()}s')
#     # plt.grid(True)
#     plt.savefig(f"{plots_dir}/reco_energy_distribution_{particle}_histogram.png")
#     plt.show()

for particle in ['muon', 'pion', 'electron', 'kaon']:
    file_path = f"outputs/cafs/caf_{particle}_output_full.root"
    root_file = uproot.open(file_path)
    tree = root_file[f'{particle}_tree']
    arrays = tree.arrays()
    data_dict = {key: arrays[key].to_numpy() for key in arrays.fields}
    df = pd.DataFrame(data_dict)
    df.set_index('index', inplace=True) 
    # bins_E_distribution = np.linspace(0, 1.5, 101)
    df_high = df[(df['E_kin_ratio_common'] >= 0.9) & (df['E_kin_ratio_common'] <= 1.2)]    
    df_low = df[df['E_kin_ratio_common'] < 0.9]

    for var in ['E', 'common_dlp_E' ,'start_x', 'start_y', 'start_z', 'end_x', 'end_y', 'end_z', 'common_dlp_start_x', 'common_dlp_start_y', 'common_dlp_start_z', 'common_dlp_end_x', 'common_dlp_end_y', 'common_dlp_end_z', 'px', 'py', 'pz', 'common_dlp_px_reco', 'common_dlp_py_reco', 'common_dlp_pz_reco', 'common_dlp_truth_overlap']:
        high_data = df_high[var].to_numpy()
        high_data = high_data[np.isfinite(high_data)]

        low_data = df_low[var].to_numpy()
        low_data = low_data[np.isfinite(low_data)]    

        plt.hist(high_data, bins=100, color='green', alpha=0.7)
        plt.xlabel(f'{var} Distribution of {particle.capitalize()}s')
        plt.ylabel('Counts')
        # plt.xlim(0, 1.5)
        plt.title(f'{var} Distribution (1.2 ≥ E_kin_ratio_common ≥ 0.9) for {particle.capitalize()}s')
        plt.savefig(f"{plots_dir}/distribution_{var}_{particle}_EkinRatioAbove09_histogram.png")
        plt.show()

        plt.hist(low_data, bins=100, color='red', alpha=0.7)
        plt.xlabel(f'{var} Distribution of {particle.capitalize()}s')
        plt.ylabel('Counts')
        # plt.xlim(0, 1.5)
        plt.title(f'{var} Distribution (E_kin_ratio_common < 0.9) for {particle.capitalize()}s')
        plt.savefig(f"{plots_dir}/distribution_{var}_{particle}_EkinRatioBelow09_histogram.png")
        plt.show()

        plt.hist(high_data, bins=100, color='green', alpha=0.7, label='1.2 ≥ E_kin_ratio_common ≥ 0.9')
        plt.hist(low_data, bins=100, color='red', alpha=0.7, label='E_kin_ratio_common < 0.9')
        plt.xlabel(f'{var} Distribution of {particle.capitalize()}s ')
        plt.ylabel('Counts')
        # plt.xlim(0, 1.5)
        plt.title(f'{var} Distribution for {particle.capitalize()}s')
        plt.legend()
        plt.savefig(f"{plots_dir}/distribution_{var}_{particle}_EkinRatio_comparison_histogram.png")
        plt.show()