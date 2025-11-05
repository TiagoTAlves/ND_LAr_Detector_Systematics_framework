import uproot
import pandas as pd
import matplotlib.pyplot as plt

file_path = "outputs/cafs/caf_muon_output_full.root"
root_file = uproot.open(file_path)
tree = root_file["muon_tree"]

arrays = tree.arrays()

data_dict = {key: arrays[key].to_numpy() for key in arrays.fields}

df = pd.DataFrame(data_dict)

df.set_index('index', inplace=True)

plt.hist(df['E_true_ratio_common'], bins=75, color='blue', alpha=0.7)
plt.xlabel('E_true_ratio_common')
plt.ylabel('Counts')
plt.xlim(0, 1.5)
plt.title('Histogram of E_true_ratio_common')
plt.grid(True)
plt.savefig("E_true_ratio_common_histogram.png")
# plt.show()

