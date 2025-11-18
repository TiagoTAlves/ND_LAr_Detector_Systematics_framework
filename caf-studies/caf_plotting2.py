import uproot
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from particle import Particle
from scipy.optimize import curve_fit
from scipy.stats import laplace, norm
from scipy.signal import find_peaks


bins_ratio = np.linspace(0, 2, 101)
bins_pos_diff = np.linspace(-30, 30, 121)
plots_dir = "plots"

# particles = ['muon', 'pion', 'electron', 'kaon', 'proton']
particles = ['proton', 'muon', 'pion']
# var_list = ['E', 'common_dlp_E', 'start_x', 'start_y', 'start_z', 'end_x', 'end_y', 'end_z',
#             'common_dlp_start_x', 'common_dlp_start_y', 'common_dlp_start_z',
#             'common_dlp_end_x', 'common_dlp_end_y', 'common_dlp_end_z',
#             'px', 'py', 'pz',
#             'common_dlp_px_reco', 'common_dlp_py_reco', 'common_dlp_pz_reco',
#             'common_dlp_truth_overlap']
var_list = ['pz']

if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)
if not os.path.exists(f"{plots_dir}/energy_distribution"):
    os.makedirs(f"{plots_dir}/energy_distribution")
if not os.path.exists(f"{plots_dir}/ratios/kin_ratio"):
    os.makedirs(f"{plots_dir}/ratios/kin_ratio")
if not os.path.exists(f"{plots_dir}/ratios/true_ratio"):
    os.makedirs(f"{plots_dir}/ratios/true_ratio")
for var in var_list:
    if not os.path.exists(f"plots/var/{var}"):
        os.makedirs(f"plots/var/{var}")
    if not os.path.exists(f"plots/var/{var}/above"):
        os.makedirs(f"plots/var/{var}/above")
    if not os.path.exists(f"plots/var/{var}/below"):
        os.makedirs(f"plots/var/{var}/below")
    if not os.path.exists(f"plots/var/{var}/compare"):
        os.makedirs(f"plots/var/{var}/compare")
for particle in particles:
    if not os.path.exists(f"plots/particle/{particle}"):
        os.makedirs(f"plots/particle/{particle}")


def load_particle_data(particle):
    file_path = f"outputs/cafs/caf_{particle}_output_full.root"
    root_file = uproot.open(file_path)
    tree = root_file[f'{particle}_tree']
    arrays = tree.arrays()
    data_dict = {key: arrays[key].to_numpy() for key in arrays.fields}
    df = pd.DataFrame(data_dict)
    df.set_index('index', inplace=True)
    return df


def plot_Ekin_ratio_for_particles(df, particle):
    if particle == 'muon':
        df = df[df['part_type'] == 1]
    data = df['E_kin_ratio_common'].to_numpy()
    data = data[np.isfinite(data)]
    plt.hist(data, bins=[0,0.25,0.5,0.75,0.9,1.1,1.25,1.5,2,5,10], color='blue', alpha=0.7)
    plt.xlabel('E_kin Ratio')
    plt.ylabel('Counts')
    plt.xlim(0, 10)
    # plt.ylim(0, 2000)
    plt.title(f'Test Histogram of E_kin Ratio for {particle.capitalize()}')
    plt.grid(True)
    plt.savefig(f"{plots_dir}/ratios/kin_ratio/E_kin_ratio_common_{particle}_histogram.png")
    plt.savefig(f"{plots_dir}/particle/{particle}/E_kin_ratio_common_{particle}_histogram.png")
    plt.close()

def plot_Etrue_ratio_for_particles(df):
    data = df['E_true_ratio_common'].to_numpy()
    data = data[np.isfinite(data)]
    plt.hist(data, bins=bins_ratio, color='blue', alpha=0.7)
    plt.xlabel('E_true Ratio')
    plt.ylabel('Counts')
    plt.xlim(0, 2)
    plt.title(f'Histogram of E_true Ratio for {particle.capitalize()}')
    plt.grid(True)
    plt.savefig(f"{plots_dir}/ratios/true_ratio/E_true_ratio_common_{particle}_histogram.png")
    plt.savefig(f"{plots_dir}/particle/{particle}/E_true_ratio_common_{particle}_histogram.png")
    plt.close()

def plot_energy_distribution(df, particle):
    plt.hist(df['common_dlp_E'], bins=np.linspace(0,0.2,51), color='blue', alpha=0.7)
    plt.hist(df['E_kin'], bins=np.linspace(0,0.2,51), color='orange', alpha=0.7)
    plt.xlabel(f'Energy Distribution of {particle.capitalize()}s [GeV]')
    plt.ylabel('Counts')
    plt.xlim(0, 0.2)
    plt.legend(['Reconstructed Energy', 'True Kinetic Energy'])
    plt.title(f'Energy Distribution of {particle.capitalize()}s')
    plt.savefig(f"{plots_dir}/particle/{particle}/reco_energy_distribution_{particle}_histogram.png")
    plt.savefig(f"{plots_dir}/energy_distribution/reco_energy_distribution_{particle}_histogram.png")
    plt.close()


def plot_position_differences(df, particle):
    for dir in ['x', 'y', 'z']:
        for pos in ['start', 'end']:
            diff = df[f'{pos}_{dir}'] - df[f'common_dlp_{pos}_{dir}']
            plt.hist(diff, bins=bins_pos_diff, color='blue', alpha=0.7)
            plt.xlabel(f'Difference in {dir.upper()} {pos.capitalize()} Position [cm]')
            plt.ylabel('Counts')
            plt.xlim(-30, 30)
            plt.title(f'Histogram of {pos.capitalize()} {dir.upper()} Position Difference for {particle.capitalize()}s')
            plt.savefig(f"{plots_dir}/particle/{particle}/{pos}_{dir}_diff_{particle}_histogram.png")
            plt.close()


def plot_ratio_cut_distributions(df, particle):
    df_high = df[(df['E_kin_ratio_common'] >= 0.9) & (df['E_kin_ratio_common'] <= 1.2)]
    df_low = df[df['E_kin_ratio_common'] <= 0.9]

    for var in var_list:
        high_data = df_high[var].to_numpy()
        high_data = high_data[np.isfinite(high_data)]
        low_data = df_low[var].to_numpy()
        low_data = low_data[np.isfinite(low_data)]

        plt.hist(high_data, bins=np.linspace(-0.4,1,51), color='green', alpha=0.7)
        plt.xlabel(f'{var} Distribution of {particle.capitalize()}s [GeV]')
        plt.ylabel('Counts')
        plt.title(f'{var} Distribution (1.1 ≥ E_kin_ratio_common ≥ 0.9) for {particle.capitalize()}s')
        plt.savefig(f"{plots_dir}/var/{var}/above/distribution_{var}_{particle}_EkinRatioAbove09_histogram.png")
        plt.savefig(f"{plots_dir}/particle/{particle}/distribution_{var}_{particle}_EkinRatioAbove09_histogram.png")
        plt.close()

        plt.hist(low_data, bins=np.linspace(-0.4,1,51), color='red', alpha=0.7)
        plt.xlabel(f'{var} Distribution of {particle.capitalize()}s')
        plt.ylabel('Counts')
        plt.title(f'{var} Distribution (E_kin_ratio_common < 0.9) for {particle.capitalize()}s [GeV]')
        plt.savefig(f"{plots_dir}/var/{var}/below/distribution_{var}_{particle}_EkinRatioBelow09_histogram.png")
        plt.savefig(f"{plots_dir}/particle/{particle}/distribution_{var}_{particle}_EkinRatioBelow09_histogram.png")
        plt.close()

        plt.hist(high_data, bins=np.linspace(-0.4,1,51), color='green', alpha=0.7, label='1.2 ≥ E_kin_ratio_common ≥ 0.9')
        plt.hist(low_data, bins=np.linspace(-0.4,1,51), color='red', alpha=0.7, label='E_kin_ratio_common < 0.9')
        plt.xlabel(f'{var} Distribution of {particle.capitalize()}s [GeV]')
        plt.ylabel('Counts')
        plt.title(f'{var} Distribution for {particle.capitalize()}s')
        plt.legend()
        plt.savefig(f"{plots_dir}/var/{var}/compare/distribution_{var}_{particle}_EkinRatio_comparison_histogram.png")
        plt.savefig(f"{plots_dir}/particle/{particle}/distribution_{var}_{particle}_EkinRatio_comparison_histogram.png")
        plt.close()

def mean_resolution_vs_ekin(df, xedges, resolution_key, plot_path, particle, label):
    bin_centers = 0.5 * (xedges[:-1] + xedges[1:])
    mean_resolutions = []
    std_resolutions = []
    for i in range(len(xedges) - 1):
        bin_mask = (df['E_kin'] >= xedges[i]) & (df['E_kin'] < xedges[i+1])
        res_in_bin = df.loc[bin_mask, resolution_key]
        if len(res_in_bin) > 0:
            mean_resolutions.append(np.mean(res_in_bin))
            std_resolutions.append(np.std(res_in_bin))
        else:
            mean_resolutions.append(np.nan)
            std_resolutions.append(np.nan)

    mean_resolutions = np.array(mean_resolutions)
    std_resolutions = np.array(std_resolutions)

    plt.scatter(bin_centers, mean_resolutions, s=2, label=label)
    plt.fill_between(bin_centers,
                     mean_resolutions - std_resolutions,
                     mean_resolutions + std_resolutions,
                     alpha=0.3, color='red')
    plt.xlabel('True Kinetic Energy [GeV]')
    plt.ylabel('Relative Mean Energy Resolution (E_rec - E_kin) / E_kin')
    plt.title(f'Relative Mean Energy Resolution vs E_kin ({particle}, {label})')
    plt.xlim(0, xedges[-1])
    plt.ylim(-1, 1)
    plt.grid()
    plt.savefig(plot_path)
    plt.close()

def Ekin_vs_Erec(df, particle):
    if particle == 'muon':
        df = df[df['part_type'] == 1]
    if particle == 'electron':
        df =df[df['part_type'] == 1]

    df = df[np.isfinite(df['E_kin_ratio_common'])]

    print(f"Number of {particle} entries for Ekin_vs_Erec: {len(df)}")

    n_entries = len(df)
    nbins_1d = max(20, int(np.sqrt(n_entries)))
    nbins_2d_x = max(20, int(np.sqrt(n_entries)/6))
    nbins_2d_y = max(20, int(np.sqrt(n_entries)/6))
    xedges_1d = np.linspace(0, 2, nbins_1d + 1)
    xedges = np.linspace(0, 1.5, nbins_2d_x + 1)
    yedges = np.linspace(-1, 2, nbins_2d_y + 1)

    vals = df['E_kin_ratio_common'].values
    mu, std = norm.fit(vals)  # global initial guess

    # region 3: 1 sigma around mean using gaussian fit on histogram counts
    region_mask3 = (vals > mu - 1 * std) & (vals < mu + 1 * std)
    region_vals3 = vals[region_mask3]
    counts3, bins3 = np.histogram(region_vals3, bins=xedges_1d)
    bin_centers3 = 0.5 * (bins3[:-1] + bins3[1:])
    p0_3 = [max(counts3), np.mean(region_vals3), np.std(region_vals3)]
    params3, _ = curve_fit(gaussian, bin_centers3, counts3, p0=p0_3)
    A3, mu_region3, std_region3 = params3

    # region 4: largest bin peak region gaussian fit on histogram counts
    counts_full, bin_edges_full = np.histogram(vals, bins=xedges_1d)
    bin_centers_full = 0.5 * (bin_edges_full[:-1] + bin_edges_full[1:])
    max_bin_index = np.argmax(counts_full)
    peak_bin_center = bin_centers_full[max_bin_index]
    lower_bound4 = max(0, peak_bin_center - 1 * std)
    upper_bound4 = peak_bin_center + 1 * std
    region_mask4 = (vals >= lower_bound4) & (vals <= upper_bound4)
    region_vals4 = vals[region_mask4]
    counts4, bins4 = np.histogram(region_vals4, bins=xedges_1d)
    bin_centers4 = 0.5 * (bins4[:-1] + bins4[1:])
    p0_4 = [max(counts4), np.mean(region_vals4), np.std(region_vals4)]
    params4, _ = curve_fit(gaussian, bin_centers4, counts4, p0=p0_4)
    A4, mu_region4, std_region4 = params4

    plt.hist(df['E_kin_ratio_common'], bins=xedges_1d, alpha=0.7)
    plt.xlabel('E_kin_ratio_common')
    plt.ylabel('Counts')
    plt.title(f'E_kin_ratio_common distribution for {particle.capitalize()}s')
    plt.savefig(f"{plots_dir}/particle/{particle}/Ekin_ratio_common_{particle}_hist.png")
    plt.close()

    plt.hist(region_vals3, bins=xedges_1d, alpha=0.6, color='g')
    x3 = np.linspace(bin_centers3.min(), bin_centers3.max(), 200)
    y3 = gaussian(x3, A3, mu_region3, std_region3)
    plt.plot(x3, y3, 'r-', linewidth=2, label=f'Gaussian fit μ={mu_region3:.3f}, σ={std_region3:.3f}')
    plt.legend()
    plt.title(f'Gaussian fit (1σ region) for {particle.capitalize()}s')
    plt.xlabel('E_kin_ratio_common')
    plt.ylabel('Counts')
    plt.savefig(f"{plots_dir}/particle/{particle}/Ekin_ratio_common_gaussfit3_{particle}.png")
    plt.close()

    plt.hist(region_vals4, bins=xedges_1d, alpha=0.6, color='g')
    x4 = np.linspace(bin_centers4.min(), bin_centers4.max(), 200)
    y4 = gaussian(x4, A4, mu_region4, std_region4)
    plt.plot(x4, y4, 'r-', linewidth=2, label=f'Gaussian fit μ={mu_region4:.3f}, σ={std_region4:.3f}')
    plt.legend()
    plt.title(f'Gaussian fit (largest bin region) for {particle.capitalize()}s')
    plt.xlabel('E_kin_ratio_common')
    plt.ylabel('Counts')
    plt.savefig(f"{plots_dir}/particle/{particle}/Ekin_ratio_common_gaussfit4_{particle}.png")
    plt.close()

    # Scatter plot, linear fit and resolutions

    plt.scatter(df['common_dlp_E'], df['E_kin'], alpha=0.5, s=0.1)

    y_fit = df['E_kin'].values
    x_fit = df['common_dlp_E'].values

    if len(x_fit) > 2:
        degree = 1
        coeffs = np.polyfit(x_fit, y_fit, degree)
        poly = np.poly1d(coeffs)

        x_curve = np.linspace(x_fit.min(), x_fit.max(), 200)
        y_curve = poly(x_curve)

        y_pred = poly(x_fit)
        ss_res = np.sum((y_fit - y_pred) ** 2)
        ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    plt.plot(x_curve, y_curve, 'r-', linewidth=2,
             label=f'Equation: y={coeffs[0]:.3f}x + {coeffs[1]:.3f}')
    print(f'{particle} fit equation: y={coeffs[0]:.3f}x + {coeffs[1]:.3f}, R²={r_squared:.4f}')
    plt.ylabel('True Kinetic Energy [GeV]')
    plt.xlabel('Reconstructed Energy [GeV]')
    plt.title(f'Scatter plot of True Kinetic Energy vs Reconstructed Energy for {particle.capitalize()}s')
    # plt.legend()
    plt.savefig(f"{plots_dir}/particle/{particle}/Ekin_vs_Erec_{particle}_scatter.png")
    plt.close()

    df['E_rec'] = 2*df['common_dlp_E'] 
    df['resolution'] = (df['E_rec'] - df['E_kin']) / df['E_kin']

    df['E_rec2'] = coeffs[0] * df['common_dlp_E'] + coeffs[1]
    df['resolution2'] = (df['E_rec2'] - df['E_kin']) / df['E_kin']

    df['E_rec3'] = df['common_dlp_E'] / mu_region3
    df['resolution3'] = (df['E_rec3'] - df['E_kin']) / df['E_kin']

    df['E_rec4'] = df['common_dlp_E'] / mu_region4
    df['resolution4'] = (df['E_rec4'] - df['E_kin']) / df['E_kin']

    # Plot resolutions histograms

    plt.hist2d(df['E_kin'], df['resolution'], alpha=0.5, bins=[xedges, yedges])
    plt.xlabel('True Kinetic Energy [GeV]')
    plt.colorbar()
    plt.ylabel('Energy Resolution (E_rec - E_kin) / E_kin')
    plt.title(f'Energy Resolution vs True Kinetic Energy ({particle.capitalize()}, No Correction)')
    plt.savefig(f"{plots_dir}/particle/{particle}/energy_resolution_{particle}_hist2d.png")
    plt.close()

    plt.hist2d(df['E_kin'], df['resolution2'], alpha=0.5, bins=[xedges, yedges])
    plt.xlabel('True Kinetic Energy [GeV]')
    plt.colorbar()
    plt.ylabel('Energy Resolution (E_rec2 - E_kin) / E_kin')
    plt.title(f'Energy Resolution vs True Kinetic Energy ({particle.capitalize()} Linear Fit Correction)')
    plt.savefig(f"{plots_dir}/particle/{particle}/energy_resolution2_{particle}_hist2d.png")
    plt.close()

    plt.hist2d(df['E_kin'], df['resolution3'], alpha=0.5, bins=[xedges, yedges])
    plt.xlabel('True Kinetic Energy [GeV]')
    plt.ylabel('Energy Resolution (E_rec3 - E_kin) / E_kin')
    plt.title(f'Energy Resolution vs True Kinetic Energy ({particle.capitalize()}, Gaussian Fit Region)')
    plt.colorbar()
    plt.savefig(f"{plots_dir}/particle/{particle}/energy_resolution3_{particle}_hist2d.png")
    plt.close()

    plt.hist2d(df['E_kin'], df['resolution4'], alpha=0.5, bins=[xedges, yedges])
    plt.xlabel('True Kinetic Energy [GeV]')
    plt.ylabel('Energy Resolution (E_rec4 - E_kin) / E_kin')
    plt.title(f'Energy Resolution vs True Kinetic Energy ({particle.capitalize()}, Gaussian Fit Near Largest Bin)')
    plt.colorbar()
    plt.savefig(f"{plots_dir}/particle/{particle}/energy_resolution4_{particle}_hist2d.png")
    plt.close()

    mean_resolution_vs_ekin(df, xedges, 'resolution', f"{plots_dir}/particle/{particle}/mean_resolution_no_{particle}.png", particle, "No Correction")
    mean_resolution_vs_ekin(df, xedges, 'resolution2', f"{plots_dir}/particle/{particle}/mean_resolution_linear_{particle}.png", particle, "Linear Fit")
    mean_resolution_vs_ekin(df, xedges, 'resolution3', f"{plots_dir}/particle/{particle}/mean_resolution_gauss3_{particle}.png", particle, "Gaussian 1σ Region")
    mean_resolution_vs_ekin(df, xedges, 'resolution4', f"{plots_dir}/particle/{particle}/mean_resolution_gauss4_{particle}.png", particle, "Gaussian fit around Largest Bin")


    plt.hist(df['E_rec'], bins=np.linspace(0, 1.5, 30) , alpha=0.7, label='No Correction', facecolor='none', edgecolor='red', histtype='step')
    plt.hist(df['E_rec3'], bins=np.linspace(0, 1.5, 30) , alpha=0.7, label='Gaussian Fit Region Correction', facecolor='none', edgecolor='blue', histtype='step')
    plt.ylabel('Counts')
    plt.title(f'Reconstructed Energy Distribution Comparison \n before and after Scaling for {particle.capitalize()}s')
    plt.legend()
    plt.xlabel('Reconstructed Energy [GeV]')
    plt.savefig(f"{plots_dir}/particle/{particle}/Erec_comparison_region_{particle}_histogram.png")
    plt.close()
    
    plt.hist(df['E_rec'], bins=np.linspace(0, 1.5, 30) , alpha=0.7, label='No Correction', facecolor='none', edgecolor='red', histtype='step')
    plt.hist(df['E_rec4'], bins=np.linspace(0, 1.5, 30) , alpha=0.7, label='Gaussian Fit around modal Bin', facecolor='none', edgecolor='blue', histtype='step')
    plt.ylabel('Counts')
    plt.title(f'Reconstructed Energy Distribution Comparison \n before and after Scaling for {particle.capitalize()}s')
    plt.legend()
    plt.xlabel('Reconstructed Energy [GeV]')
    plt.savefig(f"{plots_dir}/particle/{particle}/Erec_comparison_modal_{particle}_histogram.png")
    plt.close()

    plt.hist(df['E_rec'], bins=np.linspace(0, 1.5, 30) , alpha=0.7, label='No Correction', facecolor='none', edgecolor='red', histtype='step')
    plt.hist(df['E_rec2'], bins=np.linspace(0, 1.5, 30) , alpha=0.7, label='Linear Fit', facecolor='none', edgecolor='blue', histtype='step')
    plt.ylabel('Counts')
    plt.title(f'Reconstructed Energy Distribution Comparison \n before and after Scaling for {particle.capitalize()}s')
    plt.legend()
    plt.xlabel('Reconstructed Energy [GeV]')
    plt.savefig(f"{plots_dir}/particle/{particle}/Erec_comparison_linear_{particle}_histogram.png")
    plt.close()

def gaussian(x, amplitude, mean, sigma):
    return amplitude * np.exp(-(x - mean)**2 / (2 * sigma**2))

def proton(window=0.08):
    df = load_particle_data('proton')
    data = df['E_kin_ratio_common'].to_numpy()
    data = data[np.isfinite(data)]

    hist, bin_edges = np.histogram(data, bins=bins_ratio, density=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Estimate peak (mean) from global Gaussian fit or simply take max bin
    mu, std = 1, 0.05
    peak_min = mu - window
    peak_max = mu + window
    print(f"Proton E_kin_ratio_common peak estimated at μ={mu:.3f} with σ={std:.3f}")
    print(f"Fitting range: [{peak_min:.3f}, {peak_max:.3f}]")

    # Restrict to peak region
    peak_mask = (bin_centers >= peak_min) & (bin_centers <= peak_max)
    fit_x = bin_centers[peak_mask]
    fit_y = hist[peak_mask]
    p0 = [fit_y.max(), mu, std]
    print(f"Initial fit parameters: Amplitude={p0[0]:.3f}, Mean={p0[1]:.5f}, Sigma={p0[2]:.5f}")

    # Fit only to peak
    popt, pcov = curve_fit(gaussian, fit_x, fit_y, p0=p0)
    print(f"Fitted parameters: Amplitude={popt[0]:.3f}, Mean={popt[1]:.5f}, Sigma={popt[2]:.5f}")

    # Plot
    plt.figure(figsize=(8,6))
    plt.hist(data, bins=bins_ratio, color='blue', alpha=0.6, density=False, label='Data')
    x = np.linspace(peak_min, peak_max, 1000)
    plt.plot(x, gaussian(x, *popt), 'r-', linewidth=2, 
             label=f'Gaussian peak fit\nμ={popt[1]:.5f}, σ={popt[2]:.5f}')
    plt.xlabel('E_kin Ratio')
    plt.ylabel('Density')
    plt.xlim(0, 2)  # or [0, 2]
    plt.title('Histogram of E_kin Ratio (Gaussian Peak Region)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/ratios/kin_ratio/E_kin_ratio_common_proton_gaussian_peak_fit_histogram.png")
    plt.close()


def pion_seperation_plot():
    df = load_particle_data('pion')
    df_pi_plus = df[(df['pdg'] > 0)]
    df_pi_minus = df[(df['pdg'] < 0)]
    plt.hist(df_pi_plus['E_kin_ratio_common'], bins=bins_ratio, color='blue', alpha=0.7, label='pi+')
    plt.hist(df_pi_minus['E_kin_ratio_common'], bins=bins_ratio, color='orange', alpha=0.7, label='pi-')
    plt.xlabel('E_kin Ratio')
    plt.ylabel('Counts')
    plt.xlim(0, 2)
    plt.title(f'Histogram of E_kin Ratio for Pions')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{plots_dir}/ratios/kin_ratio/E_kin_ratio_common_pion_charge_separation_histogram.png")
    plt.close()

def muon_primary():
    df = load_particle_data('muon')
    df_primary = df[df['part_type'] == 1]
    df_secondary = df[df['part_type'] == 3]
    plt.hist(df_primary['E_kin_ratio_common'], bins=bins_ratio, color='blue', alpha=0.7, label='Primary Muons')
    plt.hist(df_secondary['E_kin_ratio_common'], bins=bins_ratio, color='orange', alpha=0.7, label='Secondary Muons')
    plt.xlabel('E_kin Ratio')
    plt.ylabel('Counts')
    plt.xlim(0, 2)
    plt.title(f'Histogram of E_kin Ratio for Primary Muons')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{plots_dir}/ratios/kin_ratio/E_kin_ratio_common_muon_primary_histogram.png")
    plt.close()

def electon_primary():
    df = load_particle_data('electron')
    df_primary = df[df['E'] > 0.03]
    df_secondary = df[df['E'] < 0.03]
    plt.hist(df_secondary['E_kin_ratio_common'], bins=bins_ratio, color='orange', alpha=0.7, label='Secondary Electrons')
    plt.hist(df_primary['E_kin_ratio_common'], bins=bins_ratio, color='blue', alpha=0.7, label='Primary Electrons')
    plt.xlabel('E_kin Ratio')
    plt.ylabel('Counts')
    plt.xlim(0, 2)
    plt.title(f'Histogram of E_kin Ratio for Primary Electrons')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{plots_dir}/ratios/kin_ratio/E_kin_ratio_common_electron_energy_histogram.png")
    plt.close()

def pion_primary():
    df = load_particle_data('pion')
    df_primary = df[df['part_type'] == 1]
    df_secondary = df[df['part_type'] == 3]
    plt.hist(df_primary['E_kin_ratio_common'], bins=bins_ratio, color='blue', alpha=0.7, label='Primary Pions')
    plt.hist(df_secondary['E_kin_ratio_common'], bins=bins_ratio, color='orange', alpha=0.7, label='Secondary Pions')
    plt.xlabel('E_kin Ratio')
    plt.ylabel('Counts')
    plt.xlim(0, 2)
    plt.title(f'Histogram of E_kin Ratio for Primary Pions')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{plots_dir}/ratios/kin_ratio/E_kin_ratio_common_pion_primary_histogram.png")
    plt.close()

def pion_momentum():
    df = load_particle_data('pion')
    df['p'] = np.sqrt(df['px']**2 + df['py']**2 + df['pz']**2)
    
    # Fix: add parentheses around each comparison
    # df_pi_plus_0p9_1p2 = df[(df['pdg'] > 0) & (df['p'] >= 0.3)]
    # df_pi_minus_0p9_1p2 = df[(df['pdg'] < 0) & (df['p'] >= 0.3)]
    # df_pi_plus_0p9 = df[(df['pdg'] > 0) & (df['p'] < 0.35)]
    # df_pi_minus_0p9 = df[(df['pdg'] < 0) & (df['p'] < 0.35)]
    for momentum in [0.1,0.15,0.2,0.25,0.3,0.35, 0.4,0.45,0.5,0.6,0.7,0.8,0.9,1.0,1.2,1.5,2.0]:
        df_pi_plus_low = df[(df['pdg'] > 0) & (df['p'] < momentum)]
        df_pi_minus_low = df[(df['pdg'] < 0) & (df['p'] < momentum)]
        plt.hist(df_pi_plus_low['E_kin_ratio_common'], bins=np.linspace(0,2,51), alpha=0.7, label=f'pi+ p< {momentum} GeV/c')
        plt.hist(df_pi_minus_low['E_kin_ratio_common'], bins=np.linspace(0,2,51), alpha=0.7, label=f'pi- p< {momentum} GeV/c')
        plt.xlabel('E_kin Ratio of Pions')
        plt.ylabel('Counts')
        plt.legend()
        plt.xlim(0, 2)
        plt.title(f'E_kin Ratio Distribution of Pions with p < {momentum} GeV/c')
        plt.savefig(f"{plots_dir}/particle/pion/E_kin_ratio_distribution_p_less_{momentum}_pion_histogram.png")
        plt.close()

    
    # plt.hist(df_pi_plus_0p9_1p2['p'], bins=np.linspace(0,3,51), color='blue', alpha=0.7, label='pi+_0.9≤E_kin_ratio_common≤1.2')
    # plt.hist(df_pi_minus_0p9_1p2['p'], bins=np.linspace(0,3,51), color='green', alpha=0.7, label='pi-_0.9≤E_kin_ratio_common≤1.2')
    # plt.hist(df_pi_plus_0p9['p'], bins=np.linspace(0,3,51), color='orange', alpha=0.7, label='pi+_E_kin_ratio_common<0.9')
    # plt.hist(df_pi_minus_0p9['p'], bins=np.linspace(0,3,51), color='red', alpha=0.7, label='pi-_E_kin_ratio_common<0.9')
    # plt.xlabel('Momentum of Pions [GeV/c]')
    # plt.ylabel('Counts')
    # plt.legend()
    # plt.xlim(0, 3)
    # plt.title(f'Pz Momentum Distribution of Pions')
    # plt.savefig(f"{plots_dir}/particle/pion/p_momentum_distribution_pion_histogram.png")
    # plt.close()

    # plt.hist(df['p'], bins=100, color='blue', alpha=0.7)
    # plt.xlabel('Momentum of Pions [GeV/c]')
    # plt.ylabel('Counts')
    # # plt.xlim(0, 3)
    # plt.title(f'P Momentum Distribution of Pions')
    # plt.savefig(f"{plots_dir}/particle/pion/p_momentum_distribution_pion_full_histogram.png")
    # plt.close()

    # plt.hist2d(df['p'], df['E_kin_ratio_common'], bins=[np.linspace(0,3,31), np.linspace(0,2,21)], cmap='magma')
    # plt.colorbar(label='Counts')
    # plt.xlabel('Momentum of Pions [GeV/c]')
    # plt.ylabel('E_kin Ratio')
    # plt.xlim(0, 3)
    # plt.ylim(0, 2)
    # plt.title(f'2D Histogram of Pion Momentum vs E_kin Ratio')
    # plt.savefig(f"{plots_dir}/particle/pion/p_momentum_vs_EkinRatio_pion_2Dhistogram.png")
    # plt.close()

    # plt.hist(df_pi_plus_0p9_1p2['E_kin_ratio_common'], bins=np.linspace(0,2,51), color='blue', alpha=0.7, label='pi+ p≤ 0.3 GeV/c')
    # plt.hist(df_pi_minus_0p9_1p2['E_kin_ratio_common'], bins=np.linspace(0,2,51), color='green', alpha=0.7, label='pi- p≤ 0.3 GeV/c')
    # plt.hist(df_pi_plus_0p9['E_kin_ratio_common'], bins=np.linspace(0,2,51), color='orange', alpha=0.7, label='pi+ p> 0.3 GeV/c')
    # plt.hist(df_pi_minus_0p9['E_kin_ratio_common'], bins=np.linspace(0,2,51), color='red', alpha=0.7, label='pi- p> 0.3 GeV/c')
    # plt.xlabel('E_kin Ratio of Pions')
    # plt.ylabel('Counts')
    # plt.legend()
    # plt.xlim(0, 2)
    # plt.title(f'E_kin Ratio Distribution of Pions')
    # plt.savefig(f"{plots_dir}/particle/pion/E_kin_ratio_distribution_charge_split_pion_histogram.png")
    # plt.close()

    # # plt.hist(df_pi_plus_0p9_1p2['E_kin_ratio_common'], bins=np.linspace(0,2,51), color='blue', alpha=0.7, label='pi+ p≤ 0.3 GeV/c')
    # # plt.hist(df_pi_minus_0p9_1p2['E_kin_ratio_common'], bins=np.linspace(0,2,51), color='green', alpha=0.7, label='pi- p≤ 0.3 GeV/c')
    # plt.hist(df_pi_plus_0p9['E_kin_ratio_common'], bins=np.linspace(0,2,51), color='orange', alpha=0.7, label='pi+ p> 0.35 GeV/c')
    # plt.hist(df_pi_minus_0p9['E_kin_ratio_common'], bins=np.linspace(0,2,51), color='red', alpha=0.7, label='pi- p> 0.35 GeV/c')
    # plt.xlabel('E_kin Ratio of Pions')
    # plt.ylabel('Counts')
    # plt.legend()
    # plt.xlim(0, 2)
    # plt.title(f'E_kin Ratio Distribution of Pions')
    # plt.savefig(f"{plots_dir}/particle/pion/E_kin_ratio_distribution_charge_split_pion<0.35_histogram.png")
    # plt.close()



if __name__ == "__main__":
    # pion_seperation_plot()
    # muon_primary()
    # electon_primary()
    # pion_primary()
    pion_momentum()
    # proton()
    for particle in particles:
        df = load_particle_data(particle)
        # Ekin_vs_Erec(df, particle)
        # plot_energy_distribution(df, particle)
        # plot_position_differences(df, particle)
        # plot_ratio_cut_distributions(df, particle)
        # plot_Ekin_ratio_for_particles(df, particle)
        # plot_Etrue_ratio_for_particles(df)