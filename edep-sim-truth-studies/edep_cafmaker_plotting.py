import glob
import uproot
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

def read_edep_sim_output(root_file_pattern, tree_name="events"):
    """Read all ROOT files matching the pattern and return a concatenated pandas DataFrame."""
    dfs = []
    for root_file in sorted(glob.glob(root_file_pattern)):
        with uproot.open(root_file) as file:
            tree = file[tree_name]
            df = tree.arrays(library="pd")
            dfs.append(df)
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        raise FileNotFoundError(f"No files matched the pattern: {root_file_pattern}")

def event_rates(df):
    """Calculate event rates based on the specified event type."""
    contained_df = df[df['is_contained'] == 1]
    uncontained_df = df[df['is_contained'] == 0]
    XVarBins = [0., 0.25, 0.5, 0.75, 1., 1.25, 1.5, 1.75, 2., 2.25, 2.5, 2.75, 3., 3.25, 3.5, 3.75, 4., 4.25, 4.5, 4.75, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 100000.]
    plt.figure()
    plt.hist(contained_df['E']/1000, bins=XVarBins, alpha=0.7, label='Contained Events (Custom Bins)')
    plt.hist(uncontained_df['E']/1000, bins=XVarBins, alpha=0.5, label='Uncontained Events (Custom Bins)')
    plt.xlabel('Energy (GeV)')
    plt.ylabel('Number of Events')
    plt.xlim(0, 10)
    plt.ylim(0, 50000)
    plt.title('Energy Distribution of Contained Events (Custom Bins)')
    plt.legend()
    plt.savefig('plots/cafs/contained_events_energy_distribution.png')
    plt.show()

def event_rates_with_uncontained_errors(df):

    sigma_cols = df.filter(like='sigma_').columns
    df[sigma_cols] = df[sigma_cols].fillna(1)
    df[sigma_cols] = df[sigma_cols].clip(upper=1)


    mask_uncontained = df['is_contained'] == 0

    df['E_shifted_+'] = df['E']
    df['E_shifted_-'] = df['E']

    df.loc[mask_uncontained, 'E_shifted_+'] = (
        df.loc[mask_uncontained, 'E'] +
        (df.loc[mask_uncontained, 'ePip'] * df.loc[mask_uncontained, 'sigma_211'])/2 +
        (df.loc[mask_uncontained, 'ePim'] * df.loc[mask_uncontained, 'sigma_-211'])/2 +
        (df.loc[mask_uncontained, 'ePi0'] * df.loc[mask_uncontained, 'sigma_111'])/2 +
        (df.loc[mask_uncontained, 'LepE'] * df.loc[mask_uncontained, 'sigma_13'])/2 +
        (df.loc[mask_uncontained, 'eP']   * df.loc[mask_uncontained, 'sigma_p'])/2 +
        (df.loc[mask_uncontained, 'eN']   * df.loc[mask_uncontained, 'sigma_n'])/2
    )

    df.loc[mask_uncontained, 'E_shifted_-'] = (
        df.loc[mask_uncontained, 'E'] +
        (df.loc[mask_uncontained, 'ePip'] * - df.loc[mask_uncontained, 'sigma_211'])/2 +
        (df.loc[mask_uncontained, 'ePim'] * - df.loc[mask_uncontained, 'sigma_-211'])/2 +
        (df.loc[mask_uncontained, 'ePi0'] * - df.loc[mask_uncontained, 'sigma_111'])/2 +
        (df.loc[mask_uncontained, 'LepE'] * - df.loc[mask_uncontained, 'sigma_13'])/2 +
        (df.loc[mask_uncontained, 'eP']   * - df.loc[mask_uncontained, 'sigma_p'])/2 +
        (df.loc[mask_uncontained, 'eN']   * - df.loc[mask_uncontained, 'sigma_n'])/2
    )

    XVarBins = np.array([
        0., 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75,
        3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0, 5.5, 6.0,
        6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 100.0
    ])

    bin_centers = (XVarBins[:-1] + XVarBins[1:]) / 2

    E_nominal_GeV = df['E'] / 1000
    E_plus_GeV    = df['E_shifted_+'] / 1000
    E_minus_GeV   = df['E_shifted_-'] / 1000
    E_contained_GeV = df[df['is_contained'] == 1]['E'] / 1000

    plt.figure()
    plt.hist(E_nominal_GeV, bins=XVarBins, alpha=0.7, label='Nominal E', histtype='step')
    plt.hist(E_plus_GeV, bins=XVarBins, alpha=0.5, label='Shifted +', histtype='step')
    plt.hist(E_minus_GeV, bins=XVarBins, alpha=0.5, label='Shifted -', histtype='step')
    plt.xlabel('Energy (GeV)')
    plt.ylabel('Number of Events')
    plt.title('Energy Distribution: Nominal, Shifted +, Shifted -')
    plt.xlim(0, 10)
    plt.ylim(0, 70000)
    plt.legend()
    plt.savefig('plots/cafs/energy_distribution_nominal_shifted.png')
    plt.show()

        # --- histogram counts ---
    nominal_counts, _ = np.histogram(E_nominal_GeV, bins=XVarBins)
    plus_counts, _    = np.histogram(E_plus_GeV, bins=XVarBins)
    minus_counts, _   = np.histogram(E_minus_GeV, bins=XVarBins)
    contained_counts, _ = np.histogram(E_contained_GeV, bins=XVarBins)

    bin_widths = np.diff(XVarBins)
    nominal_density = nominal_counts / bin_widths
    plus_density    = plus_counts / bin_widths
    minus_density   = minus_counts / bin_widths
    contained_density = contained_counts / bin_widths

    plt.figure(figsize=(8,5))
    plt.plot(bin_centers, plus_counts/ nominal_counts, label='Shifted + minus Nominal', drawstyle='steps-mid')
    plt.plot(bin_centers, minus_counts/ nominal_counts, label='Shifted - minus Nominal', drawstyle='steps-mid')
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.xlabel('Energy (GeV)')
    plt.ylabel('Δ Number of Events')
    plt.xlim(0, 10)
    # plt.ylim(-10000, 10000)
    plt.title('Bin-by-Bin Difference from Nominal')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('plots/cafs/bin_by_bin_difference_from_nominal.png')
    plt.show()

    colors = {
        'nominal': 'black',
        'plus': 'tab:blue',
        'minus': 'tab:orange',
        'contained': 'tab:green'
    }

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(5, 5), sharex=True,
        gridspec_kw={'height_ratios': [3, 1]}
    )

    # --- First subplot: absolute histograms ---
    ax1.step(bin_centers, nominal_density, where='mid', color=colors['nominal'], label='Nominal E')
    ax1.step(bin_centers, plus_density,    where='mid', color=colors['plus'], label='Shifted +')
    ax1.step(bin_centers, minus_density,   where='mid', color=colors['minus'], label='Shifted -')
    ax1.step(bin_centers, contained_density, where='mid', color=colors['contained'], label='Contained Events', linestyle='--')

    ax1.set_ylabel('Events / GeV')
    ax1.set_title('Energy Distribution & Bin-by-Bin Difference')
    ax1.set_xlim(0, 10)
    # ax1.set_ylim(0, 70000)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # --- Bottom: Differences ---
    ax2.plot(bin_centers, np.log(plus_density/ nominal_density), color=colors['plus'], drawstyle='steps-mid')
    ax2.plot(bin_centers, np.log(minus_density/ nominal_density), color=colors['minus'], drawstyle='steps-mid')
    ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax2.set_xlabel('Energy (GeV)')
    ax2.set_ylabel('log(Events/Nominal)' '\n' '/ GeV')
    ax2.set_xlim(0, 10)
    # ax2.set_ylim(-10000, 10000)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('plots/cafs/combined_energy_and_difference.png')
    plt.show()



if __name__ == "__main__":
    df = read_edep_sim_output("outputs/cafs/*.CAF.root", tree_name="caf")
    event_rates_with_uncontained_errors(df)
    print(df.head())  # Display the first few rows of the DataFrame
    print(f"Total entries: {len(df)}")  # Print total number of entries