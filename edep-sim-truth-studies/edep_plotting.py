import uproot
import pandas as pd
import os
import inspect
import matplotlib.pyplot as plt
import glob
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize



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

def get_output_dir():
    """Return the output directory as plots/{calling_function_name}."""
    frame = inspect.currentframe()
    try:
        caller = inspect.getouterframes(frame)[1].function
    finally:
        del frame
    return os.path.join("plots", caller)

def pdg_code_to_name(pdg_code):
    """Convert PDG code to particle name."""
    try:
        from particle import Particle
        particle = Particle.from_pdgid(pdg_code)
        return particle.name
    except Exception as e:
        print(f"Error converting PDG code {pdg_code}: {e}")
        return str(pdg_code)

def plot_evis_over_ekin(df):
    
    theta_bins = [(a, b) for a, b in zip(np.linspace(0, np.pi, 4)[:-1], np.linspace(0, np.pi, 4)[1:])]
    phi_bins = [(a, b) for a, b in zip(np.linspace(0, 2 * np.pi, 4)[:-1], np.linspace(0, 2 * np.pi, 4)[1:])]
    x_bins = [(-3500, 3500)]
    y_bins = [(-2500, 1400)]
    z_bins = [(4000, 9200)]
    particle_species = df['pdg'].unique()

    output_dir = get_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    for pdg in particle_species:
        df_pdg = df[df['pdg'] == pdg]
        for tmin, tmax in theta_bins:
            for xmin, xmax in x_bins:
                for ymin, ymax in y_bins:
                    for zmin, zmax in z_bins:
                        for pmin, pmax in phi_bins:
                            cut = (
                                (df_pdg['theta'] >= tmin) & (df_pdg['theta'] < tmax) &
                                (df_pdg['phi'] >= pmin) & (df_pdg['phi'] < pmax) &
                                (df_pdg['start_x'] >= xmin) & (df_pdg['start_x'] < xmax) &
                                (df_pdg['start_y'] >= ymin) & (df_pdg['start_y'] < ymax) &
                                (df_pdg['start_z'] >= zmin) & (df_pdg['start_z'] < zmax)
                            )
                            df_cut = df_pdg[cut]
                            if len(df_cut) == 0:
                                continue
                            ratio = df_cut['E_vis'] / df_cut['E_kin']
                            plt.figure(figsize=(6,4))
                            plt.hist(ratio, bins=50, range=(0,1.2), histtype='step', color='blue')
                            plt.xlabel('E_vis / E_kin')
                            plt.ylabel('Counts')
                            plt.title(f'{pdg_code_to_name(pdg)}, θ=[{tmin},{tmax}), x=[{xmin},{xmax}), y=[{ymin},{ymax}), z=[{zmin},{zmax})')
                            plt.tight_layout()
                            fname = os.path.join(
                                output_dir,
                                f"{pdg_code_to_name(pdg)}_theta{tmin}-{tmax}_x{xmin}-{xmax}_y{ymin}-{ymax}_z{zmin}-{zmax}.png"
                            )
                            plt.savefig(fname)
                            plt.close()

def plot_evis_over_etrue(df):
    
    theta_bins = [(a, b) for a, b in zip(np.linspace(0, np.pi, 4)[:-1], np.linspace(0, np.pi, 4)[1:])]
    phi_bins = [(a, b) for a, b in zip(np.linspace(0, 2 * np.pi, 4)[:-1], np.linspace(0, 2 * np.pi, 4)[1:])]    
    x_bins = [(-3500, 3500)]
    y_bins = [(-2500, 1400)]
    z_bins = [(4000, 9200)]
    particle_species = df['pdg'].unique()

    output_dir = get_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    for pdg in particle_species:
        df_pdg = df[df['pdg'] == pdg]
        for tmin, tmax in theta_bins:
            for xmin, xmax in x_bins:
                for ymin, ymax in y_bins:
                    for zmin, zmax in z_bins:
                        for pmin, pmax in phi_bins:
                            cut = (
                                (df_pdg['theta'] >= tmin) & (df_pdg['theta'] < tmax) &
                                (df_pdg['phi'] >= pmin) & (df_pdg['phi'] < pmax) &
                                (df_pdg['start_x'] >= xmin) & (df_pdg['start_x'] < xmax) &
                                (df_pdg['start_y'] >= ymin) & (df_pdg['start_y'] < ymax) &
                                (df_pdg['start_z'] >= zmin) & (df_pdg['start_z'] < zmax)
                            )
                            df_cut = df_pdg[cut]
                            if len(df_cut) == 0:
                                continue
                            ratio = df_cut['E_vis'] / df_cut['E']
                            plt.figure(figsize=(6,4))
                            plt.hist(ratio, bins=50, range=(0,1.2), histtype='step', color='blue')
                            plt.xlabel('E_vis / Etrue')
                            plt.ylabel('Counts')
                            plt.title(f'{pdg_code_to_name(pdg)}, θ=[{tmin},{tmax}), x=[{xmin},{xmax}), y=[{ymin},{ymax}), z=[{zmin},{zmax})')
                            plt.tight_layout()
                            fname = os.path.join(
                                output_dir,
                                f"{pdg_code_to_name(pdg)}_theta{tmin}-{tmax}_x{xmin}-{xmax}_y{ymin}-{ymax}_z{zmin}-{zmax}.png"
                            )
                            plt.savefig(fname)
                            plt.close()

def plot_evis_over_etrue_not_contained(df):
    particle_species = df['pdg'].unique()

    output_dir = get_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    for pdg in particle_species:
        df_pdg = df[df['pdg'] == pdg]
        cut = (
            (df_pdg['is_contained_TPC'] == 0) &
            (df_pdg['E_vis'] > 0) 
        )
        df_cut = df_pdg[cut]
        if len(df_cut) == 0:
            continue
        ratio = df_cut['E_vis'] / df_cut['E']
        plt.figure(figsize=(6,4))
        plt.hist(ratio, bins=60, range=(0,1.2), histtype='step', color='red')
        plt.xlabel('E_vis / Etrue (not contained)')
        plt.ylabel('Counts')
        plt.title(f'{pdg_code_to_name(pdg)}, not contained')
        plt.tight_layout()
        fname = os.path.join(
            output_dir,
            f"{pdg_code_to_name(pdg)}_hist_evis_over_etrue_notcontained.png"
        )
        plt.savefig(fname)
        plt.close()

def plot_evis_over_etrue_contained(df):
    particle_species = df['pdg'].unique()

    output_dir = get_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    for pdg in particle_species:
        df_pdg = df[df['pdg'] == pdg]
        cut = (
            (df_pdg['is_contained_TPC'] == 1) &
            (df_pdg['E_vis'] > 0) 
        )
        df_cut = df_pdg[cut]
        if len(df_cut) == 0:
            continue
        ratio = df_cut['E_vis'] / df_cut['E']
        plt.figure(figsize=(6,4))
        plt.hist(ratio, bins=60, range=(0,1.2), histtype='step', color='red')
        plt.xlabel('E_vis / Etrue (contained)')
        plt.ylabel('Counts')
        plt.title(f'{pdg_code_to_name(pdg)}, contained')
        plt.tight_layout()
        fname = os.path.join(
            output_dir,
            f"{pdg_code_to_name(pdg)}_hist_evis_over_etrue_contained.png"
        )
        plt.savefig(fname)
        plt.close()

def plot_evis_over_ekin_contained(df):
    particle_species = df['pdg'].unique()

    output_dir = get_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    for pdg in particle_species:
        df_pdg = df[df['pdg'] == pdg]
        cut = (
            (df_pdg['is_contained_TPC'] == 1) &
            (df_pdg['E_vis'] > 0) 
        )
        df_cut = df_pdg[cut]
        if len(df_cut) == 0:
            continue
        ratio = df_cut['E_vis'] / df_cut['E_kin']
        plt.figure(figsize=(6,4))
        plt.hist(ratio, bins=60, range=(0,1.2), histtype='step', color='red')
        plt.xlabel('E_vis / Ekin (contained in TPC)')
        plt.ylabel('Counts')
        plt.title(f'{pdg_code_to_name(pdg)}, contained')
        plt.tight_layout()
        fname = os.path.join(
            output_dir,
            f"{pdg_code_to_name(pdg)}_hist_evis_over_ekin_contained.png"
        )
        plt.savefig(fname)
        plt.close()

def plot_evis_over_ekin_not_contained(df):
    particle_species = df['pdg'].unique()

    output_dir = get_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    for pdg in particle_species:
        df_pdg = df[df['pdg'] == pdg]
        cut = (
            (df_pdg['is_contained_TPC'] == 0) &
            (df_pdg['E_vis'] > 0) 
        )
        df_cut = df_pdg[cut]
        if len(df_cut) == 0:
            continue
        ratio = df_cut['E_vis'] / df_cut['E_kin']
        plt.figure(figsize=(6,4))
        plt.hist(ratio, bins=60, range=(0,1.2), histtype='step', color='red')
        plt.xlabel('E_vis / Ekin (not contained in TPC)')
        plt.ylabel('Counts')
        plt.title(f'{pdg_code_to_name(pdg)}, not contained')
        plt.tight_layout()
        fname = os.path.join(
            output_dir,
            f"{pdg_code_to_name(pdg)}_hist_evis_over_ekin_not_contained.png"
        )
        plt.savefig(fname)
        plt.close()

def plot_evis_vs_etrue(df):
    particle_species = df['pdg'].unique()
    output_dir = get_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    for pdg in particle_species:
        df_pdg = df[df['pdg'] == pdg]
        e_kin_min, e_kin_max = df_pdg['E'].min(), df_pdg['E'].max()
        e_vis_min, e_vis_max = df_pdg['E_vis'].min(), df_pdg['E_vis'].max()
        plt.figure(figsize=(6, 5))
        if len(df_pdg) == 0:
            continue
        plt.hist2d(
            df_pdg['E'], df_pdg['E_vis'],
            bins=60,
            range=[[e_kin_min, e_kin_max], [e_vis_min, e_vis_max]],
            cmap='viridis'
        )
        plt.colorbar(label='Counts')
        plt.xlabel('E_true [MeV]')
        plt.ylabel('E_vis [MeV]')
        plt.title(f'{pdg_code_to_name(pdg)}: E_vis vs E_true')
        plt.tight_layout()

        fname = os.path.join(output_dir, f"{pdg_code_to_name(pdg)}_2D_Evis_vs_Etrue.png")
        plt.savefig(fname)
        plt.close()

def plot_evis_vs_ekin(df):
    """Plot 2D histograms of E_vis vs E_kin for each PDG species."""
    particle_species = df['pdg'].unique()
    output_dir = get_output_dir()
    os.makedirs(output_dir, exist_ok=True)

    for pdg in particle_species:
        df_pdg = df[df['pdg'] == pdg]
        e_kin_min, e_kin_max = df_pdg['E_kin'].min(), df_pdg['E_kin'].max()
        e_vis_min, e_vis_max = df_pdg['E_vis'].min(), df_pdg['E_vis'].max()

        # Skip if there's no variation in data (would cause hist2d crash)
        if len(df_pdg) == 0:
            continue
        if e_kin_min == e_kin_max or e_vis_min == e_vis_max:
            print(f"Skipping PDG {pdg_code_to_name(pdg)} due to constant or empty values in E_kin or E_vis.")
            continue

        plt.figure(figsize=(6, 5))
        plt.hist2d(
            df_pdg['E_kin'], df_pdg['E_vis'],
            bins=60,
            range=[[e_kin_min, e_kin_max], [e_vis_min, e_vis_max]],
            cmap='viridis'
        )
        plt.colorbar(label='Counts')
        plt.xlabel('E_kin [MeV]')
        plt.ylabel('E_vis [MeV]')
        plt.title(f'{pdg_code_to_name(pdg)}: E_vis vs E_kin')
        plt.tight_layout()

        fname = os.path.join(output_dir, f"{pdg_code_to_name(pdg)}_2D_Evis_vs_Ekin.png")
        plt.savefig(fname)
        plt.close()

def plot_particle_multiplicity(df):
    output_dir = get_output_dir()
    os.makedirs(output_dir, exist_ok=True)

    all_interactions = df['interaction_id'].unique()
    visible_df = df[df['E_vis'] > 0]

    particle_species = df['pdg'].unique()

    for pdg in particle_species:
        # Count number of visible particles of this pdg per interaction
        visible_pdg = visible_df[visible_df['pdg'] == pdg]
        counts = visible_pdg.groupby('interaction_id').size()

        # Map interaction_id → count, and fill missing ones with 0
        all_counts = pd.Series(0, index=all_interactions)
        all_counts.update(counts)

        multiplicities = all_counts.value_counts().sort_index()

        plt.figure(figsize=(6, 4))
        plt.bar(multiplicities.index, multiplicities.values, color='purple', edgecolor='black')
        plt.xlabel(f'Number of visible {pdg_code_to_name(pdg)} per interaction')
        plt.ylabel('Number of interactions')
        plt.title(f'Multiplicity of {pdg_code_to_name(pdg)} (E_vis > 0)')
        plt.xticks(multiplicities.index)
        plt.tight_layout()

        fname = os.path.join(output_dir, f"{pdg_code_to_name(pdg)}_multiplicity_histogram.png")
        plt.savefig(fname)
        plt.close()

def plot_dwall_vs_containment(df):

    output_dir = get_output_dir()
    os.makedirs(output_dir, exist_ok=True)

    df_filtered = df[np.isfinite(df['d_wall_TPC'])]

    plt.figure(figsize=(7, 5))
    plt.hist2d(
        df_filtered['d_wall_TPC'],
        df_filtered['is_contained_TPC'],        
        bins=[100, 2], 
        range=[[df_filtered['d_wall_TPC'].min(), df_filtered['d_wall_TPC'].max()], 
        [-0.5, 1.5]],
        cmap='viridis'
    )
    plt.colorbar(label='Counts')
    plt.xlabel('d_wall')
    plt.ylabel('is_contained_TPC')
    plt.yticks([0, 1], ['Not Contained', 'Contained'])
    plt.title('2D Histogram: d_wall vs is_contained_TPC')
    plt.tight_layout()

    fname = os.path.join(output_dir, 'hist2d_dwall_vs_is_contained_TPC.png')
    plt.savefig(fname)
    plt.close()

def plot_normalized_dwall_vs_containment_by_pdg(df):
    output_dir = get_output_dir()
    os.makedirs(output_dir, exist_ok=True)

    for pdg in df['pdg'].unique():
        df_pdg = df[df['pdg'] == pdg]

        mask = (
            np.isfinite(df_pdg['d_wall_TPC']) &
            (df_pdg['d_wall_TPC'] > 0) &
            np.isfinite(df_pdg['is_contained_TPC'])
        )
        df_filtered = df_pdg[mask]

        if df_filtered.empty:
            print(f"Skipping PDG {pdg}: no valid values.")
            continue

        # Define bins
        dmin = df_filtered['d_wall_TPC'].min()
        dmax = df_filtered['d_wall_TPC'].max()
        xbins = np.logspace(np.log10(dmin), np.log10(dmax), 50)
        ybins = [-0.5, 0.5, 1.5]  # 0: not contained, 1: contained

        # 2D histogram
        hist, xedges, yedges = np.histogram2d(
            df_filtered['d_wall_TPC'],
            df_filtered['is_contained_TPC'],
            bins=[xbins, ybins]
        )

        # Normalize across the containment axis (i.e., column-wise for each distance bin)
        with np.errstate(divide='ignore', invalid='ignore'):
            hist_norm = hist / hist.sum(axis=1, keepdims=True)
            hist_norm = np.nan_to_num(hist_norm)

        # Plot normalized histogram
        plt.figure(figsize=(7, 5))
        plt.pcolormesh(xedges, yedges, hist_norm.T, cmap='viridis', shading='auto')
        plt.colorbar(label='Fraction')
        plt.xscale('log')
        plt.xlabel('d_wall_TPC [log scale]')
        plt.ylabel('is_contained_TPC')
        plt.yticks([0, 1], ['Not Contained', 'Contained'])
        plt.title(f'PDG {pdg}: Containment Fraction vs d_wall_TPC')
        plt.tight_layout()

        fname = os.path.join(output_dir, f'norm_hist2d_dwallTPC_vs_is_contained_TPC_{pdg_code_to_name(pdg)}.png')
        plt.savefig(fname)
        plt.close()

def plot_evis_over_ekin_vs_containment_per_pdg(df):
    output_dir = get_output_dir()
    os.makedirs(output_dir, exist_ok=True)

    for pdg in df['pdg'].unique():
        df_pdg = df[
            (df['pdg'] == pdg) &
            (df['E_kin'] > 0) &
            (df['E_vis'] > 1) &
            df['is_contained_TPC'].isin([0, 1])
        ].copy()

        if df_pdg.empty:
            continue

        df_pdg['evis_over_ekin'] = df_pdg['E_vis'] / df_pdg['E_kin']

        plt.figure(figsize=(7, 5))
        plt.hist2d(
            df_pdg['evis_over_ekin'],
            df_pdg['is_contained_TPC'],
            bins=[100, 2],
            range=[[0, 1.2], [-0.5, 1.5]],
            cmap='plasma'
        )
        plt.colorbar(label='Counts')
        plt.xlabel('E_vis / E_kin')
        plt.ylabel('is_contained_TPC')
        plt.yticks([0, 1], ['Not Contained', 'Contained'])
        plt.title(f'E_vis / E_kin vs Containment for PDG {pdg}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'evis_over_ekin_vs_containment_pdg_{pdg_code_to_name(pdg)}.png'))
        plt.close()

def plot_evis_over_e_vs_containment_per_pdg(df):
    output_dir = get_output_dir()
    os.makedirs(output_dir, exist_ok=True)

    for pdg in df['pdg'].unique():
        df_pdg = df[
            (df['pdg'] == pdg) &
            (df['E'] > 0) &
            (df['E_vis'] > 1) &
            np.isfinite(df['E_vis']) &
            np.isfinite(df['E']) &
            df['is_contained_TPC'].isin([0, 1])
        ].copy()

        if df_pdg.empty:
            continue

        df_pdg['evis_over_e'] = df_pdg['E_vis'] / df_pdg['E']

        plt.figure(figsize=(7, 5))
        plt.hist2d(
            df_pdg['evis_over_e'],
            df_pdg['is_contained_TPC'],
            bins=[100, 2],
            range=[[0, 1.2], [-0.5, 1.5]],
            cmap='plasma'
        )
        plt.colorbar(label='Counts')
        plt.xlabel('E_vis / E')
        plt.ylabel('is_contained_TPC')
        plt.yticks([0, 1], ['Not Contained', 'Contained'])
        plt.title(f'E_vis / E vs Containment for PDG {pdg}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'evis_over_e_vs_containment_pdg_{pdg_code_to_name(pdg)}.png'))
        plt.close()

def plot_dwall_vs_evis_over_ekin_uncontained_per_pdg(df):
    output_dir = os.path.join(get_output_dir())
    os.makedirs(output_dir, exist_ok=True)

    for pdg in df['pdg'].unique():
        df_pdg = df[
            (df['pdg'] == pdg) &
            (df['is_contained_TPC'] == 0) &
            (df['E_kin'] > 0) &
            (df['E_vis'] > 1) &
            np.isfinite(df['d_wall_TPC'])
        ].copy()

        if df_pdg.empty:
            continue

        df_pdg['evis_over_ekin'] = df_pdg['E_vis'] / df_pdg['E_kin']

        plt.figure(figsize=(7, 5))
        plt.hist2d(
            np.log10(df_pdg['d_wall_TPC'] + 1e-3),
            df_pdg['evis_over_ekin'],
            bins=[100, 100],
            vmax= np.max(df_pdg['evis_over_ekin']),
            cmap='viridis'
        )
        plt.colorbar(label='Counts')
        plt.xlabel('log10(d_wall_TPC + 1e-3) [cm]')
        plt.ylabel('E_vis / E_kin')
        plt.title(f'PDG {pdg} (Uncontained) — d_wall_TPC vs E_vis/E_kin')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'dwall_vs_evis_over_ekin_pdg_{pdg_code_to_name(pdg)}.png'))
        plt.close()

def plot_dwall_vs_evis_over_e_uncontained_per_pdg(df):
    output_dir = os.path.join(get_output_dir())
    os.makedirs(output_dir, exist_ok=True)

    for pdg in df['pdg'].unique():
        df_pdg = df[
            (df['pdg'] == pdg) &
            (df['is_contained_TPC'] == 0) &
            (df['E'] > 0) &
            (df['E_vis'] > 1) &
            np.isfinite(df['d_wall_TPC'])
        ].copy()

        if df_pdg.empty:
            continue

        df_pdg['evis_over_e'] = df_pdg['E_vis'] / df_pdg['E']

        plt.figure(figsize=(7, 5))
        plt.hist2d(
            df_pdg['d_wall_TPC'],
            df_pdg['evis_over_e'],
            bins=[30, 30],
            vmax= np.max(df_pdg['evis_over_e']),
            cmap='viridis'
        )
        plt.colorbar(label='Counts')
        plt.xlabel('log10(d_wall_TPC + 1e-3) [cm]')
        plt.ylabel('E_vis / E')
        plt.title(f'PDG {pdg} (Uncontained) — d_wall_TPC vs E_vis/E')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'dwall_vs_evis_over_e_pdg_{pdg_code_to_name(pdg)}.png'))
        plt.close()

def plot_dwall(df):
    output_dir = os.path.join(get_output_dir())
    os.makedirs(output_dir, exist_ok=True)

    dwall = df['d_wall_TPC'].copy()
    dwall = dwall.replace([np.inf, -np.inf], np.nan)
    dwall = dwall.dropna()

    plt.figure(figsize=(7, 5))
    plt.hist(dwall, bins=100, range=(0, 4e6), histtype='step', color='black', linewidth=1.5)
    plt.xlabel("d_wall_TPC [mm]")
    plt.ylabel("Counts")
    plt.title("1D Histogram of d_wall_TPC (Inf → 9,999,999)")
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dwall_1d_all_particles.png"))
    plt.close()


def plot_evis_over_etrue_vs_etrue(df): 
    output_dir = get_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    particle_species = df['pdg'].unique()
    for pdg in particle_species:
        df_pdg = df[
            (df['pdg'] == pdg) &
            (df['E'] > 0) &
            (df['is_contained_TPC'] == 0) 
        ].copy()
        if df_pdg.empty:
            continue
        ratio = df_pdg['E_vis'] / df_pdg['E']
        e_true = df_pdg['E']
        plt.figure(figsize=(7, 5))
        # e_true_min, e_true_max = df_pdg['E'].min(), df_pdg['E'].max()
        hist, xedges, yedges = np.histogram2d(
            e_true, ratio,
            bins=[25, 25],
            range=[[1, 5000], [0, 1.2]]
        )
        hist_masked = np.ma.masked_where(hist == 0, hist)
        cmap = plt.colormaps['viridis'].copy()
        cmap.set_bad(color='white')  # This sets masked (zero) bins to white
        plt.pcolormesh(
            xedges, yedges, hist_masked.T,
            cmap=cmap,
            norm=Normalize(vmin=hist_masked.min(), vmax=hist_masked.max())
        )
        plt.colorbar(label='Counts')
        plt.xlabel('E_true [MeV]')
        plt.ylabel('E_vis / E_true')
        plt.title(f'{pdg_code_to_name(pdg)}: E_vis/E_true vs E_true')
        plt.tight_layout()
        fname = os.path.join(output_dir, f"{pdg_code_to_name(pdg)}_2D_EvisOverEtrue_vs_Etrue.png")
        plt.savefig(fname)
        plt.close()

def plot_evis_over_etrue_vs_theta(df): 
    output_dir = get_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    particle_species = df['pdg'].unique()
    for pdg in particle_species:
        df_pdg = df[
            (df['pdg'] == pdg) &
            (df['E'] > 0) &
            (df['is_contained_TPC'] == 0) 
        ].copy()
        if df_pdg.empty:
            continue
        ratio = df_pdg['E_vis'] / df_pdg['E']
        theta = df_pdg['theta']
        plt.figure(figsize=(7, 5))
        hist, xedges, yedges = np.histogram2d(
            theta, ratio,
            bins=[50, 50],
            range=[[0, np.pi], [0, 1.2]]
        )
        hist_masked = np.ma.masked_where(hist == 0, hist)
        cmap = plt.colormaps['viridis'].copy()
        cmap.set_bad(color='white')  # This sets masked (zero) bins to white
        plt.pcolormesh(
            xedges, yedges, hist_masked.T,
            cmap=cmap,
            norm=Normalize(vmin=hist_masked.min(), vmax=hist_masked.max())
        )
        plt.colorbar(label='Counts')
        plt.xlabel('theta [rad]')
        plt.ylabel('E_vis / E_true')
        plt.title(f'{pdg_code_to_name(pdg)}: E_vis/E_true vs theta')
        plt.tight_layout()
        fname = os.path.join(output_dir, f"{pdg_code_to_name(pdg)}_2D_EvisOverEtrue_vs_theta.png")
        plt.savefig(fname)
        plt.close()

def plot_evis_over_etrue_vs_phi(df): 
    output_dir = get_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    particle_species = df['pdg'].unique()
    for pdg in particle_species:
        df_pdg = df[
            (df['pdg'] == pdg) &
            (df['E'] > 0) &
            (df['is_contained_TPC'] == 0) 
        ].copy()
        if df_pdg.empty:
            continue
        ratio = df_pdg['E_vis'] / df_pdg['E']
        phi = df_pdg['phi']
        plt.figure(figsize=(7, 5))
        hist, xedges, yedges = np.histogram2d(
            phi, ratio,
            bins=[50, 50],
            range=[[-np.pi/2, np.pi/2], [0, 1.2]]
        )
        hist_masked = np.ma.masked_where(hist == 0, hist)
        cmap = plt.colormaps['viridis'].copy()
        cmap.set_bad(color='white')  # This sets masked (zero) bins to white
        plt.pcolormesh(
            xedges, yedges, hist_masked.T,
            cmap=cmap,
            norm=Normalize(vmin=hist_masked.min(), vmax=hist_masked.max())
        )
        plt.colorbar(label='Counts')
        plt.xlabel('phi [rad]')
        plt.ylabel('E_vis / E_true')
        plt.title(f'{pdg_code_to_name(pdg)}: E_vis/E_true vs phi')
        plt.tight_layout()
        fname = os.path.join(output_dir, f"{pdg_code_to_name(pdg)}_2D_EvisOverEtrue_vs_phi.png")
        plt.savefig(fname)
        plt.close()

def plot_evis_over_etrue_vs_d_wall(df): 
    output_dir = get_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    particle_species = df['pdg'].unique()
    for pdg in particle_species:
        df_pdg = df[
            (df['pdg'] == pdg) &
            (df['E'] > 0) &
            (df['is_contained_TPC'] == 1) &
            (df['start_x'] > -3478.48) & (df['start_x'] < 3478.48) &
            (df['start_y'] > -2166.71) & (df['start_y'] < 829.282) &
            (df['start_z'] > 4179.24) & (df['start_z'] < 9135.88)
        ].copy()
        if df_pdg.empty:
            continue
        ratio = df_pdg['E_vis'] / df_pdg['E']
        d_wall = df_pdg['d_wall_TPC']/1000

        plt.figure(figsize=(7, 5))
        hist, xedges, yedges = np.histogram2d(
            d_wall, ratio,
            bins=[50, 50],
            range=[[0, 5], [0, 1.2]]
        )
        hist_masked = np.ma.masked_where(hist == 0, hist)
        cmap = plt.colormaps['viridis'].copy()
        cmap.set_bad(color='white')  # This sets masked (zero) bins to white
        plt.pcolormesh(
            xedges, yedges, hist_masked.T,
            cmap=cmap,
            norm=Normalize(vmin=hist_masked.min(), vmax=hist_masked.max())
        )
        plt.colorbar(label='Counts')
        plt.xlabel('d_wall [m]')
        plt.ylabel('E_vis / E_true')
        plt.title(f'{pdg_code_to_name(pdg)}: E_vis/E_true vs d_wall')
        plt.tight_layout()
        fname = os.path.join(output_dir, f"{pdg_code_to_name(pdg)}_2D_EvisOverEtrue_vs_d_wall.png")
        plt.savefig(fname)
        plt.close()

def plot_evis_over_etrue_vs_x(df): # Check inside fiducial volume
    output_dir = get_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    particle_species = df['pdg'].unique()
    for pdg in particle_species:
        df_pdg = df[
            (df['pdg'] == pdg) &
            (df['E'] > 0) &
            (df['is_contained_TPC'] == 1) &
            (df['start_x'] > -3478.48) & (df['start_x'] < 3478.48) &
            (df['start_y'] > -2166.71) & (df['start_y'] < 829.282) &
            (df['start_z'] > 4179.24) & (df['start_z'] < 9135.88)
        ].copy()
        if df_pdg.empty:
            continue
        ratio = df_pdg['E_vis'] / df_pdg['E']
        x = df_pdg['start_x'] / 1000  # convert mm to m
        plt.figure(figsize=(7, 5))
        hist, xedges, yedges = np.histogram2d(
            x, ratio,
            bins=[50, 50],
            range=[[-3.47848, 3.47848], [0, 1.2]]
        )
        hist_masked = np.ma.masked_where(hist == 0, hist)
        cmap = plt.colormaps['viridis'].copy()
        cmap.set_bad(color='white')
        plt.pcolormesh(
            xedges, yedges, hist_masked.T,
            cmap=cmap,
            norm=Normalize(vmin=hist_masked.min(), vmax=hist_masked.max())
        )
        plt.colorbar(label='Counts')
        plt.xlabel('start_x [m]')
        plt.ylabel('E_vis / E_true')
        plt.title(f'{pdg_code_to_name(pdg)}: E_vis/E_true vs start_x')
        plt.tight_layout()
        fname = os.path.join(output_dir, f"{pdg_code_to_name(pdg)}_2D_EvisOverEtrue_vs_start_x.png")
        plt.savefig(fname)
        plt.close()

        
def plot_evis_over_etrue_vs_y(df): 
    output_dir = get_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    particle_species = df['pdg'].unique()
    for pdg in particle_species:
        df_pdg = df[
            (df['pdg'] == pdg) &
            (df['E'] > 0) &
            (df['is_contained_TPC'] == 1) &
            (df['start_x'] > -3478.48) & (df['start_x'] < 3478.48) &
            (df['start_y'] > -2166.71) & (df['start_y'] < 829.282) &
            (df['start_z'] > 4179.24) & (df['start_z'] < 9135.88)
        ].copy()
        if df_pdg.empty:
            continue
        ratio = df_pdg['E_vis'] / df_pdg['E']
        y = df_pdg['start_y'] / 1000  # convert mm to m

        plt.figure(figsize=(7, 5))
        hist, xedges, yedges = np.histogram2d(
            y, ratio,
            bins=[50, 50],
            range=[[-2.16671, 0.829282], [0, 1.2]]
        )
        hist_masked = np.ma.masked_where(hist == 0, hist)
        cmap = plt.colormaps['viridis'].copy()
        cmap.set_bad(color='white')
        plt.pcolormesh(
            xedges, yedges, hist_masked.T,
            cmap=cmap,
            norm=Normalize(vmin=hist_masked.min(), vmax=hist_masked.max())
        )
        plt.colorbar(label='Counts')
        plt.xlabel('start_y [m]')
        plt.ylabel('E_vis / E_true')
        plt.title(f'{pdg_code_to_name(pdg)}: E_vis/E_true vs start_y')
        plt.tight_layout()
        fname = os.path.join(output_dir, f"{pdg_code_to_name(pdg)}_2D_EvisOverEtrue_vs_start_y.png")
        plt.savefig(fname)
        plt.close()

def plot_evis_over_etrue_vs_z(df): 
    output_dir = get_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    particle_species = df['pdg'].unique()
    for pdg in particle_species:
        df_pdg = df[
            (df['pdg'] == pdg) &
            (df['E'] > 0) &
            (df['is_contained_TPC'] == 1) &
            (df['start_x'] > -3478.48) & (df['start_x'] < 3478.48) &
            (df['start_y'] > -2166.71) & (df['start_y'] < 829.282) &
            (df['start_z'] > 4179.24) & (df['start_z'] < 9135.88)
        ].copy()
        if df_pdg.empty:
            continue
            
        ratio = df_pdg['E_vis'] / df_pdg['E']
        z = df_pdg['start_z'] / 1000  # convert mm to m

        plt.figure(figsize=(7, 5))
        hist, xedges, yedges = np.histogram2d(
            z, ratio,
            bins=[50, 50],
            range=[[4.17924, 9.13588], [0, 1.2]]
        )
        hist_masked = np.ma.masked_where(hist == 0, hist)
        cmap = plt.colormaps['viridis'].copy()
        cmap.set_bad(color='white')
        plt.pcolormesh(
            xedges, yedges, hist_masked.T,
            cmap=cmap,
            norm=Normalize(vmin=hist_masked.min(), vmax=hist_masked.max())
        )
        plt.colorbar(label='Counts')
        plt.xlabel('start_z [m]')
        plt.ylabel('E_vis / E_true')
        plt.title(f'{pdg_code_to_name(pdg)}: E_vis/E_true vs start_z')
        plt.tight_layout()
        fname = os.path.join(output_dir, f"{pdg_code_to_name(pdg)}_2D_EvisOverEtrue_vs_start_z.png")
        plt.savefig(fname)
        plt.close()

def plot_evis_over_etrue_vs_x_not_contained(df): # Check inside fiducial volume
    output_dir = get_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    particle_species = df['pdg'].unique()
    for pdg in particle_species:
        x_exclude_half = [3000, 2000, 1000, 0, -1000, -2000, -3000]
        x_exclude_full = [3500, 2500, 1500, 500, -500, -1500, -2500]
        z_exclude = [5157.559, 6157.559, 7157.559, 8157.559]

        df_pdg = df[
            (df['pdg'] == pdg) &
            (df['E'] > 0) &
            (df['is_contained_TPC'] == 0) &
            (df['start_x'] > (-3478.48 + 500)) & (df['start_x'] < (3478.48 - 500)) &
            (df['start_y'] > (-2166.71 + 500)) & (df['start_y'] < (829.282 - 500)) &
            (df['start_z'] > 4179.24) & (df['start_z'] < (9135.88 - 1500))
        ].copy()

        # Exclude regions using DataFrame cuts
        for xc in x_exclude_half:
            df_pdg = df_pdg[~((df_pdg['start_x'] > (xc - 3.175)) & (df_pdg['start_x'] < (xc + 3.175)))]
        for xc in x_exclude_full:
            df_pdg = df_pdg[~((df_pdg['start_x'] > (xc - 28.915)) & (df_pdg['start_x'] < (xc + 28.915)))]
        for zc in z_exclude:
            df_pdg = df_pdg[~((df_pdg['start_z'] > (zc - 5)) & (df_pdg['start_z'] < (zc + 5)))]
        if df_pdg.empty:
            continue
        ratio = df_pdg['E_vis'] / df_pdg['E']
        x = df_pdg['start_x'] / 1000  # convert mm to m
        plt.figure(figsize=(7, 5))
        hist, xedges, yedges = np.histogram2d(
            x, ratio,
            bins=[50, 50],
            range=[[-3.47848, 3.47848], [0, 1.2]]
        )
        hist_masked = np.ma.masked_where(hist == 0, hist)
        cmap = plt.colormaps['viridis'].copy()
        cmap.set_bad(color='white')
        plt.pcolormesh(
            xedges, yedges, hist_masked.T,
            cmap=cmap,
            norm=Normalize(vmin=hist_masked.min(), vmax=hist_masked.max())
        )
        plt.colorbar(label='Counts')
        plt.xlabel('start_x [m]')
        plt.ylabel('E_vis / E_true')
        plt.title(f'{pdg_code_to_name(pdg)}: E_vis/E_true vs start_x_not_contained')
        plt.tight_layout()
        fname = os.path.join(output_dir, f"{pdg_code_to_name(pdg)}_2D_EvisOverEtrue_vs_start_x_not_contained.png")
        plt.savefig(fname)
        plt.close()

        
def plot_evis_over_etrue_vs_y_not_contained(df): 
    output_dir = get_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    particle_species = df['pdg'].unique()
    for pdg in particle_species:

        x_exclude_half = [3000, 2000, 1000, 0, -1000, -2000, -3000]
        x_exclude_full = [3500, 2500, 1500, 500, -500, -1500, -2500]
        z_exclude = [5157.559, 6157.559, 7157.559, 8157.559]

        df_pdg = df[
            (df['pdg'] == pdg) &
            (df['E'] > 0) &
            (df['is_contained_TPC'] == 0) &
            (df['start_x'] > (-3478.48+500)) & (df['start_x'] < (3478.48-500)) &
            (df['start_y'] > (-2166.71+500)) & (df['start_y'] < (829.282-500)) &
            (df['start_z'] > 4179.24) & (df['start_z'] < (9135.88-1500))
        ].copy()
        if df_pdg.empty:
            continue

        mask_exclude = np.ones(len(df_pdg), dtype=bool)
        for xc in x_exclude_half:
            mask_exclude &= ~((df_pdg['start_x'] > (xc - 3.175)) & (df_pdg['start_x'] < (xc + 3.175)))
        for xc in x_exclude_full:
            mask_exclude &= ~((df_pdg['start_x'] > (xc - 28.915)) & (df_pdg['start_x'] < (xc + 28.915)))
        for zc in z_exclude:
            mask_exclude &= ~((df_pdg['start_z'] > (zc - 5)) & (df_pdg['start_z'] < (zc + 5)))
        

        ratio = df_pdg['E_vis'] / df_pdg['E']
        y = df_pdg['start_y'] / 1000  # convert mm to m

        plt.figure(figsize=(7, 5))
        hist, xedges, yedges = np.histogram2d(
            y, ratio,
            bins=[50, 50],
            range=[[-2.16671, 0.829282], [0, 1.2]]
        )
        hist_masked = np.ma.masked_where(hist == 0, hist)
        cmap = plt.colormaps['viridis'].copy()
        cmap.set_bad(color='white')
        plt.pcolormesh(
            xedges, yedges, hist_masked.T,
            cmap=cmap,
            norm=Normalize(vmin=hist_masked.min(), vmax=hist_masked.max())
        )
        plt.colorbar(label='Counts')
        plt.xlabel('start_y [m]')
        plt.ylabel('E_vis / E_true')
        plt.title(f'{pdg_code_to_name(pdg)}: E_vis/E_true vs start_y_not_contained')
        plt.tight_layout()
        fname = os.path.join(output_dir, f"{pdg_code_to_name(pdg)}_2D_EvisOverEtrue_vs_start_y_not_contained.png")
        plt.savefig(fname)
        plt.close()

def plot_evis_over_etrue_vs_z_not_contained(df): 
    output_dir = get_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    particle_species = df['pdg'].unique()
    for pdg in particle_species:

        x_exclude_half = [3000, 2000, 1000, 0, -1000, -2000, -3000]
        x_exclude_full = [3500, 2500, 1500, 500, -500, -1500, -2500]
        z_exclude = [5157.559, 6157.559, 7157.559, 8157.559]

        df_pdg = df[
            (df['pdg'] == pdg) &
            (df['E'] > 0) &
            (df['is_contained_TPC'] == 0) &
            (df['start_x'] > (-3478.48+500)) & (df['start_x'] < (3478.48-500)) &
            (df['start_y'] > (-2166.71+500)) & (df['start_y'] < (829.282-500)) &
            (df['start_z'] > 4179.24) & (df['start_z'] < (9135.88-1500))
        ].copy()
        if df_pdg.empty:
            continue

        mask_exclude = np.ones(len(df_pdg), dtype=bool)
        for xc in x_exclude_half:
            mask_exclude &= ~((df_pdg['start_x'] > (xc - 3.175)) & (df_pdg['start_x'] < (xc + 3.175)))
        for xc in x_exclude_full:
            mask_exclude &= ~((df_pdg['start_x'] > (xc - 28.915)) & (df_pdg['start_x'] < (xc + 28.915)))
        for zc in z_exclude:
            mask_exclude &= ~((df_pdg['start_z'] > (zc - 5)) & (df_pdg['start_z'] < (zc + 5)))
        

        ratio = df_pdg['E_vis'] / df_pdg['E']
        z = df_pdg['start_z'] / 1000  # convert mm to m

        plt.figure(figsize=(7, 5))
        hist, xedges, yedges = np.histogram2d(
            z, ratio,
            bins=[50, 50],
            range=[[4.17924, 9.13588], [0, 1.2]]
        )
        hist_masked = np.ma.masked_where(hist == 0, hist)
        cmap = plt.colormaps['viridis'].copy()
        cmap.set_bad(color='white')
        plt.pcolormesh(
            xedges, yedges, hist_masked.T,
            cmap=cmap,
            norm=Normalize(vmin=hist_masked.min(), vmax=hist_masked.max())
        )
        plt.colorbar(label='Counts')
        plt.xlabel('start_z [m]')
        plt.ylabel('E_vis / E_true')
        plt.title(f'{pdg_code_to_name(pdg)}: E_vis/E_true vs start_z_not_contained')
        plt.tight_layout()
        fname = os.path.join(output_dir, f"{pdg_code_to_name(pdg)}_2D_EvisOverEtrue_vs_start_z_not_contained.png")
        plt.savefig(fname)
        plt.close()

def plot_evis_over_etrue_vs_x_contained(df): # Check inside fiducial volume
    output_dir = get_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    particle_species = df['pdg'].unique()
    for pdg in particle_species:

        x_exclude_half = [3000, 2000, 1000, 0, -1000, -2000, -3000]
        x_exclude_full = [3500, 2500, 1500, 500, -500, -1500, -2500]
        z_exclude = [5157.559, 6157.559, 7157.559, 8157.559]

        df_pdg = df[
            (df['pdg'] == pdg) &
            (df['E'] > 0) &
            (df['is_contained_TPC'] == 1) &
            (df['start_x'] > (-3478.48+500)) & (df['start_x'] < (3478.48-500)) &
            (df['start_y'] > (-2166.71+500)) & (df['start_y'] < (829.282-500)) &
            (df['start_z'] > 4179.24) & (df['start_z'] < (9135.88-1500))
        ].copy()
        if df_pdg.empty:
            continue

        mask_exclude = np.ones(len(df_pdg), dtype=bool)
        for xc in x_exclude_half:
            mask_exclude &= ~((df_pdg['start_x'] > (xc - 3.175)) & (df_pdg['start_x'] < (xc + 3.175)))
        for xc in x_exclude_full:
            mask_exclude &= ~((df_pdg['start_x'] > (xc - 28.915)) & (df_pdg['start_x'] < (xc + 28.915)))
        for zc in z_exclude:
            mask_exclude &= ~((df_pdg['start_z'] > (zc - 5)) & (df_pdg['start_z'] < (zc + 5)))
        

        ratio = df_pdg['E_vis'] / df_pdg['E']
        x = df_pdg['start_x'] / 1000  # convert mm to m
        plt.figure(figsize=(7, 5))
        hist, xedges, yedges = np.histogram2d(
            x, ratio,
            bins=[50, 50],
            range=[[-3.47848, 3.47848], [0, 1.2]]
        )
        hist_masked = np.ma.masked_where(hist == 0, hist)
        cmap = plt.colormaps['viridis'].copy()
        cmap.set_bad(color='white')
        plt.pcolormesh(
            xedges, yedges, hist_masked.T,
            cmap=cmap,
            norm=Normalize(vmin=hist_masked.min(), vmax=hist_masked.max())
        )
        plt.colorbar(label='Counts')
        plt.xlabel('start_x [m]')
        plt.ylabel('E_vis / E_true')
        plt.title(f'{pdg_code_to_name(pdg)}: E_vis/E_true vs start_x_contained')
        plt.tight_layout()
        fname = os.path.join(output_dir, f"{pdg_code_to_name(pdg)}_2D_EvisOverEtrue_vs_start_x_contained.png")
        plt.savefig(fname)
        plt.close()

        
def plot_evis_over_etrue_vs_y_contained(df): 
    output_dir = get_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    particle_species = df['pdg'].unique()
    for pdg in particle_species:

        x_exclude_half = [3000, 2000, 1000, 0, -1000, -2000, -3000]
        x_exclude_full = [3500, 2500, 1500, 500, -500, -1500, -2500]
        z_exclude = [5157.559, 6157.559, 7157.559, 8157.559]

        df_pdg = df[
            (df['pdg'] == pdg) &
            (df['E'] > 0) &
            (df['is_contained_TPC'] == 1) &
            (df['start_x'] > (-3478.48+500)) & (df['start_x'] < (3478.48-500)) &
            (df['start_y'] > (-2166.71+500)) & (df['start_y'] < (829.282-500)) &
            (df['start_z'] > 4179.24) & (df['start_z'] < (9135.88-1500))
        ].copy()
        if df_pdg.empty:
            continue

        mask_exclude = np.ones(len(df_pdg), dtype=bool)
        for xc in x_exclude_half:
            mask_exclude &= ~((df_pdg['start_x'] > (xc - 3.175)) & (df_pdg['start_x'] < (xc + 3.175)))
        for xc in x_exclude_full:
            mask_exclude &= ~((df_pdg['start_x'] > (xc - 28.915)) & (df_pdg['start_x'] < (xc + 28.915)))
        for zc in z_exclude:
            mask_exclude &= ~((df_pdg['start_z'] > (zc - 5)) & (df_pdg['start_z'] < (zc + 5)))

        ratio = df_pdg['E_vis'] / df_pdg['E']
        y = df_pdg['start_y'] / 1000  # convert mm to m

        plt.figure(figsize=(7, 5))
        hist, xedges, yedges = np.histogram2d(
            y, ratio,
            bins=[50, 50],
            range=[[-2.16671, 0.829282], [0, 1.2]]
        )
        hist_masked = np.ma.masked_where(hist == 0, hist)
        cmap = plt.colormaps['viridis'].copy()
        cmap.set_bad(color='white')
        plt.pcolormesh(
            xedges, yedges, hist_masked.T,
            cmap=cmap,
            norm=Normalize(vmin=hist_masked.min(), vmax=hist_masked.max())
        )
        plt.colorbar(label='Counts')
        plt.xlabel('start_y [m]')
        plt.ylabel('E_vis / E_true')
        plt.title(f'{pdg_code_to_name(pdg)}: E_vis/E_true vs start_y_contained')
        plt.tight_layout()
        fname = os.path.join(output_dir, f"{pdg_code_to_name(pdg)}_2D_EvisOverEtrue_vs_start_y_contained.png")
        plt.savefig(fname)
        plt.close()

def plot_evis_over_etrue_vs_z_contained(df): 
    output_dir = get_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    particle_species = df['pdg'].unique()
    for pdg in particle_species:

        x_exclude_half = [3000, 2000, 1000, 0, -1000, -2000, -3000]
        x_exclude_full = [3500, 2500, 1500, 500, -500, -1500, -2500]
        z_exclude = [5157.559, 6157.559, 7157.559, 8157.559]

        df_pdg = df[
            (df['pdg'] == pdg) &
            (df['E'] > 0) &
            (df['is_contained_TPC'] == 1) &
            (df['start_x'] > (-3478.48+500)) & (df['start_x'] < (3478.48-500)) &
            (df['start_y'] > (-2166.71+500)) & (df['start_y'] < (829.282-500)) &
            (df['start_z'] > 4179.24) & (df['start_z'] < (9135.88-1500))
        ].copy()
        if df_pdg.empty:
            continue

        mask_exclude = np.ones(len(df_pdg), dtype=bool)
        for xc in x_exclude_half:
            mask_exclude &= ~((df_pdg['start_x'] > (xc - 3.175)) & (df_pdg['start_x'] < (xc + 3.175)))
        for xc in x_exclude_full:
            mask_exclude &= ~((df_pdg['start_x'] > (xc - 28.915)) & (df_pdg['start_x'] < (xc + 28.915)))
        for zc in z_exclude:
            mask_exclude &= ~((df_pdg['start_z'] > (zc - 5)) & (df_pdg['start_z'] < (zc + 5)))
        
        df_pdg = df_pdg[mask_exclude]

        ratio = df_pdg['E_vis'] / df_pdg['E']
        z = df_pdg['start_z'] / 1000  # convert mm to m

        plt.figure(figsize=(7, 5))
        hist, xedges, yedges = np.histogram2d(
            z, ratio,
            bins=[50, 50],
            range=[[4.17924, 9.13588], [0, 1.2]]
        )
        hist_masked = np.ma.masked_where(hist == 0, hist)
        cmap = plt.colormaps['viridis'].copy()
        cmap.set_bad(color='white')
        plt.pcolormesh(
            xedges, yedges, hist_masked.T,
            cmap=cmap,
            norm=Normalize(vmin=hist_masked.min(), vmax=hist_masked.max())
        )
        plt.colorbar(label='Counts')
        plt.xlabel('start_z [m]')
        plt.ylabel('E_vis / E_true')
        plt.title(f'{pdg_code_to_name(pdg)}: E_vis/E_true vs start_z_contained')
        plt.tight_layout()
        fname = os.path.join(output_dir, f"{pdg_code_to_name(pdg)}_2D_EvisOverEtrue_vs_start_z_contained.png")
        plt.savefig(fname)
        plt.close()

def plot_evis_over_etrue_vs_x_not_contained_norm_columns(df):
    output_dir = get_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    particle_species = df['pdg'].unique()
    for pdg in particle_species:
        x_exclude_half = [3000, 2000, 1000, 0, -1000, -2000, -3000]
        x_exclude_full = [3500, 2500, 1500, 500, -500, -1500, -2500]
        z_exclude = [5157.559, 6157.559, 7157.559, 8157.559]

        df_pdg = df[
            (df['pdg'] == pdg) &
            (df['E'] > 0) &
            (df['is_contained_TPC'] == 0) &
            (df['start_x'] > (-3478.48 + 500)) & (df['start_x'] < (3478.48 - 500)) &
            (df['start_y'] > (-2166.71 + 500)) & (df['start_y'] < (829.282 - 500)) &
            (df['start_z'] > 4179.24) & (df['start_z'] < (9135.88 - 1500))
        ].copy()

        # Exclude regions using DataFrame cuts
        for xc in x_exclude_half:
            df_pdg = df_pdg[~((df_pdg['start_x'] > (xc - 3.175)) & (df_pdg['start_x'] < (xc + 3.175)))]
        for xc in x_exclude_full:
            df_pdg = df_pdg[~((df_pdg['start_x'] > (xc - 28.915)) & (df_pdg['start_x'] < (xc + 28.915)))]
        for zc in z_exclude:
            df_pdg = df_pdg[~((df_pdg['start_z'] > (zc - 5)) & (df_pdg['start_z'] < (zc + 5)))]
        if df_pdg.empty:
            continue
        ratio = df_pdg['E_vis'] / df_pdg['E']
        y = df_pdg['start_x'] / 1000  # convert mm to m
        plt.figure(figsize=(7, 5))
        hist, xedges, yedges = np.histogram2d(
            y, ratio,
            bins=[30, 30],
            range=[[-2.16671, 0.829282], [0, 1.2]]
        )
        # Normalize columns (for each y bin, sum over ratio bins)
        with np.errstate(divide='ignore', invalid='ignore'):
            col_sums = hist.sum(axis=1, keepdims=True)
            hist_norm = hist / col_sums
            hist_norm = np.nan_to_num(hist_norm)
        col_mask = (hist.sum(axis=1) == 0)
        mask_2d = np.tile(col_mask[:, np.newaxis], (1, hist.shape[1]))
        hist_masked = np.ma.masked_where(mask_2d, hist_norm)
        cmap = plt.colormaps['viridis'].copy()
        cmap.set_bad(color='white')
        plt.pcolormesh(
            xedges, yedges, hist_masked.T,
            cmap=cmap
        )
        plt.colorbar(label='Fraction')
        plt.xlabel('start_x [m]')
        plt.ylabel('E_vis / E_true')
        plt.title(f'{pdg_code_to_name(pdg)}: Normalized E_vis/E_true vs start_x_not_contained')
        plt.tight_layout()

        y_bin_centers = 0.5 * (xedges[:-1] + xedges[1:])
        avg_ratios = []
        for i in range(len(xedges) - 1):
            mask = (y >= xedges[i]) & (y < xedges[i+1])
            if np.any(mask):
                avg_ratios.append(ratio[mask].mean())
            else:
                avg_ratios.append(np.nan)
        avg_ratios = np.array(avg_ratios)
        valid = ~np.isnan(avg_ratios)
        plt.errorbar(
            y_bin_centers[valid], avg_ratios[valid],
            fmt='x', color='red', markersize=7, label='Average ratio'
        )
        plt.legend()

        fname = os.path.join(output_dir, f"{pdg_code_to_name(pdg)}_2D_EvisOverEtrue_vs_start_x_not_contained_norm_columns.png")
        plt.savefig(fname)
        plt.close()

def plot_evis_over_etrue_vs_x_contained_norm_columns(df):
    output_dir = get_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    particle_species = df['pdg'].unique()
    for pdg in particle_species:
        x_exclude_half = [3000, 2000, 1000, 0, -1000, -2000, -3000]
        x_exclude_full = [3500, 2500, 1500, 500, -500, -1500, -2500]
        z_exclude = [5157.559, 6157.559, 7157.559, 8157.559]

        df_pdg = df[
            (df['pdg'] == pdg) &
            (df['E'] > 0) &
            (df['is_contained_TPC'] == 1) &
            (df['start_x'] > (-3478.48 + 500)) & (df['start_x'] < (3478.48 - 500)) &
            (df['start_y'] > (-2166.71 + 500)) & (df['start_y'] < (829.282 - 500)) &
            (df['start_z'] > 4179.24) & (df['start_z'] < (9135.88 - 1500))
        ].copy()

        # Exclude regions using DataFrame cuts
        for xc in x_exclude_half:
            df_pdg = df_pdg[~((df_pdg['start_x'] > (xc - 3.175)) & (df_pdg['start_x'] < (xc + 3.175)))]
        for xc in x_exclude_full:
            df_pdg = df_pdg[~((df_pdg['start_x'] > (xc - 28.915)) & (df_pdg['start_x'] < (xc + 28.915)))]
        for zc in z_exclude:
            df_pdg = df_pdg[~((df_pdg['start_z'] > (zc - 5)) & (df_pdg['start_z'] < (zc + 5)))]
        if df_pdg.empty:
            continue
        ratio = df_pdg['E_vis'] / df_pdg['E']
        y = df_pdg['start_x'] / 1000  # convert mm to m
        plt.figure(figsize=(7, 5))
        hist, xedges, yedges = np.histogram2d(
            y, ratio,
            bins=[30, 30],
            range=[[-2.16671, 0.829282], [0, 1.2]]
        )
        # Normalize columns (for each y bin, sum over ratio bins)
        with np.errstate(divide='ignore', invalid='ignore'):
            col_sums = hist.sum(axis=1, keepdims=True)
            hist_norm = hist / col_sums
            hist_norm = np.nan_to_num(hist_norm)
        col_mask = (hist.sum(axis=1) == 0)
        mask_2d = np.tile(col_mask[:, np.newaxis], (1, hist.shape[1]))
        hist_masked = np.ma.masked_where(mask_2d, hist_norm)
        cmap = plt.colormaps['viridis'].copy()
        cmap.set_bad(color='white')
        plt.pcolormesh(
            xedges, yedges, hist_masked.T,
            cmap=cmap
        )
        plt.colorbar(label='Fraction')
        plt.xlabel('start_x [m]')
        plt.ylabel('E_vis / E_true')
        plt.title(f'{pdg_code_to_name(pdg)}: Normalized E_vis/E_true vs start_x_contained')
        plt.tight_layout()

        y_bin_centers = 0.5 * (xedges[:-1] + xedges[1:])
        avg_ratios = []
        for i in range(len(xedges) - 1):
            mask = (y >= xedges[i]) & (y < xedges[i+1])
            if np.any(mask):
                avg_ratios.append(ratio[mask].mean())
            else:
                avg_ratios.append(np.nan)
        avg_ratios = np.array(avg_ratios)
        valid = ~np.isnan(avg_ratios)
        plt.errorbar(
            y_bin_centers[valid], avg_ratios[valid],
            fmt='x', color='red', markersize=7, label='Average ratio'
        )
        plt.legend()

        fname = os.path.join(output_dir, f"{pdg_code_to_name(pdg)}_2D_EvisOverEtrue_vs_start_x_contained_norm_columns.png")
        plt.savefig(fname)
        plt.close()

def plot_evis_over_etrue_vs_y_not_contained_norm_columns(df):
    output_dir = get_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    particle_species = df['pdg'].unique()
    for pdg in particle_species:
        x_exclude_half = [3000, 2000, 1000, 0, -1000, -2000, -3000]
        x_exclude_full = [3500, 2500, 1500, 500, -500, -1500, -2500]
        z_exclude = [5157.559, 6157.559, 7157.559, 8157.559]

        df_pdg = df[
            (df['pdg'] == pdg) &
            (df['E'] > 0) &
            (df['is_contained_TPC'] == 0) &
            (df['start_x'] > (-3478.48 + 500)) & (df['start_x'] < (3478.48 - 500)) &
            (df['start_y'] > (-2166.71 + 500)) & (df['start_y'] < (829.282 - 500)) &
            (df['start_z'] > 4179.24) & (df['start_z'] < (9135.88 - 1500))
        ].copy()

        # Exclude regions using DataFrame cuts
        for xc in x_exclude_half:
            df_pdg = df_pdg[~((df_pdg['start_x'] > (xc - 3.175)) & (df_pdg['start_x'] < (xc + 3.175)))]
        for xc in x_exclude_full:
            df_pdg = df_pdg[~((df_pdg['start_x'] > (xc - 28.915)) & (df_pdg['start_x'] < (xc + 28.915)))]
        for zc in z_exclude:
            df_pdg = df_pdg[~((df_pdg['start_z'] > (zc - 5)) & (df_pdg['start_z'] < (zc + 5)))]
        if df_pdg.empty:
            continue
        ratio = df_pdg['E_vis'] / df_pdg['E']
        y = df_pdg['start_y'] / 1000  # convert mm to m
        plt.figure(figsize=(7, 5))
        hist, xedges, yedges = np.histogram2d(
            y, ratio,
            bins=[30, 30],
            range=[[-2.16671, 0.829282], [0, 1.2]]
        )
        # Normalize columns (for each y bin, sum over ratio bins)
        with np.errstate(divide='ignore', invalid='ignore'):
            col_sums = hist.sum(axis=1, keepdims=True)
            hist_norm = hist / col_sums
            hist_norm = np.nan_to_num(hist_norm)
        col_mask = (hist.sum(axis=1) == 0)
        mask_2d = np.tile(col_mask[:, np.newaxis], (1, hist.shape[1]))
        hist_masked = np.ma.masked_where(mask_2d, hist_norm)
        cmap = plt.colormaps['viridis'].copy()
        cmap.set_bad(color='white')
        plt.pcolormesh(
            xedges, yedges, hist_masked.T,
            cmap=cmap
        )
        plt.colorbar(label='Fraction')
        plt.xlabel('start_y [m]')
        plt.ylabel('E_vis / E_true')
        plt.title(f'{pdg_code_to_name(pdg)}: Normalized E_vis/E_true vs start_y_not_contained')
        plt.tight_layout()

        y_bin_centers = 0.5 * (xedges[:-1] + xedges[1:])
        avg_ratios = []
        for i in range(len(xedges) - 1):
            mask = (y >= xedges[i]) & (y < xedges[i+1])
            if np.any(mask):
                avg_ratios.append(ratio[mask].mean())
            else:
                avg_ratios.append(np.nan)
        avg_ratios = np.array(avg_ratios)
        valid = ~np.isnan(avg_ratios)
        plt.errorbar(
            y_bin_centers[valid], avg_ratios[valid],
            fmt='x', color='red', markersize=7, label='Average ratio'
        )
        plt.legend()

        fname = os.path.join(output_dir, f"{pdg_code_to_name(pdg)}_2D_EvisOverEtrue_vs_start_y_not_contained_norm_columns.png")
        plt.savefig(fname)
        plt.close()

def plot_evis_over_etrue_vs_y_contained_norm_columns(df):
    output_dir = get_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    particle_species = df['pdg'].unique()
    for pdg in particle_species:
        x_exclude_half = [3000, 2000, 1000, 0, -1000, -2000, -3000]
        x_exclude_full = [3500, 2500, 1500, 500, -500, -1500, -2500]
        z_exclude = [5157.559, 6157.559, 7157.559, 8157.559]

        df_pdg = df[
            (df['pdg'] == pdg) &
            (df['E'] > 0) &
            (df['is_contained_TPC'] == 1) &
            (df['start_x'] > (-3478.48 + 500)) & (df['start_x'] < (3478.48 - 500)) &
            (df['start_y'] > (-2166.71 + 500)) & (df['start_y'] < (829.282 - 500)) &
            (df['start_z'] > 4179.24) & (df['start_z'] < (9135.88 - 1500))
        ].copy()

        # Exclude regions using DataFrame cuts
        for xc in x_exclude_half:
            df_pdg = df_pdg[~((df_pdg['start_x'] > (xc - 3.175)) & (df_pdg['start_x'] < (xc + 3.175)))]
        for xc in x_exclude_full:
            df_pdg = df_pdg[~((df_pdg['start_x'] > (xc - 28.915)) & (df_pdg['start_x'] < (xc + 28.915)))]
        for zc in z_exclude:
            df_pdg = df_pdg[~((df_pdg['start_z'] > (zc - 5)) & (df_pdg['start_z'] < (zc + 5)))]
        if df_pdg.empty:
            continue
        ratio = df_pdg['E_vis'] / df_pdg['E']
        y = df_pdg['start_y'] / 1000  # convert mm to m
        plt.figure(figsize=(7, 5))
        hist, xedges, yedges = np.histogram2d(
            y, ratio,
            bins=[30, 30],
            range=[[-2.16671, 0.829282], [0, 1.2]]
        )
        # Normalize columns (for each y bin, sum over ratio bins)
        with np.errstate(divide='ignore', invalid='ignore'):
            col_sums = hist.sum(axis=1, keepdims=True)
            hist_norm = hist / col_sums
            hist_norm = np.nan_to_num(hist_norm)
        col_mask = (hist.sum(axis=1) == 0)
        mask_2d = np.tile(col_mask[:, np.newaxis], (1, hist.shape[1]))
        hist_masked = np.ma.masked_where(mask_2d, hist_norm)
        cmap = plt.colormaps['viridis'].copy()
        cmap.set_bad(color='white')
        plt.pcolormesh(
            xedges, yedges, hist_masked.T,
            cmap=cmap
        )
        plt.colorbar(label='Fraction')
        plt.xlabel('start_y [m]')
        plt.ylabel('E_vis / E_true')
        plt.title(f'{pdg_code_to_name(pdg)}: Normalized E_vis/E_true vs start_y_contained')
        plt.tight_layout()

        y_bin_centers = 0.5 * (xedges[:-1] + xedges[1:])
        avg_ratios = []
        for i in range(len(xedges) - 1):
            mask = (y >= xedges[i]) & (y < xedges[i+1])
            if np.any(mask):
                avg_ratios.append(ratio[mask].mean())
            else:
                avg_ratios.append(np.nan)
        avg_ratios = np.array(avg_ratios)
        valid = ~np.isnan(avg_ratios)
        plt.errorbar(
            y_bin_centers[valid], avg_ratios[valid],
            fmt='x', color='red', markersize=7, label='Average ratio'
        )
        plt.legend()

        fname = os.path.join(output_dir, f"{pdg_code_to_name(pdg)}_2D_EvisOverEtrue_vs_start_y_contained_norm_columns.png")
        plt.savefig(fname)
        plt.close()

def plot_evis_over_etrue_vs_z_not_contained_norm_columns(df):
    output_dir = get_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    particle_species = df['pdg'].unique()
    for pdg in particle_species:
        x_exclude_half = [3000, 2000, 1000, 0, -1000, -2000, -3000]
        x_exclude_full = [3500, 2500, 1500, 500, -500, -1500, -2500]
        z_exclude = [5157.559, 6157.559, 7157.559, 8157.559]

        df_pdg = df[
            (df['pdg'] == pdg) &
            (df['E'] > 0) &
            (df['is_contained_TPC'] == 0) &
            (df['start_x'] > (-3478.48 + 500)) & (df['start_x'] < (3478.48 - 500)) &
            (df['start_y'] > (-2166.71 + 500)) & (df['start_y'] < (829.282 - 500)) &
            (df['start_z'] > 4179.24) & (df['start_z'] < (9135.88 - 1500))
        ].copy()

        # Exclude regions using DataFrame cuts
        for xc in x_exclude_half:
            df_pdg = df_pdg[~((df_pdg['start_x'] > (xc - 3.175)) & (df_pdg['start_x'] < (xc + 3.175)))]
        for xc in x_exclude_full:
            df_pdg = df_pdg[~((df_pdg['start_x'] > (xc - 28.915)) & (df_pdg['start_x'] < (xc + 28.915)))]
        for zc in z_exclude:
            df_pdg = df_pdg[~((df_pdg['start_z'] > (zc - 5)) & (df_pdg['start_z'] < (zc + 5)))]
        if df_pdg.empty:
            continue
        ratio = df_pdg['E_vis'] / df_pdg['E']
        z = df_pdg['start_z'] / 1000  # convert mm to m
        plt.figure(figsize=(7, 5))
        hist, xedges, yedges = np.histogram2d(
            z, ratio,
            bins=[30, 30],
            range=[[4.17924, 9.13588], [0, 1.2]]
        )
        # Normalize columns (for each z bin, sum over ratio bins)
        with np.errstate(divide='ignore', invalid='ignore'):
            col_sums = hist.sum(axis=1, keepdims=True)
            hist_norm = hist / col_sums
            hist_norm = np.nan_to_num(hist_norm)
        col_mask = (hist.sum(axis=1) == 0)
        mask_2d = np.tile(col_mask[:, np.newaxis], (1, hist.shape[1]))
        hist_masked = np.ma.masked_where(mask_2d, hist_norm)
        cmap = plt.colormaps['viridis'].copy()
        cmap.set_bad(color='white')
        plt.pcolormesh(
            xedges, yedges, hist_masked.T,
            cmap=cmap
        )
        plt.colorbar(label='Fraction')
        plt.xlabel('start_z [m]')
        plt.ylabel('E_vis / E_true')
        plt.title(f'{pdg_code_to_name(pdg)}: Normalized E_vis/E_true vs start_z_not_contained')
        plt.tight_layout()

        z_bin_centers = 0.5 * (xedges[:-1] + xedges[1:])
        avg_ratios = []
        for i in range(len(xedges) - 1):
            mask = (z >= xedges[i]) & (z < xedges[i+1])
            if np.any(mask):
                avg_ratios.append(ratio[mask].mean())
            else:
                avg_ratios.append(np.nan)
        avg_ratios = np.array(avg_ratios)
        valid = ~np.isnan(avg_ratios)
        plt.errorbar(
            z_bin_centers[valid], avg_ratios[valid],
            fmt='x', color='red', markersize=7, label='Average ratio'
        )
        plt.legend()

        fname = os.path.join(output_dir, f"{pdg_code_to_name(pdg)}_2D_EvisOverEtrue_vs_start_z_not_contained_norm_columns.png")
        plt.savefig(fname)
        plt.close()

def plot_evis_over_etrue_vs_z_contained_norm_columns(df):
    output_dir = get_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    particle_species = df['pdg'].unique()
    for pdg in particle_species:
        x_exclude_half = [3000, 2000, 1000, 0, -1000, -2000, -3000]
        x_exclude_full = [3500, 2500, 1500, 500, -500, -1500, -2500]
        z_exclude = [5157.559, 6157.559, 7157.559, 8157.559]

        df_pdg = df[
            (df['pdg'] == pdg) &
            (df['E'] > 0) &
            (df['is_contained_TPC'] == 1) &
            (df['start_x'] > (-3478.48 + 500)) & (df['start_x'] < (3478.48 - 500)) &
            (df['start_y'] > (-2166.71 + 500)) & (df['start_y'] < (829.282 - 500)) &
            (df['start_z'] > 4179.24) & (df['start_z'] < (9135.88 - 1500))
        ].copy()

        # Exclude regions using DataFrame cuts
        for xc in x_exclude_half:
            df_pdg = df_pdg[~((df_pdg['start_x'] > (xc - 3.175)) & (df_pdg['start_x'] < (xc + 3.175)))]
        for xc in x_exclude_full:
            df_pdg = df_pdg[~((df_pdg['start_x'] > (xc - 28.915)) & (df_pdg['start_x'] < (xc + 28.915)))]
        for zc in z_exclude:
            df_pdg = df_pdg[~((df_pdg['start_z'] > (zc - 5)) & (df_pdg['start_z'] < (zc + 5)))]
        if df_pdg.empty:
            continue
        ratio = df_pdg['E_vis'] / df_pdg['E']
        z = df_pdg['start_z'] / 1000  # convert mm to m
        plt.figure(figsize=(7, 5))
        hist, xedges, yedges = np.histogram2d(
            z, ratio,
            bins=[30, 30],
            range=[[4.17924, 9.13588], [0, 1.2]]
        )
        # Normalize columns (for each z bin, sum over ratio bins)
        with np.errstate(divide='ignore', invalid='ignore'):
            col_sums = hist.sum(axis=1, keepdims=True)
            hist_norm = hist / col_sums
            hist_norm = np.nan_to_num(hist_norm)
        col_mask = (hist.sum(axis=1) == 0)
        mask_2d = np.tile(col_mask[:, np.newaxis], (1, hist.shape[1]))
        hist_masked = np.ma.masked_where(mask_2d, hist_norm)
        cmap = plt.colormaps['viridis'].copy()
        cmap.set_bad(color='white')
        plt.pcolormesh(
            xedges, yedges, hist_masked.T,
            cmap=cmap
        )
        plt.colorbar(label='Fraction')
        plt.xlabel('start_z [m]')
        plt.ylabel('E_vis / E_true')
        plt.title(f'{pdg_code_to_name(pdg)}: Normalized E_vis/E_true vs start_z_contained')
        plt.tight_layout()

        z_bin_centers = 0.5 * (xedges[:-1] + xedges[1:])
        avg_ratios = []
        for i in range(len(xedges) - 1):
            mask = (z >= xedges[i]) & (z < xedges[i+1])
            if np.any(mask):
                avg_ratios.append(ratio[mask].mean())
            else:
                avg_ratios.append(np.nan)
        avg_ratios = np.array(avg_ratios)
        valid = ~np.isnan(avg_ratios)
        plt.errorbar(
            z_bin_centers[valid], avg_ratios[valid],
            fmt='x', color='red', markersize=7, label='Average ratio'
        )
        plt.legend()

        fname = os.path.join(output_dir, f"{pdg_code_to_name(pdg)}_2D_EvisOverEtrue_vs_start_z_contained_norm_columns.png")
        plt.savefig(fname)
        plt.close()

def plot_evis_over_etrue_vs_theta_not_contained_norm_columns(df):
    output_dir = get_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    particle_species = df['pdg'].unique()
    for pdg in particle_species:
        x_exclude_half = [3000, 2000, 1000, 0, -1000, -2000, -3000]
        x_exclude_full = [3500, 2500, 1500, 500, -500, -1500, -2500]
        z_exclude = [5157.559, 6157.559, 7157.559, 8157.559]

        df_pdg = df[
            (df['pdg'] == pdg) &
            (df['E'] > 0) &
            (df['is_contained_TPC'] == 0) &
            (df['start_x'] > (-3478.48 + 500)) & (df['start_x'] < (3478.48 - 500)) &
            (df['start_y'] > (-2166.71 + 500)) & (df['start_y'] < (829.282 - 500)) &
            (df['start_z'] > 4179.24) & (df['start_z'] < (9135.88 - 1500))
        ].copy()

        # Exclude regions using DataFrame cuts
        for xc in x_exclude_half:
            df_pdg = df_pdg[~((df_pdg['start_x'] > (xc - 3.175)) & (df_pdg['start_x'] < (xc + 3.175)))]
        for xc in x_exclude_full:
            df_pdg = df_pdg[~((df_pdg['start_x'] > (xc - 28.915)) & (df_pdg['start_x'] < (xc + 28.915)))]
        for zc in z_exclude:
            df_pdg = df_pdg[~((df_pdg['start_z'] > (zc - 5)) & (df_pdg['start_z'] < (zc + 5)))]
        if df_pdg.empty:
            continue
        ratio = df_pdg['E_vis'] / df_pdg['E']
        theta = df_pdg['theta']
        plt.figure(figsize=(7, 5))
        hist, xedges, yedges = np.histogram2d(
            theta, ratio,
            bins=[30, 30],
            range=[[0, np.pi], [0, 1.2]]
        )
        # Normalize columns (for each theta bin, sum over ratio bins)
        with np.errstate(divide='ignore', invalid='ignore'):
            col_sums = hist.sum(axis=1, keepdims=True)
            hist_norm = hist / col_sums
            hist_norm = np.nan_to_num(hist_norm)
        col_mask = (hist.sum(axis=1) == 0)
        mask_2d = np.tile(col_mask[:, np.newaxis], (1, hist.shape[1]))
        hist_masked = np.ma.masked_where(mask_2d, hist_norm)
        cmap = plt.colormaps['viridis'].copy()
        cmap.set_bad(color='white')
        plt.pcolormesh(
            xedges, yedges, hist_masked.T,
            cmap=cmap
        )
        plt.colorbar(label='Fraction')
        plt.xlabel('theta [rad]')
        plt.ylabel('E_vis / E_true')
        plt.title(f'{pdg_code_to_name(pdg)}: Normalized E_vis/E_true vs theta_not_contained')
        plt.tight_layout()

        theta_bin_centers = 0.5 * (xedges[:-1] + xedges[1:])
        avg_ratios = []
        for i in range(len(xedges) - 1):
            mask = (theta >= xedges[i]) & (theta < xedges[i+1])
            if np.any(mask):
                avg_ratios.append(ratio[mask].mean())
            else:
                avg_ratios.append(np.nan)
        avg_ratios = np.array(avg_ratios)
        valid = ~np.isnan(avg_ratios)
        plt.errorbar(
            theta_bin_centers[valid], avg_ratios[valid],
            fmt='x', color='red', markersize=7, label='Average ratio'
        )
        plt.legend()

        fname = os.path.join(output_dir, f"{pdg_code_to_name(pdg)}_2D_EvisOverEtrue_vs_theta_not_contained_norm_columns.png")
        plt.savefig(fname)
        plt.close()

def plot_evis_over_etrue_vs_theta_contained_norm_columns(df):
    output_dir = get_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    particle_species = df['pdg'].unique()
    for pdg in particle_species:
        x_exclude_half = [3000, 2000, 1000, 0, -1000, -2000, -3000]
        x_exclude_full = [3500, 2500, 1500, 500, -500, -1500, -2500]
        z_exclude = [5157.559, 6157.559, 7157.559, 8157.559]

        df_pdg = df[
            (df['pdg'] == pdg) &
            (df['E'] > 0) &
            (df['is_contained_TPC'] == 1) &
            (df['start_x'] > (-3478.48 + 500)) & (df['start_x'] < (3478.48 - 500)) &
            (df['start_y'] > (-2166.71 + 500)) & (df['start_y'] < (829.282 - 500)) &
            (df['start_z'] > 4179.24) & (df['start_z'] < (9135.88 - 1500))
        ].copy()

        # Exclude regions using DataFrame cuts
        for xc in x_exclude_half:
            df_pdg = df_pdg[~((df_pdg['start_x'] > (xc - 3.175)) & (df_pdg['start_x'] < (xc + 3.175)))]
        for xc in x_exclude_full:
            df_pdg = df_pdg[~((df_pdg['start_x'] > (xc - 28.915)) & (df_pdg['start_x'] < (xc + 28.915)))]
        for zc in z_exclude:
            df_pdg = df_pdg[~((df_pdg['start_z'] > (zc - 5)) & (df_pdg['start_z'] < (zc + 5)))]
        if df_pdg.empty:
            continue
        ratio = df_pdg['E_vis'] / df_pdg['E']
        theta = df_pdg['theta']
        plt.figure(figsize=(7, 5))
        hist, xedges, yedges = np.histogram2d(
            theta, ratio,
            bins=[30, 30],
            range=[[0, np.pi], [0, 1.2]]
        )
        # Normalize columns (for each theta bin, sum over ratio bins)
        with np.errstate(divide='ignore', invalid='ignore'):
            col_sums = hist.sum(axis=1, keepdims=True)
            hist_norm = hist / col_sums
            hist_norm = np.nan_to_num(hist_norm)
        col_mask = (hist.sum(axis=1) == 0)
        mask_2d = np.tile(col_mask[:, np.newaxis], (1, hist.shape[1]))
        hist_masked = np.ma.masked_where(mask_2d, hist_norm)
        cmap = plt.colormaps['viridis'].copy()
        cmap.set_bad(color='white')
        plt.pcolormesh(
            xedges, yedges, hist_masked.T,
            cmap=cmap
        )
        plt.colorbar(label='Fraction')
        plt.xlabel('theta [rad]')
        plt.ylabel('E_vis / E_true')
        plt.title(f'{pdg_code_to_name(pdg)}: Normalized E_vis/E_true vs theta_contained')
        plt.tight_layout()

        theta_bin_centers = 0.5 * (xedges[:-1] + xedges[1:])
        avg_ratios = []
        for i in range(len(xedges) - 1):
            mask = (theta >= xedges[i]) & (theta < xedges[i+1])
            if np.any(mask):
                avg_ratios.append(ratio[mask].mean())
            else:
                avg_ratios.append(np.nan)
        avg_ratios = np.array(avg_ratios)
        valid = ~np.isnan(avg_ratios)
        plt.errorbar(
            theta_bin_centers[valid], avg_ratios[valid],
            fmt='x', color='red', markersize=7, label='Average ratio'
        )
        plt.legend()

        fname = os.path.join(output_dir, f"{pdg_code_to_name(pdg)}_2D_EvisOverEtrue_vs_theta_contained_norm_columns.png")
        plt.savefig(fname)
        plt.close()

def plot_evis_over_etrue_vs_dwall_not_contained_norm_columns(df):
    output_dir = get_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    particle_species = df['pdg'].unique()
    for pdg in particle_species:
        x_exclude_half = [3000, 2000, 1000, 0, -1000, -2000, -3000]
        x_exclude_full = [3500, 2500, 1500, 500, -500, -1500, -2500]
        z_exclude = [5157.559, 6157.559, 7157.559, 8157.559]

        df_pdg = df[
            (df['pdg'] == pdg) &
            (df['E'] > 0) &
            (df['is_contained_TPC'] == 0) &
            (df['start_x'] > (-3478.48 + 500)) & (df['start_x'] < (3478.48 - 500)) &
            (df['start_y'] > (-2166.71 + 500)) & (df['start_y'] < (829.282 - 500)) &
            (df['start_z'] > 4179.24) & (df['start_z'] < (9135.88 - 1500))
        ].copy()

        # Exclude regions using DataFrame cuts
        for xc in x_exclude_half:
            df_pdg = df_pdg[~((df_pdg['start_x'] > (xc - 3.175)) & (df_pdg['start_x'] < (xc + 3.175)))]
        for xc in x_exclude_full:
            df_pdg = df_pdg[~((df_pdg['start_x'] > (xc - 28.915)) & (df_pdg['start_x'] < (xc + 28.915)))]
        for zc in z_exclude:
            df_pdg = df_pdg[~((df_pdg['start_z'] > (zc - 5)) & (df_pdg['start_z'] < (zc + 5)))]
        if df_pdg.empty:
            continue
        ratio = df_pdg['E_vis'] / df_pdg['E']
        d_wall = df_pdg['d_wall_TPC'] / 1000  # mm to m
        plt.figure(figsize=(7, 5))
        hist, xedges, yedges = np.histogram2d(
            d_wall, ratio,
            bins=[50, 50],
            range=[[0, 7], [0, 1.2]]
        )
        # Normalize columns (for each d_wall bin, sum over ratio bins)
        with np.errstate(divide='ignore', invalid='ignore'):
            col_sums = hist.sum(axis=1, keepdims=True)
            hist_norm = hist / col_sums
            hist_norm = np.nan_to_num(hist_norm)
        col_mask = (hist.sum(axis=1) == 0)
        mask_2d = np.tile(col_mask[:, np.newaxis], (1, hist.shape[1]))
        hist_masked = np.ma.masked_where(mask_2d, hist_norm)
        cmap = plt.colormaps['viridis'].copy()
        cmap.set_bad(color='white')
        plt.pcolormesh(
            xedges, yedges, hist_masked.T,
            cmap=cmap
        )
        plt.colorbar(label='Fraction')
        plt.xlabel('d_wall [m]')
        plt.ylabel('E_vis / E_true')
        plt.title(f'{pdg_code_to_name(pdg)}: Normalized E_vis/E_true vs d_wall_not_contained')
        plt.tight_layout()

        d_bin_centers = 0.5 * (xedges[:-1] + xedges[1:])
        avg_ratios = []
        medians = []
        modes = []
        for i in range(len(xedges) - 1):
            mask = (d_wall >= xedges[i]) & (d_wall < xedges[i+1])
            if np.any(mask):
                avg_ratios.append(ratio[mask].mean())
                medians.append(np.median(ratio[mask]))
                # Mode: find the bin with the most entries in this d_wall bin
                hist_counts, bin_edges = np.histogram(ratio[mask], bins=yedges)
                mode_bin_idx = np.argmax(hist_counts)
                # Use bin center for mode
                mode_val = 0.5 * (bin_edges[mode_bin_idx] + bin_edges[mode_bin_idx+1])
                modes.append(mode_val)
            else:
                avg_ratios.append(np.nan)
                medians.append(np.nan)
                modes.append(np.nan)
        avg_ratios = np.array(avg_ratios)
        medians = np.array(medians)
        modes = np.array(modes)
        valid = ~np.isnan(avg_ratios)
        plt.errorbar(
            d_bin_centers[valid], avg_ratios[valid],
            fmt='x', color='red', markersize=7, label='Mean'
        )
        plt.errorbar(
            d_bin_centers[valid], medians[valid],
            fmt='o', color='blue', markersize=5, label='Median'
        )
        plt.errorbar(
            d_bin_centers[valid], modes[valid],
            fmt='s', color='black', markersize=5, label='Mode (bin center)'
        )
        plt.legend()

        fname = os.path.join(output_dir, f"{pdg_code_to_name(pdg)}_2D_EvisOverEtrue_vs_dwall_not_contained_norm_columns.png")
        plt.savefig(fname)
        plt.close()

def plot_evis_over_etrue_vs_dwall_contained_norm_columns(df):
    output_dir = get_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    particle_species = df['pdg'].unique()
    for pdg in particle_species:
        x_exclude_half = [3000, 2000, 1000, 0, -1000, -2000, -3000]
        x_exclude_full = [3500, 2500, 1500, 500, -500, -1500, -2500]
        z_exclude = [5157.559, 6157.559, 7157.559, 8157.559]

        df_pdg = df[
            (df['pdg'] == pdg) &
            (df['E'] > 0) &
            (df['is_contained_TPC'] == 1) &
            (df['start_x'] > (-3478.48 + 500)) & (df['start_x'] < (3478.48 - 500)) &
            (df['start_y'] > (-2166.71 + 500)) & (df['start_y'] < (829.282 - 500)) &
            (df['start_z'] > 4179.24) & (df['start_z'] < (9135.88 - 1500))
        ].copy()

        # Exclude regions using DataFrame cuts
        for xc in x_exclude_half:
            df_pdg = df_pdg[~((df_pdg['start_x'] > (xc - 3.175)) & (df_pdg['start_x'] < (xc + 3.175)))]
        for xc in x_exclude_full:
            df_pdg = df_pdg[~((df_pdg['start_x'] > (xc - 28.915)) & (df_pdg['start_x'] < (xc + 28.915)))]
        for zc in z_exclude:
            df_pdg = df_pdg[~((df_pdg['start_z'] > (zc - 5)) & (df_pdg['start_z'] < (zc + 5)))]
        if df_pdg.empty:
            continue
        ratio = df_pdg['E_vis'] / df_pdg['E']
        d_wall = df_pdg['d_wall_TPC'] / 1000  # mm to m
        plt.figure(figsize=(7, 5))
        hist, xedges, yedges = np.histogram2d(
            d_wall, ratio,
            bins=[50, 50],
            range=[[0, 7], [0, 1.2]]
        )
        # Normalize columns (for each d_wall bin, sum over ratio bins)
        with np.errstate(divide='ignore', invalid='ignore'):
            col_sums = hist.sum(axis=1, keepdims=True)
            hist_norm = hist / col_sums
            hist_norm = np.nan_to_num(hist_norm)
        col_mask = (hist.sum(axis=1) == 0)
        mask_2d = np.tile(col_mask[:, np.newaxis], (1, hist.shape[1]))
        hist_masked = np.ma.masked_where(mask_2d, hist_norm)
        cmap = plt.colormaps['viridis'].copy()
        cmap.set_bad(color='white')
        plt.pcolormesh(
            xedges, yedges, hist_masked.T,
            cmap=cmap
        )
        plt.colorbar(label='Fraction')
        plt.xlabel('d_wall [m]')
        plt.ylabel('E_vis / E_true')
        plt.title(f'{pdg_code_to_name(pdg)}: Normalized E_vis/E_true vs d_wall_contained')
        plt.tight_layout()

        d_bin_centers = 0.5 * (xedges[:-1] + xedges[1:])
        avg_ratios = []
        medians = []
        modes = []
        for i in range(len(xedges) - 1):
            mask = (d_wall >= xedges[i]) & (d_wall < xedges[i+1])
            if np.any(mask):
                avg_ratios.append(ratio[mask].mean())
                medians.append(np.median(ratio[mask]))
                # Mode: find the bin with the most entries in this d_wall bin
                hist_counts, bin_edges = np.histogram(ratio[mask], bins=yedges)
                mode_bin_idx = np.argmax(hist_counts)
                # Use bin center for mode
                mode_val = 0.5 * (bin_edges[mode_bin_idx] + bin_edges[mode_bin_idx+1])
                modes.append(mode_val)
            else:
                avg_ratios.append(np.nan)
                medians.append(np.nan)
                modes.append(np.nan)
        avg_ratios = np.array(avg_ratios)
        medians = np.array(medians)
        modes = np.array(modes)
        valid = ~np.isnan(avg_ratios)
        plt.errorbar(
            d_bin_centers[valid], avg_ratios[valid],
            fmt='x', color='red', markersize=7, label='Mean'
        )
        plt.errorbar(
            d_bin_centers[valid], medians[valid],
            fmt='o', color='blue', markersize=5, label='Median'
        )
        plt.errorbar(
            d_bin_centers[valid], modes[valid],
            fmt='s', color='black', markersize=5, label='Mode (bin center)'
        )
        plt.legend()

        fname = os.path.join(output_dir, f"{pdg_code_to_name(pdg)}_2D_EvisOverEtrue_vs_dwall_not_contained_norm_columns.png")
        plt.savefig(fname)
        plt.close()

def plot_dwall_vs_momentum_avg_evis_over_etrue(df):
    output_dir = get_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    particle_species = df['pdg'].unique()
    for pdg in particle_species:
        x_exclude_half = [3000, 2000, 1000, 0, -1000, -2000, -3000]
        x_exclude_full = [3500, 2500, 1500, 500, -500, -1500, -2500]
        z_exclude = [5157.559, 6157.559, 7157.559, 8157.559]

        df_pdg = df[
            (df['pdg'] == pdg) &
            (df['E'] > 0) &
            # (df['is_contained_TPC'] == 1) &
            (df['start_x'] > (-3478.48 + 500)) & (df['start_x'] < (3478.48 - 500)) &
            (df['start_y'] > (-2166.71 + 500)) & (df['start_y'] < (829.282 - 500)) &
            (df['start_z'] > 4179.24) & (df['start_z'] < (9135.88 - 1500))
        ].copy()

        for xc in x_exclude_half:
            df_pdg = df_pdg[~((df_pdg['start_x'] > (xc - 3.175)) & (df_pdg['start_x'] < (xc + 3.175)))]
        for xc in x_exclude_full:
            df_pdg = df_pdg[~((df_pdg['start_x'] > (xc - 28.915)) & (df_pdg['start_x'] < (xc + 28.915)))]
        for zc in z_exclude:
            df_pdg = df_pdg[~((df_pdg['start_z'] > (zc - 5)) & (df_pdg['start_z'] < (zc + 5)))]
        if df_pdg.empty:
            continue

        # Calculate momentum if not present
        if 'p' in df_pdg.columns:
            p = df_pdg['p']
        else:
            print(f"Momentum not found for PDG {pdg_code_to_name(pdg)}. Skipping.")
            continue

        d_wall = df_pdg['d_wall_TPC'] / 1000  # mm to m
        ratio = df_pdg['E_vis'] / df_pdg['E']

        # Define bins
        d_bins = np.linspace(0, 7, 20)
        p_bins = np.linspace(p.min(), p.max(), 20)

        # Compute average E_vis/E_true in each bin
        avg = np.full((len(d_bins)-1, len(p_bins)-1), np.nan)
        for i in range(len(d_bins)-1):
            for j in range(len(p_bins)-1):
                mask = (
                    (d_wall >= d_bins[i]) & (d_wall < d_bins[i+1]) &
                    (p >= p_bins[j]) & (p < p_bins[j+1])
                )
                if np.any(mask):
                    avg[i, j] = ratio[mask].mean()

        plt.figure(figsize=(8, 6))
        cmap = plt.colormaps['viridis'].copy()
        cmap.set_bad(color='white')
        mesh = plt.pcolormesh(
            d_bins, p_bins, avg.T,
            cmap=cmap,
            vmin=0, vmax=1.2,
            shading='auto'
        )
        plt.colorbar(label='Average E_vis / E_true')
        plt.xlabel('d_wall [m]')
        plt.ylabel('Momentum [MeV/c]')
        plt.title(f'{pdg_code_to_name(pdg)}: <E_vis/E_true> vs d_wall vs momentum')
        plt.tight_layout()

        # Annotate each bin with the value (3 significant figures)
        for i in range(len(d_bins)-1):
            for j in range(len(p_bins)-1):
                value = avg[i, j]
                if not np.isnan(value):
                    x = 0.5 * (d_bins[i] + d_bins[i+1])
                    y = 0.5 * (p_bins[j] + p_bins[j+1])
                    plt.text(
                        x, y, f"{value:.3g}",
                        color='black', ha='center', va='center', fontsize=6
                    )

        fname = os.path.join(output_dir, f"{pdg_code_to_name(pdg)}_avg_EvisOverEtrue_vs_dwall_vs_p.png")
        plt.savefig(fname)
        plt.close()

def ratio_vs_variable_norm_columns(df, var, containment, bins_x=50, bins_y=50, var_range=None):
    """
    Generalized 2D normalized column plot for ratio (E_vis/E_true) vs a variable.
    - df: DataFrame
    - var: variable name for x-axis (e.g. 'd_wall_TPC', 'start_x', etc.)
    - containment: 1 (contained) or 0 (not contained)
    - bins: number of bins or [xbins, ybins]
    - var_range: (min, max) for var axis
    - ratio_range: (min, max) for ratio axis
    """
    if containment == 1:
        containment_str = "contained"
    else:
        containment_str = "not_contained"

    output_dir = os.path.join(get_output_dir(), f"{var}_vs_ratio_{containment_str}")
    os.makedirs(output_dir, exist_ok=True)
    particle_species = df['pdg'].unique()

    for pdg in particle_species:
        x_exclude_half = [3000, 2000, 1000, 0, -1000, -2000, -3000]
        x_exclude_full = [3500, 2500, 1500, 500, -500, -1500, -2500]
        z_exclude = [5157.559, 6157.559, 7157.559, 8157.559]

        df_pdg = df[
            (df['pdg'] == pdg) &
            (df['E'] > 0) &
            (df['is_contained_TPC'] == containment) &
            (df['start_x'] > (-3478.48 + 500)) & (df['start_x'] < (3478.48 - 500)) &
            (df['start_y'] > (-2166.71 + 500)) & (df['start_y'] < (829.282 - 500)) &
            (df['start_z'] > 4179.24) & (df['start_z'] < (9135.88 - 1500))
        ].copy()

        for xc in x_exclude_half:
            df_pdg = df_pdg[~((df_pdg['start_x'] > (xc - 3.175)) & (df_pdg['start_x'] < (xc + 3.175)))]
        for xc in x_exclude_full:
            df_pdg = df_pdg[~((df_pdg['start_x'] > (xc - 28.915)) & (df_pdg['start_x'] < (xc + 28.915)))]
        for zc in z_exclude:
            df_pdg = df_pdg[~((df_pdg['start_z'] > (zc - 5)) & (df_pdg['start_z'] < (zc + 5)))]
        if df_pdg.empty or var not in df_pdg.columns:
            continue

        ratio = df_pdg['E_vis'] / df_pdg['E']
        if var == 'd_wall_TPC':
            x = df_pdg[var] / 1000 
            default_range = (0, 7)
        elif var == 'start_x':
            x = df_pdg[var] / 1000
            default_range = (-3.47848, 3.47848)
        elif var == 'start_y':
            x = df_pdg[var] / 1000
            default_range = (-2.16671, 0.829282)
        elif var == 'start_z':
            x = df_pdg[var] / 1000
            default_range = (4.17924, 9.13588)
        else:
            x = df_pdg[var]
            default_range = (df_pdg[var].min(), df_pdg[var].max())
        x_range = var_range if var_range is not None else default_range

        plt.figure(figsize=(7, 5))
        hist, xedges, yedges = np.histogram2d(
            x, ratio,
            bins=[bins_x, bins_y],
            range=[x_range, (0, 1.2)]
        )
        with np.errstate(divide='ignore', invalid='ignore'):
            col_sums = hist.sum(axis=1, keepdims=True)
            hist_norm = hist / col_sums
            hist_norm = np.nan_to_num(hist_norm)
        col_mask = (hist.sum(axis=1) == 0)
        mask_2d = np.tile(col_mask[:, np.newaxis], (1, hist.shape[1]))
        hist_masked = np.ma.masked_where(mask_2d, hist_norm)
        cmap = plt.colormaps['viridis'].copy()
        cmap.set_bad(color='white')
        plt.pcolormesh(
            xedges, yedges, hist_masked.T,
            cmap=cmap
        )
        plt.colorbar(label='Column Normalised Counts')
        plt.xlabel(f'{var}')
        plt.ylabel('E_vis / E_true')
        plt.title(f'{containment_str} Energy Ratio vs {var} for {pdg_code_to_name(pdg)}:')
        plt.tight_layout()

        x_bin_centers = 0.5 * (xedges[:-1] + xedges[1:])
        avg_ratios = []
        for i in range(len(xedges) - 1):
            mask = (x >= xedges[i]) & (x < xedges[i+1])
            if np.any(mask):
                avg_ratios.append(ratio[mask].mean())
            else:
                avg_ratios.append(np.nan)
        avg_ratios = np.array(avg_ratios)
        valid = ~np.isnan(avg_ratios)
        plt.errorbar(
            x_bin_centers[valid], avg_ratios[valid],
            fmt='x', color='red', markersize=7, label='Average ratio'
        )
        plt.legend()

        fname = os.path.join(output_dir, f"{pdg_code_to_name(pdg)}_2D_ratio_vs_{var}_{containment}_norm_columns.png")
        plt.savefig(fname)
        plt.close()

def avg_ratio_vs_vars_2d(df, var_x, var_y, bins_x=20, bins_y=20, range_x=None, range_y=None, ratio_denominator='E', containment=None):
    """
    Generalized 2D plot of average (ratio_numerator/ratio_denominator) in bins of var_x vs var_y.
    - df: DataFrame
    - var_x, var_y: variable names for axes (e.g. 'd_wall_TPC', 'p', etc.)
    - bins_x, bins_y: number of bins or bin edges for each axis
    - range_x, range_y: (min, max) for each axis
    - ratio_numerator, ratio_denominator: columns for the ratio (default: E_vis/E)
    - containment: None (no cut), 1 (contained), or 0 (not contained)
    """
    if containment == 1:
        containment_str = "contained"
    elif containment == 0:
        containment_str = "not_contained"
    else:
        containment_str = "all"
    output_dir = os.path.join(
        get_output_dir(), f"{var_x}_vs_{var_y}_avg_ratio_{containment_str}"
    )
    os.makedirs(output_dir, exist_ok=True)
    particle_species = df['pdg'].unique()

    for pdg in particle_species:
        x_exclude_half = [3000, 2000, 1000, 0, -1000, -2000, -3000]
        x_exclude_full = [3500, 2500, 1500, 500, -500, -1500, -2500]
        z_exclude = [5157.559, 6157.559, 7157.559, 8157.559]

        mask = (
            (df['pdg'] == pdg) &
            (df['E'] > 0) &
            (df['start_x'] > (-3478.48 + 500)) & (df['start_x'] < (3478.48 - 500)) &
            (df['start_y'] > (-2166.71 + 500)) & (df['start_y'] < (829.282 - 500)) &
            (df['start_z'] > 4179.24) & (df['start_z'] < (9135.88 - 1500))
        )
        if containment is not None:
            mask &= (df['is_contained_TPC'] == containment)
        df_pdg = df[mask].copy()

        for xc in x_exclude_half:
            df_pdg = df_pdg[~((df_pdg['start_x'] > (xc - 3.175)) & (df_pdg['start_x'] < (xc + 3.175)))]
        for xc in x_exclude_full:
            df_pdg = df_pdg[~((df_pdg['start_x'] > (xc - 28.915)) & (df_pdg['start_x'] < (xc + 28.915)))]
        for zc in z_exclude:
            df_pdg = df_pdg[~((df_pdg['start_z'] > (zc - 5)) & (df_pdg['start_z'] < (zc + 5)))]
        if df_pdg.empty or var_x not in df_pdg.columns or var_y not in df_pdg.columns:
            continue

        x = df_pdg[var_x]
        y = df_pdg[var_y]
        if var_x in ['d_wall_TPC', 'start_x', 'start_y', 'start_z']:
            x = x / 1000
        if var_y in ['d_wall_TPC', 'start_x', 'start_y', 'start_z']:
            y = y / 1000

        # if range_x is None:
        #     if var_x == 'd_wall_TPC':
        #         range_x = (0, 7)
        #     elif var_x == 'start_x':
        #         range_x = (-3.47848, 3.47848)
        #     elif var_x == 'start_y':
        #         range_x = (-2.16671, 0.829282)
        #     elif var_x == 'start_z':
        #         range_x = (4.17924, 9.13588)
        #     else:
        #         range_x = (x.min(), x.max())
        # if range_y is None:
        #     if var_y == 'd_wall_TPC':
        #         range_y = (0, 7)
        #     elif var_y == 'start_x':
        #         range_y = (-3.47848, 3.47848)
        #     elif var_y == 'start_y':
        #         range_y = (-2.16671, 0.829282)
        #     elif var_y == 'start_z':
        #         range_y = (4.17924, 9.13588)
        #     else:
        #         range_y = (y.min(), y.max())

        ratio = df_pdg['E_vis'] / df_pdg[ratio_denominator]

        x_bins = np.linspace(*range_x, bins_x+1) if isinstance(bins_x, int) else bins_x
        y_bins = np.linspace(*range_y, bins_y+1) if isinstance(bins_y, int) else bins_y

        avg = np.full((len(x_bins)-1, len(y_bins)-1), np.nan)
        for i in range(len(x_bins)-1):
            for j in range(len(y_bins)-1):
                mask = (
                    (x >= x_bins[i]) & (x < x_bins[i+1]) &
                    (y >= y_bins[j]) & (y < y_bins[j+1])
                )
                if np.any(mask):
                    avg[i, j] = ratio[mask].mean()

        plt.figure(figsize=(8, 6))
        cmap = plt.colormaps['viridis'].copy()
        cmap.set_bad(color='white')
        mesh = plt.pcolormesh(
            x_bins, y_bins, avg.T,
            cmap=cmap,
            vmin=0, vmax=1.2,
            shading='auto'
        )
        plt.colorbar(label=f'Average E_vis/ {ratio_denominator}')
        plt.xlabel(var_x)
        plt.ylabel(var_y)
        plt.title(f'<E_vis/{ratio_denominator}> vs {var_x} vs {var_y} for {pdg_code_to_name(pdg)}')
        plt.tight_layout()

        for i in range(len(x_bins)-1):
            for j in range(len(y_bins)-1):
                value = avg[i, j]
                if not np.isnan(value):
                    x_center = 0.5 * (x_bins[i] + x_bins[i+1])
                    y_center = 0.5 * (y_bins[j] + y_bins[j+1])
                    plt.text(
                        x_center, y_center, f"{value:.3g}",
                        color='black', ha='center', va='center', fontsize=6
                    )

        fname = os.path.join(output_dir, f"{pdg_code_to_name(pdg)}_avg_E_vis_over_{ratio_denominator}_vs_{var_x}_vs_{var_y}_{containment_str}.png")
        plt.savefig(fname)
        plt.close()

if __name__ == "__main__":
    # df = read_edep_sim_output("outputs/edep_sim_output.root")
    # plot_evis_over_ekin(read_edep_sim_output("outputs/edep_sim_output_chunk*.root", tree_name="events"))
    # plot_evis_over_etrue(read_edep_sim_output("outputs/edep_sim_output_chunk*.root", tree_name="events"))
    # plot_evis_over_etrue_not_contained(read_edep_sim_output("outputs/edep_sim_output_chunk*.root", tree_name="events"))
    # plot_evis_over_etrue_tiTue12032432$contained(read_edep_sim_output("outputs/edep_sim_output_chunk*.root", tree_name="events"))
    # plot_evis_over_ekin_contained(read_edep_sim_output("outputs/edep_sim_output_chunk*.root", tree_name="events"))
    # plot_evis_over_ekin_not_contained(read_edep_sim_output("outputs/edep_sim_output_chunk*.root", tree_name="events"))
    # plot_evis_vs_etrue(read_edep_sim_output("outputs/edep_sim_output_chunk*.root", tree_name="events"))
    # plot_evis_vs_ekin(read_edep_sim_output("outputs/edep_sim_output_chunk*.root", tree_name="events"))
    # plot_particle_multiplicity(read_edep_sim_output("outputs/edep_sim_output_chunk*.root", tree_name="events"))
    # plot_dwall_vs_containment(read_edep_sim_output("outputs/edep_sim_output_chunk*.root", tree_name="events"))
    # plot_normalized_dwall_vs_containment_by_pdg(read_edep_sim_output("outputs/edep_sim_output_chunk*.root", tree_name="events"))
    # plot_evis_over_e_vs_containment_per_pdg(read_edep_sim_output("outputs/edep_sim_output_chunk*.root", tree_name="events"))
    # plot_evis_over_ekin_vs_containment_per_pdg(read_edep_sim_output("outputs/edep_sim_output_chunk*.root", tree_name="events"))
    # plot_dwall_vs_evis_over_e_uncontained_per_pdg(read_edep_sim_output("outputs/edep_sim_output_chunk*.root", tree_name="events"))
    # plot_dwall_vs_evis_over_ekin_uncontained_per_pdg(read_edep_sim_output("outputs/edep_sim_output_chunk*.root", tree_name="events"))
    # plot_dwall(read_edep_sim_output("outputs/edep_sim_output_chunk*.root", tree_name="events"))
    # plot_evis_over_etrue_vs_etrue(read_edep_sim_output("outputs/edep_sim_output_chunk*mm, tree_name="events"))
    # plot_evis_over_etrue_vs_theta(read_edep_sim_output("outputs/edep_sim_output_chunk*.root", tree_name="events"))
    # plot_evis_over_etrue_vs_phi(read_edep_sim_output("outputs/edep_sim_output_chunk*.root", tree_name="events"))
    # plot_evis_over_etrue_vs_d_wall(read_edep_sim_output("outputs/edep_sim_output_chunk*.root", tree_name="events"))
    # plot_evis_over_etrue_vs_x_not_contained(read_edep_sim_output("outputs/edep_sim_output_chunk*.root", tree_name="events"))
    # plot_evis_over_etrue_vs_y_not_contained(read_edep_sim_output("outputs/edep_sim_output_chunk*.root", tree_name="events"))
    # plot_evis_over_etrue_vs_z_not_contained(read_edep_sim_output("outputs/edep_sim_output_chunk*.root", tree_name="events"))
    # plot_evis_over_etrue_vs_x_contained(read_edep_sim_output("outputs/edep_sim_output_chunk*.root", tree_name="events"))
    # plot_evis_over_etrue_vs_y_contained(read_edep_sim_output("outputs/edep_sim_output_chunk*.root", tree_name="events"))
    # plot_evis_over_etrue_vs_z_contained(read_edep_sim_output("outputs/edep_sim_output_chunk*.root", tree_name="events"))
    # plot_evis_over_etrue_vs_x_not_contained_norm_columns(read_edep_sim_output("outputs/edep_sim_output_chunk*.root", tree_name="events"))
    # plot_evis_over_etrue_vs_x_contained_norm_columns(read_edep_sim_output("outputs/edep_sim_output_chunk*.root", tree_name="events"))
    # plot_evis_over_etrue_vs_y_not_contained_norm_columns(read_edep_sim_output("outputs/edep_sim_output_chunk*.root", tree_name="events"))
    # plot_evis_over_etrue_vs_y_contained_norm_columns(read_edep_sim_output("outputs/edep_sim_output_chunk*.root", tree_name="events"))
    # plot_evis_over_etrue_vs_z_not_contained_norm_columns(read_edep_sim_output("outputs/edep_sim_output_chunk*.root", tree_name="events"))
    # plot_evis_over_etrue_vs_z_contained_norm_columns(read_edep_sim_output("outputs/edep_sim_output_chunk*.root", tree_name="events"))
    # plot_evis_over_etrue_vs_theta_not_contained_norm_columns(read_edep_sim_output("outputs/edep_sim_output_chunk*.root", tree_name="events"))
    # plot_evis_over_etrue_vs_theta_contained_norm_columns(read_edep_sim_output("outputs/edep_sim_output_chunk*.root", tree_name="events"))
    # plot_evis_over_etrue_vs_dwall_not_contained_norm_columns(read_edep_sim_output("outputs/edep_sim_output_chunk*.root", tree_name="events"))
    # plot_evis_over_etrue_vs_dwall_contained_norm_columns(read_edep_sim_output("outputs/edep_sim_output_chunk*.root", tree_name="events"))
    # plot_dwall_vs_momentum_avg_evis_over_etrue(read_edep_sim_output("outputs/edep_sim_output_chunk*.root", tree_name="events"))
    ratio_vs_variable_norm_columns(
        read_edep_sim_output("outputs/edep_sim_output_chunk*.root", tree_name="events"),
        var='d_wall_TPC',
        containment=0,
        bins_x=50,
        bins_y=50,
        var_range=(0, 7)
    )
    avg_ratio_vs_vars_2d(
        df=read_edep_sim_output("outputs/edep_sim_output_chunk*.root", tree_name="events"),
        var_x='d_wall_TPC',
        var_y='p',
        bins_x=20,
        bins_y=20,
        range_x=(0, 7),
        range_y=(0, 60000),
        ratio_denominator='E',
        containment=None  # or 1 or 0
    )
    pass