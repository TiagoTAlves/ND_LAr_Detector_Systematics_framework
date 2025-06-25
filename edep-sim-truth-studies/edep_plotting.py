import uproot
import pandas as pd
import os
import inspect
import matplotlib.pyplot as plt
import glob
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from itertools import combinations
from scipy.stats import norm
from scipy.optimize import curve_fit


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

def ratio_vs_variable_norm_columns(df, var, containment=None, bins_x=50, bins_y=50, var_range=None, volume="active"):
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
    elif containment == 0:
        containment_str = "not_contained"
    else:
        containment_str = "all"

    output_dir = os.path.join(get_output_dir(), f"{var}_vs_ratio_{containment_str}")
    os.makedirs(output_dir, exist_ok=True)
    particle_species = df['pdg'].unique()

    for pdg in particle_species:
        x_exclude_half = [3000, 2000, 1000, 0, -1000, -2000, -3000]
        x_exclude_full = [3500, 2500, 1500, 500, -500, -1500, -2500]
        z_exclude = [5157.559, 6157.559, 7157.559, 8157.559]

        if volume == "active":
            mask = (
                (df['pdg'] == pdg) &
                (df['E'] > 0) &
                (df['start_x'] > (-3478.48)) & (df['start_x'] < (3478.48)) &
                (df['start_y'] > (-2166.71)) & (df['start_y'] < (829.282)) &
                (df['start_z'] > 4179.24) & (df['start_z'] < (9135.88)) &
                (df['E_vis'] > 0)
            )
        elif volume == "fiducial":
            mask = (
                (df['pdg'] == pdg) &
                (df['E'] > 0) &
                (df['start_x'] > (-3478.48 + 500)) & (df['start_x'] < (3478.48 - 500)) &
                (df['start_y'] > (-2166.71 + 500)) & (df['start_y'] < (829.282 - 500)) &
                (df['start_z'] > 4179.24) & (df['start_z'] < (9135.88 - 1500)) &
                (df['E_vis'] > 0) 
            )
        else:
            raise ValueError(f"Unknown volume: {volume}. Please specify 'active' or 'fiducial'.")
        if containment is not None:
            mask &= (df['is_contained_TPC'] == containment)
        df_pdg = df[mask].copy()

        for xc in x_exclude_half:
            df_pdg = df_pdg[~((df_pdg['start_x'] > (xc - 3.175)) & (df_pdg['start_x'] < (xc + 3.175)))]
        for xc in x_exclude_full:
            df_pdg = df_pdg[~((df_pdg['start_x'] > (xc - 28.915)) & (df_pdg['start_x'] < (xc + 28.915)))]
        for zc in z_exclude:
            df_pdg = df_pdg[~((df_pdg['start_z'] > (zc - 5)) & (df_pdg['start_z'] < (zc + 5)))]
        if df_pdg.empty or var not in df_pdg.columns:
            continue

        if var in ['d_wall_TPC', 'start_x', 'start_y', 'start_z']:
            x = df_pdg[var]
            x = x / 1000

        ratio = df_pdg['E_vis'] / df_pdg['E']
        if var == 'd_wall_TPC':
            default_range = (0, 7)
        elif var == 'start_x':
            default_range = (-3.47848, 3.47848)
        elif var == 'start_y':
            default_range = (-2.16671, 0.829282)
        elif var == 'start_z':
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
        std_ratios = []
        for i in range(len(xedges) - 1):
            mask = (x >= xedges[i]) & (x < xedges[i+1])
        plt.errorbar(
            x_bin_centers[valid], avg_ratios[valid],
            fmt='x', color='red', markersize=7, label='Average ratio ± std'
        )
        plt.legend()

        fname = os.path.join(output_dir, f"{pdg_code_to_name(pdg)}_2D_ratio_vs_{var}_{containment}_{volume}_norm_columns.png")
        plt.savefig(fname)
        plt.close()

def avg_ratio_vs_vars_2d(df, var_x, var_y, bins_x=20, bins_y=20, range_x=None, range_y=None, ratio_denominator='E', containment=None, volume="active"):
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

        if volume == "active":
            mask = (
            (df['pdg'] == pdg) &
            (df['E'] > 0) &
            (df['start_x'] > (-3478.48)) & (df['start_x'] < (3478.48)) &
            (df['start_y'] > (-2166.71)) & (df['start_y'] < (829.282)) &
            (df['start_z'] > 4179.24) & (df['start_z'] < (9135.88)) &
            (df['E_vis'] > 0)
            )
        elif volume == "fiducial":
            mask = (
            (df['pdg'] == pdg) &
            (df['E'] > 0) &
            (df['start_x'] > (-3478.48 + 500)) & (df['start_x'] < (3478.48 - 500)) &
            (df['start_y'] > (-2166.71 + 500)) & (df['start_y'] < (829.282 - 500)) &
            (df['start_z'] > 4179.24) & (df['start_z'] < (9135.88 - 1500)) &
            (df['E_vis'] > 0) 
            )
        else:
            raise ValueError(f"Unknown volume: {volume}. Please specify 'active' or 'fiducial'.")
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

        ratio = df_pdg['E_vis'] / df_pdg[ratio_denominator]

        if range_x is None:
            if var_x == 'd_wall_TPC':
                range_x = (0, 7)
            elif var_x == 'start_x':
                range_x = (-3.47848, 3.47848)
            elif var_x == 'start_y':
                range_x = (-2.16671, 0.829282)
            elif var_x == 'start_z':
                range_x = (4.17924, 9.13588)
            else:
                range_x = (x.min(), x.max())
        if range_y is None:
            if var_y == 'd_wall_TPC':
                range_y = (0, 7)
            elif var_y == 'start_x':
                range_y = (-3.47848, 3.47848)
            elif var_y == 'start_y':
                range_y = (-2.16671, 0.829282)
            elif var_y == 'start_z':
                range_y = (4.17924, 9.13588)
            else:
                range_y = (y.min(), y.max())

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
        plt.title(f'<E_vis/{ratio_denominator}> vs {var_x} vs {var_y} for {containment_str} {pdg_code_to_name(pdg)}')
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

        fname = os.path.join(output_dir, f"{pdg_code_to_name(pdg)}_avg_E_vis_over_{ratio_denominator}_vs_{var_x}_vs_{var_y}_{containment_str}_{volume}.png")
        plt.savefig(fname)
        plt.close()

def std_ratio_vs_vars_2d(df, var_x, var_y, bins_x=20, bins_y=20, range_x=None, range_y=None, ratio_denominator='E', containment=None, volume="active"):
    """
    Generalized 2D plot of std(ratio) in bins of var_x vs var_y.
    - df: DataFrame
    - var_x, var_y: variable names for axes (e.g. 'd_wall_TPC', 'p', etc.)
    - bins_x, bins_y: number of bins or bin edges for each axis
    - range_x, range_y: (min, max) for each axis
    - ratio_denominator: denominator column for the ratio (numerator is E_vis)
    - containment: None (no cut), 1 (contained), or 0 (not contained)
    """
    if containment == 1:
        containment_str = "contained"
    elif containment == 0:
        containment_str = "not_contained"
    else:
        containment_str = "all"
    output_dir = os.path.join(
        get_output_dir(), f"{var_x}_vs_{var_y}_std_ratio_{containment_str}"
    )
    os.makedirs(output_dir, exist_ok=True)
    particle_species = df['pdg'].unique()

    for pdg in particle_species:
        x_exclude_half = [3000, 2000, 1000, 0, -1000, -2000, -3000]
        x_exclude_full = [3500, 2500, 1500, 500, -500, -1500, -2500]
        z_exclude = [5157.559, 6157.559, 7157.559, 8157.559]

        if volume == "active":
            mask = (
            (df['pdg'] == pdg) &
            (df['E'] > 0) &
            (df['start_x'] > (-3478.48)) & (df['start_x'] < (3478.48)) &
            (df['start_y'] > (-2166.71)) & (df['start_y'] < (829.282)) &
            (df['start_z'] > 4179.24) & (df['start_z'] < (9135.88)) &
            (df['E_vis'] > 0)
            )
        elif volume == "fiducial":
            mask = (
            (df['pdg'] == pdg) &
            (df['E'] > 0) &
            (df['start_x'] > (-3478.48 + 500)) & (df['start_x'] < (3478.48 - 500)) &
            (df['start_y'] > (-2166.71 + 500)) & (df['start_y'] < (829.282 - 500)) &
            (df['start_z'] > 4179.24) & (df['start_z'] < (9135.88 - 1500)) &
            (df['E_vis'] > 0) 
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

        ratio = df_pdg['E_vis'] / df_pdg[ratio_denominator]

        if range_x is None:
            if var_x == 'd_wall_TPC':
                range_x = (0, 7)
            elif var_x == 'start_x':
                range_x = (-3.47848, 3.47848)
            elif var_x == 'start_y':
                range_x = (-2.16671, 0.829282)
            elif var_x == 'start_z':
                range_x = (4.17924, 9.13588)
            else:
                range_x = (x.min(), x.max())
        if range_y is None:
            if var_y == 'd_wall_TPC':
                range_y = (0, 7)
            elif var_y == 'start_x':
                range_y = (-3.47848, 3.47848)
            elif var_y == 'start_y':
                range_y = (-2.16671, 0.829282)
            elif var_y == 'start_z':
                range_y = (4.17924, 9.13588)
            else:
                range_y = (y.min(), y.max())

        x_bins = np.linspace(*range_x, bins_x+1) if isinstance(bins_x, int) else bins_x
        y_bins = np.linspace(*range_y, bins_y+1) if isinstance(bins_y, int) else bins_y

        std = np.full((len(x_bins)-1, len(y_bins)-1), np.nan)
        for i in range(len(x_bins)-1):
            for j in range(len(y_bins)-1):
                mask = (
                    (x >= x_bins[i]) & (x < x_bins[i+1]) &
                    (y >= y_bins[j]) & (y < y_bins[j+1])
                )
                if np.any(mask):
                    std[i, j] = ratio[mask].std()

        plt.figure(figsize=(8, 6))
        cmap = plt.colormaps['viridis'].copy()
        cmap.set_bad(color='white')
        mesh = plt.pcolormesh(
            x_bins, y_bins, std.T,
            cmap=cmap,
            shading='auto'
        )
        plt.colorbar(label=f'Std(E_vis/{ratio_denominator})')
        plt.xlabel(var_x)
        plt.ylabel(var_y)
        plt.title(f'Std(E_vis/{ratio_denominator}) vs {var_x} vs {var_y} for {containment_str} {pdg_code_to_name(pdg)}')
        plt.tight_layout()

        for i in range(len(x_bins)-1):
            for j in range(len(y_bins)-1):
                value = std[i, j]
                if not np.isnan(value):
                    x_center = 0.5 * (x_bins[i] + x_bins[i+1])
                    y_center = 0.5 * (y_bins[j] + y_bins[j+1])
                    plt.text(
                        x_center, y_center, f"{value:.3g}",
                        color='black', ha='center', va='center', fontsize=6
                    )

        fname = os.path.join(output_dir, f"{pdg_code_to_name(pdg)}_std_E_vis_over_{ratio_denominator}_vs_{var_x}_vs_{var_y}_{containment_str}_{volume}.png")
        plt.savefig(fname)
        plt.close()


def counts_vs_vars_2d(df, var_x, var_y, bins_x=20, bins_y=20, range_x=None, range_y=None, containment=None, volume="active"):
    """
    Generalized 2D counts plot in bins of var_x vs var_y.
    - df: DataFrame
    - var_x, var_y: variable names for axes (e.g. 'd_wall_TPC', 'p', etc.)
    - bins_x, bins_y: number of bins or bin edges for each axis
    - range_x, range_y: (min, max) for each axis
    - containment: None (no cut), 1 (contained), or 0 (not contained)
    """
    if containment == 1:
        containment_str = "contained"
    elif containment == 0:
        containment_str = "not_contained"
    else:
        containment_str = "all"
    output_dir = os.path.join(
        get_output_dir(), f"{var_x}_vs_{var_y}_counts_{containment_str}"
    )
    os.makedirs(output_dir, exist_ok=True)
    particle_species = df['pdg'].unique()

    for pdg in particle_species:
        x_exclude_half = [3000, 2000, 1000, 0, -1000, -2000, -3000]
        x_exclude_full = [3500, 2500, 1500, 500, -500, -1500, -2500]
        z_exclude = [5157.559, 6157.559, 7157.559, 8157.559]

        if volume == "active":
            mask = (
            (df['pdg'] == pdg) &
            (df['E'] > 0) &
            (df['start_x'] > (-3478.48)) & (df['start_x'] < (3478.48)) &
            (df['start_y'] > (-2166.71)) & (df['start_y'] < (829.282)) &
            (df['start_z'] > 4179.24) & (df['start_z'] < (9135.88)) &
            (df['E_vis'] > 0)
            )
        elif volume == "fiducial":
            mask = (
            (df['pdg'] == pdg) &
            (df['E'] > 0) &
            (df['start_x'] > (-3478.48 + 500)) & (df['start_x'] < (3478.48 - 500)) &
            (df['start_y'] > (-2166.71 + 500)) & (df['start_y'] < (829.282 - 500)) &
            (df['start_z'] > 4179.24) & (df['start_z'] < (9135.88 - 1500)) &
            (df['E_vis'] > 0) 
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

        if range_x is None:
            if var_x == 'd_wall_TPC':
                range_x = (0, 7)
            elif var_x == 'start_x':
                range_x = (-3.47848, 3.47848)
            elif var_x == 'start_y':
                range_x = (-2.16671, 0.829282)
            elif var_x == 'start_z':
                range_x = (4.17924, 9.13588)
            else:
                range_x = (x.min(), x.max())
        if range_y is None:
            if var_y == 'd_wall_TPC':
                range_y = (0, 7)
            elif var_y == 'start_x':
                range_y = (-3.47848, 3.47848)
            elif var_y == 'start_y':
                range_y = (-2.16671, 0.829282)
            elif var_y == 'start_z':
                range_y = (4.17924, 9.13588)
            else:
                range_y = (y.min(), y.max())

        x_bins = np.linspace(*range_x, bins_x+1) if isinstance(bins_x, int) else bins_x
        y_bins = np.linspace(*range_y, bins_y+1) if isinstance(bins_y, int) else bins_y

        counts, _, _ = np.histogram2d(x, y, bins=[x_bins, y_bins])

        counts_masked = np.ma.masked_where(counts == 0, counts)

        plt.figure(figsize=(8, 6))
        cmap = plt.colormaps['viridis'].copy()
        cmap.set_bad(color='white')
        mesh = plt.pcolormesh(
            x_bins, y_bins, counts_masked.T,
            cmap=cmap,
            shading='auto'
        )
        plt.colorbar(label='Counts')
        plt.xlabel(var_x)
        plt.ylabel(var_y)
        plt.title(f'Counts vs {var_x} vs {var_y} for {containment_str} {pdg_code_to_name(pdg)}, {volume} volume')
        plt.tight_layout()

        for i in range(len(x_bins)-1):
            for j in range(len(y_bins)-1):
                value = counts[i, j]
                if value > 0:
                    x_center = 0.5 * (x_bins[i] + x_bins[i+1])
                    y_center = 0.5 * (y_bins[j] + y_bins[j+1])
                    plt.text(
                        x_center, y_center, f"{int(value)}",
                        color='black', ha='center', va='center', fontsize=6
                    )

        fname = os.path.join(output_dir, f"{pdg_code_to_name(pdg)}_counts_{var_x}_vs_{var_y}_{containment_str}_{volume}.png")
        plt.savefig(fname)
        plt.close()

def heaviside_gaussian(x, mu, sigma, norm):
    return norm * np.exp(-0.5 * ((x - mu) / sigma) ** 2) * (x <= 1)

def specific_bin_hist(df, pdg, var_x, var_y, range_x, range_y, ratio_denominator='E', containment=None, bins=12, volume="active"):
    if containment == 1:
        containment_str = "contained"
    elif containment == 0:
        containment_str = "not_contained"
    else:
        containment_str = "all"
    output_dir = os.path.join(
        get_output_dir(), f"{var_x}_vs_{var_y}_counts_{containment_str}"
    )
    os.makedirs(output_dir, exist_ok=True)


    x_exclude_half = [3000, 2000, 1000, 0, -1000, -2000, -3000]
    x_exclude_full = [3500, 2500, 1500, 500, -500, -1500, -2500]
    z_exclude = [5157.559, 6157.559, 7157.559, 8157.559]


    if volume == "active":
        mask = (
        (df['pdg'] == pdg) &
        (df['E'] > 0) &
        (df['start_x'] > (-3478.48)) & (df['start_x'] < (3478.48)) &
        (df['start_y'] > (-2166.71)) & (df['start_y'] < (829.282)) &
        (df['start_z'] > 4179.24) & (df['start_z'] < (9135.88)) &
        (df['E_vis'] > 0)
        )
    elif volume == "fiducial":
        mask = (
        (df['pdg'] == pdg) &
        (df['E'] > 0) &
        (df['start_x'] > (-3478.48 + 500)) & (df['start_x'] < (3478.48 - 500)) &
        (df['start_y'] > (-2166.71 + 500)) & (df['start_y'] < (829.282 - 500)) &
        (df['start_z'] > 4179.24) & (df['start_z'] < (9135.88 - 1500)) &
        (df['E_vis'] > 0) 
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


    x = df_pdg[var_x] / 1000 if var_x in ['d_wall_TPC', 'start_x', 'start_y', 'start_z'] else df_pdg[var_x]
    y = df_pdg[var_y] / 1000 if var_y in ['d_wall_TPC', 'start_x', 'start_y', 'start_z'] else df_pdg[var_y]
    x_min, x_max = range_x
    y_min, y_max = range_y

    if var_y in ['d_wall_TPC','start_x', 'start_y', 'start_z']:
        y = df_pdg[var_y] / 1000
    else:
        y = df_pdg[var_y]
    y_min, y_max = range_y

    df_cut = df_pdg[(x >= x_min) & (x < x_max) & (y >= y_min) & (y < y_max)]
    if df_cut.empty:
        print(f"No entries for {pdg_code_to_name(pdg)} in selected bin.")
        return

    ratio = df_cut['E_vis'] / df_cut[ratio_denominator]

    counts, bin_edges = np.histogram(ratio, bins=bins, range=(0, 1.2))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])


    fit_mask = bin_centers <= 1
    x_fit = bin_centers[fit_mask]
    y_fit = counts[fit_mask]

    # Initial guesses: mu, sigma, norm
    p0 = [ratio.mean(), ratio.std(), y_fit.max()]



    plt.figure(figsize=(7, 5))
    plt.hist(ratio, bins=bins, range=(0, 1.2), histtype='step', color='blue')
    plt.xlabel(f'E_vis / {ratio_denominator}')
    plt.ylabel('Counts')
    plt.title(f'{pdg_code_to_name(pdg)}: E_vis/{ratio_denominator} in {var_x}=[{range_x[0]}, {range_x[1]}], {var_y}=[{range_y[0]}, {range_y[1]}]')
    try:
        popt, pcov = curve_fit(heaviside_gaussian, x_fit, y_fit, p0=p0)
        mu, sigma, norm_factor = popt
        fit_label = f'Heaviside Gaussian fit\n$\mu$={mu:.3f}, $\sigma$={sigma:.3f}'
        x_vals = np.linspace(0, 1.2, 200)
        plt.plot(
            x_vals,
            heaviside_gaussian(x_vals, *popt),
            'r--',
            label=fit_label
        )
        n_samples = int(np.sum(y_fit))
        samples = []
        while len(samples) < n_samples:
            r = np.random.normal(mu, sigma, n_samples)
            r = r[r <= 1]
            samples.extend(r.tolist())
        samples = np.array(samples[:n_samples])
        plt.hist(samples, bins=bins, range=(0, 1.2), histtype='step', color='green', label='Random Throws', alpha=0.7)

    except Exception as e:
        print(f"Fit failed: {e}")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(output_dir, f"{pdg_code_to_name(pdg)}_ratio_{ratio_denominator}_hist_{var_x}_{range_x[0]}_{range_x[1]}_{var_y}_{range_y[0]}_{range_y[1]}_{containment_str}_{volume}.png")
    plt.savefig(fname)
    plt.close()

if __name__ == "__main__":
    df = read_edep_sim_output("outputs/edep_sim_output_chunk*.root", tree_name="events")
    variables = ['d_wall_TPC', 'p', 'start_x', 'start_y', 'start_z', 'E', 'E_vis', ]
    # for var in variables:
        # for containment in [0, 1]:
            # ratio_vs_variable_norm_columns(
                # read_edep_sim_output("outputs/edep_sim_output_chunk*.root", tree_name="events"),
                # var='d_wall_TPC',
                # containment=containment,
                # bins_x=50,
                # bins_y=50,
                # var_range=(0, 7)
            # )
    for containment in [0, 1]:
        ratio_vs_variable_norm_columns(
            # read_edep_sim_output("outputs/edep_sim_output_chunk*.root", tree_name="events"),
            df=df,
            var='start_x',
            containment=containment,
            bins_x=50,
            bins_y=50,
            volume="fiducial"
        )

    
    # for var_x, var_y in combinations(variables, 2):
    #     for containment in [0, 1, all]:
    #         avg_ratio_vs_vars_2d(
    #             df=df,
    #             var_x=var_x,
    #             var_y=var_y,
    #             bins_x=20,
    #             bins_y=20,
    #             range_x=None,
    #             range_y=None,
    #             ratio_denominator='E',
    #             containment=containment,
    #             volume="fiducial"
    #         )
    #         std_ratio_vs_vars_2d(
    #             df=df,
    #             var_x=var_x,
    #             var_y=var_y,
    #             bins_x=20,
    #             bins_y=20,
    #             range_x=None,
    #             range_y=None,
    #             ratio_denominator='E',
    #             containment=containment,
    #             volume="fiducial"
    #         )
    #         counts_vs_vars_2d(
    #             df=df,
    #             var_x=var_x,
    #             var_y=var_y,
    #             bins_x=20,
    #             bins_y=20,
    #             range_x=None,
    #             range_y=None,
    #             containment=containment,
    #             volume="fiducial"
    #         )
    # counts_vs_vars_2d(
    #     df=df,
    #     var_x='d_wall_TPC',
    #     var_y='p',
    #     bins_x=14,
    #     bins_y=20,
    #     range_x=None,
    #     range_y=(0,5000),
    #     volume="fiducial",
    #     containment=1
    # )
    # avg_ratio_vs_vars_2d(
    #     df=df,
    #     var_x='d_wall_TPC',
    #     var_y='p',
    #     bins_x=14,
    #     bins_y=20,
    #     range_x=None,
    #     range_y=(0,5000),
    #     ratio_denominator='E',
    #     volume="fiducial",
    #     containment=0
    # )
    # std_ratio_vs_vars_2d(
    #     df=df,
    #     var_x='d_wall_TPC',
    #     var_y='p',
    #     bins_x=14,
    #     bins_y=20,
    #     range_x=None,
    #     range_y=(0,5000),
    #     ratio_denominator='E',
    #     volume="fiducial",
    #     containment=0
    # )
    # specific_bin_hist(
    #     df=df,
    #     pdg=211,
    #     var_x='d_wall_TPC',
    #     var_y='p',
    #     range_x=(1.5, 2.0),
    #     range_y=(500, 750),
    #     ratio_denominator='E',
    #     # containment='all',
    #     # bins=84,
    #     volume="fiducial"
    # )
    pass