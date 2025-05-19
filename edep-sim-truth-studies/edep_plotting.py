import uproot
import pandas as pd
import os
import inspect
import matplotlib.pyplot as plt

def read_edep_sim_output(root_file, tree_name="events"):
    """Read ROOT file and return a pandas DataFrame."""
    with uproot.open(root_file) as file:
        tree = file[tree_name]
        df = tree.arrays(library="pd")
    return df

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
    
    theta_bins = [(0, 3.2)]
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
                        cut = (
                            (df_pdg['theta'] >= tmin) & (df_pdg['theta'] < tmax) &
                            (df_pdg['x'] >= xmin) & (df_pdg['x'] < xmax) &
                            (df_pdg['y'] >= ymin) & (df_pdg['y'] < ymax) &
                            (df_pdg['z'] >= zmin) & (df_pdg['z'] < zmax)
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
    
    theta_bins = [(0, 3.2)]
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
                        cut = (
                            (df_pdg['theta'] >= tmin) & (df_pdg['theta'] < tmax) &
                            (df_pdg['x'] >= xmin) & (df_pdg['x'] < xmax) &
                            (df_pdg['y'] >= ymin) & (df_pdg['y'] < ymax) &
                            (df_pdg['z'] >= zmin) & (df_pdg['z'] < zmax)
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
            (df_pdg['is_contained'] == 0) &
            (df_pdg['E_vis'] > 0) 
        )
        df_cut = df_pdg[cut]
        if len(df_cut) == 0:
            continue
        ratio = df_cut['E_vis'] / df_cut['E']
        plt.figure(figsize=(6,4))
        plt.hist(ratio, bins=50, range=(0,1.2), histtype='step', color='red')
        plt.xlabel('E_vis / Etrue (not contained)')
        plt.ylabel('Counts')
        plt.title(f'{pdg_code_to_name(pdg)}, not contained')
        plt.tight_layout()
        fname = os.path.join(
            output_dir,
            f"hist_evis_over_etrue_notcontained_{pdg_code_to_name(pdg)}.png"
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
        plt.hist(ratio, bins=50, range=(0,1.2), histtype='step', color='red')
        plt.xlabel('E_vis / Etrue (contained)')
        plt.ylabel('Counts')
        plt.title(f'{pdg_code_to_name(pdg)}, contained')
        plt.tight_layout()
        fname = os.path.join(
            output_dir,
            f"hist_evis_over_etrue_contained_{pdg_code_to_name(pdg)}.png"
        )
        plt.savefig(fname)
        plt.close()

if __name__ == "__main__":
    # df = read_edep_sim_output("outputs/edep_sim_output.root")
    plot_evis_over_ekin(read_edep_sim_output("outputs/edep_sim_output.root", tree_name="events"))
    plot_evis_over_etrue(read_edep_sim_output("outputs/edep_sim_output.root", tree_name="events"))
    plot_evis_over_etrue_not_contained(read_edep_sim_output("outputs/edep_sim_output.root", tree_name="events"))
    plot_evis_over_etrue_contained(read_edep_sim_output("outputs/edep_sim_output.root", tree_name="events"))
    pass