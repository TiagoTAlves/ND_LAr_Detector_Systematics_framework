import glob
import os
import subprocess
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot
from scipy.optimize import curve_fit
from scipy.stats import kstest, norm
from typing import Optional

plt.rcParams.update({
    "font.size": 15,
    "font.family": "DejaVu Sans",
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT_DIR / "thesis_plots"
ROOT_PATTERN = "outputs/cafs/*.CAF.root"
EDEP_ROOT_PATTERN = "outputs/edep/*.root"

PARTICLE_SPECS = {
    "muon": {
        "label": "Muons",
        "display": "μ⁻",
        "energy_terms": [("LepE", "sigma_13")],
        "notes": "Uses the leptonic energy branch as the main proxy for muon energy response.",
    },
    "charged_pion": {
        "label": "Charged pions",
        "display": "π⁺/π⁻",
        "energy_terms": [("ePip", "sigma_211"), ("ePim", "sigma_-211")],
        "notes": "Combines charged-pion energy components with the corresponding sigma shifts.",
    },
    "neutral_pion": {
        "label": "Neutral pions",
        "display": "π⁰",
        "energy_terms": [("ePi0", "sigma_111")],
        "notes": "Uses the neutral-pion energy branch as the main response proxy.",
    },
    "photon": {
        "label": "Photons",
        "display": "γ",
        "energy_terms": [("ePi0", "sigma_111")],
        "notes": "Uses an electromagnetic-energy proxy because the current CAF schema does not expose a dedicated photon branch.",
    },
    "proton": {
        "label": "Protons",
        "display": "p",
        "energy_terms": [("eP", "sigma_p")],
        "notes": "Uses the proton energy branch and its associated sigma variation.",
    },
    "neutron": {
        "label": "Neutrons",
        "display": "n",
        "energy_terms": [("eN", "sigma_n")],
        "notes": "Uses the neutron energy branch and its associated sigma variation.",
    },
    "electron": {
        "label": "Electrons",
        "display": "e⁻",
        "energy_terms": [("LepE", "sigma_11")],
        "notes": "Uses the leptonic energy branch as a proxy for electron energy response.",
    },
}


def read_caf_data(pattern: str = ROOT_PATTERN) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No CAF files matched {pattern}")

    frames = []
    for path in files:
        with uproot.open(path) as handle:
            tree = handle["caf"]
            frames.append(tree.arrays(library="pd"))
    if not frames:
        raise RuntimeError("No CAF data was loaded")
    return pd.concat(frames, ignore_index=True)


def read_edep_data(pattern: str = EDEP_ROOT_PATTERN) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No EDEP files matched {pattern}")

    frames = []
    for path in files:
        with uproot.open(path) as handle:
            tree = handle["events"]
            frames.append(tree.arrays(library="pd"))
    if not frames:
        raise RuntimeError("No EDEP data was loaded")
    return pd.concat(frames, ignore_index=True)


def save_pdf_compatible_plot(filename: str, fig=None) -> str:
    if fig is None:
        fig = plt.gcf()

    for ax in fig.get_axes():
        for coll in ax.findobj(matplotlib.collections.PathCollection):
            coll.set_zorder(10)
            coll.set_alpha(1.0)
        for line in ax.get_lines():
            line.set_zorder(11)
        for poly in ax.findobj(matplotlib.collections.PolyCollection):
            poly.set_zorder(1)
            poly.set_alpha(1.0)
        for art in ax.findobj(matplotlib.artist.Artist):
            art.set_alpha(1.0)

    base = os.path.splitext(filename)[0]
    temp_pdf = f"{base}_temp.pdf"
    final_pdf = f"{base}.pdf"
    os.makedirs(os.path.dirname(final_pdf), exist_ok=True)

    fig.savefig(temp_pdf, format="pdf", dpi=600, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    gs_cmd = [
        "gs",
        "-q",
        "-dPDFA=1",
        "-dBATCH",
        "-dNOPAUSE",
        "-dNOOUTERSAVE",
        "-dDetectDuplicateImage",
        "-r300",
        "-sDEVICE=pdfwrite",
        "-dPDFACompatibilityPolicy=1",
        "-sProcessColorModel=DeviceCMYK",
        "-sColorConversionStrategy=UseDeviceIndependentColor",
        "-dEmbedAllFonts=true",
        "-dSubsetFonts=true",
        "-sDefaultRGBProfile=default_rgb.icc",
        "-dCompressFonts=true",
        f"-sOutputFile={final_pdf}",
        temp_pdf,
    ]

    try:
        subprocess.run(gs_cmd, check=True, capture_output=True, text=True)
        if os.path.exists(temp_pdf):
            os.remove(temp_pdf)
        print(f"Validated PDF/A: {final_pdf}")
    except subprocess.CalledProcessError as exc:
        if os.path.exists(temp_pdf):
            os.rename(temp_pdf, final_pdf)
            print(f"Ghostscript failed. Saved fallback (non-compliant): {final_pdf}")
        print(f"GS detail: {exc.stderr}")

    return final_pdf


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    sigma_cols = [col for col in prepared.columns if col.startswith("sigma_")]
    for col in sigma_cols:
        prepared[col] = pd.to_numeric(prepared[col], errors="coerce").fillna(1.0)
        prepared[col] = prepared[col].clip(upper=1.0)

    for col in ["E", "erec", "LepE", "HadE", "ePip", "ePim", "ePi0", "eN", "eP"]:
        if col in prepared.columns:
            prepared[col] = pd.to_numeric(prepared[col], errors="coerce")

    prepared["is_contained"] = prepared["is_contained"].astype(int)
    return prepared


def build_shifted_energy(df: pd.DataFrame, spec: dict) -> tuple[np.ndarray, np.ndarray]:
    base_energy = df["E"].astype(float)
    plus = base_energy.copy()
    minus = base_energy.copy()
    for component, sigma in spec["energy_terms"]:
        if component in df.columns and sigma in df.columns:
            weights = df[sigma].fillna(1.0).clip(upper=1.0)
            values = df[component].fillna(0.0)
            plus += 0.5 * values * weights
            minus -= 0.5 * values * weights
    return plus.to_numpy(), minus.to_numpy()


def _prepare_edep_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    for col in ["E", "E_vis", "p", "d_wall_TPC", "start_x", "start_y", "start_z", "is_contained_TPC"]:
        if col in prepared.columns:
            prepared[col] = pd.to_numeric(prepared[col], errors="coerce")
    if "pdg" in prepared.columns:
        prepared["pdg"] = pd.to_numeric(prepared["pdg"], errors="coerce")
    return prepared


def _get_edep_species_mask(df: pd.DataFrame, species_key: str) -> pd.Series:
    species_pdgs = {
        "muon": {13, -13},
        "charged_pion": {211, -211},
        "neutral_pion": {111},
        "photon": {22},
        "proton": {2212},
        "neutron": {2112},
        "electron": {11, -11},
    }
    if "pdg" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["pdg"].isin(species_pdgs.get(species_key, set()))


def _filter_edep_for_volume(df: pd.DataFrame, containment=None, volume: str = "fiducial") -> pd.DataFrame:
    if volume == "active":
        mask = (
            (df["E"].fillna(0) > 0) &
            (df["start_x"].fillna(0) > -3478.48) & (df["start_x"].fillna(0) < 3478.48) &
            (df["start_y"].fillna(0) > -2166.71) & (df["start_y"].fillna(0) < 829.282) &
            (df["start_z"].fillna(0) > 4179.24) & (df["start_z"].fillna(0) < 9135.88) &
            (df["E_vis"].fillna(0) > 0)
        )
    elif volume == "fiducial":
        mask = (
            (df["E"].fillna(0) > 0) &
            (df["start_x"].fillna(0) > -3478.48 + 500) & (df["start_x"].fillna(0) < 3478.48 - 500) &
            (df["start_y"].fillna(0) > -2166.71 + 500) & (df["start_y"].fillna(0) < 829.282 - 500) &
            (df["start_z"].fillna(0) > 4179.24) & (df["start_z"].fillna(0) < 9135.88 - 1500) &
            (df["E_vis"].fillna(0) > 0)
        )
    else:
        raise ValueError(f"Unknown volume: {volume}")

    if containment is not None:
        mask &= (df["is_contained_TPC"].fillna(0) == containment)

    filtered = df[mask].copy()
    x_exclude_half = [3000, 2000, 1000, 0, -1000, -2000, -3000]
    x_exclude_full = [3500, 2500, 1500, 500, -500, -1500, -2500]
    z_exclude = [5157.559, 6157.559, 7157.559, 8157.559]

    for xc in x_exclude_half:
        filtered = filtered[~((filtered["start_x"] > (xc - 3.175)) & (filtered["start_x"] < (xc + 3.175)))]
    for xc in x_exclude_full:
        filtered = filtered[~((filtered["start_x"] > (xc - 28.915)) & (filtered["start_x"] < (xc + 28.915)))]
    for zc in z_exclude:
        filtered = filtered[~((filtered["start_z"] > (zc - 5)) & (filtered["start_z"] < (zc + 5)))]
    return filtered


def _make_bins(values: np.ndarray, n_bins: int = 30) -> np.ndarray:
    vmax = np.nanmax(values) if np.nanmax(values) > 0 else 10.0
    upper = max(10.0, float(vmax) * 1.05)
    return np.linspace(0.0, upper, n_bins + 1)


def plot_energy_response(df: pd.DataFrame, species_key: str, spec: dict, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    contained = df["is_contained"] == 1
    uncontained = df["is_contained"] == 0

    nominal_geV = df["E"].fillna(0) / 1000.0
    erec_geV = df["erec"].fillna(0) / 1000.0
    bins = _make_bins(np.r_[nominal_geV, erec_geV])

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.hist(nominal_geV[contained], bins=bins, histtype="step", color="#1f77b4", linewidth=2.0, label="True energy (contained)")
    ax.hist(nominal_geV[uncontained], bins=bins, histtype="step", color="#ff7f0e", linewidth=2.0, label="True energy (uncontained)")
    ax.hist(erec_geV[contained], bins=bins, histtype="step", linestyle="--", color="#2ca02c", linewidth=2.0, label="Reconstructed energy (contained)")
    ax.hist(erec_geV[uncontained], bins=bins, histtype="step", linestyle="--", color="#d62728", linewidth=2.0, label="Reconstructed energy (uncontained)")
    ax.set_xlabel("Energy (GeV)")
    ax.set_ylabel("Entries")
    ax.set_title(f"Energy response for {spec['display']} ({spec['label']})")
    ax.legend(loc="best", frameon=False)
    ax.grid(alpha=0.25)
    save_pdf_compatible_plot(str(outdir / f"{species_key}_fit_energy_response_contained_vs_uncontained_fiducial.pdf"), fig)


def plot_counts(df: pd.DataFrame, species_key: str, spec: dict, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    counts = [int((df["is_contained"] == 1).sum()), int((df["is_contained"] == 0).sum())]
    labels = ["Contained", "Uncontained"]

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.bar(labels, counts, color=["#2ca02c", "#d62728"], edgecolor="black", linewidth=0.8)
    ax.set_ylabel("Count")
    ax.set_title(f"Containment counts for {spec['display']} ({spec['label']})")
    ax.grid(axis="y", alpha=0.25)
    save_pdf_compatible_plot(str(outdir / f"{species_key}_counts_energy_response_contained_vs_uncontained_fiducial.pdf"), fig)


def plot_ratio_histogram(df: pd.DataFrame, species_key: str, spec: dict, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    mask = df["E"].fillna(0) > 0
    ratio = (df.loc[mask, "erec"] / df.loc[mask, "E"]).replace([np.inf, -np.inf], np.nan).dropna()
    contained_ratio = ratio[df.loc[mask, "is_contained"] == 1]
    uncontained_ratio = ratio[df.loc[mask, "is_contained"] == 0]

    bins = np.linspace(0.0, 2.0, 25)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.hist(contained_ratio, bins=bins, histtype="step", color="#2ca02c", linewidth=2.0, label="Contained")
    ax.hist(uncontained_ratio, bins=bins, histtype="step", color="#d62728", linewidth=2.0, label="Uncontained")
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Reconstructed energy / true energy")
    ax.set_ylabel("Entries")
    ax.set_title(f"Energy-ratio distribution for {spec['display']} ({spec['label']})")
    ax.legend(loc="best", frameon=False)
    ax.grid(alpha=0.25)
    save_pdf_compatible_plot(str(outdir / f"{species_key}_ratio_energy_hist_contained_vs_uncontained_fiducial.pdf"), fig)


def plot_shift_summary(df: pd.DataFrame, species_key: str, spec: dict, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    plus, minus = build_shifted_energy(df, spec)
    nominal = df["E"].fillna(0) / 1000.0
    bins = _make_bins(np.r_[nominal, plus / 1000.0, minus / 1000.0])

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.hist(nominal, bins=bins, histtype="step", color="#1f77b4", linewidth=2.0, label="Nominal")
    ax.hist(plus / 1000.0, bins=bins, histtype="step", color="#ff7f0e", linewidth=2.0, label="Shifted +")
    ax.hist(minus / 1000.0, bins=bins, histtype="step", color="#9467bd", linewidth=2.0, label="Shifted -")
    ax.set_xlabel("Energy (GeV)")
    ax.set_ylabel("Entries")
    ax.set_title(f"Shifted-energy response for {spec['display']} ({spec['label']})")
    ax.legend(loc="best", frameon=False)
    ax.grid(alpha=0.25)
    save_pdf_compatible_plot(str(outdir / f"{species_key}_hist_evis_over_ekin_contained_vs_uncontained.pdf"), fig)


def plot_combined_summary(df: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.8, 5.7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

    nominal = df["E"].fillna(0) / 1000.0
    erec = df["erec"].fillna(0) / 1000.0
    bins = _make_bins(np.r_[nominal, erec])

    ax1.hist(nominal, bins=bins, histtype="step", color="#1f77b4", linewidth=2.0, label="True energy")
    ax1.hist(erec, bins=bins, histtype="step", color="#2ca02c", linewidth=2.0, label="Reconstructed energy")
    ax1.set_ylabel("Entries")
    ax1.set_title("Combined energy-response summary")
    ax1.legend(loc="best", frameon=False)
    ax1.grid(alpha=0.25)

    diff = (df["erec"] - df["E"]) / 1000.0
    ax2.hist(diff, bins=30, color="#d62728", alpha=0.8)
    ax2.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax2.set_xlabel("Reconstructed energy minus true energy (GeV)")
    ax2.set_ylabel("Entries")
    ax2.grid(alpha=0.25)

    save_pdf_compatible_plot(str(outdir / "combined_energy_and_difference.pdf"), fig)


def select_species_subset(df: pd.DataFrame, species_key: str) -> pd.DataFrame:
    subset = df.copy()
    subset = subset[subset["E"].fillna(0) > 0]
    subset = subset[subset["erec"].fillna(0) >= 0]

    if species_key == "muon":
        subset = subset[subset["LepE"].fillna(0) > 0]
    elif species_key == "charged_pion":
        subset = subset[((subset["ePip"].fillna(0) > 0) | (subset["ePim"].fillna(0) > 0))]
    elif species_key == "neutral_pion":
        subset = subset[subset["ePi0"].fillna(0) > 0]
    elif species_key == "photon":
        subset = subset[subset["ePi0"].fillna(0) > 0]
    elif species_key == "proton":
        subset = subset[subset["eP"].fillna(0) > 0]
    elif species_key == "neutron":
        subset = subset[subset["eN"].fillna(0) > 0]
    elif species_key == "electron":
        subset = subset[subset["LepE"].fillna(0) > 0]

    return subset


def _get_scales(df: pd.DataFrame, var: str) -> tuple[np.ndarray, str]:
    values = pd.to_numeric(df[var], errors="coerce").fillna(0.0)
    if var in {"E", "erec", "LepE", "HadE", "ePip", "ePim", "ePi0", "eN", "eP"}:
        values = values / 1000.0
    elif var in {"d_wall_TPC", "start_x", "start_y", "start_z"}:
        values = values / 1000.0
    elif var == "p":
        values = values / 1000.0

    if var in {"E", "erec", "LepE", "HadE", "ePip", "ePim", "ePi0", "eN", "eP"}:
        label = "Energy (GeV)"
    elif var == "p":
        label = "Momentum (GeV/c)"
    elif var in {"d_wall_TPC", "start_x", "start_y", "start_z"}:
        label = "Distance to Wall(m)" if var == "d_wall_TPC" else "Position (m)"
    else:
        label = var
    return values.to_numpy(), label


def _get_axis_range(var: str, values: np.ndarray) -> tuple[float, float]:
    if var == "d_wall_TPC":
        return (0.0, 7.0)
    if var == "p":
        return (0.0, 5.0)
    if var == "start_x":
        return (-3.47848, 3.47848)
    if var == "start_y":
        return (-2.16671, 0.829282)
    if var == "start_z":
        return (4.17924, 9.13588)

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return (0.0, 1.0)
    positive = finite[finite >= 0]
    if positive.size == 0:
        return (0.0, 1.0)
    upper = float(np.nanmax(positive))
    if upper <= 0:
        upper = 1.0
    return (0.0, max(upper, 1.0))


def _get_ratio_values(df: pd.DataFrame) -> np.ndarray:
    if "E_vis" in df.columns and "E" in df.columns:
        ratio = (df["E_vis"].fillna(0.0) / df["E"].fillna(0.0)).replace([np.inf, -np.inf], np.nan)
        return ratio.to_numpy()
    if "erec" in df.columns and "E" in df.columns:
        ratio = (df["erec"].fillna(0.0) / df["E"].fillna(0.0)).replace([np.inf, -np.inf], np.nan)
        return ratio.to_numpy()
    return np.array([], dtype=float)


def _select_edep_species_subset(df: pd.DataFrame, species_key: str) -> pd.DataFrame:
    subset = df.copy()
    subset = subset[subset["E"].fillna(0) > 0]
    subset = subset[subset["E_vis"].fillna(0) >= 0]
    subset = subset[_get_edep_species_mask(subset, species_key)]
    return subset


def _build_2d_bin_edges(x_range: tuple[float, float], y_range: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    x_edges = np.arange(x_range[0], x_range[1] + 0.5, 0.5)
    y_edges = np.arange(y_range[0], y_range[1] + 0.25, 0.25)
    if x_edges[-1] < x_range[1]:
        x_edges = np.append(x_edges, x_range[1])
    if y_edges[-1] < y_range[1]:
        y_edges = np.append(y_edges, y_range[1])
    return x_edges, y_edges


def _make_2d_plot(df: pd.DataFrame, species_key: str, spec: dict, outdir: Path, kind: str, containment: Optional[int] = None) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    subset = df.copy()
    
    if containment == 1:
        label_suffix = "contained "
    elif containment == 0:
        label_suffix = "uncontained"
    else:
        label_suffix = "total"

    x_vals, x_label = _get_scales(subset, "d_wall_TPC")
    y_vals, y_label = _get_scales(subset, "p")
    mask = np.isfinite(x_vals) & np.isfinite(y_vals) & (x_vals >= 0) & (y_vals >= 0)
    x_vals = x_vals[mask]
    y_vals = y_vals[mask]
    ratio = _get_ratio_values(subset.loc[mask]).astype(float)
    ratio = ratio[np.isfinite(ratio)]

    x_range = _get_axis_range("d_wall_TPC", x_vals)
    y_range = _get_axis_range("p", y_vals)
    x_bins, y_bins = _build_2d_bin_edges(x_range, y_range)

    if kind == "counts":
        counts, _, _ = np.histogram2d(x_vals, y_vals, bins=[x_bins, y_bins])
        counts_masked = np.ma.masked_where(counts == 0, counts)
        fig, ax = plt.subplots(figsize=(7.0, 5.2))
        cmap = plt.colormaps["viridis"].copy()
        cmap.set_bad(color="white")
        mesh = ax.pcolormesh(x_bins, y_bins, counts_masked.T, cmap=cmap, shading="auto")
        fig.colorbar(mesh, ax=ax, label="Counts")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_xlim(*x_range)
        ax.set_ylim(*y_range)
        ax.set_title(f"Counts in the d_wall_TPC-p plane for {spec['display']} ({spec['label']}) - {label_suffix}")
        ax.grid(False)
        for i in range(counts.shape[0]):
            for j in range(counts.shape[1]):
                value = counts[i, j]
                if value > 0:
                    x_center = 0.5 * (x_bins[i] + x_bins[i + 1])
                    y_center = 0.5 * (y_bins[j] + y_bins[j + 1])
                    ax.text(x_center, y_center, f"{int(value)}", ha="center", va="center", fontsize=6, color="black")
        save_pdf_compatible_plot(str(outdir / f"{species_key}_counts_d_wall_TPC_vs_p_{label_suffix}.pdf"), fig)
        return

    mu_map = np.full((len(x_bins) - 1, len(y_bins) - 1), np.nan)
    sigma_map = np.full((len(x_bins) - 1, len(y_bins) - 1), np.nan)

    def heaviside_gaussian(x, mu, sigma, norm):
        return norm * np.exp(-0.5 * ((x - mu) / sigma) ** 2) * (x <= 1.0)

    for i in range(len(x_bins) - 1):
        for j in range(len(y_bins) - 1):
            in_bin = (x_vals >= x_bins[i]) & (x_vals < x_bins[i + 1]) & (y_vals >= y_bins[j]) & (y_vals < y_bins[j + 1])
            if np.sum(in_bin) < 8:
                continue
            ratio_bin = ratio[in_bin]
            if ratio_bin.size < 5:
                continue
            counts, bin_edges = np.histogram(ratio_bin, bins=12, range=(0.0, 1.2))
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            fit_mask = (bin_centers <= 1.0) & (counts > 0)
            if np.sum(fit_mask) < 4:
                continue
            try:
                popt, _ = curve_fit(heaviside_gaussian, bin_centers[fit_mask], counts[fit_mask], p0=[ratio_bin.mean(), max(ratio_bin.std(), 0.05), counts.max()])
                mu_map[i, j] = popt[0]
                sigma_map[i, j] = popt[1]
            except Exception:
                continue

    if kind == "fit":
        fig, ax = plt.subplots(figsize=(7.0, 5.2))
        cmap = plt.colormaps["viridis"].copy()
        cmap.set_bad(color="white")
        mesh = ax.pcolormesh(x_bins, y_bins, mu_map.T, cmap=cmap, shading="auto", vmin=0.0, vmax=1.0)
        fig.colorbar(mesh, ax=ax, label="Fit mean")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_xlim(*x_range)
        ax.set_ylim(*y_range)
        ax.set_title(f"Mean reconstructed-to-true ratio for {spec['display']} ({spec['label']}) - {label_suffix}")
        ax.grid(False)
        for i in range(mu_map.shape[0]):
            for j in range(mu_map.shape[1]):
                value = mu_map[i, j]
                if np.isfinite(value):
                    x_center = 0.5 * (x_bins[i] + x_bins[i + 1])
                    y_center = 0.5 * (y_bins[j] + y_bins[j + 1])
                    ax.text(x_center, y_center, f"{value:.2f}", ha="center", va="center", fontsize=6, color="black")
        save_pdf_compatible_plot(str(outdir / f"{species_key}_fit_mu_d_wall_TPC_vs_p_{label_suffix}.pdf"), fig)

    if kind == "std":
        fig, ax = plt.subplots(figsize=(7.0, 5.2))
        cmap = plt.colormaps["viridis"].copy()
        cmap.set_bad(color="white")
        mesh = ax.pcolormesh(x_bins, y_bins, sigma_map.T, cmap=cmap, shading="auto")
        fig.colorbar(mesh, ax=ax, label="Fit sigma")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_xlim(*x_range)
        ax.set_ylim(*y_range)
        ax.set_title(f"Fit sigma for {spec['display']} ({spec['label']}) - {label_suffix}")
        ax.grid(False)
        for i in range(sigma_map.shape[0]):
            for j in range(sigma_map.shape[1]):
                value = sigma_map[i, j]
                if np.isfinite(value):
                    x_center = 0.5 * (x_bins[i] + x_bins[i + 1])
                    y_center = 0.5 * (y_bins[j] + y_bins[j + 1])
                    ax.text(x_center, y_center, f"{value:.2f}", ha="center", va="center", fontsize=6, color="black")
        save_pdf_compatible_plot(str(outdir / f"{species_key}_std_ratio_d_wall_TPC_vs_p_{label_suffix}.pdf"), fig)


def plot_counts_vs_vars_2d(df: pd.DataFrame, species_key: str, spec: dict, outdir: Path, var_x: str = "d_wall_TPC", var_y: str = "p") -> None:
    return None


def plot_fit_vs_vars_2d(df: pd.DataFrame, species_key: str, spec: dict, outdir: Path, var_x: str = "d_wall_TPC", var_y: str = "p", bins_ratio: int = 6) -> None:
    return None


def plot_std_ratio_vs_vars_2d(df: pd.DataFrame, species_key: str, spec: dict, outdir: Path, var_x: str = "d_wall_TPC", var_y: str = "p") -> None:
    return None


def plot_specific_bin_hist(df: pd.DataFrame, species_key: str, spec: dict, outdir: Path, bins: int = 12) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    species_mask = _get_edep_species_mask(df, species_key)
    species_df = df.loc[species_mask].copy()
    if species_df.empty:
        return

    x_ranges = [(0,0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 3.5), (3.5, 4.0), (4.0, 4.5), (4.5, 5.0), (5.0, 5.5), (5.5, 6.0), (6.0, 6.5), (6.5, 7.0)]
    y_ranges = [
        (0, 0.25),
        (0.25, 0.5),
        (0.5, 0.75),
        (0.75, 1.0),
        (1.0, 1.25),
        (1.25, 1.5),
        (1.5, 1.75),
        (1.75, 2.0),
        (2.0, 2.25),
        (2.25, 2.5),
        (2.5, 2.75),
        (2.75, 3.0),
        (3.0, 3.25),
        (3.25, 3.5),
        (3.5, 3.75),
        (3.75, 4.0),
        (4.0, 4.25),
        (4.25, 4.5),
        (4.5, 4.75),
        (4.75, 5.0)
    ]

    for containment in [0, 1, "total"]:
        if containment == "total":
            containment_df = _filter_edep_for_volume(species_df, volume="fiducial")
        else:
            containment_df = _filter_edep_for_volume(species_df, containment=containment, volume="fiducial")
        if containment_df.empty:
            continue

        for x_range in x_ranges:
            for y_range in y_ranges:
                x_vals = containment_df["d_wall_TPC"].fillna(0.0) / 1000.0
                y_vals = containment_df["p"].fillna(0.0) / 1000.0
                mask = (
                    (x_vals >= x_range[0]) & (x_vals < x_range[1]) &
                    (y_vals >= y_range[0]) & (y_vals < y_range[1]) &
                    (containment_df["E"].fillna(0) > 0) &
                    (containment_df["E_vis"].fillna(0) > 0)
                )
                window_df = containment_df.loc[mask].copy()
                if window_df.shape[0] < 5:
                    continue

                ratio = window_df["E_vis"].fillna(0.0) / window_df["E"].fillna(0.0)
                ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
                if ratio.size < 5:
                    continue

                counts, bin_edges = np.histogram(ratio, bins=bins, range=(0.0, 1.2))
                bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                fit_mask = (bin_centers <= 1.0) & (counts > 0)
                if np.sum(fit_mask) < 4:
                    continue

                def heaviside_gaussian(x, mu, sigma, norm):
                    return norm * np.exp(-0.5 * ((x - mu) / sigma) ** 2) * (x <= 1.0)

                try:
                    popt, _ = curve_fit(
                        heaviside_gaussian,
                        bin_centers[fit_mask],
                        counts[fit_mask],
                        p0=[ratio.mean(), max(ratio.std(), 0.05), counts.max()],
                    )
                    mu, sigma, _ = popt
                    x_grid = np.linspace(0.0, 1.2, 200)
                    fit_cdf = lambda x: np.clip(norm.cdf(x, loc=mu, scale=sigma), 0.0, 1.0)
                    ks_stat, p_value = kstest(ratio.to_numpy(), fit_cdf)
                except Exception:
                    continue

                fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=400)
                ax.hist(
                    ratio,
                    bins=bins,
                    range=(0.0, 1.2),
                    histtype="step",
                    linewidth=2.2,
                    color="#1f77b4",
                    label="All events",
                    zorder=3,
                )
                ax.plot(
                    x_grid,
                    heaviside_gaussian(x_grid, *popt),
                    color="#d62728",
                    linewidth=2.2,
                    solid_capstyle="round",
                    solid_joinstyle="round",
                    label=f"Fit: $\mu$={mu:.3f}, \n $\sigma$={sigma:.3f}",
                    zorder=4,
                )
                ax.axvline(1.0, color="black", linestyle="--", linewidth=1.0, zorder=2)
                ax.set_xlabel("E_vis / E", fontsize=11)
                ax.set_ylabel("Entries", fontsize=11)
                ax.set_xlim(0.0, 1.2)
                containment_label = "Contained" if containment == 1 else "Uncontained" if containment == 0 else "All events"
                ax.set_title(
                    f"{spec['display']} ({spec['label']}) | {containment_label}\n"
                    f"d_wall_TPC = [{x_range[0]:.1f}, {x_range[1]:.1f}] m | p = [{y_range[0]:.2f}, {y_range[1]:.2f}] GeV/c",
                    fontsize=10.5,
                    fontweight="semibold",
                    pad=8,
                    linespacing=1.15,
                )
                ax.grid(False)
                ax.text(
                    0.02,
                    0.95,
                    f"KS = {ks_stat:.3f}\np = {p_value:.3g}",
                    transform=ax.transAxes,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9),
                    fontsize=9,
                )
                fig.subplots_adjust(left=0.10, right=0.78, top=0.86, bottom=0.16)
                ax.legend(
                    loc="center left",
                    bbox_to_anchor=(1.01, 0.9),
                    frameon=True,
                    facecolor="white",
                    edgecolor="#bdbdbd",
                    framealpha=0.95,
                    fancybox=True,
                    borderpad=0.8,
                    fontsize=9,
                )
                save_pdf_compatible_plot(
                    str(outdir /f"{species_key}" /f"{species_key}_{'contained' if containment == 1 else 'not_contained' if containment == 0 else 'total'}_dwall_{x_range[0]:.1f}_{x_range[1]:.1f}_p_{y_range[0]:.2f}_{y_range[1]:.2f}.pdf"),
                    fig,
                )


def plot_E_kin_ratio(edep_df: pd.DataFrame, species_key: str, spec: dict, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    species_mask = _get_edep_species_mask(edep_df, species_key)
    species_df = edep_df.loc[species_mask].copy()
    if species_df.empty:
        return

    for containment in [0, 1, "total"]:
        if containment == "total":
            # containment_df = _filter_edep_for_volume(species_df, volume="fiducial")
            containment_df = species_df.copy()
        else:
            containment_df = _filter_edep_for_volume(species_df, containment=containment, volume="fiducial")
        if containment_df.empty:
            continue

        ratio = (containment_df["E_vis"].fillna(0.0) / containment_df["E_kin"].fillna(0.0)).replace([np.inf, -np.inf], np.nan).dropna()
        if ratio.size < 5:
            return

        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        fig.subplots_adjust(
            left=0.16,
            right=0.96,
            bottom=0.15,
            top=0.90,
        )
        ax.hist(ratio, bins=np.linspace(0.0, 1.2, 61), histtype="step", color="#1f77b4", linewidth=2.0)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3), useMathText=True)        
        ax.axvline(1.0, color="black", linestyle="--", linewidth=1.0)
        ax.set_xlabel(f"Visible Energy / Kinetic Energy, $R_{{\\mathrm{{kin}}}}$")
        ax.set_ylabel("Number of Particles")
        ax.set_title(f"Kinetic Energy-ratio distribution for \n {'contained' if containment == 1 else 'uncontained' if containment == 0 else 'all'} {spec['display']} ({spec['label']})")
        ax.grid(alpha=0.25)
        save_pdf_compatible_plot(str(outdir / f"{species_key}_{'contained' if containment == 1 else 'not_contained' if containment == 0 else 'total'}_kin_ratio_energy_hist.pdf"), fig)

def plot_E_true_ratio(edep_df: pd.DataFrame, species_key: str, spec: dict, outdir: Path) -> None:


    outdir.mkdir(parents=True, exist_ok=True)
    species_mask = _get_edep_species_mask(edep_df, species_key)
    species_df = edep_df.loc[species_mask].copy()
    if species_df.empty:
        return

    for containment in [0, 1, "total"]:
        if containment == "total":
            # containment_df = _filter_edep_for_volume(species_df, volume="fiducial")
            containment_df = species_df.copy()
        else:
            containment_df = _filter_edep_for_volume(species_df, containment=containment, volume="fiducial")
        if containment_df.empty:
            continue

        ratio = (containment_df["E_vis"].fillna(0.0) / containment_df["E"].fillna(0.0)).replace([np.inf, -np.inf], np.nan).dropna()
        if ratio.size < 5:
            return

        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        fig.subplots_adjust(
            left=0.16,
            right=0.96,
            bottom=0.15,
            top=0.90,
        )
        ax.hist(ratio, bins=np.linspace(0.0, 1.2, 61), histtype="step", color="#1f77b4", linewidth=2.0)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(4, 4), useMathText=True)        
        ax.axvline(1.0, color="black", linestyle="--", linewidth=1.0)
        ax.set_xlabel(f"Visible Energy / True Energy, $R_{{\\mathrm{{true}}}}$")
        ax.set_ylabel("Number of Particles")
        ax.set_title(f"True Energy-ratio distribution for \n{'contained' if containment == 1 else 'uncontained' if containment == 0 else 'all'} {spec['display']} ({spec['label']})")
        ax.grid(alpha=0.25)
        save_pdf_compatible_plot(str(outdir / f"{species_key}_{'contained' if containment == 1 else 'not_contained' if containment == 0 else 'total'}_true_ratio_energy_hist.pdf"), fig)

def main() -> None:
    df = prepare_dataframe(read_caf_data())
    edep_df = _prepare_edep_dataframe(read_edep_data())
    output_root = OUTPUT_ROOT
    output_root.mkdir(parents=True, exist_ok=True)

    for subdir_name in ["energy_response", "containment_summary", "summary", "2d_counts", "2d_fit", "2d_std", "specific_bin", "ratio"]:
        (output_root / subdir_name).mkdir(parents=True, exist_ok=True)

    # for species_key, spec in PARTICLE_SPECS.items():
    #     subset = select_species_subset(df, species_key)
    #     if subset.empty:
    #         continue
    #     plot_energy_response(subset, species_key, spec, output_root / "energy_response")
    #     plot_counts(subset, species_key, spec, output_root / "containment_summary")
    #     plot_ratio_histogram(subset, species_key, spec, output_root / "energy_response")
    #     plot_shift_summary(subset, species_key, spec, output_root / "energy_response")

    for species_key, spec in PARTICLE_SPECS.items():

        plot_E_kin_ratio(edep_df, species_key, spec, output_root / "ratio")
        plot_E_true_ratio(edep_df, species_key, spec, output_root / "ratio")
    
    #     for containment in [None, 0, 1]:
        
    #         # Apply fiducial-volume and containment cuts first
    #         filtered = _filter_edep_for_volume(
    #             edep_df,
    #             containment=containment,
    #             volume="fiducial",
    #         )
    
    #         # Then select the particle species
    #         edep_subset = _select_edep_species_subset(filtered, species_key)
    
    #         if edep_subset.empty:
    #             continue
    
    #         _make_2d_plot(
    #             edep_subset,
    #             species_key,
    #             spec,
    #             output_root / "2d_counts",
    #             kind="counts",
    #             containment=containment,
    #         )
    
    #         _make_2d_plot(
    #             edep_subset,
    #             species_key,
    #             spec,
    #             output_root / "2d_fit",
    #             kind="fit",
    #             containment=containment,
    #         )
    
    #         _make_2d_plot(
    #             edep_subset,
    #             species_key,
    #             spec,
    #             output_root / "2d_std",
    #             kind="std",
    #             containment=containment,
    #         )
    # for species_key, spec in PARTICLE_SPECS.items():
    #     plot_specific_bin_hist(edep_df, species_key, spec, output_root / "specific_bin", bins=12)

    plot_combined_summary(df, output_root / "summary")
    print(f"Wrote thesis-ready plots to {output_root}")


if __name__ == "__main__":
    main()
