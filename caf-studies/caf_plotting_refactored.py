"""
Energy resolution and kinetic energy ratio analysis for particle detection.
Performs fitting, plotting, and energy correction studies.
"""

import uproot
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
from particle import Particle
from scipy.optimize import curve_fit, least_squares
from scipy.stats import laplace, norm, crystalball, chi2 as chi2dist, binned_statistic
import math

# Suppress harmless runtime warnings
warnings.filterwarnings('ignore', message='divide by zero encountered in divide', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='Degrees of freedom <= 0 for slice', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='invalid value encountered in scalar divide', category=RuntimeWarning)

# Try to import iminuit for Minuit χ² minimization
try:
    from iminuit import Minuit
    use_minuit = True
except Exception:
    Minuit = None
    use_minuit = False
    print("iminuit not available; Minuit-based fits will fall back to scipy.curve_fit.")

# ============================================================================
# Configuration
# ============================================================================

bins_ratio = np.linspace(0, 2, 101)
bins_pos_diff = np.linspace(-30, 30, 121)
plots_dir = "test_plots"
particles = ['photon', 'electron', 'muon', 'pion', 'kaon', 'proton']
var_list = [
    'E', 'common_dlp_E', 'start_x', 'start_y', 'start_z', 'end_x', 'end_y', 'end_z',
    'common_dlp_start_x', 'common_dlp_start_y', 'common_dlp_start_z',
    'common_dlp_end_x', 'common_dlp_end_y', 'common_dlp_end_z',
    'px', 'py', 'pz',
    'common_dlp_px_reco', 'common_dlp_py_reco', 'common_dlp_pz_reco',
    'common_dlp_truth_overlap'
]

# ============================================================================
# Directory Setup
# ============================================================================

def setup_directories():
    """Create all necessary output directories."""
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(f"{plots_dir}/energy_distribution", exist_ok=True)
    os.makedirs(f"{plots_dir}/ratios/kin_ratio", exist_ok=True)
    os.makedirs(f"{plots_dir}/ratios/true_ratio", exist_ok=True)
    
    for var in var_list:
        for subdir in ['', '/above', '/below', '/compare']:
            os.makedirs(f"{plots_dir}/var/{var}{subdir}", exist_ok=True)
    
    for particle in particles:
        os.makedirs(f"{plots_dir}/particle/{particle}/test", exist_ok=True)
        os.makedirs(f"{plots_dir}/particle/{particle}/module_crossings", exist_ok=True)

setup_directories()

# ============================================================================
# Model Functions
# ============================================================================

def gaussian(x, amplitude, mean, sigma):
    """Gaussian model function."""
    return amplitude * np.exp(-(x - mean)**2 / (2 * sigma**2))

def two_gaussian_mixture(x, A1, mu1, sigma1, A2, mu2, sigma2):
    """Two-component Gaussian mixture model."""
    component1 = A1 * np.exp(-0.5 * ((x - mu1) / sigma1)**2)
    component2 = A2 * np.exp(-0.5 * ((x - mu2) / sigma2)**2)
    return component1 + component2

def resolution_model(E_rec, a, b, c):
    """Energy resolution model: sqrt(a^2 + (b/√E)^2 + (c/E)^2)."""
    return np.sqrt(a**2 + (b / np.sqrt(E_rec))**2 + (c / E_rec)**2)

def crystalball_pdf(x, A, beta, m, mu, sigma):
    """Crystal Ball PDF model using scipy."""
    return A * crystalball.pdf(x, beta, m, loc=mu, scale=sigma)

def cauchy(x, A, x0, gamma, offset):
    """Cauchy/Lorentzian model."""
    return A * (gamma**2 / ((x - x0)**2 + gamma**2)) + offset

# ============================================================================
# Data Loading
# ============================================================================

def load_particle_data(particle):
    """Load particle data from ROOT file."""
    file_path = f"outputs/cafs/caf_{particle}_output_full.root"
    root_file = uproot.open(file_path)
    tree = root_file[f'{particle}_tree']
    arrays = tree.arrays()
    data_dict = {key: arrays[key].to_numpy() for key in arrays.fields}
    df = pd.DataFrame(data_dict)
    df.set_index('index', inplace=True)
    return df

# ============================================================================
# Fitting Functions (Minuit-based with Scipy fallback)
# ============================================================================

def minuit_linear_fit(x, y):
    """Perform weighted linear fit y = a*x + b using Minuit or scipy."""
    x = np.asarray(x)
    y = np.asarray(y)
    n = x.size
    if n < 2:
        raise ValueError("Not enough points to fit")

    a0, b0 = np.polyfit(x, y, 1)
    residuals0 = y - (a0 * x + b0)
    sigma = np.std(residuals0, ddof=1) if residuals0.size > 1 else 1.0
    if sigma <= 0:
        sigma = 1.0

    def chi2(a, b):
        model = a * x + b
        return float(np.sum(((y - model) / sigma) ** 2))

    if use_minuit and Minuit is not None:
        try:
            m = Minuit(chi2, a=a0, b=b0)
            m.errordef = 1.0
            m.migrad()
            try:
                m.hesse()
            except Exception:
                pass
            a = float(m.values["a"])
            b = float(m.values["b"])
            a_err = float(m.errors["a"]) if "a" in m.errors else np.nan
            b_err = float(m.errors["b"]) if "b" in m.errors else np.nan
            chi2_val = float(m.fval)
        except Exception as e:
            print(f"Minuit fit failed: {e}. Falling back to np.polyfit.")
            a, b = a0, b0
            a_err, b_err = np.nan, np.nan
            y_pred = a * x + b
            chi2_val = float(np.sum(((y - y_pred) / sigma) ** 2))
    else:
        a, b = a0, b0
        a_err, b_err = np.nan, np.nan
        y_pred = a * x + b
        chi2_val = float(np.sum(((y - y_pred) / sigma) ** 2))

    dof = max(1, n - 2)
    redchi2 = chi2_val / dof if dof > 0 else np.nan
    y_pred = a * x + b
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    return {
        "coeffs": np.array([a, b]),
        "errors": np.array([a_err, b_err]),
        "r2": r2,
        "chi2": chi2_val,
        "dof": dof,
        "redchi2": redchi2
    }

def minuit_fit_gaussian(x, y, p0=None, sigma=None):
    """Fit Gaussian to histogram counts."""
    x = np.asarray(x)
    y = np.asarray(y)
    if sigma is None:
        yerr = np.sqrt(np.clip(y, 1.0, None))
    else:
        yerr = np.asarray(sigma)

    if p0 is None:
        A0 = np.max(y) if y.size > 0 else 1.0
        mu0 = np.sum(x * y) / np.sum(y) if np.sum(y) > 0 else np.mean(x)
        sigma0 = np.std(x) if x.size > 1 else 0.1
    else:
        A0, mu0, sigma0 = p0

    def chi2_fn(amplitude, mean, sigma_par):
        if sigma_par <= 0:
            return 1e12
        model = gaussian(x, amplitude, mean, sigma_par)
        return float(np.sum(((y - model) / yerr) ** 2))

    if use_minuit and Minuit is not None:
        try:
            m = Minuit(chi2_fn, amplitude=A0, mean=mu0, sigma_par=sigma0)
            m.errordef = 1.0
            m.migrad()
            try:
                m.hesse()
            except Exception:
                pass
            params = np.array([float(m.values["amplitude"]), float(m.values["mean"]), float(m.values["sigma_par"])])
            errors = np.array([
                float(m.errors["amplitude"]) if "amplitude" in m.errors else np.nan,
                float(m.errors["mean"]) if "mean" in m.errors else np.nan,
                float(m.errors["sigma_par"]) if "sigma_par" in m.errors else np.nan
            ])
            chi2_val = float(m.fval)
            success = True
            message = "Minuit"
        except Exception as e:
            success = False
            params = np.array([A0, mu0, sigma0])
            errors = np.array([np.nan, np.nan, np.nan])
            chi2_val = np.nan
            message = f"Minuit failed: {e}"
    else:
        try:
            popt, pcov = curve_fit(gaussian, x, y, p0=[A0, mu0, sigma0], sigma=yerr, 
                                   absolute_sigma=True, maxfev=20000)
            params = np.asarray(popt)
            errors = np.sqrt(np.diag(pcov)) if pcov is not None else np.array([np.nan, np.nan, np.nan])
            model = gaussian(x, *params)
            chi2_val = float(np.sum(((y - model) / yerr) ** 2))
            success = True
            message = "curve_fit"
        except Exception as e:
            success = False
            params = np.array([A0, mu0, sigma0])
            errors = np.array([np.nan, np.nan, np.nan])
            chi2_val = np.nan
            message = f"curve_fit failed: {e}"

    dof = max(1, x.size - 3)
    redchi2 = chi2_val / dof if dof > 0 and np.isfinite(chi2_val) else np.nan
    return {
        "params": params,
        "errors": errors,
        "chi2": chi2_val,
        "dof": dof,
        "redchi2": redchi2,
        "success": success,
        "message": message
    }

def minuit_fit_cauchy(x, y, p0=None, sigma=None):
    """Fit Cauchy/Lorentzian model to histogram counts."""
    x = np.asarray(x)
    y = np.asarray(y)
    if sigma is None:
        yerr = np.sqrt(np.clip(y, 1.0, None))
    else:
        yerr = np.asarray(sigma)

    if p0 is None:
        A0 = np.sum(y)
        x0 = np.sum(x * y) / np.sum(y) if np.sum(y) > 0 else np.mean(x)
        gamma0 = np.std(x) if x.size > 1 else 0.05
        offset0 = 0.0
        p0 = [A0, x0, gamma0, offset0]

    def chi2_fn(A, x0, gamma, offset):
        if gamma <= 0:
            return 1e12
        model = cauchy(x, A, x0, gamma, offset)
        return float(np.sum(((y - model) / yerr) ** 2))

    if use_minuit and Minuit is not None:
        try:
            m = Minuit(chi2_fn, A=p0[0], x0=p0[1], gamma=p0[2], offset=p0[3])
            m.errordef = 1.0
            m.migrad()
            try:
                m.hesse()
            except Exception:
                pass
            params = np.array([float(m.values[k]) for k in ("A", "x0", "gamma", "offset")])
            errors = np.array([
                float(m.errors[k]) if k in m.errors else np.nan
                for k in ("A", "x0", "gamma", "offset")
            ])
            chi2_val = float(m.fval)
            success = True
            message = "Minuit"
        except Exception as e:
            success = False
            params = np.array(p0)
            errors = np.array([np.nan] * 4)
            chi2_val = np.nan
            message = f"Minuit failed: {e}"
    else:
        try:
            popt, pcov = curve_fit(cauchy, x, y, p0=p0, sigma=yerr, absolute_sigma=True, maxfev=20000)
            params = np.asarray(popt)
            errors = np.sqrt(np.diag(pcov)) if pcov is not None else np.array([np.nan] * 4)
            model = cauchy(x, *params)
            chi2_val = float(np.sum(((y - model) / yerr) ** 2))
            success = True
            message = "curve_fit"
        except Exception as e:
            success = False
            params = np.array(p0)
            errors = np.array([np.nan] * 4)
            chi2_val = np.nan
            message = f"curve_fit failed: {e}"

    dof = max(1, x.size - 4)
    redchi2 = chi2_val / dof if dof > 0 and np.isfinite(chi2_val) else np.nan
    return {
        "params": params,
        "errors": errors,
        "chi2": chi2_val,
        "dof": dof,
        "redchi2": redchi2,
        "success": success,
        "message": message
    }

def minuit_fit_func(func, x, y, p0, bounds=None, sigma=None, names=None):
    """Generic Minuit-based least-squares wrapper."""
    x = np.asarray(x)
    y = np.asarray(y)
    if sigma is None:
        yerr = np.sqrt(np.clip(y, 1.0, None))
    else:
        yerr = np.asarray(sigma)
    if names is None:
        names = [f"p{i}" for i in range(len(p0))]

    def in_bounds(params):
        if bounds is None:
            return True
        low, high = bounds
        return all(lo <= val <= hi for val, lo, hi in zip(params, low, high))

    def chi2_arr(params):
        params = np.atleast_1d(np.asarray(params))
        try:
            if not in_bounds(params):
                return 1e12
        except Exception:
            return 1e12
        model = func(x, *params)
        return float(np.sum(((y - model) / yerr) ** 2))

    if use_minuit and Minuit is not None:
        try:
            if hasattr(Minuit, 'from_array_func'):
                m = Minuit.from_array_func(chi2_arr, p0, name=names)
            else:
                args_def = ", ".join(names)
                list_args = ", ".join(names)
                src = f"def _chi2({args_def}):\n    return chi2_arr([{list_args}])"
                local = {}
                exec(src, {'chi2_arr': chi2_arr}, local)
                chi2_named = local['_chi2']
                init_kwargs = {n: float(p0[i]) for i, n in enumerate(names)}
                m = Minuit(chi2_named, **init_kwargs)
            m.errordef = 1.0
            m.migrad()
            try:
                m.hesse()
            except Exception:
                pass
            params = np.array([float(m.values[n]) for n in names])
            errors = np.array([
                float(m.errors[n]) if n in m.errors else np.nan
                for n in names
            ])
            chi2_val = float(m.fval)
            success = True
            message = "Minuit"
        except Exception as e:
            success = False
            params = np.array(p0)
            errors = np.array([np.nan] * len(p0))
            chi2_val = np.nan
            message = f"Minuit failed: {e}"
    else:
        try:
            popt, pcov = curve_fit(func, x, y, p0=p0, sigma=yerr, 
                                   absolute_sigma=True, bounds=bounds, maxfev=20000)
            params = np.asarray(popt)
            errors = np.sqrt(np.diag(pcov)) if pcov is not None else np.array([np.nan] * len(p0))
            model = func(x, *params)
            chi2_val = float(np.sum(((y - model) / yerr) ** 2))
            success = True
            message = "curve_fit"
        except Exception as e:
            success = False
            params = np.array(p0)
            errors = np.array([np.nan] * len(p0))
            chi2_val = np.nan
            message = f"curve_fit failed: {e}"

    dof = max(1, x.size - len(p0))
    redchi2 = chi2_val / dof if dof > 0 and np.isfinite(chi2_val) else np.nan
    return {
        "params": params,
        "errors": errors,
        "chi2": chi2_val,
        "dof": dof,
        "redchi2": redchi2,
        "success": success,
        "message": message
    }

def minuit_fit_crystalball(x, y, p0=None, sigma=None, bounds=None):
    """Fit Crystal Ball model to histogram counts using scipy.optimize.curve_fit.
    
    Parameters: [A, beta, m, mu, sigma_par]
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if sigma is None:
        yerr = np.sqrt(np.clip(y, 1.0, None))
    else:
        yerr = np.asarray(sigma)

    if p0 is None:
        A0 = np.sum(y)
        beta0 = 2.0
        m0 = 2.0
        sigma0 = np.std(x) if x.size > 1 else 0.05
        mu0 = np.sum(x * y) / np.sum(y) if np.sum(y) > 0 else np.mean(x)
        p0 = [A0, beta0, m0, mu0, sigma0]

    try:
        if bounds is None:
            popt, pcov = curve_fit(crystalball_pdf, x, y, p0=p0, sigma=yerr, 
                                   absolute_sigma=True, maxfev=20000)
        else:
            popt, pcov = curve_fit(crystalball_pdf, x, y, p0=p0, sigma=yerr, 
                                   absolute_sigma=True, bounds=bounds, maxfev=20000)
        params = np.asarray(popt)
        errors = np.sqrt(np.diag(pcov)) if pcov is not None else np.array([np.nan] * len(p0))
        model = crystalball_pdf(x, *params)
        chi2_val = float(np.sum(((y - model) / yerr) ** 2))
        success = True
        message = "curve_fit"
    except Exception as e:
        success = False
        params = np.array(p0)
        errors = np.array([np.nan] * len(p0))
        chi2_val = np.nan
        message = f"curve_fit failed: {e}"

    dof = max(1, x.size - len(p0))
    redchi2 = chi2_val / dof if dof > 0 and np.isfinite(chi2_val) else np.nan
    return {
        "params": params,
        "errors": errors,
        "chi2": chi2_val,
        "dof": dof,
        "redchi2": redchi2,
        "success": success,
        "message": message
    }

# ============================================================================
# Resolution Statistics
# ============================================================================

def compute_resolution_stats(df, xedges, resolution_key):
    """Compute mean and std resolution in energy bins."""
    bin_centers = 0.5 * (xedges[:-1] + xedges[1:])
    mean_resolutions = []
    std_resolutions = []
    counts = []

    for i in range(len(xedges) - 1):
        bin_mask = (df['E_kin'] >= xedges[i]) & (df['E_kin'] < xedges[i + 1])
        res_in_bin = df.loc[bin_mask, resolution_key].to_numpy()
        if res_in_bin.size > 0:
            mean_resolutions.append(np.mean(res_in_bin))
            std_resolutions.append(np.std(res_in_bin, ddof=1))
            counts.append(res_in_bin.size)
        else:
            mean_resolutions.append(np.nan)
            std_resolutions.append(np.nan)
            counts.append(0)

    return np.array(bin_centers), np.array(mean_resolutions), np.array(std_resolutions), np.array(counts)

def fit_distributions_per_bin(df, xedges, resolution_key, n_bins_per_distribution=50, 
                              fit_types=['gaussian', 'cauchy', 'crystalball']):
    """
    Fit distributions to each energy bin and extract mean/sigma from best fits.
    
    Parameters:
    -----------
    df : DataFrame
        Data with E_kin and resolution columns
    xedges : array
        Energy bin edges
    resolution_key : str
        Column name for resolution values
    n_bins_per_distribution : int
        Number of bins for histogram within each energy bin
    fit_types : list
        Types of distributions to fit: 'gaussian', 'cauchy', 'crystalball'
    
    Returns:
    --------
    dict with keys:
        'bin_centers': energy bin centers
        'fit_means': mean values from best fits
        'fit_sigmas': sigma values from best fits
        'fit_errors': error on fitted parameters
        'fit_types_used': which fit was best for each bin
        'redchi2_values': reduced chi² for best fit
        'counts': number of events in each bin
    """
    bin_centers = 0.5 * (xedges[:-1] + xedges[1:])
    fit_means = []
    fit_sigmas = []
    fit_errors = []
    fit_types_used = []
    redchi2_values = []
    counts = []
    
    for i in range(len(xedges) - 1):
        bin_mask = (df['E_kin'] >= xedges[i]) & (df['E_kin'] < xedges[i + 1])
        res_in_bin = df.loc[bin_mask, resolution_key].to_numpy()
        
        if res_in_bin.size < 5:
            # Not enough data to fit
            fit_means.append(np.nan)
            fit_sigmas.append(np.nan)
            fit_errors.append([np.nan, np.nan])
            fit_types_used.append('none')
            redchi2_values.append(np.nan)
            counts.append(res_in_bin.size)
            continue
        
        # Create histogram for this bin
        hist_range = (np.nanpercentile(res_in_bin, 1), np.nanpercentile(res_in_bin, 99))
        hist_vals, hist_edges = np.histogram(res_in_bin, bins=n_bins_per_distribution, range=hist_range)
        hist_centers = 0.5 * (hist_edges[:-1] + hist_edges[1:])
        yerr = np.sqrt(np.clip(hist_vals, 1.0, None))
        mask_nonzero = hist_vals > 0
        
        if np.sum(mask_nonzero) < 3:
            # Not enough non-zero bins to fit
            fit_means.append(np.nan)
            fit_sigmas.append(np.nan)
            fit_errors.append([np.nan, np.nan])
            fit_types_used.append('none')
            redchi2_values.append(np.nan)
            counts.append(res_in_bin.size)
            continue
        
        # Initial guesses
        mu_init = np.mean(res_in_bin)
        sigma_init = np.std(res_in_bin)
        
        best_fit = None
        best_redchi2 = np.inf
        best_type = 'none'
        
        # Try Gaussian fit
        if 'gaussian' in fit_types:
            try:
                res_g = minuit_fit_gaussian(
                    hist_centers[mask_nonzero], hist_vals[mask_nonzero],
                    p0=[np.max(hist_vals), mu_init, sigma_init],
                    sigma=yerr[mask_nonzero]
                )
                if res_g['success'] and np.isfinite(res_g['redchi2']):
                    if res_g['redchi2'] < best_redchi2:
                        best_redchi2 = res_g['redchi2']
                        best_fit = res_g
                        best_type = 'gaussian'
            except Exception:
                pass
        
        # Try Cauchy fit
        if 'cauchy' in fit_types:
            try:
                res_c = minuit_fit_cauchy(
                    hist_centers[mask_nonzero], hist_vals[mask_nonzero],
                    p0=[np.sum(hist_vals), mu_init, max(1e-3, sigma_init), 0.0],
                    sigma=yerr[mask_nonzero]
                )
                if res_c['success'] and np.isfinite(res_c['redchi2']):
                    if res_c['redchi2'] < best_redchi2:
                        best_redchi2 = res_c['redchi2']
                        best_fit = res_c
                        best_type = 'cauchy'
            except Exception:
                pass
        
        # Try Crystal Ball fit
        if 'crystalball' in fit_types:
            try:
                # Crystal Ball params: [A, N, m, sigma, mu]
                res_cb = minuit_fit_crystalball(
                    hist_centers[mask_nonzero], hist_vals[mask_nonzero],
                    p0=[np.max(hist_vals), 1.0, 0.5, sigma_init, mu_init],
                    sigma=yerr[mask_nonzero]
                )
                if res_cb['success'] and np.isfinite(res_cb['redchi2']):
                    if res_cb['redchi2'] < best_redchi2:
                        best_redchi2 = res_cb['redchi2']
                        best_fit = res_cb
                        best_type = 'crystalball'
            except Exception:
                pass
        
        # Extract results
        if best_fit is not None and best_type != 'none':
            if best_type == 'gaussian':
                # params: [amplitude, mean, sigma]
                fit_mean = best_fit['params'][1]
                fit_sigma = best_fit['params'][2]
                fit_err = best_fit['errors'][1:3]
            elif best_type == 'cauchy':
                # params: [A, x0, gamma, offset]
                fit_mean = best_fit['params'][1]
                fit_sigma = best_fit['params'][2]
                fit_err = best_fit['errors'][1:3]
            elif best_type == 'crystalball':
                # params: [A, N, m, sigma, mu]
                fit_mean = best_fit['params'][4]
                fit_sigma = best_fit['params'][3]
                fit_err = best_fit['errors'][3:5]
            
            fit_means.append(fit_mean)
            fit_sigmas.append(fit_sigma)
            fit_errors.append(fit_err)
            fit_types_used.append(best_type)
            redchi2_values.append(best_redchi2)
        else:
            fit_means.append(np.nan)
            fit_sigmas.append(np.nan)
            fit_errors.append([np.nan, np.nan])
            fit_types_used.append('none')
            redchi2_values.append(np.nan)
        
        counts.append(res_in_bin.size)
    
    return {
        'bin_centers': np.array(bin_centers),
        'fit_means': np.array(fit_means),
        'fit_sigmas': np.array(fit_sigmas),
        'fit_errors': np.array(fit_errors),
        'fit_types_used': fit_types_used,
        'redchi2_values': np.array(redchi2_values),
        'counts': np.array(counts)
    }

def plot_binned_fit_resolution(df, xedges, resolution_key, plot_path, particle, label,
                               n_bins_per_distribution=50, fit_types=['gaussian', 'cauchy', 'crystalball']):
    """
    Plot fitted distribution means and sigmas from binned analysis.
    
    Creates a figure with two subplots:
    - Left: Mean values vs energy
    - Right: Sigma values vs energy
    """
    fit_results = fit_distributions_per_bin(
        df, xedges, resolution_key, 
        n_bins_per_distribution=n_bins_per_distribution,
        fit_types=fit_types
    )
    
    bin_centers = fit_results['bin_centers']
    fit_means = fit_results['fit_means']
    fit_sigmas = fit_results['fit_sigmas']
    fit_errors = fit_results['fit_errors']
    counts = fit_results['counts']
    
    # Filter valid points
    valid_mask = np.isfinite(fit_means) & np.isfinite(fit_sigmas) & (counts > 0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot means
    if np.any(valid_mask):
        mean_errors = fit_errors[valid_mask, 0] if len(fit_errors) > 0 else None
        ax1.errorbar(bin_centers[valid_mask], fit_means[valid_mask], 
                    yerr=mean_errors, fmt='o', markersize=6, capsize=3, capthick=1,
                    label='Fitted means', alpha=0.8)
    ax1.set_xlabel('True Kinetic Energy [GeV]')
    ax1.set_ylabel('Mean from Distribution Fit')
    ax1.set_title(f'Mean Energy Resolution vs $E_{{kin}}$ ({particle}, {label})')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot sigmas
    if np.any(valid_mask):
        sigma_errors = fit_errors[valid_mask, 1] if len(fit_errors) > 0 else None
        ax2.errorbar(bin_centers[valid_mask], fit_sigmas[valid_mask],
                    yerr=sigma_errors, fmt='o', markersize=6, capsize=3, capthick=1,
                    label='Fitted sigmas', alpha=0.8)
    ax2.set_xlabel('True Kinetic Energy [GeV]')
    ax2.set_ylabel('Sigma from Distribution Fit')
    ax2.set_title(f'Sigma Energy Resolution vs $E_{{kin}}$ ({particle}, {label})')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=100)
    plt.close()

def fit_mean_resolution_vs_ekin(bin_centers, mean_resolutions, std_resolutions, counts, xedges, 
                                 plot_path, particle, label):
    """Plot mean resolution vs energy."""
    mask = (
        np.isfinite(bin_centers) & np.isfinite(mean_resolutions) & 
        np.isfinite(std_resolutions) & (counts > 1) & (bin_centers > 0)
    )
    
    plt.figure(figsize=(12, 6))
    plt.scatter(bin_centers, mean_resolutions, s=8, label='Bin mean')
    plt.fill_between(bin_centers, mean_resolutions - std_resolutions, 
                     mean_resolutions + std_resolutions, alpha=0.3, color='red', label='σ band')
    plt.xlabel('True Kinetic Energy [GeV]')
    plt.ylabel('Relative Mean Energy Resolution\n$(E_{rec} - E_{kin}) / E_{kin}$')
    plt.title(f'Relative Mean Energy Resolution vs $E_{{kin}}$\n({particle}, {label})')
    plt.xlim(0, xedges[-1])
    plt.grid()
    plt.ylim(-2, 2)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

def fit_std_resolution_vs_ekin(bin_centers, std_resolutions, counts, xedges, plot_path, 
                                particle, label, p0=(0.0, 0.0, 0.0)):
    """Plot std deviation of resolution vs energy with model fitting."""
    mask = (
        np.isfinite(bin_centers) & np.isfinite(std_resolutions) & 
        (counts > 1) & (bin_centers > 0)
    )
    bin_centers = np.asarray(bin_centers)
    std_resolutions = np.asarray(std_resolutions)
    counts = np.asarray(counts)

    x_all = bin_centers[mask]
    y_all = std_resolutions[mask]
    counts_all = counts[mask]

    min_counts = 8
    good = counts_all >= min_counts
    if not np.any(good):
        print(f"Skipping std-fit for {particle}: no bins with >= {min_counts} entries")
        return np.array([np.nan, np.nan, np.nan])

    x_fit = x_all[good]
    y_fit = y_all[good]
    counts_fit = counts_all[good]

    sigma_y = y_fit / np.sqrt(2.0 * counts_fit)
    finite_sig = sigma_y[np.isfinite(sigma_y) & (sigma_y > 0)]
    if finite_sig.size == 0:
        sigma_y = np.ones_like(sigma_y) * np.nanmedian(np.abs(y_fit))
    else:
        sigma_y = np.where(np.isfinite(sigma_y) & (sigma_y > 0), sigma_y, np.median(finite_sig))

    def sigma_model(E, a, b, c):
        E_safe = np.where(E <= 0, 1e-6, E)
        return (a + (b / np.sqrt(E_safe)) + (c * np.sqrt(E_safe))) / E_safe

    if p0 != (0.0, 0.0, 0.0) and np.all(np.isfinite(p0)):
        p0_use = tuple(p0)
    else:
        try:
            Y2 = y_fit**2
            X = np.vstack([np.ones_like(x_fit), 1.0/x_fit, 1.0/(x_fit**2)]).T
            coeffs, *_ = np.linalg.lstsq(X, Y2, rcond=None)
            coeffs = np.clip(coeffs, 1e-12, None)
            p0_use = tuple(np.sqrt(coeffs))
        except Exception:
            p0_use = (np.nanmedian(y_fit), 0.1, 0.01)
        p0_use = tuple(np.where(np.isfinite(p0_use), p0_use, (np.nanmedian(y_fit), 0.1, 0.01)))

    lower = [0.0, 0.0, 0.0]
    upper = [1000, 1000, 1000]

    res_stage1 = minuit_fit_func(sigma_model, x_fit, y_fit, p0=list(p0_use), 
                                  bounds=(lower, upper), sigma=sigma_y, names=['a', 'b', 'c'])
    
    popt = res_stage1['params'] if res_stage1['success'] else np.array([np.nan, np.nan, np.nan])

    if x_fit.size > 0 and np.all(np.isfinite(popt)):
        x_smooth = np.linspace(np.min(x_fit), np.max(x_fit), 500)
    else:
        x_smooth = np.linspace(0.01, xedges[-1] if np.isfinite(xedges[-1]) else 1.0, 200)
    y_smooth = sigma_model(x_smooth, *popt)

    # Compute χ²/dof
    chi2_text = ""
    try:
        valid = np.isfinite(sigma_y) & (sigma_y > 0)
        if x_fit.size > 0 and np.all(np.isfinite(popt)) and np.any(valid):
            model_vals = sigma_model(x_fit[valid], *popt)
            resid = (y_fit[valid] - model_vals) / sigma_y[valid]
            chi2_val = np.sum(resid**2)
            dof = max(1, np.count_nonzero(valid) - len(popt))
            red_chi2 = chi2_val / dof
            chi2_text = f"\nχ²/dof={chi2_val:.1f}/{dof}={red_chi2:.3f}"
    except Exception:
        chi2_text = ""

    plt.figure(figsize=(12, 6))
    plt.errorbar(x_fit, y_fit, yerr=sigma_y, fmt='o', markersize=4, capsize=3, capthick=1, 
                 label='Std dev per bin', alpha=0.8)
    plt.plot(x_smooth, y_smooth, color='k', 
             label=(f'Fit: p0={popt[0]*100:.2f}%, p1={popt[1]*100:.2f}%, p2={popt[2]*100:.2f}%{chi2_text}'))
    plt.xlabel('True Kinetic Energy [GeV]')
    plt.ylabel('Std dev of Energy Resolution\n$(E_{rec} - E_{kin}) / E_{kin}$')
    plt.title(f'Std Dev of Resolution vs $E_{{kin}}$ \n({particle}, {label})')
    plt.xlim(0, xedges[-1])
    plt.ylim(-1, 5)
    plt.grid()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

def mean_std_resolution_plots(df, xedges, resolution_key, mean_path, std_path, particle, label, p0=(0.0, 0.0, 0.0)):
    """Generate both mean and std resolution plots."""
    bin_centers, mean_res, std_res, counts = compute_resolution_stats(df, xedges, resolution_key)
    fit_mean_resolution_vs_ekin(bin_centers, mean_res, std_res, counts, xedges, mean_path, particle, label)
    fit_std_resolution_vs_ekin(bin_centers, std_res, counts, xedges, std_path, particle, label, p0=p0)

# ============================================================================
# Plotting Utilities
# ============================================================================

def save_hist2d(x, y, xedges, yedges, xlabel, ylabel, title, filename):
    """Save 2D histogram."""
    plt.hist2d(x, y, alpha=0.5, bins=[xedges, yedges])
    plt.xlabel(xlabel)
    plt.colorbar()
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename)
    plt.close()

def save_scatter(x, y, xlabel, ylabel, title, filename, s=0.5, alpha=0.5):
    """Save scatter plot."""
    plt.scatter(x, y, s=s, alpha=alpha)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename)
    plt.close()

def save_reco_comparison(df, particle, outpath, include_linear=True, extra_pairs=None):
    """Compare reconstructed energy distributions with various corrections."""
    plt.figure(figsize=(12, 6))
    plt.hist(df['E_rec'], bins=np.linspace(0, 1.5, 30), alpha=0.7, label='No Correction', 
             facecolor='none', edgecolor='red', histtype='step')
    if include_linear and 'E_rec2' in df.columns:
        plt.hist(df['E_rec2'], bins=np.linspace(0, 1.5, 30), alpha=0.7, label='Linear Fit', 
                 facecolor='none', edgecolor='blue', histtype='step')
    if extra_pairs is not None:
        for col, label, color in extra_pairs:
            if col in df.columns:
                plt.hist(df[col], bins=np.linspace(0, 1.5, 30), alpha=0.7, label=label, 
                         facecolor='none', edgecolor=color, histtype='step')

    plt.xlabel('Reconstructed Energy [GeV]')
    plt.ylabel('Counts')
    plt.title(f'Reconstructed Energy Distribution Comparison \n before and after Scaling for {particle.capitalize()}s')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

# ============================================================================
# Main Analysis Functions
# ============================================================================

def plot_Ekin_ratio_for_particles(df, particle):
    """Plot E_kin ratio histogram."""
    if particle == 'muon':
        df = df[df['part_type'] == 1]
    data = (df['E_kin_ratio_common']-1).to_numpy()
    data = data[np.isfinite(data)]
    
    plt.hist(data, bins=[0, 0.25, 0.5, 0.75, 0.9, 1.1, 1.25, 1.5, 2, 5, 10], 
             color='blue', alpha=0.7)
    plt.xlabel('$(E_{rec} - E_{kin}) / E_{kin}$')
    plt.ylabel('Counts')
    plt.xlim(0, 10)
    plt.title(f'Test Histogram of E_kin Ratio for {particle.capitalize()}')
    plt.grid(True)
    plt.savefig(f"{plots_dir}/ratios/kin_ratio/E_kin_ratio_common_{particle}_histogram.png")
    plt.savefig(f"{plots_dir}/particle/{particle}/E_kin_ratio_common_{particle}_histogram.png")
    plt.close()

def plot_Etrue_ratio_for_particles(df, particle):
    """Plot E_true ratio histogram."""
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
    """Plot reconstructed vs true energy distributions."""
    plt.figure(figsize=(12, 6))
    plt.hist(df['common_dlp_E'], bins=np.linspace(0, 0.2, 51), color='blue', alpha=0.7)
    plt.hist(df['E_kin'], bins=np.linspace(0, 0.2, 51), color='orange', alpha=0.7)
    plt.xlabel(f'Energy Distribution of {particle.capitalize()}s [GeV]')
    plt.ylabel('Counts')
    plt.xlim(0, 0.2)
    plt.legend(['Reconstructed Energy', 'True Kinetic Energy'], loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(f'Energy Distribution of {particle.capitalize()}s')
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/particle/{particle}/reco_energy_distribution_{particle}_histogram.png")
    plt.savefig(f"{plots_dir}/energy_distribution/reco_energy_distribution_{particle}_histogram.png")
    plt.close()

def plot_position_differences(df, particle):
    """Plot position differences between reconstructed and true."""
    for direction in ['x', 'y', 'z']:
        for position in ['start', 'end']:
            diff = df[f'{position}_{direction}'] - df[f'common_dlp_{position}_{direction}']
            plt.hist(diff, bins=bins_pos_diff, color='blue', alpha=0.7)
            plt.xlabel(f'Difference in {direction.upper()} {position.capitalize()} Position [cm]')
            plt.ylabel('Counts')
            plt.xlim(-30, 30)
            plt.title(f'Histogram of {position.capitalize()} {direction.upper()} Position Difference for {particle.capitalize()}s')
            plt.savefig(f"{plots_dir}/particle/{particle}/{position}_{direction}_diff_{particle}_histogram.png")
            plt.close()

def plot_ratio_cut_distributions(df, particle):
    """Plot distributions of variables split by E_kin_ratio cuts."""
    df_high = df[(df['E_kin_ratio_common'] >= 0.9) & (df['E_kin_ratio_common'] <= 1.2)]
    df_low = df[df['E_kin_ratio_common'] <= 0.9]

    for var in var_list:
        high_data = df_high[var].to_numpy()
        high_data = high_data[np.isfinite(high_data)]
        low_data = df_low[var].to_numpy()
        low_data = low_data[np.isfinite(low_data)]

        # High ratio plot
        plt.hist(high_data, bins=np.linspace(-0.4, 1, 51), color='green', alpha=0.7)
        plt.xlabel(f'{var} Distribution of {particle.capitalize()}s [GeV]')
        plt.ylabel('Counts')
        plt.title(f'{var} Distribution (1.2 ≥ E_kin_ratio_common ≥ 0.9) for {particle.capitalize()}s')
        plt.savefig(f"{plots_dir}/var/{var}/above/distribution_{var}_{particle}_EkinRatioAbove09_histogram.png")
        plt.savefig(f"{plots_dir}/particle/{particle}/distribution_{var}_{particle}_EkinRatioAbove09_histogram.png")
        plt.close()

        # Low ratio plot
        plt.hist(low_data, bins=np.linspace(-0.4, 1, 51), color='red', alpha=0.7)
        plt.xlabel(f'{var} Distribution of {particle.capitalize()}s')
        plt.ylabel('Counts')
        plt.title(f'{var} Distribution (E_kin_ratio_common < 0.9) for {particle.capitalize()}s [GeV]')
        plt.savefig(f"{plots_dir}/var/{var}/below/distribution_{var}_{particle}_EkinRatioBelow09_histogram.png")
        plt.savefig(f"{plots_dir}/particle/{particle}/distribution_{var}_{particle}_EkinRatioBelow09_histogram.png")
        plt.close()

        # Combined comparison
        plt.figure(figsize=(12, 6))
        plt.hist(high_data, bins=np.linspace(-0.4, 1, 51), color='green', alpha=0.7, 
                 label='1.2 ≥ E_kin_ratio_common ≥ 0.9')
        plt.hist(low_data, bins=np.linspace(-0.4, 1, 51), color='red', alpha=0.7, 
                 label='E_kin_ratio_common < 0.9')
        plt.xlabel(f'{var} Distribution of {particle.capitalize()}s [GeV]')
        plt.ylabel('Counts')
        plt.title(f'{var} Distribution for {particle.capitalize()}s')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/var/{var}/compare/distribution_{var}_{particle}_EkinRatio_comparison_histogram.png")
        plt.savefig(f"{plots_dir}/particle/{particle}/distribution_{var}_{particle}_EkinRatio_comparison_histogram.png")
        plt.close()

def fit_gaussian_regions(vals_full, xedges_1d, particle):
    """Fit Gaussian to 1σ region and peak region of data."""
    mu, std = norm.fit(vals_full)

    # Region 1: 1σ around mean
    region_mask3 = (vals_full > mu - 1 * std) & (vals_full < mu + 1 * std)
    region_vals3 = vals_full[region_mask3]
    counts3, bins3 = np.histogram(region_vals3, bins=xedges_1d)
    bin_centers3 = 0.5 * (bins3[:-1] + bins3[1:])
    p0_3 = [max(counts3), np.mean(region_vals3), np.std(region_vals3)]
    res3 = minuit_fit_gaussian(bin_centers3, counts3, p0=p0_3)
    A3, mu_region3, std_region3 = res3['params']

    if res3['success']:
        print(f"{particle} Gaussian fit (1σ region): χ²/dof={res3['chi2']:.1f}/{res3['dof']}={res3['redchi2']:.3f}")

    # Region 2: Peak region
    counts_full, bin_edges_full = np.histogram(vals_full, bins=xedges_1d)
    bin_centers_full = 0.5 * (bin_edges_full[:-1] + bin_edges_full[1:])
    max_bin_index = np.argmax(counts_full)
    peak_bin_center = bin_centers_full[max_bin_index]
    lower_bound4 = max(0, peak_bin_center - 1 * std)
    upper_bound4 = peak_bin_center + 1 * std
    region_mask4 = (vals_full >= lower_bound4) & (vals_full <= upper_bound4)
    region_vals4 = vals_full[region_mask4]
    counts4, bins4 = np.histogram(region_vals4, bins=xedges_1d)
    bin_centers4 = 0.5 * (bins4[:-1] + bins4[1:])
    p0_4 = [max(counts4), np.mean(region_vals4), np.std(region_vals4)]
    res4 = minuit_fit_gaussian(bin_centers4, counts4, p0=p0_4)
    A4, mu_region4, std_region4 = res4['params']

    if res4['success']:
        print(f"{particle} Gaussian fit (peak region): χ²/dof={res4['chi2']:.1f}/{res4['dof']}={res4['redchi2']:.3f}")

    return (A3, mu_region3, std_region3, res3), (A4, mu_region4, std_region4, res4)

def plot_gaussian_fits(vals_full, xedges_1d, particle, fit_results):
    """Plot Gaussian fits to data."""
    (A3, mu_region3, std_region3, res3), (A4, mu_region4, std_region4, res4) = fit_results
    
    # Plot region 1
    plt.figure(figsize=(12, 6))
    region_mask3 = (vals_full > mu_region3 - 1 * std_region3) & (vals_full < mu_region3 + 1 * std_region3)
    plt.hist(vals_full[region_mask3], bins=xedges_1d, alpha=0.6, color='g')
    x3 = np.linspace(-1, 1, 200)
    y3 = gaussian(x3, A3, mu_region3, std_region3)
    plt.plot(x3, y3, 'r-', linewidth=2, 
             label=(f'Gaussian fit μ={mu_region3:.3f}, σ={std_region3:.3f}\n'
                    f'χ²/dof={res3["chi2"]:.1f}/{res3["dof"]}={res3["redchi2"]:.3f}'))
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.title(f'Gaussian fit (1σ region) for {particle.capitalize()}s')
    plt.xlabel('E_kin_ratio_common')
    plt.ylabel('Counts')
    plt.savefig(f"{plots_dir}/particle/{particle}/Ekin_ratio_common_gaussfit1_{particle}.png")
    plt.close()

    # Plot region 2
    plt.figure(figsize=(12, 6))
    region_mask4 = (vals_full >= mu_region4 - 1 * std_region4) & (vals_full <= mu_region4 + 1 * std_region4)
    plt.hist(vals_full[region_mask4], bins=xedges_1d, alpha=0.6, color='g')
    x4 = np.linspace(-1, 1, 200)
    y4 = gaussian(x4, A4, mu_region4, std_region4)
    plt.plot(x4, y4, 'r-', linewidth=2, 
             label=(f'Gaussian fit μ={mu_region4:.3f}, σ={std_region4:.3f}\n'
                    f'χ²/dof={res4["chi2"]:.1f}/{res4["dof"]}={res4["redchi2"]:.3f}'))
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.title(f'Gaussian fit (peak region) for {particle.capitalize()}s')
    plt.xlabel('E_kin_ratio_common')
    plt.ylabel('Counts')
    plt.savefig(f"{plots_dir}/particle/{particle}/Ekin_ratio_common_gaussfit2_{particle}.png")
    plt.close()

def compute_binwise_scales(df, particle, ekin_edges, xedges_1d, out_dir, bins=50, min_entries=10):
    """Compute per-energy-bin scale factors using multiple fitting methods.
    
    Returns:
        scale_factors : array of scale factors per bin
        fit_means : array of best-fit means per bin
        fit_sigmas : array of best-fit sigmas per bin
        fit_types_used : list of which fit was best for each bin
    """
    # Convert E_kin_ratio_common to resolution: (E_kin - E_true) / E_true = E_kin_ratio_common - 1
    df_copy = df.copy()
    df_copy['energy_resolution'] = df_copy['E_kin_ratio_common'] - 1.0
    
    ekin_bin_indices = np.digitize(df_copy['E_kin'].values, ekin_edges) - 1
    n_bins = len(ekin_edges) - 1
    scale_factors = np.ones(n_bins)
    fit_means = np.full(n_bins, np.nan)
    fit_sigmas = np.full(n_bins, np.nan)
    fit_types_used = ['none'] * n_bins

    for j in range(n_bins):
        mask_j = ekin_bin_indices == j
        if not np.any(mask_j):
            continue

        vals_j_raw = df_copy.loc[mask_j, 'energy_resolution'].dropna().values
        vals_j = vals_j_raw[(vals_j_raw >= -1.0) & (vals_j_raw <= 1.0)]
        if vals_j.size < min_entries:
            print(f"Bin {j} has only {vals_j.size} entries in resolution [-1,1], skipping")
            continue

        hist, be = np.histogram(vals_j, bins=bins, range=(-1.0, 1.0))
        centers = 0.5 * (be[:-1] + be[1:])
        yerr = np.sqrt(np.clip(hist, 1.0, None))
        mask_fit = hist > 0

        # Gaussian fit
        gauss_ok = False
        mu_gauss = np.mean(vals_j)
        sigma_gauss = np.std(vals_j)
        chi2_g = dof_g = redchi2_g = np.nan
        try:
            res_g = minuit_fit_gaussian(centers[mask_fit], hist[mask_fit], 
                                       p0=[hist.max(), mu_gauss, sigma_gauss], sigma=yerr[mask_fit])
            popt_g = res_g['params'] if res_g['success'] else None
            chi2_g = res_g['chi2']
            dof_g = res_g['dof']
            redchi2_g = res_g['redchi2']
            gauss_ok = res_g['success']
        except Exception:
            popt_g = None

        # Cauchy fit
        cauchy_ok = False
        chi2_c = dof_c = redchi2_c = np.nan
        try:
            res_c = minuit_fit_cauchy(centers[mask_fit], hist[mask_fit],
                                     p0=[np.sum(hist), mu_gauss, max(1e-3, sigma_gauss), 0.0],
                                     sigma=yerr[mask_fit])
            popt_c = res_c['params'] if res_c['success'] else None
            chi2_c = res_c['chi2']
            dof_c = res_c['dof']
            redchi2_c = res_c['redchi2']
            cauchy_ok = res_c['success']
        except Exception:
            popt_c = None

        # CrystalBall fit
        cb_ok = False
        chi2_cb = dof_cb = redchi2_cb = np.nan
        try:
            res_cb = minuit_fit_crystalball(centers[mask_fit], hist[mask_fit],
                                           p0=[np.sum(hist), 2.0, 2.0, mu_gauss, sigma_gauss],
                                           sigma=yerr[mask_fit])
            popt_cb = res_cb['params'] if res_cb['success'] else None
            chi2_cb = res_cb['chi2']
            dof_cb = res_cb['dof']
            redchi2_cb = res_cb['redchi2']
            cb_ok = res_cb['success']
        except Exception:
            popt_cb = None

        # Choose best fit based on lowest reduced chi-squared
        best_chi2 = np.inf
        best_fit_type = 'none'
        best_mean = np.nan
        best_sigma = np.nan
        
        if gauss_ok and popt_g is not None and np.isfinite(redchi2_g):
            if redchi2_g < best_chi2:
                best_chi2 = redchi2_g
                best_fit_type = 'gaussian'
                best_mean = popt_g[1]
                best_sigma = popt_g[2]
        
        if cauchy_ok and popt_c is not None and np.isfinite(redchi2_c):
            if redchi2_c < best_chi2:
                best_chi2 = redchi2_c
                best_fit_type = 'cauchy'
                best_mean = popt_c[1]
                best_sigma = popt_c[2]
        
        if cb_ok and popt_cb is not None and np.isfinite(redchi2_cb):
            if redchi2_cb < best_chi2:
                best_chi2 = redchi2_cb
                best_fit_type = 'crystalball'
                best_mean = popt_cb[3]
                best_sigma = popt_cb[4]
        
        # Store fit results
        fit_means[j] = best_mean
        fit_sigmas[j] = best_sigma
        fit_types_used[j] = best_fit_type
        
        # Choose scale factor (use best fit mean, or fallback to sample mean)
        chosen_scale = np.mean(vals_j)
        if np.isfinite(best_mean):
            chosen_scale = best_mean
        
        scale_factors[j] = chosen_scale
        print(f'Bin {j} [{ekin_edges[j]:.3f}, {ekin_edges[j+1]:.3f}] GeV: {vals_j.size} entries -> scale={chosen_scale:.4f} (best fit: {best_fit_type})')

        # Plot histogram with fits
        plt.figure(figsize=(12, 6))
        plt.hist(vals_j, bins=np.linspace(-1, 1, 50), alpha=0.4, label=f'Data (n={len(vals_j)})')
        xplot = np.linspace(-1.0, 1.0, 1000)
        
        if gauss_ok and popt_g is not None:
            y_g = gaussian(xplot, *popt_g)
            plt.plot(xplot, y_g, 'r-', lw=2, label=f'Gaussian μ={popt_g[1]:.4f}, σ={popt_g[2]:.4f}\nχ²/dof={chi2_g:.1f}/{dof_g}={redchi2_g:.3f}')
        
        if cauchy_ok and popt_c is not None:
            y_c = cauchy(xplot, *popt_c)
            plt.plot(xplot, y_c, 'm--', lw=2, label=f'Cauchy x0={popt_c[1]:.4f}, γ={popt_c[2]:.4f}\nχ²/dof={chi2_c:.1f}/{dof_c}={redchi2_c:.3f}')
        
        if cb_ok and popt_cb is not None:
            y_cb = crystalball_pdf(xplot, *popt_cb)
            plt.plot(xplot, y_cb, 'b-.', lw=2, label=f'CB μ={popt_cb[3]:.4f}, σ={popt_cb[4]:.4f}\nχ²/dof={chi2_cb:.1f}/{dof_cb}={redchi2_cb:.3f}')

        plt.xlabel('Energy Resolution $(E_{rec} - E_{kin}) / E_{kin}$')
        
        plt.xlim(-1, 1)
        plt.ylim(0, np.max(hist) * 1.2)
        plt.ylabel('Counts')
        plt.title(f'{particle.capitalize()} - Bin {j}: {ekin_edges[j]:.3f} to {ekin_edges[j+1]:.3f} GeV')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig(f"{out_dir}/Ekin_ratio_binwise_bin{j}.png")
        plt.close()

    return scale_factors, fit_means, fit_sigmas, fit_types_used

def Ekin_vs_Erec(df, particle):
    """Main energy kinetic vs reconstructed analysis."""
    # Filter by track length
    df['reco_track_length'] = np.sqrt(
        (df['common_dlp_start_x'] - df['common_dlp_end_x'])**2 +
        (df['common_dlp_start_y'] - df['common_dlp_end_y'])**2 +
        (df['common_dlp_start_z'] - df['common_dlp_end_z'])**2
    )
    
    n_total = len(df)
    n_short = len(df[df['reco_track_length'] < 0.1])
    df = df[df['reco_track_length'] >= 0.1]
    
    print(f"\n{particle.upper()} - Track Length Filtering: {n_short}/{n_total} removed\n")

    if particle == 'muon':
        df = df[df['part_type'] == 1]

    df = df[np.isfinite(df['E_kin_ratio_common'])]
    print(f"Number of {particle} entries: {len(df)}")

    # Setup binning
    n_entries = len(df)
    nbins_1d = max(20, int(np.sqrt(n_entries)))
    xedges_1d = np.linspace(-1, 1, nbins_1d + 1)
    xedges_1d_ratio = np.linspace(0, 2, nbins_1d + 1)
    yedges = np.linspace(-1, 2, 30)
    xedges_2d = np.array([0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1, 1.5])

    vals = df['E_kin_ratio_common'].values
    vals_full = vals[(vals >= 0.0) & (vals <= 2.0)]
    if vals_full.size == 0:
        print(f"No entries in [0,2]; skipping")
        return

    # Fit Gaussian to different regions
    fit_results = fit_gaussian_regions(vals_full, xedges_1d_ratio, particle)
    (A3, mu_region3, std_region3, res3), (A4, mu_region4, std_region4, res4) = fit_results
    plot_gaussian_fits(vals_full, xedges_1d_ratio, particle, fit_results)

    # Plot clipped distribution
    plt.hist(vals_full, bins=xedges_1d_ratio, alpha=0.7)
    plt.xlabel('E_kin_ratio_common')
    plt.ylabel('Counts')
    plt.title(f'E_kin_ratio_common distribution for {particle.capitalize()}s')
    plt.savefig(f"{plots_dir}/particle/{particle}/Ekin_ratio_common_{particle}_hist.png")
    plt.close()

    # CrystalBall fit on full distribution
    counts_full, bin_edges_full = np.histogram(vals_full, bins=xedges_1d)
    bin_centers_full = 0.5 * (bin_edges_full[:-1] + bin_edges_full[1:])
    mask_full = counts_full > 0
    x_fit_cb = bin_centers_full[mask_full]
    y_fit_cb = counts_full[mask_full]

    res_cb = minuit_fit_crystalball(x_fit_cb, y_fit_cb, 
                                    p0=[y_fit_cb.sum(), 2.0, 2.0, mu_region3, std_region3])
    cb_ok = res_cb['success']
    params_cb = res_cb['params'] if cb_ok else None
    
    if not cb_ok:
        print(f"CB fit failed: {res_cb.get('message', 'Unknown error')}")
    
    if cb_ok:
        print(f"{particle} CB fit: A={params_cb[0]:.1f}, β={params_cb[1]:.2f}, m={params_cb[2]:.2f}, "
              f"μ={params_cb[3]:.3f}, σ={params_cb[4]:.3f}, χ²/dof={res_cb['chi2']:.1f}/{res_cb['dof']}={res_cb['redchi2']:.3f}")

    # Plot CrystalBall fit
    plt.figure(figsize=(12, 6))
    plt.hist(vals_full, bins=xedges_1d, alpha=0.6, color='g', label='Data')
    x_cb = np.linspace(bin_centers_full.min(), bin_centers_full.max(), 400)
    if cb_ok:
        y_cb = crystalball_pdf(x_cb, *params_cb)
        chi2_text = f"χ²/dof={res_cb['chi2']:.1f}/{res_cb['dof']}={res_cb['redchi2']:.3f}"
        plt.plot(x_cb, y_cb, 'b-', linewidth=2,
                label=f'CB: μ={params_cb[3]:.3f}, σ={params_cb[4]:.3f}\n{chi2_text}')
    plt.xlabel('E_kin_ratio_common')
    plt.ylabel('Counts')
    plt.title(f'E_kin_ratio_common with CB fit ({particle.capitalize()})')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/particle/{particle}/Ekin_ratio_common_crystalball_{particle}.png")
    plt.close()

    # Cauchy fit
    res_c = minuit_fit_cauchy(x_fit_cb, y_fit_cb,
                             p0=[y_fit_cb.sum(), mu_region3, std_region3, 0.0])
    cauchy_ok = res_c['success']
    params_cauchy = res_c['params'] if cauchy_ok else None

    if cauchy_ok:
        print(f"{particle} Cauchy fit: A={params_cauchy[0]:.1f}, x0={params_cauchy[1]:.3f}, "
              f"γ={params_cauchy[2]:.3f}, χ²/dof={res_c['chi2']:.1f}/{res_c['dof']}={res_c['redchi2']:.3f}")

    # Plot Cauchy fit
    plt.figure(figsize=(12, 6))
    plt.hist(vals_full, bins=xedges_1d, alpha=0.6, color='g', label='Data')
    x_c = np.linspace(bin_centers_full.min(), bin_centers_full.max(), 400)
    if cauchy_ok:
        y_c = cauchy(x_c, *params_cauchy)
        chi2_text = f"χ²/dof={res_c['chi2']:.1f}/{res_c['dof']}={res_c['redchi2']:.3f}"
        plt.plot(x_c, y_c, 'm-', linewidth=2,
                label=f'Cauchy: x0={params_cauchy[1]:.3f}, γ={params_cauchy[2]:.3f}\n{chi2_text}')
    plt.xlabel('E_kin_ratio_common')
    plt.ylabel('Counts')
    plt.title(f'E_kin_ratio_common with Cauchy fit ({particle.capitalize()})')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/particle/{particle}/Ekin_ratio_common_cauchy_{particle}.png")
    plt.close()

    # Compute binwise scales
    ekin_edges = np.array([0, 0.1, 0.2, 0.3, 0.6, 1.5])
    out_dir = f"{plots_dir}/particle/{particle}"
    scale_factors, fit_means, fit_sigmas, fit_types_used = compute_binwise_scales(df, particle, ekin_edges, xedges_1d, out_dir)

    # Create mean and std resolution plots from binwise fitted distributions
    bin_centers = 0.5 * (ekin_edges[:-1] + ekin_edges[1:])
    valid_mask = np.isfinite(fit_means) & np.isfinite(fit_sigmas) & (bin_centers > 0)
    
    # Fit resolution model to mean values
    if np.any(valid_mask):
        x_fit_mean = bin_centers[valid_mask]
        y_fit_mean = fit_means[valid_mask]
        
        # Fit resolution model: sqrt(a^2 + (b/sqrt(E))^2 + (c/E)^2)
        try:
            popt_mean, pcov_mean = curve_fit(
                resolution_model, x_fit_mean, y_fit_mean,
                p0=[0.01, 0.01, 0.01], maxfev=10000
            )
            mean_fit_valid = True
        except Exception:
            mean_fit_valid = False
            popt_mean = np.array([0.01, 0.01, 0.01])
    else:
        mean_fit_valid = False
        popt_mean = np.array([0.01, 0.01, 0.01])
    
    # Fit resolution model to sigma values
    if np.any(valid_mask):
        x_fit_sigma = bin_centers[valid_mask]
        y_fit_sigma = fit_sigmas[valid_mask]
        
        try:
            popt_sigma, pcov_sigma = curve_fit(
                resolution_model, x_fit_sigma, y_fit_sigma,
                p0=[0.01, 0.01, 0.01], maxfev=10000
            )
            sigma_fit_valid = True
        except Exception:
            sigma_fit_valid = False
            popt_sigma = np.array([0.01, 0.01, 0.01])
    else:
        sigma_fit_valid = False
        popt_sigma = np.array([0.01, 0.01, 0.01])
    
    # Plot mean resolution from best fits
    mask = (
        np.isfinite(bin_centers) & np.isfinite(fit_means) & 
        np.isfinite(fit_sigmas) & (bin_centers > 0)
    )
    
    plt.figure(figsize=(12, 6))
    if np.any(mask):
        plt.scatter(bin_centers[mask], fit_means[mask], s=8, label='Bin mean')
        plt.fill_between(bin_centers[mask], fit_means[mask] - fit_sigmas[mask], 
                         fit_means[mask] + fit_sigmas[mask], alpha=0.3, color='red', label='σ band')
    
    plt.xlabel('True Kinetic Energy [GeV]')
    plt.ylabel('Relative Mean Energy Resolution\n$(E_{rec} - E_{kin}) / E_{kin}$')
    plt.title(f'Relative Mean Energy Resolution vs $E_{{kin}}$\n({particle}, binwise fitted)')
    plt.xlim(0, ekin_edges[-1])
    plt.grid()
    plt.ylim(-2, 2)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/particle/{particle}/binwise_mean_resolution_{particle}.png")
    plt.close()
    
    # Plot sigma resolution from best fits with fit
    mask_sigma = (
        np.isfinite(bin_centers) & np.isfinite(fit_sigmas) & 
        (bin_centers > 0)
    )
    
    x_fit = bin_centers[mask_sigma]
    y_fit = fit_sigmas[mask_sigma]
    counts_fit = np.ones_like(x_fit) * 10  # Approximate counts per bin
    
    sigma_y = y_fit / np.sqrt(2.0 * counts_fit)
    finite_sig = sigma_y[np.isfinite(sigma_y) & (sigma_y > 0)]
    if finite_sig.size == 0:
        sigma_y = np.ones_like(sigma_y) * np.nanmedian(np.abs(y_fit))
    else:
        sigma_y = np.where(np.isfinite(sigma_y) & (sigma_y > 0), sigma_y, np.median(finite_sig))
    
    # Compute chi-squared for sigma fit
    chi2_text = ""
    if sigma_fit_valid and np.all(np.isfinite(popt_sigma)) and x_fit.size > 0:
        try:
            model_vals = resolution_model(x_fit, *popt_sigma)
            valid_chi2 = np.isfinite(sigma_y) & (sigma_y > 0)
            if np.any(valid_chi2):
                resid = (y_fit[valid_chi2] - model_vals[valid_chi2]) / sigma_y[valid_chi2]
                chi2_val = np.sum(resid**2)
                dof = max(1, np.count_nonzero(valid_chi2) - len(popt_sigma))
                red_chi2 = chi2_val / dof
                chi2_text = f"\nχ²/dof={chi2_val:.1f}/{dof}={red_chi2:.3f}"
        except Exception:
            chi2_text = ""
    
    if x_fit.size > 0 and np.all(np.isfinite(popt_sigma)):
        x_smooth = np.linspace(np.min(x_fit), np.max(x_fit), 500)
    else:
        x_smooth = np.linspace(0.01, ekin_edges[-1] if np.isfinite(ekin_edges[-1]) else 1.0, 200)
    y_smooth = resolution_model(x_smooth, *popt_sigma)
    
    plt.figure(figsize=(12, 6))
    if x_fit.size > 0:
        plt.errorbar(x_fit, y_fit, yerr=sigma_y, fmt='o', markersize=4, capsize=3, capthick=1, 
                     label='Std dev per bin', alpha=0.8)
        if sigma_fit_valid and np.all(np.isfinite(popt_sigma)):
            plt.plot(x_smooth, y_smooth, color='k', 
                label=(f'Fit: p0={popt_sigma[0]*100:.2f}%, p1={popt_sigma[1]*100:.2f}%, p2={popt_sigma[2]*100:.2f}%{chi2_text}'))
    
    plt.xlabel('True Kinetic Energy [GeV]')
    plt.ylabel('Std dev of Energy Resolution\n$(E_{rec} - E_{kin}) / E_{kin}$')
    plt.title(f'Std Dev of Resolution vs $E_{{kin}}$\n({particle}, binwise fitted)')
    plt.xlim(0, ekin_edges[-1])
    plt.ylim(-1, 5)
    plt.grid()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/particle/{particle}/binwise_sigma_resolution_{particle}.png")
    plt.close()

    # Map scales to events
    ekin_bin_indices = np.digitize(df['E_kin'].values, ekin_edges) - 1
    bin_scales = np.ones_like(df['E_kin'].values, dtype=float)
    valid_mask = (ekin_bin_indices >= 0) & (ekin_bin_indices < len(scale_factors))
    bin_scales[valid_mask] = scale_factors[ekin_bin_indices[valid_mask]]

    # Linear fit on E_rec vs E_kin
    x_fit = df['common_dlp_E'].values
    y_fit = df['E_kin'].values

    if len(x_fit) > 2:
        try:
            res = minuit_linear_fit(x_fit, y_fit)
            coeffs = res['coeffs']
            r_squared = res['r2']
            chi2_lin = res['chi2']
            dof_lin = res['dof']
            redchi2_lin = res['redchi2']
            fit_valid = True
        except Exception as e:
            print(f'Minuit fit failed: {e}. Fallback to polyfit.')
            coeffs = np.polyfit(x_fit, y_fit, 1)
            fit_valid = True

        # Plot scatter and fit
        plt.scatter(x_fit, y_fit, s=0.5, alpha=0.5)
        if fit_valid:
            x_curve = np.linspace(x_fit.min(), x_fit.max(), 200)
            y_curve = coeffs[0] * x_curve + coeffs[1]
            chi2_text = f"χ²/dof={res['chi2']:.1f}/{res['dof']}={res['redchi2']:.3f}, R²={res['r2']:.4f}"
            plt.plot(x_curve, y_curve, 'r-', linewidth=2,
                    label=f'y={coeffs[0]:.3f}x+{coeffs[1]:.3f}\n{chi2_text}')
        plt.xlabel('Reconstructed Energy [GeV]')
        plt.ylabel('True Kinetic Energy [GeV]')
        plt.title(f'E_kin vs E_rec for {particle.capitalize()}s')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlim(0, 0.4)
        plt.ylim(0, 1)
        plt.savefig(f"{plots_dir}/particle/{particle}/Ekin_vs_Erec_{particle}_scatter.png")
        plt.close()
        print(f'{particle} fit: y={coeffs[0]:.3f}x+{coeffs[1]:.3f}, χ²/dof={res["chi2"]:.1f}/{res["dof"]}={res["redchi2"]:.3f}')

    # Create resolution columns
    df['E_rec'] = 2 * df['common_dlp_E']
    df['resolution'] = (df['E_rec'] - df['E_kin']) / df['E_kin']

    df['E_rec2'] = coeffs[0] * df['common_dlp_E'] + coeffs[1]
    df['resolution2'] = (df['E_rec2'] - df['E_kin']) / df['E_kin']

    df['E_rec3'] = df['common_dlp_E'] / mu_region3
    df['resolution3'] = (df['E_rec3'] - df['E_kin']) / df['E_kin']

    df['E_rec4'] = df['common_dlp_E'] / mu_region4
    df['resolution4'] = (df['E_rec4'] - df['E_kin']) / df['E_kin']

    df['E_rec5'] = df['common_dlp_E'] / bin_scales
    df['resolution5'] = (df['E_rec5'] - df['E_kin']) / df['E_kin']

    if cb_ok and params_cb is not None:
        mu_cb = params_cb[3]
    else:
        mu_cb = mu_region3
    df['E_rec_cb'] = df['common_dlp_E'] / mu_cb
    df['resolution_cb'] = (df['E_rec_cb'] - df['E_kin']) / df['E_kin']

    if cauchy_ok and params_cauchy is not None:
        x0_c = params_cauchy[1]
    else:
        x0_c = mu_region3
    df['E_rec_cauchy'] = df['common_dlp_E'] / x0_c
    df['resolution_cauchy'] = (df['E_rec_cauchy'] - df['E_kin']) / df['E_kin']

    # Save 2D resolution histograms
    for res_col, label in [
        ('resolution', 'No Correction'),
        ('resolution2', 'Linear Fit'),
        ('resolution3', 'Gaussian 1σ'),
        ('resolution4', 'Gaussian Peak'),
        ('resolution5', 'Bin-wise'),
        ('resolution_cb', 'CrystalBall'),
        ('resolution_cauchy', 'Cauchy')
    ]:
        save_hist2d(df['E_kin'], df[res_col], xedges_2d, yedges, 'True E [GeV]',
                    'Resolution', f'Energy Resolution ({label})', 
                    f"{plots_dir}/particle/{particle}/energy_resolution_{res_col}_{particle}_hist2d.png")

    # Save scatter plots
    mask = df['E_kin'] > 0
    if mask.any():
        df_sub = df[mask]
        for res_col, label in [
            ('resolution', 'No Correction'),
            ('resolution2', 'Linear Fit'),
            ('resolution3', 'Gaussian 1σ'),
            ('resolution4', 'Gaussian Peak'),
            ('resolution5', 'Bin-wise'),
            ('resolution_cb', 'CrystalBall'),
            ('resolution_cauchy', 'Cauchy')
        ]:
            save_scatter(df_sub['E_kin'], df_sub[res_col], 'True E [GeV]',
                        f'Resolution', f'Energy Resolution ({label})',
                        f"{plots_dir}/particle/{particle}/energy_resolution_{res_col}_{particle}_scatter.png")

    # Mean/std resolution plots
    guesses_p0 = {'muon': (0.02, 0.02, 0.02), 'pion': (0.05, 0.05, 0.05), 'proton': (0.1, 0.1, 0.1)}
    p0 = guesses_p0.get(particle, (0.05, 0.05, 0.05))

    for res_col, label in [
        ('resolution', 'No Correction'),
        ('resolution2', 'Linear Fit'),
        ('resolution3', 'Gaussian 1σ'),
        ('resolution4', 'Gaussian Peak'),
        ('resolution5', 'Bin-wise'),
        ('resolution_cb', 'CrystalBall'),
        ('resolution_cauchy', 'Cauchy')
    ]:
        mean_std_resolution_plots(
            df, xedges_2d, res_col,
            f"{plots_dir}/particle/{particle}/mean_resolution_{res_col}_{particle}.png",
            f"{plots_dir}/particle/{particle}/std_resolution_{res_col}_{particle}.png",
            particle, label, p0=p0
        )

    # Comparison histograms
    save_reco_comparison(df, particle, f"{plots_dir}/particle/{particle}/Erec_comparison_{particle}.png",
                        include_linear=True,
                        extra_pairs=[('E_rec_cb', 'CrystalBall', 'purple'),
                                     ('E_rec_cauchy', 'Cauchy', 'magenta')])

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    for particle in particles:
        df = load_particle_data(particle)
        Ekin_vs_Erec(df, particle)
