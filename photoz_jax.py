#!/usr/bin/env python3
"""
photoz_jax.py  --  JAX/GPU photometric-redshift template fitting

Replicates the SPHEREx C++ photo-z pipeline (fitting_tools.cpp / photo_z.cpp)
using batched matrix operations on GPU via JAX.

Algorithm
---------
For each (object o, model m) pair the C++ computes the optimal-scale chi-squared:

    chi2_om = C_o  -  B_om^2 / D_om

where
    C_o   = sum_f  F_o^f^2 * w_o^f              (per-object scalar, precomputed once)
    B_om  = sum_f  F_o^f * f_m^f * w_o^f        (obs x model inner product)
    D_om  = sum_f  (f_m^f)^2 * w_o^f            (weighted model self-product)

B and D are both batched matrix multiplies:
    B = (F_obs * W_obs)  @  F_models.T      [N_obj, N_m]
    D =  W_obs           @  (F_models^2).T  [N_obj, N_m]

Probability density at redshift z:
    P(z) = (1/N_m) * sum_m  P_tmpl(m) * exp(-chi2_om / 2)

where N_m = number of models per z-bin (10080 for the fineEBV grid).
The C++ divides prob_density by kn_models before storing (fitting_tools.cpp
line ~430).  Zero-probability models contribute 0 to the sum (matching C++
`continue` logic), but still count toward N_m.

Usage
-----
  # Run the fitter:
  python photoz_jax.py fit  <model_grid_glob>  <obs_phot>  <template_prob>  \\
                            <output_redshifts>  [--pdfs <pdfs_file>]

  # Verify against an existing C++ run:
  python photoz_jax.py verify  <model_grid_glob>  <obs_phot>  <template_prob>  \\
                               --cpp-pdfs <cpp_pdfs>  [--cpp-redshifts <cpp_z>]
"""

import argparse
import glob as glob_module
import math
import os
import struct
import sys
import time
from typing import Dict, List, Optional, Tuple

import functools
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ---------------------------------------------------------------------------
# JAX import -- disable memory pre-allocation so we control VRAM manually
# ---------------------------------------------------------------------------
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
from jax import jit

# Force full IEEE float32 precision for matrix multiplies.
# A100 GPUs default to TF32 (10-bit mantissa) which causes ~0.1% errors in
# the B and D matmuls — the core of the chi2 calculation — and thereby
# diverges from the C++ output.  "highest" selects float32 ≥ float32 GEMM.
jax.config.update("jax_default_matmul_precision", "highest")

# Enable float64 (x64) support for chi² and PDF calculations.
# C++ uses double precision, so JAX must match to reproduce results exactly.
_PHOTOZ_PRECISION = os.environ.get("PHOTOZ_PRECISION", "float64").lower()
if _PHOTOZ_PRECISION not in ("float64", "float32", "bfloat16"):
    raise ValueError(f"Invalid PHOTOZ_PRECISION '{_PHOTOZ_PRECISION}'; valid: float64,float32,bfloat16")
jax.config.update("jax_enable_x64", _PHOTOZ_PRECISION == "float64")

# ---------------------------------------------------------------------------
# Binary format magic numbers (from grid_tools.cpp)
# ---------------------------------------------------------------------------
_NFLT_MAGIC = 1953261166   # "nflt" ASCII, little-endian uint64
_ZBIN_MAGIC  = 1852400250   # "zbin" ASCII, little-endian uint64


# ===========================================================================
# Section 1: Binary grid parser
# ===========================================================================

def parse_photoz_grid(paths) -> dict:
    """
    Parse one or more .photoz binary model-grid files.

    Binary format (SavePhotoZBinary in grid_tools.cpp):
        uint64   NFLT magic
        uint64   n_filters
        -- repeated for each redshift bin --
        uint64   ZBIN magic
        float32  redshift
        -- repeated for each model at this z --
        int32    template_id  (kn_metadata_ = 1 word stored as float, read as int)
        float32  × n_filters  (L1-normalised fluxes)

    Multiple files are stitched in ascending-z order.

    Parameters
    ----------
    paths : str or list of str
        Glob pattern, single path, or list of paths.

    Returns
    -------
    dict with keys:
        redshifts    : float32 [N_z]
        model_fluxes : float32 [N_z, N_m, N_f]  (L1-normalised, sum over f ≈ 1)
        template_ids : int32   [N_z, N_m]
        n_filters    : int
    """
    if isinstance(paths, str):
        expanded = sorted(glob_module.glob(paths))
        if not expanded:
            # Single explicit path
            expanded = [paths]
    else:
        expanded = list(paths)

    if not expanded:
        raise FileNotFoundError(f"No .photoz grid files found: {paths}")

    all_redshifts: List[float]     = []
    all_fluxes:    List[np.ndarray] = []   # each element: [N_m, N_f]
    all_tids:      List[np.ndarray] = []   # each element: [N_m]
    n_filters_ref: Optional[int]   = None

    for path in expanded:
        size_gb = os.path.getsize(path) / 1e9
        print(f"  [parse] reading {os.path.basename(path)}  ({size_gb:.2f} GB)", flush=True)
        z_bins, fluxes_by_z, tids_by_z, n_filters = _parse_single_photoz(path)

        if n_filters_ref is None:
            n_filters_ref = n_filters
        elif n_filters != n_filters_ref:
            raise ValueError(
                f"{path}: n_filters={n_filters} differs from earlier files "
                f"({n_filters_ref})"
            )

        all_redshifts.extend(z_bins)
        all_fluxes.extend(fluxes_by_z)
        all_tids.extend(tids_by_z)

    # Sort by redshift (should already be ascending across files, but be safe)
    order = np.argsort(all_redshifts)
    all_redshifts_sorted = np.array(all_redshifts, dtype=np.float32)[order]
    all_fluxes_sorted    = [all_fluxes[i] for i in order]
    all_tids_sorted      = [all_tids[i]   for i in order]

    # Deduplicate overlapping z-bins at set boundaries
    # (e.g., set1 ends at z=1.0, set2 starts at z=1.0 → keep only one)
    unique_mask = np.ones(len(all_redshifts_sorted), dtype=bool)
    for i in range(1, len(all_redshifts_sorted)):
        if np.abs(all_redshifts_sorted[i] - all_redshifts_sorted[i-1]) < 1e-6:
            unique_mask[i] = False  # Mark duplicate for removal
    
    n_duplicates = (~unique_mask).sum()
    if n_duplicates > 0:
        print(f"  [parse] removing {n_duplicates} duplicate z-bins at set boundaries", flush=True)
    
    all_redshifts = all_redshifts_sorted[unique_mask]
    all_fluxes    = [all_fluxes_sorted[i] for i in range(len(all_fluxes_sorted)) if unique_mask[i]]
    all_tids      = [all_tids_sorted[i]   for i in range(len(all_tids_sorted))   if unique_mask[i]]

    # Validate uniform model count per z-bin (required by jax.lax.scan)
    n_models_per_z = {arr.shape[0] for arr in all_fluxes}
    if len(n_models_per_z) != 1:
        raise ValueError(
            f"Unequal model counts across z-bins: {n_models_per_z}. "
            "jax.lax.scan requires uniform shape."
        )

    model_fluxes = np.stack(all_fluxes, axis=0).astype(np.float32)  # [N_z, N_m, N_f]
    template_ids = np.stack(all_tids,   axis=0).astype(np.int32)    # [N_z, N_m]

    N_z, N_m, N_f = model_fluxes.shape
    print(
        f"  [parse] {N_z} z-bins × {N_m} models/z × {N_f} filters.  "
        f"z in [{all_redshifts[0]:.4f}, {all_redshifts[-1]:.4f}]",
        flush=True,
    )

    return {
        "redshifts"   : all_redshifts,
        "model_fluxes": model_fluxes,
        "template_ids": template_ids,
        "n_filters"   : N_f,
    }


def _parse_single_photoz(
    path: str,
) -> Tuple[List[float], List[np.ndarray], List[np.ndarray], int]:
    """
    Parse one .photoz binary file.

    Returns
    -------
    z_bins      : list of float  (one per z-bin in this file)
    fluxes_by_z : list of float32 arrays [N_m, N_f]
    tids_by_z   : list of int32  arrays [N_m]
    n_filters   : int
    """
    with open(path, "rb") as fh:
        raw = fh.read()

    offset = 0

    def read_u64() -> int:
        nonlocal offset
        v = struct.unpack_from("<Q", raw, offset)[0]
        offset += 8
        return v

    def read_f32() -> float:
        nonlocal offset
        v = struct.unpack_from("<f", raw, offset)[0]
        offset += 4
        return v

    def read_i32() -> int:
        nonlocal offset
        v = struct.unpack_from("<i", raw, offset)[0]
        offset += 4
        return v

    magic = read_u64()
    if magic != _NFLT_MAGIC:
        raise ValueError(
            f"{path}: bad magic {magic:#x} (expected {_NFLT_MAGIC:#x}). "
            "Not a .photoz file?"
        )

    n_filters = int(read_u64())
    file_size = len(raw)

    z_bins:      List[float]     = []
    fluxes_by_z: List[np.ndarray] = []
    tids_by_z:   List[np.ndarray] = []

    cur_fluxes: List[Tuple] = []
    cur_tids:   List[int]   = []
    cur_z: Optional[float]  = None

    while offset < file_size - 5:
        if offset + 8 > file_size:
            break

        # Peek at next 8 bytes to check for ZBIN marker
        next_u64 = struct.unpack_from("<Q", raw, offset)[0]

        if next_u64 == _ZBIN_MAGIC:
            # Flush previous z-bin
            if cur_z is not None and cur_fluxes:
                z_bins.append(cur_z)
                fluxes_by_z.append(
                    np.array(cur_fluxes, dtype=np.float32)   # [N_m, N_f]
                )
                tids_by_z.append(
                    np.array(cur_tids, dtype=np.int32)        # [N_m]
                )
                cur_fluxes = []
                cur_tids   = []

            offset += 8          # consume the ZBIN magic
            cur_z = read_f32()   # read the redshift value

        else:
            # Model record: int32 template_id + n_filters × float32
            tid    = read_i32()
            fluxes = struct.unpack_from(f"<{n_filters}f", raw, offset)
            offset += 4 * n_filters
            cur_fluxes.append(fluxes)
            cur_tids.append(tid)

    # Flush the final z-bin
    if cur_z is not None and cur_fluxes:
        z_bins.append(cur_z)
        fluxes_by_z.append(np.array(cur_fluxes, dtype=np.float32))
        tids_by_z.append(np.array(cur_tids, dtype=np.int32))

    return z_bins, fluxes_by_z, tids_by_z, n_filters


# ===========================================================================
# Section 2: I/O helpers
# ===========================================================================

def read_obs_phot(path: str):
    """
    Read observed photometry in C++ text format:
        id  xpos  ypos  f0  e0  f1  e1  ...

    Weights are 1/sigma^2.  Also computes per-object summary statistics
    needed for the SaveRedshifts output columns.

    Returns
    -------
    ids         : int64  [N_obj]
    xpos        : float32 [N_obj]
    ypos        : float32 [N_obj]
    obs_fluxes  : float32 [N_obj, N_f]
    obs_weights : float32 [N_obj, N_f]   (1/sigma^2)
    obj_stats   : dict of float32 [N_obj]
    """
    print(f"  [io] reading observations: {path}", flush=True)
    # Some large forecast catalogs can contain a small number of malformed
    # rows (inconsistent column counts). Parse line-by-line and skip bad rows.
    rows = []
    expected_cols = None
    skipped = 0
    with open(path, "r") as fh:
        for lineno, line in enumerate(fh, start=1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if expected_cols is None:
                expected_cols = len(parts)
            if len(parts) != expected_cols:
                skipped += 1
                continue
            rows.append(parts)

    if expected_cols is None or len(rows) == 0:
        raise ValueError(f"No valid photometry rows found in {path}")

    raw = np.asarray(rows, dtype=np.float64)

    ids  = raw[:, 0].astype(np.int64)
    xpos = raw[:, 1].astype(np.float32)
    ypos = raw[:, 2].astype(np.float32)

    # Columns 3,5,7,... are fluxes; columns 4,6,8,... are 1-sigma errors
    fluxes = raw[:, 3::2].astype(np.float32)   # [N_obj, N_f]
    sigmas = raw[:, 4::2].astype(np.float32)   # [N_obj, N_f]

    weights = (1.0 / sigmas) ** 2              # w_i = 1/sigma_i^2

    # Pre-compute per-object scalars used in SaveRedshifts output columns,
    # matching ReadSourceSeds in fitting_tools.cpp exactly.
    obj_stats = dict(
        total_flux             = np.sum(fluxes,            axis=-1).astype(np.float32),
        total_squared_flux     = np.sum(fluxes ** 2,       axis=-1).astype(np.float32),
        total_weighted_sq_flux = np.sum(fluxes ** 2 * weights, axis=-1).astype(np.float32),
        weighted_flux          = np.sum(fluxes * weights,  axis=-1).astype(np.float32),
        total_weight           = np.sum(weights,           axis=-1).astype(np.float32),
        total_noise            = np.sum(sigmas ** 2,       axis=-1).astype(np.float32),  # sum(sigma^2)
        snr                    = np.sum(fluxes / sigmas,   axis=-1).astype(np.float32),  # sum(F/sigma)
    )

    if skipped > 0:
        print(
            f"  [io] warning: skipped {skipped} malformed row(s) with inconsistent column count",
            flush=True,
        )

    print(f"  [io] {len(ids)} objects, {fluxes.shape[1]} filters", flush=True)
    return ids, xpos, ypos, fluxes, weights, obj_stats


def _compute_source_snr(
    fluxes: np.ndarray,
    weights: np.ndarray,
    mode: str,
) -> np.ndarray:
    """Compute per-source SNR for optional object filtering."""
    mode_norm = str(mode).lower()
    sigmas = 1.0 / np.sqrt(np.maximum(weights, 1e-30))

    if mode_norm == "sum":
        return np.sum(fluxes / np.maximum(sigmas, 1e-30), axis=-1).astype(np.float32)
    if mode_norm == "quadrature":
        return np.sqrt(np.sum((fluxes / np.maximum(sigmas, 1e-30)) ** 2, axis=-1)).astype(np.float32)
    if mode_norm == "total_flux":
        total_flux = np.sum(fluxes, axis=-1)
        total_unc = np.sqrt(np.sum(sigmas ** 2, axis=-1))
        return (total_flux / np.maximum(total_unc, 1e-30)).astype(np.float32)

    raise ValueError(
        f"Unsupported source SNR mode '{mode}'. Valid: sum, quadrature, total_flux"
    )


def _apply_source_snr_filter(
    ids: np.ndarray,
    xpos: np.ndarray,
    ypos: np.ndarray,
    fluxes: np.ndarray,
    weights: np.ndarray,
    obj_stats: dict,
    min_source_snr: Optional[float],
    source_snr_mode: str,
    source_snr_max_filters: Optional[int] = None,
):
    """Optionally filter sources by a source-SNR threshold."""
    if min_source_snr is None:
        return ids, xpos, ypos, fluxes, weights, obj_stats

    if source_snr_max_filters is None:
        fluxes_snr = fluxes
        weights_snr = weights
        snr_filter_desc = "all"
    else:
        n_use = int(source_snr_max_filters)
        if n_use <= 0:
            raise ValueError("source_snr_max_filters must be a positive integer")
        if n_use > fluxes.shape[1]:
            print(
                f"  [snr] requested source_snr_max_filters={n_use} but only "
                f"{fluxes.shape[1]} filters available; using all filters",
                flush=True,
            )
            n_use = fluxes.shape[1]
        fluxes_snr = fluxes[:, :n_use]
        weights_snr = weights[:, :n_use]
        snr_filter_desc = f"first_{n_use}"

    source_snr = _compute_source_snr(fluxes_snr, weights_snr, source_snr_mode)
    keep = np.isfinite(source_snr) & (source_snr >= float(min_source_snr))
    n_keep = int(np.sum(keep))
    n_all = int(len(keep))
    frac = 100.0 * n_keep / max(n_all, 1)
    print(
        f"  [snr] mode={source_snr_mode}, filters={snr_filter_desc}, "
        f"min_source_snr={float(min_source_snr):.3f}: "
        f"keeping {n_keep}/{n_all} objects ({frac:.2f}%)",
        flush=True,
    )

    filtered_stats = {k: v[keep] for k, v in obj_stats.items()}
    filtered_stats["source_snr_filter"] = source_snr[keep].astype(np.float32)
    return ids[keep], xpos[keep], ypos[keep], fluxes[keep], weights[keep], filtered_stats


def read_template_probs(path: str, max_template_id: Optional[int] = None) -> np.ndarray:
    """
    Read template probability file (two columns: template_id  probability).

    Returns float32 array indexed by template_id (index 0 is unused).
    All un-listed template_ids receive probability 0.
    """
    data = np.loadtxt(path, dtype=np.float64)
    if data.ndim == 1:
        data = data[None, :]

    tids  = data[:, 0].astype(int)
    probs = data[:, 1].astype(np.float32)

    if max_template_id is None:
        max_template_id = int(tids.max())

    prob_array = np.zeros(max_template_id + 1, dtype=np.float32)
    prob_array[tids] = probs
    return prob_array   # index by template_id


def write_photoz_results(
    path: str,
    ids: np.ndarray,
    xpos: np.ndarray,
    ypos: np.ndarray,
    moments: dict,
    obj_stats: dict,
    n_filters: int,
) -> None:
    """
    Write C++-compatible redshift output, matching SaveRedshifts column order:

    id xpos ypos  z_mean z_std skewness kurtosis pdf_sum
    peak_density z_peak chi2_at_peak min_chi2 mean_chi2
    best_tmpl_peak best_tmpl_overall
    mean_flux weighted_mean_flux rms_weighted_flux
    total_weighted_sq_flux snr_sq total_flux_snr per_filter_snr

    NOTE:
    Column 4 (z_std) is populated with sigma_68 = 0.5 * HPDI_68 width
    derived from P(z), so downstream scripts remain column-compatible.
    """
    s = obj_stats
    m = moments
    N = len(ids)

    # Vectorise the derived columns that require square roots
    mean_flux       = s["total_flux"]             / n_filters
    wt_mean_flux    = s["weighted_flux"]           / s["total_weight"]
    rms_wt_flux     = np.sqrt(np.maximum(s["total_weighted_sq_flux"] / s["total_weight"], 0.0))
    snr_sq          = s["total_squared_flux"]      / s["total_noise"]
    total_flux_snr  = s["total_flux"]              / np.sqrt(s["total_weight"])
    per_filter_snr  = s["snr"]                     / n_filters

    lines = []
    for i in range(N):
        line = (
            f"{ids[i]} "
            f"{xpos[i]:.4f} {ypos[i]:.4f} "
            f"{m['z_mean'][i]:.8f} {m['sigma_68'][i]:.8f} "
            f"{m['skewness'][i]:.8f} {m['kurtosis'][i]:.8f} "
            f"{m['pdf_sum'][i]:.8f} "
            f"{m['peak_density'][i]:.8f} {m['z_peak'][i]:.8f} "
            f"{m['chi2_at_peak'][i]:.8f} {m['min_chi2'][i]:.8f} "
            f"{m['mean_chi2'][i]:.8f} "
            f"{m['best_tmpl_peak'][i]} {m['best_tmpl_overall'][i]} "
            f"{mean_flux[i]:.8f} "
            f"{wt_mean_flux[i]:.8f} "
            f"{rms_wt_flux[i]:.8f} "
            f"{s['total_weighted_sq_flux'][i]:.8f} "
            f"{snr_sq[i]:.8f} "
            f"{total_flux_snr[i]:.8f} "
            f"{per_filter_snr[i]:.8f}"
        )
        lines.append(line)

    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"  [io] wrote {N} redshifts to {path}", flush=True)


def write_pdfs(
    path: str,
    ids: np.ndarray,
    xpos: np.ndarray,
    ypos: np.ndarray,
    redshifts: np.ndarray,
    pdfs: np.ndarray,
) -> None:
    """
    Write C++-compatible PDF output, matching SavePDFs format:

    Header:  # id xpos ypos redshifts:  z0 z1 z2 ...
    Body:    id xpos ypos p0 p1 p2 ...
    """
    N_obj = len(ids)
    N_z   = len(redshifts)
    print(
        f"  [io] writing PDFs ({N_obj} objects × {N_z} z-bins) → {path}",
        flush=True,
    )

    z_str = " ".join(f"{z}" for z in redshifts)

    with open(path, "w") as fh:
        fh.write(f"# id xpos ypos redshifts:  {z_str}\n")
        for i in range(N_obj):
            p_str = " ".join(f"{p:.8e}" for p in pdfs[i])
            fh.write(f"{ids[i]} {xpos[i]:.4f} {ypos[i]:.4f} {p_str}\n")

    print(f"  [io] done writing PDFs", flush=True)


def write_chi2_perz(
    path: str,
    ids: np.ndarray,
    xpos: np.ndarray,
    ypos: np.ndarray,
    redshifts: np.ndarray,
    chi2_perz: np.ndarray,   # float32 [N_obj, N_z]  (all_min_chi2_z)
) -> None:
    """
    Write per-z minimum chi² in the same format as write_pdfs:

    Header:  # id xpos ypos redshifts:  z0 z1 z2 ...
    Body:    id xpos ypos chi2_0 chi2_1 ...

    Mirrors the C++ SaveChiSquared output (auto_chi2 companion to SavePDFs).
    """
    N_obj = len(ids)
    N_z   = len(redshifts)
    print(
        f"  [io] writing chi\u00b2 per-z ({N_obj} objects \u00d7 {N_z} z-bins) \u2192 {path}",
        flush=True,
    )

    z_str = " ".join(f"{z}" for z in redshifts)

    with open(path, "w") as fh:
        fh.write(f"# id xpos ypos redshifts: {z_str}\n")
        for i in range(N_obj):
            c_str = " ".join(f"{c:.6e}" for c in chi2_perz[i])
            fh.write(f"{ids[i]} {xpos[i]:.4f} {ypos[i]:.4f} {c_str}\n")

    print(f"  [io] done writing chi\u00b2 per-z", flush=True)


def _chi2_path_from_pdf_path(pdf_path: str) -> str:
    """Derive chi² companion filename: insert '_chi2' before last '.' (or append)."""
    import os
    stem, ext = os.path.splitext(pdf_path)
    return stem + "_chi2" + ext


def read_pdfs(path: str):
    """
    Read a PDF file written by write_pdfs (or C++ SavePDFs).

    Returns
    -------
    ids       : int64  [N_obj]
    xpos      : float32 [N_obj]
    ypos      : float32 [N_obj]
    redshifts : float32 [N_z]
    pdfs      : float64 [N_obj, N_z]
    """
    ids_list, xpos_list, ypos_list, pdfs_list = [], [], [], []
    redshifts = None

    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                # "# id xpos ypos redshifts:  z0 z1 ..."
                z_part = line.split("redshifts:")[1].split()
                redshifts = np.array(z_part, dtype=np.float32)
            else:
                tokens = line.split()
                ids_list.append(int(tokens[0]))
                xpos_list.append(float(tokens[1]))
                ypos_list.append(float(tokens[2]))
                pdfs_list.append(np.array(tokens[3:], dtype=np.float64))

    return (
        np.array(ids_list, dtype=np.int64),
        np.array(xpos_list, dtype=np.float32),
        np.array(ypos_list, dtype=np.float32),
        redshifts,
        np.array(pdfs_list, dtype=np.float64),
    )


# ===========================================================================
# Section 3: JAX fitting kernel
# ===========================================================================

def _build_tmpl_prob_vector(
    template_ids: np.ndarray,       # [N_z, N_m]
    template_prob_array: np.ndarray, # [max_tid+1]
) -> np.ndarray:
    """
    Return a [N_m] float32 array of template probabilities in model order.

    The template order is identical across every z-bin (verified by the
    C++ grid builder), so we use the first z-bin's template_ids.
    """
    tids_first_z = template_ids[0]                           # [N_m]
    return template_prob_array[tids_first_z].astype(np.float32)  # [N_m]


def compute_pdfs_jax(
    obs_fluxes: np.ndarray,         # float32 [N_obj, N_f]
    obs_weights: np.ndarray,         # float32 [N_obj, N_f]
    model_fluxes: np.ndarray,        # float32 [N_z, N_m, N_f]
    template_ids: np.ndarray,        # int32   [N_z, N_m]
    template_prob_array: np.ndarray, # float32 [max_tid+1]
    batch_obj: int = 10_000,
    multicore: bool = False,
    z_chunk: int = 2,
    precision: str = 'float64',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-object P(z) PDFs using JAX on GPU.

    Two execution paths selected by ``multicore``:

    Single-GPU  (multicore=False, default)
        lax.scan over N_z/z_chunk chunks (z_chunk=2 → 751 steps instead of 1501).
        Within each scan step, vmap processes z_chunk z-bins in parallel.
        Objects are processed in Python batches of size batch_obj
        (batch_obj=0 means all objects at once).

    Multi-GPU   (multicore=True)
        pmap across all available devices (4×A100 on Perlmutter).
        Same z-chunked vmap+scan within each device.
        batch_obj is interpreted as the *total* objects per outer step across
        all GPUs; each device receives batch_obj//n_dev objects per step.
        (batch_obj=0 means all objects at once.)

    Memory guidance (40 GB A100, N_m=10080)
        Peak working memory per device per scan step:
            grid (F + F²): ~12.4 GB  (constant)
            intermediates: z_chunk × (batch_per_dev/50000) × 8.1 GB
        Default z_chunk=2, batch_obj=50000 → ~28.5 GB single-GPU,
        ~17 GB multi-GPU (batch_per_dev=12500).  Increase z_chunk
        only if using a smaller batch_obj.

    The core chi-squared formula is:
        chi2_om = C_o  -  B_om^2 / D_om
    where
        C_o  = sum_f (F_o^f)^2 * w_o^f
        B_om = (F_obs * W_obs) @ F_models.T    [N_obj, N_m]
        D_om = W_obs            @ (F_models^2).T [N_obj, N_m]

    Models with zero template probability are excluded from chi2
    argmin / best_template tracking (matching C++ `continue` logic).

    Parameters
    ----------
    obs_fluxes          : float32 [N_obj, N_f]
    obs_weights         : float32 [N_obj, N_f]
    model_fluxes        : float32 [N_z, N_m, N_f]
    template_ids        : int32   [N_z, N_m]
    template_prob_array : float32 [max_tid+1]
    batch_obj           : objects per GPU batch for single-GPU path
                          (0 = all objects at once)
    multicore           : use pmap across all available GPUs
    z_chunk             : number of z-bins to vmap over per scan step

    Returns
    -------
    pdfs            : float64 [N_obj, N_z]  -- full P(z) array
    min_chi2_perz   : float32 [N_obj, N_z]  -- min chi2 across models at each z
    best_tmpl_perz  : int32   [N_obj, N_z]  -- best-fit template_id at each z
    """
    N_obj, N_f = obs_fluxes.shape
    N_z,  N_m, _ = model_fluxes.shape

    tmpl_probs_np   = _build_tmpl_prob_vector(template_ids, template_prob_array)  # [N_m]
    nonzero_mask_np = (tmpl_probs_np > 0.0).astype(np.float32)                   # [N_m]
    tids_np         = template_ids[0]                                             # [N_m]

    t0 = time.time()

    # Precision control
    prec = str(precision).lower()
    if prec not in ("float64", "float32", "bfloat16"):
        raise ValueError(f"Unsupported precision '{precision}'")
    # accumulation dtype (used for chi2, exp, sums)
    acc_dtype = jnp.float64 if prec == "float64" else jnp.float32
    # store dtype for large grids on-device (bfloat16 saves memory)
    store_dtype = jnp.bfloat16 if prec == "bfloat16" else jnp.float32

    _INF = acc_dtype(1e30)
    _EPS = acc_dtype(1e-30)

    # ==================================================================
    # Single-GPU path  (original lax.scan over individual z-bins)
    # ==================================================================
    if not multicore:
        print(
            f"  [jax] transferring grid to GPU  "
            f"(fluxes {model_fluxes.nbytes/1e9:.2f} GB)...",
            flush=True,
        )
        f_grid_base = jnp.array(model_fluxes)                # [N_z, N_m, N_f]
        if prec == "bfloat16":
            f_grid_dev = f_grid_base.astype(jnp.bfloat16)
            f_sq_grid_dev = (f_grid_base ** 2).astype(jnp.bfloat16)
        elif prec == "float64":
            f_grid_dev = f_grid_base.astype(jnp.float64)
            f_sq_grid_dev = (f_grid_base ** 2).astype(jnp.float64)
        else:
            f_grid_dev = f_grid_base.astype(jnp.float32)
            f_sq_grid_dev = (f_grid_base ** 2).astype(jnp.float32)
        tmpl_probs_dev   = jnp.array(tmpl_probs_np).astype(acc_dtype)  # [N_m]
        nonzero_mask_dev = jnp.array(nonzero_mask_np)
        print(f"  [jax] grid on GPU in {time.time()-t0:.1f}s", flush=True)

        @jit
        def process_batch(F_obs, W_obs, f_grid, f_sq_grid, tmpl_probs, nonzero_mask):
            # Choose accumulation dtype based on selected precision
            A = acc_dtype
            # Upcast observations to accumulation dtype
            F_a = F_obs.astype(A)
            W_a = W_obs.astype(A)
            C_obs = jnp.sum(F_a ** 2 * W_a, axis=-1)    # [B] in accumulation dtype
            FW = F_a * W_a                              # [B, N_f]

            def scan_step(carry, z_slice):
                f_z, f_sq_z = z_slice
                f_z_a = f_z.astype(A)
                f_sq_z_a = f_sq_z.astype(A)
                B_a = FW @ f_z_a.T
                D_a = W_a @ f_sq_z_a.T
                chi2 = C_obs[:, None] - B_a ** 2 / jnp.maximum(D_a, A(_EPS))
                # P(z) in accumulation dtype
                p_z = jnp.sum(tmpl_probs[None, :].astype(A) * jnp.exp(-0.5 * chi2), axis=-1) / N_m
                # For storage, provide min_chi2 in float32 (matches downstream expectations)
                chi2_store = chi2 if A == jnp.float32 else chi2.astype(jnp.float32)
                chi2_masked = jnp.where(nonzero_mask[None, :] > 0, chi2_store, _INF.astype(jnp.float32))
                min_chi2_z = jnp.min(chi2_masked, axis=-1)
                best_m_z = jnp.argmin(chi2_masked, axis=-1)
                return None, (p_z, min_chi2_z, best_m_z)

            _, (pdfs_z, min_chi2_z, best_m_z) = jax.lax.scan(
                scan_step, None, (f_grid, f_sq_grid),
            )
            return pdfs_z, min_chi2_z, best_m_z  # each [N_z, B]

        all_pdfs        = np.zeros((N_obj, N_z), dtype=np.float64)
        all_min_chi2_z  = np.zeros((N_obj, N_z), dtype=np.float32)
        all_best_tmpl_z = np.zeros((N_obj, N_z), dtype=np.int32)

        t_start   = time.time()
        eff_batch = batch_obj if batch_obj > 0 else N_obj
        n_batches = math.ceil(N_obj / eff_batch)

        for b in range(n_batches):
            lo = b * eff_batch
            hi = min(lo + eff_batch, N_obj)
            B  = hi - lo

            F_b = jnp.array(obs_fluxes [lo:hi])
            W_b = jnp.array(obs_weights[lo:hi])

            print(
                f"  [jax single] batch {b+1}/{n_batches}  "
                f"obj {lo}–{hi-1}  ({B} objects)",
                flush=True,
            )
            t_b = time.time()
            p_j, mc_j, bm_j = process_batch(
                F_b, W_b, f_grid_dev, f_sq_grid_dev, tmpl_probs_dev, nonzero_mask_dev
            )
            jax.block_until_ready(p_j)
            print(f"         finished in {time.time()-t_b:.2f}s", flush=True)

            pdfs_z_np   = np.array(p_j,  dtype=np.float64)   # [N_z, B]
            min_chi2_np = np.array(mc_j, dtype=np.float32)
            best_m_np   = np.array(bm_j, dtype=np.int32)

            all_pdfs       [lo:hi] = pdfs_z_np.T
            all_min_chi2_z [lo:hi] = min_chi2_np.T
            all_best_tmpl_z[lo:hi] = tids_np[best_m_np].T

        print(
            f"  [jax single] all batches done — {time.time()-t_start:.2f}s total",
            flush=True,
        )
        return all_pdfs, all_min_chi2_z, all_best_tmpl_z

    # ==================================================================
    # Multi-GPU path  (pmap across all local devices)
    # ==================================================================

    # Pad N_z to a multiple of z_chunk for the vmapped multi-GPU scan
    N_z_pad   = math.ceil(N_z / z_chunk) * z_chunk
    N_chunks  = N_z_pad // z_chunk
    z_pad_amt = N_z_pad - N_z

    print(
        f"  [jax] preparing grid  "
        f"(fluxes {model_fluxes.nbytes/1e9:.2f} GB, z_chunk={z_chunk})...",
        flush=True,
    )

    f_np = model_fluxes
    if z_pad_amt > 0:
        pad  = np.zeros((z_pad_amt, N_m, N_f), dtype=np.float32)
        f_np = np.concatenate([f_np, pad], axis=0)
    f_chunks_np = f_np.reshape(N_chunks, z_chunk, N_m, N_f)

    print(f"  [jax] grid prepared in {time.time()-t0:.1f}s", flush=True)

    # ==================================================================
    # Multi-GPU path  (pmap across all local devices)
    # ==================================================================
    devices = jax.local_devices()
    n_dev   = len(devices)
    print(f"  [jax multi] using {n_dev} devices", flush=True)

    # Per-device batch size.  batch_obj is the *total* objects per outer step
    # across all devices, so each device sees batch_obj // n_dev objects.
    # This bounds intermediate memory to z_chunk × (batch_per_dev/50k) × 8.1 GB.
    if batch_obj > 0:
        batch_per_dev = max(1, math.ceil(batch_obj / n_dev))
    else:
        batch_per_dev = math.ceil(N_obj / n_dev)
    total_step = n_dev * batch_per_dev

    # Pad N_obj to a multiple of total_step for clean looping
    n_obj_pad = math.ceil(N_obj / total_step) * total_step
    obs_f_pad = np.zeros((n_obj_pad, N_f), dtype=np.float32)
    obs_w_pad = np.zeros((n_obj_pad, N_f), dtype=np.float32)
    obs_f_pad[:N_obj] = obs_fluxes
    obs_w_pad[:N_obj] = obs_weights

    # Replicate grid to all devices once (outside the batch loop)
    f_chunks_arr    = jnp.array(f_chunks_np)
    f_sq_chunks_arr = f_chunks_arr ** 2  # compute on GPU, saves 6 GB CPU
    f_chunks_rep    = jax.device_put_replicated(f_chunks_arr,     devices)
    f_sq_chunks_rep = jax.device_put_replicated(f_sq_chunks_arr,  devices)
    tp_rep          = jax.device_put_replicated(jnp.array(tmpl_probs_np),   devices)
    nm_rep          = jax.device_put_replicated(jnp.array(nonzero_mask_np), devices)
    del f_chunks_arr, f_sq_chunks_arr

    @functools.partial(jax.pmap, in_axes=(0, 0, 0, 0, 0, 0))
    def process_all_pmap(F_dev, W_dev, f_chunks_loc, f_sq_chunks_loc, tp_loc, nm_loc):
        """
        Per-device computation (pmap strips the leading device axis).
        F_dev / W_dev      : [batch_per_dev, N_f]
        f_chunks_loc       : [N_chunks, z_chunk, N_m, N_f]
        Returns each: [N_chunks, z_chunk, batch_per_dev]
        """
        C_obs = jnp.sum(F_dev ** 2 * W_dev, axis=-1)  # [batch_per_dev]
        FW    = F_dev * W_dev                          # [batch_per_dev, N_f]

        def one_z(f_z, f_sq_z):
            B_om  = FW    @ f_z.T                               # [batch_per_dev, N_m]
            D_om  = W_dev @ f_sq_z.T                           # [batch_per_dev, N_m]
            chi2  = C_obs[:, None] - B_om ** 2 / jnp.maximum(D_om, _EPS)
            # P(z) = (1/N_m) Σᵢ P(Tᵢ) exp(-χ²/2)  — C++ divides by kn_models
            p_z   = jnp.sum(
                tp_loc[None, :] * jnp.exp(-0.5 * chi2), axis=-1
            ) / N_m
            chi2m = jnp.where(nm_loc[None, :] > 0, chi2, _INF)
            return p_z, jnp.min(chi2m, axis=-1), jnp.argmin(chi2m, axis=-1)

        def scan_step(carry, chunk):
            f_c, f_sq_c = chunk
            p_c, mc_c, bm_c = jax.vmap(one_z)(f_c, f_sq_c)  # [z_chunk, batch_per_dev]
            return None, (p_c, mc_c, bm_c)

        _, (p_c, mc_c, bm_c) = jax.lax.scan(
            scan_step, None, (f_chunks_loc, f_sq_chunks_loc)
        )
        return p_c, mc_c, bm_c  # each [N_chunks, z_chunk, batch_per_dev]

    all_pdfs        = np.zeros((N_obj, N_z), dtype=np.float64)
    all_min_chi2_z  = np.zeros((N_obj, N_z), dtype=np.float32)
    all_best_tmpl_z = np.zeros((N_obj, N_z), dtype=np.int32)

    n_outer = n_obj_pad // total_step
    t_start = time.time()

    for b in range(n_outer):
        lo = b * total_step
        hi = lo + total_step
        n_valid = min(hi, N_obj) - lo   # number of real objects in this step

        # Shape for pmap: [n_dev, batch_per_dev, N_f]
        F_slice = obs_f_pad[lo:hi].reshape(n_dev, batch_per_dev, N_f)
        W_slice = obs_w_pad[lo:hi].reshape(n_dev, batch_per_dev, N_f)
        F_sh = jax.device_put_sharded(list(F_slice), devices)
        W_sh = jax.device_put_sharded(list(W_slice), devices)

        print(
            f"  [jax multi] batch {b+1}/{n_outer}  "
            f"obj {lo}–{lo+n_valid-1}  ({n_valid} valid, {n_dev} GPUs)",
            flush=True,
        )
        t_b = time.time()
        p_rep, mc_rep, bm_rep = process_all_pmap(
            F_sh, W_sh, f_chunks_rep, f_sq_chunks_rep, tp_rep, nm_rep,
        )
        jax.block_until_ready(p_rep)
        print(f"         finished in {time.time()-t_b:.2f}s", flush=True)

        # Gather: [n_dev, N_chunks, z_chunk, batch_per_dev]
        #   → [n_dev, batch_per_dev, N_chunks, z_chunk]  (transpose)
        #   → [total_step, N_z_pad]                       (reshape)
        #   → [:, :N_z]                                   (trim z padding)
        def gather(arr_dev, dtype):
            arr = np.array(arr_dev, dtype=dtype)          # [n_dev, N_chunks, z_chunk, B_pd]
            arr = arr.transpose(0, 3, 1, 2)               # [n_dev, B_pd, N_chunks, z_chunk]
            arr = arr.reshape(total_step, N_z_pad)        # [total_step, N_z_pad]
            return arr[:n_valid, :N_z]                    # only valid objects, valid z

        p_np  = gather(p_rep,  np.float64)
        mc_np = gather(mc_rep, np.float32)
        bm_np = gather(bm_rep, np.int32)

        all_pdfs       [lo:lo+n_valid] = p_np
        all_min_chi2_z [lo:lo+n_valid] = mc_np
        all_best_tmpl_z[lo:lo+n_valid] = tids_np[bm_np]

    print(
        f"  [jax multi] done — {time.time()-t_start:.2f}s total ({n_outer} outer batch(es))",
        flush=True,
    )
    return all_pdfs, all_min_chi2_z, all_best_tmpl_z


# ===========================================================================
# Section 4: Redshift moment computation
# ===========================================================================

def _sigma68_from_hpdi(
    z: np.ndarray,                # float64 [N_z], ascending grid
    p: np.ndarray,                # float64 [N_obj, N_z], non-negative density
    pdf_sum: np.ndarray,          # float64 [N_obj]
    valid: np.ndarray,            # bool [N_obj], pdf_sum > 0
    mass: float = 0.68,
) -> np.ndarray:
    """
    Compute sigma_68 from the 68% HPDI set on a discrete z-grid.

    Implementation: for each object, sort bins by descending density, include
    bins until cumulative mass reaches `mass * pdf_sum`, then define
    HPDI width as max(z_selected) - min(z_selected). Return sigma_68 = 0.5*width.
    """
    N_obj, N_z = p.shape
    if N_obj == 0:
        return np.zeros(0, dtype=np.float64)

    # Sort each PDF row by descending density.
    order = np.argsort(-p, axis=1)                         # [N_obj, N_z]
    p_sorted = np.take_along_axis(p, order, axis=1)        # [N_obj, N_z]
    cdf_sorted = np.cumsum(p_sorted, axis=1)               # [N_obj, N_z]

    target = mass * pdf_sum                                # [N_obj]
    reached = cdf_sorted >= target[:, None]                # [N_obj, N_z]
    # For valid rows there is always at least one True; invalid rows are masked later.
    k = np.argmax(reached, axis=1)                         # [N_obj]

    rank = np.arange(N_z)[None, :]                         # [1, N_z]
    keep = rank <= k[:, None]                              # [N_obj, N_z]

    z_sorted = np.take_along_axis(z[None, :], order, axis=1)  # [N_obj, N_z]
    z_min = np.min(np.where(keep, z_sorted, np.inf), axis=1)
    z_max = np.max(np.where(keep, z_sorted, -np.inf), axis=1)

    width = z_max - z_min
    sigma_68 = 0.5 * width
    sigma_68 = np.where(valid, sigma_68, np.nan)
    return sigma_68.astype(np.float64)

def compute_redshift_moments(
    redshifts: np.ndarray,       # float32 [N_z]
    pdfs: np.ndarray,            # float64 [N_obj, N_z]
    min_chi2_perz: np.ndarray,   # float32 [N_obj, N_z]
    best_tmpl_perz: np.ndarray,  # int32   [N_obj, N_z]
) -> dict:
    """
    Compute redshift statistics, replicating C++ SaveRedshifts exactly.

    Moments:
        z_mean  = sum(z*p) / sum(p)
        sigma_68 = 0.5 * HPDI_68_width(P(z))
        skewness = sum((z-zmean)^3 * p) / sum(p) / z_std^3
        kurtosis = sum((z-zmean)^4 * p) / sum(p) / z_std^4

    mean_chi2 = sum(min_chi2_z) / sum(p)
        -- NOTE: C++ sums per-z min_chi2 WITHOUT density weighting,
           then divides by pdf_sum.  We replicate this exactly.

    All quantities are set to -9.0 for objects where pdf_sum == 0 or
    any statistic is non-finite, matching C++ sentinel behaviour.

    Returns dict of float32 [N_obj] arrays plus int32 best_template arrays.
    """
    z = redshifts.astype(np.float64)   # [N_z]
    p = pdfs.astype(np.float64)        # [N_obj, N_z]
    N_obj = p.shape[0]
    SENTINEL = np.float32(-9.0)

    pdf_sum   = np.sum(p, axis=-1)                  # [N_obj]
    # Gate on pdf_sum > 0: with 102 filters, even good fits give chi2 ~ 50-100,
    # so exp(-chi2/2) is tiny (~1e-22) and pdf_sum is typically 1e-15 to 1e-25.
    # The C++ sentinel applies only when pdf_sum is truly zero (no valid fits).
    # Moments are ratios (e.g. z_mean = sum(z*P)/sum(P)) so absolute magnitude cancels;
    # the _mask() function catches any non-finite results from near-zero pdf_sum.
    valid     = pdf_sum > 0                         # [N_obj] bool
    inv_pdf   = np.where(valid, 1.0 / np.where(valid, pdf_sum, 1.0), 0.0)

    # Mean redshift
    z_mean = np.sum(p * z[None, :], axis=-1) * inv_pdf   # [N_obj]

    # Central moments (raw, i.e. not yet divided by pdf_sum)
    dz  = z[None, :] - z_mean[:, None]   # [N_obj, N_z]
    dz2 = dz ** 2

    m2_raw = np.sum(p * dz2,       axis=-1)   # [N_obj]
    m3_raw = np.sum(p * dz2 * dz,  axis=-1)
    m4_raw = np.sum(p * dz2 ** 2,  axis=-1)

    # Standard deviation (matches C++ second_moment * inv_pdf_sum formula)
    z_var = m2_raw * inv_pdf
    z_std_moment = np.sqrt(np.maximum(z_var, 0.0))

    # HPDI-based uncertainty used for saved z_std column downstream.
    sigma_68 = _sigma68_from_hpdi(z, p, pdf_sum, valid, mass=0.68)

    # Valid for higher moments: need both valid pdf_sum and non-zero std
    valid_std = valid & (z_std_moment > 1e-30)
    inv_std   = np.where(valid_std, 1.0 / np.where(valid_std, z_std_moment, 1.0), 0.0)

    # Skewness = (sum diff^3 * p / pdf_sum) / std^3
    skewness = m3_raw * inv_pdf * inv_std ** 3

    # Kurtosis = (sum diff^4 * p / pdf_sum) / std^4
    kurtosis = m4_raw * inv_pdf * inv_std ** 4

    # mean_chi2: C++ sums per-z minimum chi2 (NOT density-weighted), then /pdf_sum
    chi2_sum  = np.sum(min_chi2_perz.astype(np.float64), axis=-1)   # [N_obj]
    mean_chi2 = chi2_sum * inv_pdf

    # Global minimum chi2 across all z-bins, and corresponding template
    min_chi2_global   = np.min(min_chi2_perz,  axis=-1)              # [N_obj]
    best_z_idx_global = np.argmin(min_chi2_perz, axis=-1)            # [N_obj]
    arange            = np.arange(N_obj)
    best_tmpl_overall = best_tmpl_perz[arange, best_z_idx_global]    # [N_obj]

    # Peak PDF z-bin
    peak_z_idx    = np.argmax(p, axis=-1)                            # [N_obj]
    peak_density  = p[arange, peak_z_idx]
    z_peak        = z[peak_z_idx]
    chi2_at_peak  = min_chi2_perz[arange, peak_z_idx]
    best_tmpl_peak = best_tmpl_perz[arange, peak_z_idx]

    # ------------------------------------------------------------------
    # Apply sentinel -9.0 for invalid objects, matching C++ behaviour
    # ------------------------------------------------------------------
    def _mask(arr: np.ndarray, gate: np.ndarray) -> np.ndarray:
        """Replace entries where gate is False with SENTINEL; ensure finite."""
        out = np.where(gate, arr, SENTINEL)
        out = np.where(np.isfinite(out), out, SENTINEL)
        return out.astype(np.float32)

    return dict(
        z_mean         = _mask(z_mean,    valid),
        sigma_68       = _mask(sigma_68,  valid),
        z_std          = _mask(sigma_68,  valid),  # backward-compatible key
        skewness       = _mask(skewness,  valid_std),
        kurtosis       = _mask(kurtosis,  valid_std),
        pdf_sum        = _mask(pdf_sum,   valid),
        peak_density   = _mask(peak_density, valid),
        z_peak         = _mask(z_peak,    valid),
        chi2_at_peak   = _mask(chi2_at_peak.astype(np.float64), valid),
        min_chi2       = min_chi2_global.astype(np.float32),  # not gated (always valid)
        mean_chi2      = _mask(mean_chi2,  valid),
        best_tmpl_peak    = best_tmpl_peak,        # int32 [N_obj]
        best_tmpl_overall = best_tmpl_overall,     # int32 [N_obj]
    )


# ===========================================================================
# Section 5: Verification against C++ output
# ===========================================================================

def verify_against_cpp(
    model_grid_paths,
    obs_phot_path: str,
    template_prob_path: str,
    cpp_pdfs_path: Optional[str] = None,
    cpp_redshifts_path: Optional[str] = None,
    batch_obj: int = 10_000,
    multicore: bool = False,
    precision: str = 'float64',
    min_source_snr: Optional[float] = None,
    source_snr_mode: str = 'quadrature',
    source_snr_max_filters: Optional[int] = None,
) -> None:
    """
    Run JAX fitter on the same inputs as a previous C++ run and compare.

    Primary verification: element-wise PDF differences.
    Secondary:           column-by-column redshift moment differences.
    """
    print("\n=== VERIFICATION MODE ===\n", flush=True)

    if cpp_pdfs_path is None and cpp_redshifts_path is None:
        raise ValueError(
            "At least one of --cpp-pdfs or --cpp-redshifts must be provided."
        )

    # --- Parse inputs -------------------------------------------------------
    grid = parse_photoz_grid(model_grid_paths)
    ids, xpos, ypos, obs_fluxes, obs_weights, obj_stats = read_obs_phot(obs_phot_path)
    ids, xpos, ypos, obs_fluxes, obs_weights, obj_stats = _apply_source_snr_filter(
        ids,
        xpos,
        ypos,
        obs_fluxes,
        obs_weights,
        obj_stats,
        min_source_snr=min_source_snr,
        source_snr_mode=source_snr_mode,
        source_snr_max_filters=source_snr_max_filters,
    )
    template_prob_array = read_template_probs(
        template_prob_path,
        max_template_id=int(grid["template_ids"].max()),
    )

    # --- Run JAX ------------------------------------------------------------
    print("\n--- Running JAX fitting ---", flush=True)
    pdfs_jax, min_chi2_perz, best_tmpl_perz = compute_pdfs_jax(
        obs_fluxes, obs_weights,
        grid["model_fluxes"], grid["template_ids"],
        template_prob_array,
        batch_obj=batch_obj,
        multicore=multicore,
        precision=precision,
    )

    # --- Primary: PDF comparison --------------------------------------------
    if cpp_pdfs_path and os.path.exists(cpp_pdfs_path):
        print("\n--- PDF comparison ---", flush=True)
        _, _, _, z_cpp, pdfs_cpp = read_pdfs(cpp_pdfs_path)

        z_jax = grid["redshifts"]

        # Align z-bins if files do not cover the same range
        if len(z_cpp) != len(z_jax):
            print(
                f"  WARNING: C++ PDFs have {len(z_cpp)} z-bins, "
                f"JAX has {len(z_jax)}.  Comparing common z-bins only.",
                flush=True,
            )
            # Find common z-bins (round to 5 decimals for float comparison)
            z_jax_rounded = np.round(z_jax.astype(float), 5)
            z_cpp_rounded = np.round(z_cpp.astype(float), 5)
            
            # Keep only z-bins that appear in both
            common_z = np.intersect1d(z_jax_rounded, z_cpp_rounded)
            keep_jax = np.isin(z_jax_rounded, common_z)
            keep_cpp = np.isin(z_cpp_rounded, common_z)
            
            pdfs_jax_cmp = pdfs_jax[:, keep_jax]
            pdfs_cpp_cmp = pdfs_cpp[:, keep_cpp]
            print(f"  Comparing {keep_jax.sum()} common z-bins.", flush=True)
        else:
            pdfs_jax_cmp = pdfs_jax
            pdfs_cpp_cmp = pdfs_cpp

        diff = np.abs(pdfs_jax_cmp - pdfs_cpp_cmp)
        print(
            f"  |ΔPDF|  max={diff.max():.3e}  "
            f"mean={diff.mean():.3e}  "
            f"median={np.median(diff):.3e}",
            flush=True,
        )
        
        # Relative diff only where C++ PDF is significant (> 1e-10)
        mask_significant = pdfs_cpp_cmp > 1e-10
        if mask_significant.any():
            rel_diff_sig = diff[mask_significant] / pdfs_cpp_cmp[mask_significant]
            print(
                f"  |ΔPDF|/|PDF_cpp| (where P_cpp > 1e-10)  "
                f"max={rel_diff_sig.max():.3e}  mean={rel_diff_sig.mean():.3e}",
                flush=True,
            )
        else:
            print(f"  |ΔPDF|/|PDF_cpp|  (no significant PDFs to compare)", flush=True)
    else:
        pdfs_jax_cmp = None

    # --- Secondary: moment comparison ---------------------------------------
    if cpp_redshifts_path and os.path.exists(cpp_redshifts_path):
        print("\n--- Moment comparison ---", flush=True)
        moments_jax = compute_redshift_moments(
            grid["redshifts"], pdfs_jax, min_chi2_perz, best_tmpl_perz
        )
        cpp_data = np.loadtxt(cpp_redshifts_path)
        if cpp_data.ndim == 1:
            cpp_data = cpp_data[None, :]

        # Objects with valid z estimates in C++ output (not sentinel -9)
        valid_cpp = cpp_data[:, 3] > -8.0
        print(f"  Valid objects in C++ output: {valid_cpp.sum()} / {len(valid_cpp)}", flush=True)

        # (column_name, C++ column index, JAX moments key)
        comparisons = [
            ("z_mean",       3, "z_mean"),
            ("z_std",        4, "z_std"),
            ("skewness",     5, "skewness"),
            ("kurtosis",     6, "kurtosis"),
            ("pdf_sum",      7, "pdf_sum"),
            ("peak_density", 8, "peak_density"),
            ("z_peak",       9, "z_peak"),
            ("chi2_at_peak", 10, "chi2_at_peak"),
            ("min_chi2",     11, "min_chi2"),
            ("mean_chi2",    12, "mean_chi2"),
        ]

        for name, cidx, key in comparisons:
            jax_vals = moments_jax[key][valid_cpp].astype(np.float64)
            cpp_vals = cpp_data[valid_cpp, cidx]
            
            # Filter out inf/nan values
            finite_mask = np.isfinite(jax_vals) & np.isfinite(cpp_vals)
            if not finite_mask.any():
                print(f"  {name:<20s}  (no finite values to compare)", flush=True)
                continue
            
            jax_vals_clean = jax_vals[finite_mask]
            cpp_vals_clean = cpp_vals[finite_mask]
            d = np.abs(jax_vals_clean - cpp_vals_clean)
            
            n_compared = len(d)
            n_skipped = len(jax_vals) - n_compared
            status = f" ({n_compared} objs" + (f", {n_skipped} inf/nan)" if n_skipped > 0 else ")")
            
            print(
                f"  {name:<20s}  max|Δ|={d.max():.3e}  "
                f"mean|Δ|={d.mean():.3e}{status}",
                flush=True,
            )

    elif cpp_pdfs_path and os.path.exists(cpp_pdfs_path):
        # No C++ moments file for this grid — derive moments from C++ PDFs instead.
        print("\n--- Moment comparison (derived from C++ PDFs) ---", flush=True)
        print("  Note: chi2 columns skipped (not in PDF file).", flush=True)
        _, _, _, z_cpp2, pdfs_cpp2 = read_pdfs(cpp_pdfs_path)
        cpp_moments2 = _moments_from_pdfs(z_cpp2, pdfs_cpp2)
        # Build a fake cpp_data array matching write_photoz_results column layout
        N2 = len(cpp_moments2["z_mean"])
        cpp_data2 = np.full((N2, 15), -9.0, dtype=np.float64)
        cpp_data2[:, 3] = cpp_moments2["z_mean"]
        cpp_data2[:, 4] = cpp_moments2["z_std"]
        cpp_data2[:, 5] = cpp_moments2["skewness"]
        cpp_data2[:, 6] = cpp_moments2["kurtosis"]
        cpp_data2[:, 7] = cpp_moments2["pdf_sum"]
        cpp_data2[:, 8] = cpp_moments2["peak_density"]
        cpp_data2[:, 9] = cpp_moments2["z_peak"]
        # Compute JAX moments from grid + JAX PDFs
        moments_jax2 = compute_redshift_moments(
            grid["redshifts"], pdfs_jax, min_chi2_perz, best_tmpl_perz
        )
        pdf_comparisons = [
            ("z_mean",       3, "z_mean"),
            ("z_std",        4, "z_std"),
            ("skewness",     5, "skewness"),
            ("kurtosis",     6, "kurtosis"),
            ("pdf_sum",      7, "pdf_sum"),
            ("peak_density", 8, "peak_density"),
            ("z_peak",       9, "z_peak"),
        ]
        valid2 = cpp_data2[:, 3] > -8.0
        print(f"  Valid C++ objects (derived): {valid2.sum()} / {N2}", flush=True)
        for name, cidx, key in pdf_comparisons:
            jax_v = moments_jax2[key][valid2].astype(np.float64)
            cpp_v = cpp_data2[valid2, cidx]
            fm = np.isfinite(jax_v) & np.isfinite(cpp_v) & (cpp_v > -8.0)
            if not fm.any():
                print(f"  {name:<20s}  (no finite values)", flush=True)
                continue
            d = np.abs(jax_v[fm] - cpp_v[fm])
            print(
                f"  {name:<20s}  max|Δ|={d.max():.3e}  "
                f"mean|Δ|={d.mean():.3e}  ({len(d)} objs)",
                flush=True,
            )

    print("\n=== VERIFICATION COMPLETE ===\n", flush=True)


def _moments_from_pdfs(z_grid: np.ndarray, pdfs: np.ndarray) -> dict:
    """
    Compute redshift moments purely from a PDF array (no chi2 quantities).
    Used to derive C++ moments when only the C++ PDF file is available.
    Returns dict of float32 [N_obj]: z_mean, sigma_68, z_std, skewness, kurtosis,
    pdf_sum, peak_density, z_peak.
    """
    z = z_grid.astype(np.float64)
    p = pdfs.astype(np.float64)
    N_obj = p.shape[0]
    SENTINEL = np.float32(-9.0)

    pdf_sum = np.sum(p, axis=-1)
    valid   = pdf_sum > 0
    inv_pdf = np.where(valid, 1.0 / np.where(valid, pdf_sum, 1.0), 0.0)

    z_mean = np.sum(p * z[None, :], axis=-1) * inv_pdf
    dz     = z[None, :] - z_mean[:, None]
    dz2    = dz ** 2
    m2_raw = np.sum(p * dz2,      axis=-1)
    m3_raw = np.sum(p * dz2 * dz, axis=-1)
    m4_raw = np.sum(p * dz2 ** 2, axis=-1)

    z_std_moment = np.sqrt(np.maximum(m2_raw * inv_pdf, 0.0))
    sigma_68 = _sigma68_from_hpdi(z, p, pdf_sum, valid, mass=0.68)
    valid_std = valid & (z_std_moment > 1e-30)
    inv_std   = np.where(valid_std, 1.0 / np.where(valid_std, z_std_moment, 1.0), 0.0)
    skewness  = m3_raw * inv_pdf * inv_std ** 3
    kurtosis  = m4_raw * inv_pdf * inv_std ** 4

    arange       = np.arange(N_obj)
    peak_z_idx   = np.argmax(p, axis=-1)
    peak_density = p[arange, peak_z_idx]
    z_peak       = z[peak_z_idx]

    def _mask(arr, gate):
        out = np.where(gate, arr, SENTINEL)
        out = np.where(np.isfinite(out), out, SENTINEL)
        return out.astype(np.float32)

    return {
        "z_mean":       _mask(z_mean,        valid),
        "sigma_68":     _mask(sigma_68,      valid),
        "z_std":        _mask(sigma_68,      valid),  # backward-compatible key
        "skewness":     _mask(skewness,      valid_std),
        "kurtosis":     _mask(kurtosis,      valid_std),
        "pdf_sum":      _mask(pdf_sum,       valid),
        "peak_density": _mask(peak_density,  valid),
        "z_peak":       _mask(z_peak,        valid),
    }


def compare_existing_outputs(
    jax_pdfs_path: Optional[str] = None,
    jax_redshifts_path: Optional[str] = None,
    cpp_pdfs_path: Optional[str] = None,
    cpp_redshifts_path: Optional[str] = None,
    true_z_path: Optional[str] = None,
    plot_dir: Optional[str] = None,
    derive_from_pdfs: bool = True,  # NEW: force moment derivation from PDFs
) -> None:
    """
    Compare existing JAX outputs against C++ outputs without re-running fitting.
    
    At least one JAX output and one C++ output must be provided.
    Optionally save comparison plots to plot_dir.
    If true_z_path is provided, adds photoz vs true_z comparisons for both JAX and C++.
    
    If derive_from_pdfs=True (default), moments are re-derived from PDFs for both
    JAX and C++ to ensure consistent computation, avoiding any file format differences.
    """
    print("\n=== COMPARE MODE (using existing outputs) ===\n", flush=True)
    
    if jax_pdfs_path is None and jax_redshifts_path is None:
        raise ValueError(
            "At least one of --jax-pdfs or --jax-redshifts must be provided."
        )
    if cpp_pdfs_path is None and cpp_redshifts_path is None:
        raise ValueError(
            "At least one of --cpp-pdfs or --cpp-redshifts must be provided."
        )
    
    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
        print(f"Saving plots to: {plot_dir}\n", flush=True)
    
    # --- Primary: PDF comparison --------------------------------------------
    pdfs_jax_cmp, pdfs_cpp_cmp = None, None
    if jax_pdfs_path and cpp_pdfs_path:
        print("\n--- PDF comparison ---", flush=True)
        ids_jax, _, _, z_jax, pdfs_jax = read_pdfs(jax_pdfs_path)
        ids_cpp, _, _, z_cpp, pdfs_cpp = read_pdfs(cpp_pdfs_path)
        
        if len(ids_jax) != len(ids_cpp):
            print(f"  WARNING: JAX has {len(ids_jax)} objects, C++ has {len(ids_cpp)}", flush=True)
        
        # Align z-bins if files do not cover the same range
        if len(z_cpp) != len(z_jax):
            print(
                f"  WARNING: C++ PDFs have {len(z_cpp)} z-bins, "
                f"JAX has {len(z_jax)}.  Comparing common z-bins only.",
                flush=True,
            )
            # Find common z-bins (round to 5 decimals for float comparison)
            z_jax_rounded = np.round(z_jax.astype(float), 5)
            z_cpp_rounded = np.round(z_cpp.astype(float), 5)
            
            # Keep only z-bins that appear in both
            common_z = np.intersect1d(z_jax_rounded, z_cpp_rounded)
            keep_jax = np.isin(z_jax_rounded, common_z)
            keep_cpp = np.isin(z_cpp_rounded, common_z)
            
            pdfs_jax_cmp = pdfs_jax[:, keep_jax]
            pdfs_cpp_cmp = pdfs_cpp[:, keep_cpp]
            z_bins_cmp = z_jax[keep_jax]
            print(f"  Comparing {keep_jax.sum()} common z-bins.", flush=True)
        else:
            pdfs_jax_cmp = pdfs_jax
            pdfs_cpp_cmp = pdfs_cpp
            z_bins_cmp = z_jax
        
        diff = np.abs(pdfs_jax_cmp - pdfs_cpp_cmp)
        print(
            f"  |ΔPDF|  max={diff.max():.3e}  "
            f"mean={diff.mean():.3e}  "
            f"median={np.median(diff):.3e}",
            flush=True,
        )
        
        # Relative diff only where C++ PDF is significant (> 1e-10)
        mask_significant = pdfs_cpp_cmp > 1e-10
        if mask_significant.any():
            rel_diff_sig = diff[mask_significant] / pdfs_cpp_cmp[mask_significant]
            print(
                f"  |ΔPDF|/|PDF_cpp| (where P_cpp > 1e-10)  "
                f"max={rel_diff_sig.max():.3e}  mean={rel_diff_sig.mean():.3e}",
                flush=True,
            )
        else:
            print(f"  |ΔPDF|/|PDF_cpp|  (no significant PDFs to compare)", flush=True)
        
        # Save PDF comparison plot if requested
        if plot_dir:
            _plot_pdf_comparison(
                pdfs_jax_cmp, pdfs_cpp_cmp, z_bins_cmp,
                plot_dir
            )
    
    # --- Secondary: PDF normalization diagnostics ---------------------------
    if jax_pdfs_path and cpp_pdfs_path and pdfs_jax_cmp is not None and pdfs_cpp_cmp is not None:
        print("\n--- PDF Normalization Diagnostics ---", flush=True)
        
        # Check PDF sums (should match between JAX and C++)
        jax_pdf_sums = np.sum(pdfs_jax_cmp, axis=-1)
        cpp_pdf_sums = np.sum(pdfs_cpp_cmp, axis=-1)
        
        print(f"  JAX PDF sums:  mean={np.mean(jax_pdf_sums):.3e}  "
              f"median={np.median(jax_pdf_sums):.3e}  "
              f"range=[{np.min(jax_pdf_sums):.3e}, {np.max(jax_pdf_sums):.3e}]", flush=True)
        print(f"  C++ PDF sums:  mean={np.mean(cpp_pdf_sums):.3e}  "
              f"median={np.median(cpp_pdf_sums):.3e}  "
              f"range=[{np.min(cpp_pdf_sums):.3e}, {np.max(cpp_pdf_sums):.3e}]", flush=True)
        
        # Ratio of PDF sums
        valid_both_sums = (jax_pdf_sums > 0) & (cpp_pdf_sums > 0)
        if valid_both_sums.sum() > 0:
            ratio_sums = jax_pdf_sums[valid_both_sums] / cpp_pdf_sums[valid_both_sums]
            print(f"  JAX/C++ PDF sum ratio:  mean={np.mean(ratio_sums):.6f}  "
                  f"median={np.median(ratio_sums):.6f}  "
                  f"std={np.std(ratio_sums):.6f}", flush=True)
            if np.abs(np.median(ratio_sums) - 1.0) > 0.01:
                print(f"  WARNING: PDF normalizations differ by ~{100*(np.median(ratio_sums)-1):.2f}%", flush=True)
                print(f"           This could affect quality cuts and selection.", flush=True)
    
    # --- Tertiary: moment comparison (derive from PDFs for consistency) -----
    jax_data, cpp_data, valid_both = None, None, None
    jax_zmean_pdf, cpp_zmean_pdf = None, None  # moments derived from PDFs
    jax_zstd_pdf, cpp_zstd_pdf = None, None
    
    if derive_from_pdfs and jax_pdfs_path and cpp_pdfs_path:
        print("\n--- Moment comparison (deriving from PDFs for apples-to-apples) ---", flush=True)
        print("  NOTE: Using PDF-derived moments for both JAX and C++ to ensure consistent computation.", flush=True)
        
        # Derive moments from PDFs for both
        ids_jax_mom, _, _, z_jax_mom, pdfs_jax_mom = read_pdfs(jax_pdfs_path)
        ids_cpp_mom, _, _, z_cpp_mom, pdfs_cpp_mom = read_pdfs(cpp_pdfs_path)
        
        # Align z-bins if needed (same logic as PDF comparison)
        if len(z_cpp_mom) != len(z_jax_mom):
            z_jax_rounded = np.round(z_jax_mom.astype(float), 5)
            z_cpp_rounded = np.round(z_cpp_mom.astype(float), 5)
            common_z = np.intersect1d(z_jax_rounded, z_cpp_rounded)
            keep_jax = np.isin(z_jax_rounded, common_z)
            keep_cpp = np.isin(z_cpp_rounded, common_z)
            pdfs_jax_mom = pdfs_jax_mom[:, keep_jax]
            pdfs_cpp_mom = pdfs_cpp_mom[:, keep_cpp]
            z_aligned = z_jax_mom[keep_jax]
        else:
            z_aligned = z_jax_mom
        
        print(f"  Deriving JAX moments from PDFs...", flush=True)
        jax_moments = _moments_from_pdfs(z_aligned, pdfs_jax_mom)
        print(f"  Deriving C++ moments from PDFs...", flush=True)
        cpp_moments = _moments_from_pdfs(z_aligned, pdfs_cpp_mom)
        
        # Store for later use
        jax_zmean_pdf = jax_moments["z_mean"]
        jax_zstd_pdf = jax_moments["z_std"]
        cpp_zmean_pdf = cpp_moments["z_mean"]
        cpp_zstd_pdf = cpp_moments["z_std"]
        
        # Selection diagnostics
        valid_jax = jax_moments["z_mean"] > -8.0
        valid_cpp = cpp_moments["z_mean"] > -8.0
        valid_both = valid_jax & valid_cpp
        
        print(f"  Valid objects:  JAX={valid_jax.sum()}  C++={valid_cpp.sum()}  both={valid_both.sum()}  total={len(valid_jax)}", flush=True)
        if valid_jax.sum() != valid_cpp.sum():
            n_jax_only = (valid_jax & ~valid_cpp).sum()
            n_cpp_only = (valid_cpp & ~valid_jax).sum()
            print(f"  WARNING: Selection differs!  JAX-only={n_jax_only}  C++-only={n_cpp_only}", flush=True)
            print(f"           This suggests different quality cuts or numerical precision issues.", flush=True)
        
        # Compare moments
        comparisons_pdf = [
            ("z_mean",       "z_mean"),
            ("z_std",        "z_std"),
            ("skewness",     "skewness"),
            ("kurtosis",     "kurtosis"),
            ("pdf_sum",      "pdf_sum"),
            ("peak_density", "peak_density"),
            ("z_peak",       "z_peak"),
        ]
        
        for name, key in comparisons_pdf:
            jax_v = jax_moments[key][valid_both]
            cpp_v = cpp_moments[key][valid_both]
            fm = np.isfinite(jax_v) & np.isfinite(cpp_v) & (cpp_v > -8.0) & (jax_v > -8.0)
            if not fm.any():
                print(f"  {name:<20s}  (no finite values)", flush=True)
                continue
            d = np.abs(jax_v[fm] - cpp_v[fm])
            n_cmp = len(d)
            n_skip = valid_both.sum() - n_cmp
            status = f" ({n_cmp} objs" + (f", {n_skip} invalid)" if n_skip > 0 else ")")
            print(
                f"  {name:<20s}  max|Δ|={d.max():.3e}  "
                f"mean|Δ|={d.mean():.3e}{status}",
                flush=True,
            )
        
        # Package as "data" arrays for compatibility with plotting code below
        N_obj = len(jax_moments["z_mean"])
        jax_data = np.full((N_obj, 15), -9.0, dtype=np.float64)
        cpp_data = np.full((N_obj, 15), -9.0, dtype=np.float64)
        
        jax_data[:, 3] = jax_moments["z_mean"]
        jax_data[:, 4] = jax_moments["z_std"]
        jax_data[:, 5] = jax_moments["skewness"]
        jax_data[:, 6] = jax_moments["kurtosis"]
        jax_data[:, 7] = jax_moments["pdf_sum"]
        
        cpp_data[:, 3] = cpp_moments["z_mean"]
        cpp_data[:, 4] = cpp_moments["z_std"]
        cpp_data[:, 5] = cpp_moments["skewness"]
        cpp_data[:, 6] = cpp_moments["kurtosis"]
        cpp_data[:, 7] = cpp_moments["pdf_sum"]

        if plot_dir:
            _plot_jax_vs_cpp_scatter(
                jax_zmean_pdf[valid_both], cpp_zmean_pdf[valid_both], plot_dir
            )
            _plot_moment_comparison(jax_data, cpp_data, valid_both, [
                ("z_mean", 3), ("z_std", 4), ("skewness", 5),
                ("kurtosis", 6), ("pdf_sum", 7),
            ], plot_dir)

    elif jax_redshifts_path and cpp_redshifts_path:
        print("\n--- Moment comparison ---", flush=True)
        jax_data = np.loadtxt(jax_redshifts_path)
        cpp_data = np.loadtxt(cpp_redshifts_path)
        
        if jax_data.ndim == 1:
            jax_data = jax_data[None, :]
        if cpp_data.ndim == 1:
            cpp_data = cpp_data[None, :]
        
        # Assumes row-by-row correspondence (same order as input photometry)
        # Note: IDs may differ due to C++ formatting bug; comparison is order-based.
        if len(jax_data) != len(cpp_data):
            print(f"  WARNING: row count mismatch (JAX={len(jax_data)}, C++={len(cpp_data)})", flush=True)
        
        # Objects with valid z estimates (not sentinel -9)
        valid_both = (cpp_data[:, 3] > -8.0) & (jax_data[:, 3] > -8.0)
        print(f"  Valid objects for comparison: {valid_both.sum()} / {len(valid_both)}", flush=True)
        
        # (column_name, column index)
        comparisons = [
            ("z_mean",       3),
            ("z_std",        4),
            ("skewness",     5),
            ("kurtosis",     6),
            ("pdf_sum",      7),
            ("peak_density", 8),
            ("z_peak",       9),
            ("chi2_at_peak", 10),
            ("min_chi2",     11),
            ("mean_chi2",    12),
        ]
        
        for name, cidx in comparisons:
            jax_vals = jax_data[valid_both, cidx]
            cpp_vals = cpp_data[valid_both, cidx]
            
            # Filter out inf/nan values
            finite_mask = np.isfinite(jax_vals) & np.isfinite(cpp_vals)
            if not finite_mask.any():
                print(f"  {name:<20s}  (no finite values to compare)", flush=True)
                continue
            
            jax_vals_clean = jax_vals[finite_mask]
            cpp_vals_clean = cpp_vals[finite_mask]
            d = np.abs(jax_vals_clean - cpp_vals_clean)
            
            n_compared = len(d)
            n_skipped = len(jax_vals) - n_compared
            status = f" ({n_compared} objs" + (f", {n_skipped} inf/nan)" if n_skipped > 0 else ")")
            
            print(
                f"  {name:<20s}  max|Δ|={d.max():.3e}  "
                f"mean|Δ|={d.mean():.3e}{status}",
                flush=True,
            )
        
        # Save moment comparison plots if requested
        if plot_dir:
            _plot_moment_comparison(jax_data, cpp_data, valid_both, comparisons, plot_dir)
            _plot_jax_vs_cpp_scatter(
                jax_data[valid_both, 3], cpp_data[valid_both, 3], plot_dir
            )

    elif jax_redshifts_path and cpp_pdfs_path:
        # No C++ moments file — derive moments from C++ PDFs for apples-to-apples comparison.
        print("\n--- Moment comparison (derived from C++ PDFs) ---", flush=True)
        print("  Note: chi2 columns skipped (not available from PDF file).", flush=True)
        _, _, _, z_cpp_m, pdfs_cpp_m = read_pdfs(cpp_pdfs_path)
        cpp_mom = _moments_from_pdfs(z_cpp_m, pdfs_cpp_m)

        jax_data = np.loadtxt(jax_redshifts_path)
        if jax_data.ndim == 1:
            jax_data = jax_data[None, :]

        N_m = len(cpp_mom["z_mean"])
        cpp_data = np.full((N_m, 15), -9.0, dtype=np.float64)
        cpp_data[:, 3] = cpp_mom["z_mean"]
        cpp_data[:, 4] = cpp_mom["z_std"]
        cpp_data[:, 5] = cpp_mom["skewness"]
        cpp_data[:, 6] = cpp_mom["kurtosis"]
        cpp_data[:, 7] = cpp_mom["pdf_sum"]
        cpp_data[:, 8] = cpp_mom["peak_density"]
        cpp_data[:, 9] = cpp_mom["z_peak"]

        valid_both = (cpp_data[:, 3] > -8.0) & (jax_data[:, 3] > -8.0)
        print(f"  Valid objects for comparison: {valid_both.sum()} / {len(valid_both)}", flush=True)

        pdf_comparisons = [
            ("z_mean",       3),
            ("z_std",        4),
            ("skewness",     5),
            ("kurtosis",     6),
            ("pdf_sum",      7),
            ("peak_density", 8),
            ("z_peak",       9),
        ]
        for name, cidx in pdf_comparisons:
            jax_v = jax_data[valid_both, cidx]
            cpp_v = cpp_data[valid_both, cidx]
            fm = np.isfinite(jax_v) & np.isfinite(cpp_v) & (cpp_v > -8.0)
            if not fm.any():
                print(f"  {name:<20s}  (no finite values)", flush=True)
                continue
            d = np.abs(jax_v[fm] - cpp_v[fm])
            n_cmp = len(d)
            n_skip = valid_both.sum() - n_cmp
            status = f" ({n_cmp} objs" + (f", {n_skip} nan)" if n_skip > 0 else ")")
            print(
                f"  {name:<20s}  max|Δ|={d.max():.3e}  "
                f"mean|Δ|={d.mean():.3e}{status}",
                flush=True,
            )
        if plot_dir:
            _plot_moment_comparison(jax_data, cpp_data, valid_both, pdf_comparisons, plot_dir)
            _plot_jax_vs_cpp_scatter(
                jax_data[valid_both, 3], cpp_data[valid_both, 3], plot_dir
            )

    # --- True redshift comparison (if provided) ----------------------------
    if true_z_path:
        print("\n--- Photo-z vs True Redshift comparison ---", flush=True)
        try:
            # catgrid_info format: col 0 = object ID, col 4 = true redshift.
            # The file covers the full catalog; join by ID to handle subsets.
            _info = np.loadtxt(true_z_path, usecols=(0, 4))
            _id_to_truez = {int(round(row[0])): row[1] for row in _info}
            print(f"  Loaded {len(_id_to_truez)} id→z entries from {os.path.basename(true_z_path)}", flush=True)

            # Resolve object IDs for this comparison run.
            # ids_jax_mom is set by the derive_from_pdfs branch; ids_jax by the
            # PDF comparison branch.  Fall back to a fresh read if neither exists.
            if "ids_jax_mom" in dir() and ids_jax_mom is not None:
                _obj_ids = ids_jax_mom
            elif "ids_jax" in dir() and ids_jax is not None:
                _obj_ids = ids_jax
            elif jax_pdfs_path:
                _obj_ids, _, _, _, _ = read_pdfs(jax_pdfs_path)
            else:
                _obj_ids, _, _, _, _ = read_pdfs(cpp_pdfs_path)

            true_z = np.array([_id_to_truez.get(int(round(float(i))), np.nan)
                                for i in _obj_ids])
            n_matched = int(np.isfinite(true_z).sum())
            n_missing = len(true_z) - n_matched
            print(f"  Matched {n_matched}/{len(true_z)} objects by ID"
                  + (f"  ({n_missing} IDs not found in catgrid_info)" if n_missing else ""),
                  flush=True)
            
            # Compare JAX vs true if JAX redshifts available
            jax_zmean, jax_zstd = None, None
            if jax_zmean_pdf is not None and jax_zstd_pdf is not None:
                # Use PDF-derived moments if available (more reliable)
                print(f"  Using PDF-derived JAX moments for true-z comparison", flush=True)
                jax_zmean = jax_zmean_pdf
                jax_zstd = jax_zstd_pdf
            elif jax_redshifts_path:
                jax_data_true = np.loadtxt(jax_redshifts_path) if 'jax_data' not in locals() else jax_data
                if jax_data_true.ndim == 1:
                    jax_data_true = jax_data_true[None, :]
                jax_zmean = jax_data_true[:, 3]
                jax_zstd = jax_data_true[:, 4]  # z_std (uncertainty)
                # Filter outliers: remove invalid redshifts (z < 0 or z > 10) and true_z <= 0
                valid_jax_true = (jax_zmean >= 0) & (jax_zmean < 10) & (true_z > 0) & (true_z < 10)
                if valid_jax_true.sum() > 0:
                    dz_jax = jax_zmean[valid_jax_true] - true_z[valid_jax_true]
                    dz_norm_jax = dz_jax / (1.0 + true_z[valid_jax_true])
                    bias_jax = np.median(dz_norm_jax)
                    nmad_jax = 1.4821 * np.median(np.abs(dz_norm_jax - bias_jax))
                    outlier_jax = np.abs(dz_norm_jax) > 0.15
                    print(f"  JAX:  {valid_jax_true.sum()} valid  "
                          f"bias={bias_jax:.4f}  "
                          f"NMAD={nmad_jax:.4f}  "
                          f"outliers={outlier_jax.sum()} ({100*outlier_jax.mean():.1f}%)",
                          flush=True)
            
            # Compare C++ vs true if C++ redshifts available
            cpp_zmean, cpp_zstd = None, None
            if cpp_zmean_pdf is not None and cpp_zstd_pdf is not None:
                # Use PDF-derived moments if available (more reliable)
                print(f"  Using PDF-derived C++ moments for true-z comparison", flush=True)
                cpp_zmean = cpp_zmean_pdf
                cpp_zstd = cpp_zstd_pdf
            elif cpp_redshifts_path or (jax_redshifts_path and cpp_pdfs_path):
                if 'cpp_data' in locals() and cpp_data is not None:
                    cpp_zmean = cpp_data[:, 3]
                    cpp_zstd = cpp_data[:, 4]  # z_std (uncertainty)
                elif cpp_redshifts_path:
                    cpp_data_true = np.loadtxt(cpp_redshifts_path)
                    if cpp_data_true.ndim == 1:
                        cpp_data_true = cpp_data_true[None, :]
                    cpp_zmean = cpp_data_true[:, 3]
                    cpp_zstd = cpp_data_true[:, 4]
                else:
                    # Derive from PDFs if only PDFs available
                    _, _, _, z_true_cpp, pdfs_true_cpp = read_pdfs(cpp_pdfs_path)
                    cpp_mom_true = _moments_from_pdfs(z_true_cpp, pdfs_true_cpp)
                    cpp_zmean = cpp_mom_true["z_mean"]
                    cpp_zstd = cpp_mom_true["z_std"]
                
                # Filter outliers: remove invalid redshifts (z < 0 or z > 10) and true_z <= 0
                valid_cpp_true = (cpp_zmean >= 0) & (cpp_zmean < 10) & (true_z > 0) & (true_z < 10)
                if valid_cpp_true.sum() > 0:
                    dz_cpp = cpp_zmean[valid_cpp_true] - true_z[valid_cpp_true]
                    dz_norm_cpp = dz_cpp / (1.0 + true_z[valid_cpp_true])
                    bias_cpp = np.median(dz_norm_cpp)
                    nmad_cpp = 1.4821 * np.median(np.abs(dz_norm_cpp - bias_cpp))
                    outlier_cpp = np.abs(dz_norm_cpp) > 0.15
                    print(f"  C++:  {valid_cpp_true.sum()} valid  "
                          f"bias={bias_cpp:.4f}  "
                          f"NMAD={nmad_cpp:.4f}  "
                          f"outliers={outlier_cpp.sum()} ({100*outlier_cpp.mean():.1f}%)",
                          flush=True)
            
            # Generate plots if requested
            if plot_dir and (jax_redshifts_path or cpp_redshifts_path or cpp_pdfs_path):
                # 1. Basic hexbin + histogram comparison
                _plot_truez_comparison(
                    true_z,
                    jax_zmean if jax_redshifts_path else None,
                    cpp_zmean if (cpp_redshifts_path or cpp_pdfs_path) else None,
                    plot_dir
                )
                
                # 2. Multi-panel uncertainty-binned comparison (like PAE analysis)
                if jax_zmean is not None and cpp_zmean is not None and jax_zstd is not None and cpp_zstd is not None:
                    _plot_multisel_comparison(
                        true_z, jax_zmean, jax_zstd, cpp_zmean, cpp_zstd, plot_dir
                    )
                    
                    # 3. Bias vs z_true by uncertainty bins
                    _plot_bias_vs_ztrue(
                        true_z, jax_zmean, jax_zstd, cpp_zmean, cpp_zstd, plot_dir
                    )
        except Exception as e:
            print(f"  WARNING: Could not load true redshifts: {e}", flush=True)

    print("\n=== COMPARISON COMPLETE ===\n", flush=True)


def _plot_pdf_comparison(pdfs_jax, pdfs_cpp, z_bins, plot_dir):
    """Generate PDF comparison plots."""
    print("  Generating PDF comparison plots...", flush=True)
    
    # 1. Population-level: histogram of PDF differences
    diff = np.abs(pdfs_jax - pdfs_cpp)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Absolute differences
    axes[0].hist(diff.flatten(), bins=100, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('|JAX PDF - C++ PDF|')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'PDF Absolute Differences\nmax={diff.max():.3e}, mean={diff.mean():.3e}')
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)
    
    # Relative differences
    rel_diff = diff / (np.abs(pdfs_cpp) + 1e-300)
    rel_diff_masked = rel_diff[pdfs_cpp > 1e-10]  # Only where C++ has significant probability
    if len(rel_diff_masked) > 0:
        axes[1].hist(np.clip(rel_diff_masked, 0, 10), bins=100, alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('|ΔP| / |P_C++| (clipped at 10)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'PDF Relative Differences (P_C++ > 1e-10)\nmax={rel_diff_masked.max():.3e}')
        axes[1].set_yscale('log')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'pdf_diff_histogram.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Per-object: sample PDFs for a few objects
    n_samples = min(12, pdfs_jax.shape[0])
    sample_idx = np.linspace(0, pdfs_jax.shape[0]-1, n_samples, dtype=int)
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    axes = axes.flatten()
    
    for i, obj_idx in enumerate(sample_idx):
        ax = axes[i]
        ax.plot(z_bins, pdfs_cpp[obj_idx], 'b-', label='C++', alpha=0.7, linewidth=1.5)
        ax.plot(z_bins, pdfs_jax[obj_idx], 'r--', label='JAX', alpha=0.7, linewidth=1.5)
        ax.set_xlabel('Redshift')
        ax.set_ylabel('P(z)')
        ax.set_title(f'Object {obj_idx}')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)
    
    plt.suptitle('Sample PDF Comparisons (JAX vs C++)', fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'pdf_samples.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Heatmap: PDF differences vs (object, redshift)
    # Subsample for visualization
    n_obj_plot = min(500, pdfs_jax.shape[0])
    obj_step = max(1, pdfs_jax.shape[0] // n_obj_plot)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(
        diff[::obj_step, :],
        aspect='auto',
        cmap='viridis',
        extent=[z_bins.min(), z_bins.max(), 0, n_obj_plot],
        vmax=np.percentile(diff, 99),
        interpolation='nearest'
    )
    ax.set_xlabel('Redshift')
    ax.set_ylabel('Object Index (subsampled)')
    ax.set_title('PDF Absolute Differences |JAX - C++|')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('|ΔP(z)|')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'pdf_diff_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    Saved: pdf_diff_histogram.png, pdf_samples.png, pdf_diff_heatmap.png", flush=True)


def _plot_moment_comparison(jax_data, cpp_data, valid_both, comparisons, plot_dir):
    """Generate moment comparison plots."""
    print("  Generating moment comparison plots...", flush=True)
    
    # Create multi-panel figure with scatter plots
    n_comp = len(comparisons)
    n_cols = 3
    n_rows = (n_comp + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(15, 4*n_rows))
    gs = GridSpec(n_rows, n_cols, figure=fig)
    
    for i, (name, cidx) in enumerate(comparisons):
        row = i // n_cols
        col = i % n_cols
        ax = fig.add_subplot(gs[row, col])
        
        jax_vals = jax_data[valid_both, cidx]
        cpp_vals = cpp_data[valid_both, cidx]
        
        # Filter for finite values before plotting
        finite_mask = np.isfinite(jax_vals) & np.isfinite(cpp_vals)
        if not finite_mask.any():
            ax.text(0.5, 0.5, f'{name}\n(no finite values)', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        
        jax_vals_clean = jax_vals[finite_mask]
        cpp_vals_clean = cpp_vals[finite_mask]
        diff_clean = jax_vals_clean - cpp_vals_clean
        
        # Scatter plot
        ax.scatter(cpp_vals_clean, jax_vals_clean, alpha=0.3, s=1, rasterized=True)
        
        # 1:1 line and reasonable axis limits based on percentiles
        combined = np.concatenate([cpp_vals_clean, jax_vals_clean])
        p1, p99 = np.percentile(combined, [1.0, 99.0])
        # small guard if data nearly constant
        if not np.isfinite(p1) or not np.isfinite(p99):
            p1, p99 = combined.min(), combined.max()
        if p99 <= p1:
            span = abs(p1) if p1 != 0 else 1.0
            p1 -= 0.5 * span
            p99 += 0.5 * span
        margin = 0.05 * (p99 - p1)
        vmin = p1 - margin
        vmax = p99 + margin
        ax.plot([vmin, vmax], [vmin, vmax], 'r--', alpha=0.5, linewidth=1)
        ax.set_xlim(vmin, vmax)
        ax.set_ylim(vmin, vmax)
        
        ax.set_xlabel(f'C++ {name}')
        ax.set_ylabel(f'JAX {name}')
        ax.set_title(f'{name}\nmax|Δ|={np.abs(diff_clean).max():.3e}')
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        textstr = f'mean Δ: {diff_clean.mean():.3e}\nstd Δ: {diff_clean.std():.3e}'
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
                verticalalignment='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Per-Object Moment Comparisons (JAX vs C++)', fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'moment_scatter_all.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Population-level: histograms of differences
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    if n_comp == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, (name, cidx) in enumerate(comparisons):
        ax = axes[i]
        jax_vals = jax_data[valid_both, cidx]
        cpp_vals = cpp_data[valid_both, cidx]
        
        # Filter for finite values before computing differences
        finite_mask = np.isfinite(jax_vals) & np.isfinite(cpp_vals)
        if not finite_mask.any():
            ax.text(0.5, 0.5, f'{name}\n(no finite values)', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        
        diff = jax_vals[finite_mask] - cpp_vals[finite_mask]
        
        # Only plot if we have reasonable data
        if len(diff) > 0 and np.isfinite(diff).all():
            # Use percentile-based clipping for histogram range to avoid extreme outliers
            d_p1, d_p99 = np.percentile(diff, [1.0, 99.0])
            if not np.isfinite(d_p1) or not np.isfinite(d_p99):
                hist_range = None
            else:
                if d_p99 <= d_p1:
                    d_p1 -= 1.0
                    d_p99 += 1.0
                hist_range = (d_p1 - 0.05 * (d_p99 - d_p1), d_p99 + 0.05 * (d_p99 - d_p1))

            if hist_range is not None:
                ax.hist(diff, bins=100, range=hist_range, alpha=0.7, edgecolor='black')
            else:
                ax.hist(diff, bins=100, alpha=0.7, edgecolor='black')

            ax.set_xlabel(f'JAX - C++')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{name} Differences\nmax|Δ|={np.abs(diff).max():.3e}, mean={diff.mean():.3e}')
            ax.axvline(0, color='r', linestyle='--', alpha=0.5)
            ax.grid(True, alpha=0.3)
            if diff.std() > 0:
                ax.set_yscale('log')
        else:
            ax.text(0.5, 0.5, f'{name}\n(invalid differences)', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
    
    # Hide unused subplots
    for i in range(n_comp, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Population-Level Moment Difference Histograms', fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'moment_diff_histogram.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # N(z) comparison - redshift distribution
    print("  Generating N(z) comparison...", flush=True)
    z_mean_jax = jax_data[valid_both, 3]  # z_mean is column 3
    z_mean_cpp = cpp_data[valid_both, 3]
    
    # Filter for finite z_mean values
    valid_z_jax = (z_mean_jax > -8.0) & np.isfinite(z_mean_jax)
    valid_z_cpp = (z_mean_cpp > -8.0) & np.isfinite(z_mean_cpp)
    
    z_jax_clean = z_mean_jax[valid_z_jax]
    z_cpp_clean = z_mean_cpp[valid_z_cpp]
    
    if len(z_jax_clean) > 0 and len(z_cpp_clean) > 0:
        # Determine common binning using robust percentiles to avoid extreme outliers
        combined_z = np.concatenate([z_jax_clean, z_cpp_clean])
        try:
            z_min, z_max = np.percentile(combined_z, [0.1, 99.9])
        except Exception:
            z_min, z_max = combined_z.min(), combined_z.max()
        if not np.isfinite(z_min) or not np.isfinite(z_max) or z_max <= z_min:
            z_min, z_max = combined_z.min(), combined_z.max()
        # add small margin
        margin = 0.02 * (z_max - z_min) if (z_max - z_min) > 0 else 0.02
        z_min -= margin
        z_max += margin
        z_bins = np.linspace(z_min, z_max, 100)
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Top panel: overlaid histograms
        ax = axes[0]
        ax.hist(z_cpp_clean, bins=z_bins, alpha=0.6, label='C++', color='blue', edgecolor='black')
        ax.hist(z_jax_clean, bins=z_bins, alpha=0.6, label='JAX', color='red', edgecolor='black')
        ax.set_xlabel('z_mean')
        ax.set_ylabel('N (number of objects)')
        ax.set_title(f'Redshift Distribution N(z)\nC++: {len(z_cpp_clean)} objs, JAX: {len(z_jax_clean)} objs')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Bottom panel: fractional difference (JAX - C++) / C++
        ax = axes[1]
        n_cpp, _ = np.histogram(z_cpp_clean, bins=z_bins)
        n_jax, _ = np.histogram(z_jax_clean, bins=z_bins)

        # Avoid division by zero
        valid_bins = n_cpp > 0
        frac_diff = np.zeros_like(n_cpp, dtype=float)
        frac_diff[valid_bins] = (n_jax[valid_bins] - n_cpp[valid_bins]) / n_cpp[valid_bins]

        bin_centers = 0.5 * (z_bins[:-1] + z_bins[1:])
        # Clamp fractional differences for plotting using percentiles and an absolute cap
        finite_fd = frac_diff[np.isfinite(frac_diff)]
        if finite_fd.size > 0:
            q1, q99 = np.percentile(finite_fd, [1.0, 99.0])
            fd_min = max(-2.0, q1 - 0.05 * (q99 - q1))
            fd_max = min(2.0, q99 + 0.05 * (q99 - q1))
            if fd_max <= fd_min:
                fd_min, fd_max = -1.0, 1.0
        else:
            fd_min, fd_max = -1.0, 1.0

        ax.plot(bin_centers, np.clip(frac_diff, fd_min, fd_max), 'o-', color='green', markersize=3)
        ax.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('z_mean')
        ax.set_ylabel('(N_JAX - N_C++) / N_C++')
        ax.set_title('Fractional Difference in N(z) (clipped for plotting)')
        ax.set_ylim(fd_min, fd_max)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'nz_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"    Saved: nz_comparison.png", flush=True)
    
    print(f"    Saved: moment_scatter_all.png, moment_diff_histogram.png", flush=True)


def _plot_jax_vs_cpp_scatter(z_jax, z_cpp, plot_dir):
    """One-to-one scatter plot of JAX vs C++ recovered redshifts with residual panel."""
    print("  Generating JAX vs C++ redshift scatter plot...", flush=True)

    valid = np.isfinite(z_jax) & np.isfinite(z_cpp) & (z_jax > -8.0) & (z_cpp > -8.0)
    zj = z_jax[valid]
    zc = z_cpp[valid]

    if len(zj) == 0:
        print("  WARNING: no valid objects for JAX vs C++ scatter plot", flush=True)
        return

    resid = zj - zc
    bias  = np.median(resid)
    nmad  = 1.4821 * np.median(np.abs(resid - bias))
    outlier_frac = np.mean(np.abs(resid) > 0.1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                             gridspec_kw={"width_ratios": [3, 1]})

    # Left: hexbin scatter
    ax = axes[0]
    z_all = np.concatenate([zj, zc])
    p1, p99 = np.percentile(z_all, [0.5, 99.5])
    if not (np.isfinite(p1) and np.isfinite(p99) and p99 > p1):
        p1, p99 = z_all.min(), z_all.max()
    margin = 0.05 * (p99 - p1)
    vmin, vmax = p1 - margin, p99 + margin

    hb = ax.hexbin(zc, zj, gridsize=80, cmap="plasma", bins="log",
                   extent=[vmin, vmax, vmin, vmax])
    ax.plot([vmin, vmax], [vmin, vmax], "r--", linewidth=1.5, label="1:1")
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_xlabel(r"$z_\mathrm{C++}$", fontsize=13)
    ax.set_ylabel(r"$z_\mathrm{JAX}$", fontsize=13)
    ax.set_title(f"JAX vs C++ recovered photo-z  ({len(zj):,} objects)", fontsize=13)
    plt.colorbar(hb, ax=ax, label="count (log scale)")
    ax.legend(fontsize=11)
    stats_text = (
        f"bias  = {bias:+.5f}\n"
        f"NMAD  = {nmad:.5f}\n"
        f"out.  = {100*outlier_frac:.2f}%  (|Δz|>0.1)"
    )
    ax.text(0.03, 0.97, stats_text, transform=ax.transAxes,
            va="top", ha="left", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.grid(True, alpha=0.3)

    # Right: residual histogram (horizontal)
    ax2 = axes[1]
    rng = np.percentile(np.abs(resid), 99.5) * 1.5
    rng = max(rng, 1e-6)  # guard against constant data
    bins = np.linspace(-rng, rng, 80)
    ax2.hist(resid, bins=bins, orientation="horizontal",
             alpha=0.7, edgecolor="black", density=True)
    ax2.axhline(0,    color="r",     linestyle="--", linewidth=1.5)
    ax2.axhline(bias, color="green", linestyle=":",  linewidth=1.5,
                label=f"median={bias:+.4f}")
    ax2.set_ylabel(r"$z_\mathrm{JAX} - z_\mathrm{C++}$", fontsize=12)
    ax2.set_xlabel("density", fontsize=11)
    ax2.set_title("Residuals", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(plot_dir, "jax_vs_cpp_redshift_scatter.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: jax_vs_cpp_redshift_scatter.png", flush=True)


def _plot_truez_comparison(true_z, jax_zmean, cpp_zmean, plot_dir):
    """Generate photo-z vs true-z comparison plots (hexbin scatter + histogram)."""
    print("  Generating photo-z vs true-z plots...", flush=True)
    
    has_jax = jax_zmean is not None
    has_cpp = cpp_zmean is not None
    n_methods = int(has_jax) + int(has_cpp)
    
    if n_methods == 0:
        return
    
    # Create figure with 2 rows: top=hexbin scatter, bottom=histogram
    fig = plt.figure(figsize=(6*n_methods, 10))
    gs = GridSpec(2, n_methods, height_ratios=[1.2, 1], hspace=0.3, wspace=0.3)
    
    method_idx = 0
    dz_norm_all = []  # Store for histogram comparison
    labels_all = []
    
    for name, z_est in [("JAX", jax_zmean), ("C++", cpp_zmean)]:
        if z_est is None:
            continue
        
        # Filter outliers: remove invalid redshifts (z < 0 or z > 10)
        valid = (z_est >= 0) & (z_est < 10) & (true_z > 0) & (true_z < 10)
        z_true_v = true_z[valid]
        z_est_v = z_est[valid]
        dz = z_est_v - z_true_v
        dz_norm = dz / (1.0 + z_true_v)
        
        # Store for histogram
        dz_norm_all.append(dz_norm)
        labels_all.append(name)
        
        # --- Top panel: hexbin scatter plot ---
        ax_scatter = fig.add_subplot(gs[0, method_idx])
        
        # Hexbin plot (log scale)
        hb = ax_scatter.hexbin(
            z_true_v, z_est_v,
            gridsize=50, cmap='magma', mincnt=1, bins='log',
            extent=[0, 3, 0, 3]
        )
        
        # 1:1 line
        ax_scatter.plot([0, 3], [0, 3], 'k--', lw=2, alpha=0.5, label='1:1')
        
        # Statistics (using NMAD like PAE analysis)
        bias = np.median(dz_norm)
        nmad = 1.4821 * np.median(np.abs(dz_norm - bias))
        outliers = (np.abs(dz_norm) > 0.15).sum()
        outlier_frac = 100 * outliers / len(dz_norm)
        
        # Annotate with stats
        stats_text = (
            f'N = {len(dz_norm)}\n'
            f'bias = {bias:.4f}\n'
            f'NMAD = {nmad:.4f}\n'
            f'$f_{{outlier}}^{{15\%}} = {outlier_frac:.1f}$%'
        )
        ax_scatter.text(
            0.05, 0.95, stats_text, transform=ax_scatter.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray', linewidth=0.5)
        )
        
        ax_scatter.set_xlabel('True Redshift', fontsize=13)
        ax_scatter.set_ylabel(f'{name} Photo-z', fontsize=13)
        ax_scatter.set_title(f'{name} vs True Redshift', fontsize=14, fontweight='bold')
        ax_scatter.set_xlim(0, 3)
        ax_scatter.set_ylim(0, 3)
        ax_scatter.grid(True, alpha=0.3)
        ax_scatter.legend(loc='lower right', fontsize=10)
        
        # Colorbar
        cbar = plt.colorbar(hb, ax=ax_scatter, label='log$_{10}$(N objects)')
        
        method_idx += 1
    
    # --- Bottom panel: Δz/(1+z) histogram comparison ---
    ax_hist = fig.add_subplot(gs[1, :])
    
    colors_hist = ['#1f77b4', '#ff7f0e']  # blue for JAX, orange for C++
    for i, (dz_norm, label) in enumerate(zip(dz_norm_all, labels_all)):
        ax_hist.hist(
            dz_norm, bins=100, range=(-0.3, 0.3),
            alpha=0.6, color=colors_hist[i], label=label,
            edgecolor='black', linewidth=0.5
        )
    
    ax_hist.axvline(0, color='k', linestyle='--', lw=2, alpha=0.5, label='Zero bias')
    ax_hist.set_xlabel('$\Delta z / (1 + z_{true})$', fontsize=13)
    ax_hist.set_ylabel('N objects', fontsize=13)
    ax_hist.set_title('Redshift Error Distribution', fontsize=14, fontweight='bold')
    ax_hist.legend(fontsize=12)
    ax_hist.grid(True, alpha=0.3, axis='y')
    
    plt.savefig(os.path.join(plot_dir, 'photoz_vs_truez.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("    Saved: photoz_vs_truez.png", flush=True)


def _plot_multisel_comparison(true_z, jax_zmean, jax_zstd, cpp_zmean, cpp_zstd, plot_dir):
    """
    Multi-panel uncertainty-binned comparison (like PAE compare_pae_tf_redshifts_multisel).
    
    Creates 2×4 grid:
    - Row 0: JAX results in different σ_z/(1+z) bins
    - Row 1: C++ results in different σ_z/(1+z) bins
    - Columns: [<0.01, <0.03, <0.1, <0.2]
    """
    print("  Generating multi-panel uncertainty-binned comparison...", flush=True)
    
    sig_z_thresholds = [0.01, 0.03, 0.1, 0.2]
    n_cols = len(sig_z_thresholds)
    z_min, z_max = 0.0, 3.0
    gridsize = 50
    
    linsp = np.linspace(z_min, z_max, 100)
    
    fig, axes = plt.subplots(nrows=2, ncols=n_cols, figsize=(14, 7))
    fig.subplots_adjust(hspace=0.12, wspace=0.12, top=0.94, bottom=0.08,
                        left=0.07, right=0.90)
    
    hb_last = None
    method_data = [
        (jax_zmean, jax_zstd, 'JAX', 0),
        (cpp_zmean, cpp_zstd, 'C++', 1),
    ]
    
    for col, threshold in enumerate(sig_z_thresholds):
        for zmean, zstd, method, row in method_data:
            ax = axes[row, col]
            
            # Fractional uncertainty
            sigz = zstd / (1.0 + zmean)
            
            # Filter: valid redshifts + uncertainty threshold
            m = (zmean >= 0) & (zmean < 10) & (true_z > 0) & (true_z < 10) & \
                np.isfinite(zmean) & np.isfinite(true_z) & np.isfinite(sigz) & \
                (sigz < threshold)
            
            z_hat = zmean[m]
            zt_m = true_z[m]
            sz_m = sigz[m]
            
            ax.plot(linsp, linsp, 'k--', lw=1.5, alpha=0.5, zorder=10)
            ax.set_xlim(z_min, z_max)
            ax.set_ylim(z_min, z_max)
            ax.grid(alpha=0.3)
            
            if len(z_hat) > 0:
                hb = ax.hexbin(zt_m, z_hat, gridsize=gridsize, cmap='magma',
                               mincnt=1, bins='log',
                               extent=[z_min, z_max, z_min, z_max])
                hb_last = hb
                
                # Compute statistics
                dz_norm = (z_hat - zt_m) / (1.0 + zt_m)
                bias = np.median(dz_norm)
                nmad = 1.4821 * np.median(np.abs(dz_norm - bias))
                # 3-sigma outlier fraction (robust sigma estimate using NMAD)
                outlier_mask = np.abs(dz_norm - bias) > 3.0 * nmad
                outl_rate = outlier_mask.sum() / len(dz_norm)
                med_sigz = np.median(sz_m)
                n_src = int(m.sum())
                
                # Stats text (upper left)
                stat_str = (f'N = {n_src}\n'
                            f'bias = {bias:.4f}\n'
                            f'NMAD = {nmad:.4f}\n'
                            f'$\\sigma_{{med}}$ = {med_sigz:.4f}\n'
                            f'$f_{{out}}^{{3\\sigma}}$ = {100*outl_rate:.1f}%')
                ax.text(0.05, 0.96, stat_str, fontsize=8,
                        transform=ax.transAxes, va='top',
                        bbox=dict(boxstyle='round', facecolor='white',
                                  alpha=0.85, edgecolor='gray', lw=0.5))
            else:
                ax.text(0.5, 0.5, 'No sources', ha='center', va='center',
                        transform=ax.transAxes, fontsize=10)
            
            # Selection label (bottom right, beige box)
            sel_label = f'$\\sigma_{{z/(1+z)}}^{{{method}}} < {threshold}$'
            ax.text(0.97, 0.04, sel_label, fontsize=9,
                    transform=ax.transAxes, ha='right', va='bottom',
                    bbox=dict(boxstyle='round', facecolor='cornsilk',
                              alpha=0.95, edgecolor='gray', lw=0.5))
            
            # Axis labels
            if col == 0:
                ax.set_ylabel(f'$\\hat{{z}}_{{{method}}}$', fontsize=13)
            else:
                ax.tick_params(labelleft=False)
            
            if row == 1:
                ax.set_xlabel('$z_{{\\rm true}}$', fontsize=13)
            else:
                ax.tick_params(labelbottom=False)
    
    # Shared colorbar
    if hb_last is not None:
        cbar_ax = fig.add_axes([0.92, 0.08, 0.015, 0.86])
        fig.colorbar(hb_last, cax=cbar_ax, label='log$_{10}$(counts)')
    
    plt.savefig(os.path.join(plot_dir, 'photoz_vs_truez_multisel.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("    Saved: photoz_vs_truez_multisel.png", flush=True)


def _plot_bias_vs_ztrue(true_z, jax_zmean, jax_zstd, cpp_zmean, cpp_zstd, plot_dir):
    """Bias vs z_true by uncertainty bins (like PAE plot_bias_vs_ztrue_by_sigz_bins)."""
    print("  Generating bias vs z_true plots...", flush=True)
    
    sigz_bin_edges = [0, 0.003, 0.01, 0.03, 0.1, 0.2]
    n_sigma_bins = len(sigz_bin_edges) - 1
    colors_list = plt.cm.viridis(np.linspace(0.1, 0.9, n_sigma_bins))
    
    z_min, z_max = 0.0, 3.0
    n_ztrue_bins = 12
    ztrue_edges = np.linspace(z_min, z_max, n_ztrue_bins + 1)
    ztrue_centers = 0.5 * (ztrue_edges[:-1] + ztrue_edges[1:])
    
    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(13, 5), sharey=True)
    
    method_data = [
        (jax_zmean, jax_zstd, 'JAX', 0),
        (cpp_zmean, cpp_zstd, 'C++', 1),
    ]
    
    for zmean, zstd, method, idx in method_data:
        ax = axes[idx]
        
        # Fractional uncertainty
        sigz = zstd / (1.0 + zmean)
        
        # Valid data
        valid = (zmean >= 0) & (zmean < 10) & (true_z > 0) & (true_z < 10) & \
                np.isfinite(zmean) & np.isfinite(true_z) & np.isfinite(sigz)
        
        bias_all = (zmean - true_z) / (1.0 + true_z)
        
        for k in range(n_sigma_bins):
            lo, hi = sigz_bin_edges[k], sigz_bin_edges[k + 1]
            m = valid & (sigz >= lo) & (sigz < hi)
            
            med_bias = np.full(n_ztrue_bins, np.nan)
            pct16 = np.full(n_ztrue_bins, np.nan)
            pct84 = np.full(n_ztrue_bins, np.nan)
            
            for j in range(n_ztrue_bins):
                zlo, zhi = ztrue_edges[j], ztrue_edges[j + 1]
                in_zbin = m & (true_z >= zlo) & (true_z < zhi)
                n_in_bin = in_zbin.sum()
                
                if n_in_bin >= 5:
                    b = bias_all[in_zbin]
                    med_bias[j] = np.median(b)
                    pct16[j] = np.percentile(b, 16)
                    pct84[j] = np.percentile(b, 84)
            
            # Build label
            if lo == 0:
                lbl = f'$\\sigma < {hi}$  (N={m.sum()})'
            else:
                lbl = f'${lo} \\leq \\sigma < {hi}$  (N={m.sum()})'
            
            good = np.isfinite(med_bias)
            if not good.any():
                continue
            
            ax.plot(ztrue_centers[good], med_bias[good],
                    color=colors_list[k], lw=2, label=lbl)
            ax.fill_between(ztrue_centers[good], pct16[good], pct84[good],
                            color=colors_list[k], alpha=0.15)
        
        ax.axhline(0, color='k', lw=1.5, ls='--', alpha=0.6)
        ax.set_xlim(z_min, z_max)
        ax.set_xlabel('$z_{{\\rm true}}$', fontsize=14)
        ax.set_title(method, fontsize=15, fontweight='bold')
        ax.grid(alpha=0.35)
        ax.legend(fontsize=9, loc='lower right', framealpha=0.9)
    
    axes[0].set_ylabel('Median bias $(\\hat{z} - z_{{\\rm true}})/(1 + z_{{\\rm true}})$',
                       fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'bias_vs_ztrue_by_sigz.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("    Saved: bias_vs_ztrue_by_sigz.png", flush=True)


# ===========================================================================
# Section 6: Full pipeline runner
# ===========================================================================

def run(
    model_grid_paths,
    obs_phot_path: str,
    template_prob_path: str,
    output_redshift_path: str,
    output_pdfs_path: Optional[str] = None,
    batch_obj: int = 10_000,
    multicore: bool = False,
    precision: str = 'float64',
    min_source_snr: Optional[float] = None,
    source_snr_mode: str = 'quadrature',
    source_snr_max_filters: Optional[int] = None,
) -> None:
    """End-to-end: parse grid → read obs → fit → moments → save."""

    t_total = time.time()
    print("\n=== SPHEREx Photo-z  (JAX/GPU) ===\n", flush=True)

    # 1. Parse model grid
    print("[1/4] Parsing model grid...", flush=True)
    grid         = parse_photoz_grid(model_grid_paths)
    redshifts    = grid["redshifts"]
    model_fluxes = grid["model_fluxes"]
    template_ids = grid["template_ids"]
    n_filters    = grid["n_filters"]

    # 2. Read observations
    print("\n[2/4] Reading observed photometry...", flush=True)
    ids, xpos, ypos, obs_fluxes, obs_weights, obj_stats = read_obs_phot(obs_phot_path)
    ids, xpos, ypos, obs_fluxes, obs_weights, obj_stats = _apply_source_snr_filter(
        ids,
        xpos,
        ypos,
        obs_fluxes,
        obs_weights,
        obj_stats,
        min_source_snr=min_source_snr,
        source_snr_mode=source_snr_mode,
        source_snr_max_filters=source_snr_max_filters,
    )

    if obs_fluxes.shape[1] != n_filters:
        raise ValueError(
            f"Obs photometry has {obs_fluxes.shape[1]} filters but grid has {n_filters}."
        )

    # 3. Read template probabilities
    template_prob_array = read_template_probs(
        template_prob_path,
        max_template_id=int(template_ids.max()),
    )

    # 4. GPU fitting
    print("\n[3/4] Computing PDFs on GPU...", flush=True)
    pdfs, min_chi2_perz, best_tmpl_perz = compute_pdfs_jax(
        obs_fluxes, obs_weights,
        model_fluxes, template_ids,
        template_prob_array,
        batch_obj=batch_obj,
        multicore=multicore,
        precision=precision,
    )

    # Diagnostic: check PDF output sanity before moment computation
    pdf_sums = pdfs.sum(axis=-1)  # [N_obj]
    n_nonzero_pdf = (pdf_sums > 0).sum()
    print(f"  [diag] PDF sums > 0: {n_nonzero_pdf} / {len(pdf_sums)}", flush=True)
    print(f"  [diag] PDF sums range: [{pdf_sums.min():.3e}, {pdf_sums.max():.3e}]", flush=True)
    print(f"  [diag] pdfs shape: {pdfs.shape}, dtype: {pdfs.dtype}", flush=True)
    print(f"  [diag] pdfs[0, :5] = {pdfs[0, :5]}", flush=True)
    print(f"  [diag] pdfs[0].max() = {pdfs[0].max():.3e}, pdfs[0].sum() = {pdfs[0].sum():.3e}", flush=True)
    # Check if all PDFs are zero (indicates a fundamental computation bug)
    if n_nonzero_pdf < 10:
        print(f"  [diag] WARNING: Almost all PDFs are zero! Checking min_chi2...", flush=True)
        print(f"  [diag] min_chi2 range: [{min_chi2_perz.min():.3e}, {min_chi2_perz.max():.3e}]", flush=True)
        print(f"  [diag] min_chi2[0, :5] = {min_chi2_perz[0, :5]}", flush=True)

    # 5. Moments + output
    print("\n[4/4] Computing moments and saving...", flush=True)
    moments = compute_redshift_moments(
        redshifts, pdfs, min_chi2_perz, best_tmpl_perz
    )
    write_photoz_results(
        output_redshift_path, ids, xpos, ypos, moments, obj_stats, n_filters
    )

    if output_pdfs_path:
        write_pdfs(output_pdfs_path, ids, xpos, ypos, redshifts, pdfs)
        # chi2_out_path = _chi2_path_from_pdf_path(output_pdfs_path)
        # write_chi2_perz(chi2_out_path, ids, xpos, ypos, redshifts, min_chi2_perz)

    elapsed = time.time() - t_total
    print(f"\nTotal wall time: {elapsed:.1f}s", flush=True)
    print(f"Throughput: {len(ids)/elapsed:.0f} objects/s", flush=True)
    print(f"Results: {output_redshift_path}", flush=True)
    if output_pdfs_path:
        print(f"PDFs:    {output_pdfs_path}", flush=True)
        # print(f"Chi2:    {_chi2_path_from_pdf_path(output_pdfs_path)}", flush=True)


# ===========================================================================
# Section 7: CLI
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="photoz_jax",
        description="JAX/GPU template-fitting photo-z  (SPHEREx pipeline)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="cmd")

    # ---- fit ----------------------------------------------------------------
    p_fit = sub.add_parser(
        "fit",
        help="Run photo-z fitting on observed photometry",
    )
    p_fit.add_argument(
        "model_grid",
        help="Path or glob pattern to .photoz model grid file(s)",
    )
    p_fit.add_argument("obs_phot",      help="Observed photometry (text, C++ format)")
    p_fit.add_argument("template_prob", help="Template probability file")
    p_fit.add_argument("output",        help="Output redshift file")
    p_fit.add_argument(
        "--pdfs",
        default=None,
        metavar="FILE",
        help="(optional) save full P(z) arrays here",
    )
    p_fit.add_argument(
        "--batch-obj",
        type=int,
        default=10_000,
        metavar="N",
        help="Objects per GPU batch (default: 10000)",
    )
    p_fit.add_argument(
        "--precision",
        choices=["float64", "float32", "bfloat16"],
        default="float64",
        help="Numeric precision tier: float64 (default), float32, or bfloat16 (store)",
    )
    p_fit.add_argument(
        "--multicore",
        action="store_true",
        default=False,
        help="Use pmap across all available GPUs (multi-GPU path)",
    )
    p_fit.add_argument(
        "--min-source-snr",
        type=float,
        default=None,
        help="Optional source-level SNR threshold; objects below this value are excluded",
    )
    p_fit.add_argument(
        "--source-snr-mode",
        choices=["sum", "quadrature", "total_flux"],
        default="quadrature",
        help=(
            "SNR definition for --min-source-snr: "
            "sum=sum(F/sigma), quadrature=sqrt(sum((F/sigma)^2)), "
            "total_flux=sum(F)/sqrt(sum(sigma^2))"
        ),
    )
    p_fit.add_argument(
        "--source-snr-max-filters",
        type=int,
        default=None,
        help=(
            "Optional: compute --min-source-snr using only the first N photometric "
            "filters (for example, N=102 to use SPHEREx-only bands)"
        ),
    )

    # ---- verify -------------------------------------------------------------
    p_vfy = sub.add_parser(
        "verify",
        help="Compare JAX output against existing C++ results",
    )
    p_vfy.add_argument("model_grid",    help="Path or glob to .photoz grid file(s)")
    p_vfy.add_argument("obs_phot",      help="Observed photometry file")
    p_vfy.add_argument("template_prob", help="Template probability file")
    p_vfy.add_argument(
        "--cpp-pdfs",
        default=None,
        metavar="FILE",
        help="C++ PDF output file (primary comparison)",
    )
    p_vfy.add_argument(
        "--cpp-redshifts",
        default=None,
        metavar="FILE",
        help="C++ redshift output file (secondary comparison)",
    )
    p_vfy.add_argument("--batch-obj", type=int, default=10_000)
    p_vfy.add_argument(
        "--multicore",
        action="store_true",
        default=False,
        help="Use pmap across all available GPUs (multi-GPU path)",
    )
    p_vfy.add_argument(
        "--precision",
        choices=["float64", "float32", "bfloat16"],
        default="float64",
        help="Numeric precision tier for the JAX run",
    )
    p_vfy.add_argument(
        "--min-source-snr",
        type=float,
        default=None,
        help="Optional source-level SNR threshold applied before verification",
    )
    p_vfy.add_argument(
        "--source-snr-mode",
        choices=["sum", "quadrature", "total_flux"],
        default="quadrature",
        help="SNR definition for --min-source-snr",
    )
    p_vfy.add_argument(
        "--source-snr-max-filters",
        type=int,
        default=None,
        help="Optional: compute --min-source-snr using only the first N filters",
    )

    # ---- compare ------------------------------------------------------------
    p_cmp = sub.add_parser(
        "compare",
        help="Compare existing JAX and C++ outputs (no re-fitting)",
    )
    p_cmp.add_argument(
        "--jax-pdfs",
        default=None,
        metavar="FILE",
        help="JAX PDF output file",
    )
    p_cmp.add_argument(
        "--jax-redshifts",
        default=None,
        metavar="FILE",
        help="JAX redshift output file",
    )
    p_cmp.add_argument(
        "--cpp-pdfs",
        default=None,
        metavar="FILE",
        help="C++ PDF output file",
    )
    p_cmp.add_argument(
        "--cpp-redshifts",
        default=None,
        metavar="FILE",
        help="C++ redshift output file",
    )
    p_cmp.add_argument(
        "--plot-dir",
        default=None,
        metavar="DIR",
        help="Directory to save comparison plots (optional)",
    )
    p_cmp.add_argument(
        "--true-z",
        default=None,
        metavar="FILE",
        help="catgrid_info file (col 0 = object ID, col 4 = true redshift); "
             "objects are matched by ID so the full catalog file can be passed "
             "even when comparing a subset",
    )
    p_cmp.add_argument(
        "--derive-from-pdfs",
        action="store_true",
        default=True,
        help="Derive moments from PDFs for both JAX/C++ (ensures identical computation, default=True)",
    )
    p_cmp.add_argument(
        "--no-derive-from-pdfs",
        dest="derive_from_pdfs",
        action="store_false",
        help="Use moment files as-is instead of deriving from PDFs",
    )

    args = parser.parse_args()

    # Apply precision selection from CLI (overrides PHOTOZ_PRECISION env var)
    if hasattr(args, 'precision'):
        chosen = args.precision
        print(f"[config] selected precision: {chosen}", flush=True)
        jax.config.update("jax_enable_x64", chosen == "float64")

    if args.cmd == "fit":
        run(
            model_grid_paths     = args.model_grid,
            obs_phot_path        = args.obs_phot,
            template_prob_path   = args.template_prob,
            output_redshift_path = args.output,
            output_pdfs_path     = args.pdfs,
            batch_obj            = args.batch_obj,
            multicore            = args.multicore,
            precision            = args.precision,
            min_source_snr       = args.min_source_snr,
            source_snr_mode      = args.source_snr_mode,
            source_snr_max_filters = args.source_snr_max_filters,
        )

    elif args.cmd == "verify":
        verify_against_cpp(
            model_grid_paths   = args.model_grid,
            obs_phot_path      = args.obs_phot,
            template_prob_path = args.template_prob,
            cpp_pdfs_path      = args.cpp_pdfs,
            cpp_redshifts_path = args.cpp_redshifts,
            batch_obj          = args.batch_obj,
            multicore          = args.multicore,
            precision          = args.precision,
            min_source_snr     = args.min_source_snr,
            source_snr_mode    = args.source_snr_mode,
            source_snr_max_filters = args.source_snr_max_filters,
        )
    
    elif args.cmd == "compare":
        compare_existing_outputs(
            jax_pdfs_path      = args.jax_pdfs,
            jax_redshifts_path = args.jax_redshifts,
            cpp_pdfs_path      = args.cpp_pdfs,
            cpp_redshifts_path = args.cpp_redshifts,
            true_z_path        = args.true_z,
            plot_dir           = args.plot_dir,
            derive_from_pdfs   = args.derive_from_pdfs,
        )

    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
