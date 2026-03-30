# template_fitter_jax

Python/JAX implementation of a GPU-supported photometric redshift template fitter.

## Current scope

At the moment this implementation is for the chi2 model evaluation given a precomputed grid, however the grid pre-computation may be added at a later point.

## Included files

- `photoz_jax.py`: Main JAX fitter and compare/verify utilities
- `run_jax_photoz.sh`: Simple wrapper script for `fit` and optional `verify`
- `plotting_fns.py`: Plot helpers currently kept with the implementation

## Setup

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Install GPU-enabled JAX following the official JAX instructions for your CUDA version.

## Quick usage

Run the wrapper script with your own paths as environment variables:

```bash
MODEL_GRID_GLOB="/path/to/grid/*.photoz" \
OBS_PHOT="/path/to/obs_phot.dat" \
TEMPLATE_PROB="/path/to/template_prob.dat" \
OUTPUT_REDSHIFTS="/path/to/output_redshifts.out" \
OUTPUT_PDFS="/path/to/output_pdfs.out" \
./run_jax_photoz.sh
```

Optional verification against C++ PDFs:

```bash
CPP_PDFS="/path/to/cpp_pdfs.out" ./run_jax_photoz.sh
```

For command-level options and all modes (`fit`, `verify`, `compare`), run:

```bash
python photoz_jax.py -h
```
