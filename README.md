# Land Tax Project

Author: Jiacheng Li

## Repository layout

The repository is organized into the following top-level folders:

- `src/land_tax/`: Python package containing the core steady-state and dynamics solvers.
- `scripts/`: Executable scripts that orchestrate calibration and produce figures.
- `data/params/`: JSON files that store parameter configurations used by the models.
- `data/solutions/`: Saved steady-state solutions produced by the scripts.
- `data/dynamics/`: Intermediate results exchanged with the Julia transition dynamics scripts.
- `notebooks/`: Jupyter notebooks for exploratory work and ad-hoc analysis.
- `julia/`: Julia implementations of the transition dynamics solver.
- `docs/`: Documentation and derivations (LyX/PDF files).

## Running the main analysis

1. Install the Python dependencies used by the project (NumPy, pandas, Matplotlib, SciPy, PrettyTable).
2. Execute the main script from the repository root:
   ```bash
   python scripts/main.py
   ```
   The script automatically makes the `src/` directory importable and writes calibrated parameters
   to `data/params/` and steady-state solutions to `data/solutions/`.

## Transition dynamics workflow

1. Run the Python pre-processing script to export the initial and terminal steady states:
   ```bash
   PYTHONPATH=src python -m land_tax.dynamics
   ```
   This populates JSON files inside `data/dynamics/`.
2. Solve the transition paths with the Julia scripts in `julia/`, which read from the same
   directory.
3. Use `PYTHONPATH=src python -m land_tax.plot_dynamics` to generate the comparative plots once the
   Julia output CSVs are available.

## Legacy notes

The historical research notes from the original project remain below for reference:

- `tau_K_inv = 0.02` is the calibration benchmark; `tau_K_inv = 0.0` is the benchmark for the steady
  state comparative static analysis.
- Questions for discussion with Etienne and Alain:
  1. For the comparative static analysis of taxes, shall we use `tau_K_inv = 0.02` as the benchmark
     or `tau_K_inv = 0.0`?
- Additional observations:
  1. The problem is solved – missing $K$ in tax on capital stock.
  2. But when we put $(1+\omega)$ instead of $\omega$, tax on capital will first increase worker's
     consumption and then decrease it – to be addressed.
  3. First best implies not uniform tax rate on land values, but tax rate * price – to be included.
