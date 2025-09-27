# Assignment 10 — Leakage Assessment with Non-Specific Welch’s t-test
**Student**: *Omidi Zadeh*

Implements the **non-specific Welch’s t-test** up to order 3 on the provided `Traces.dat` with 100,000 traces (id + 1000 points). Two groups: **fixed** (id=0) vs **random** (id=1).

## How it works
For order *m* in {1,2,3}, we transform each point as **y = x^m** and compare the group means via **Welch’s t-test** for each of the 1000 time samples individually (univariate). We stream the file once in blocks and maintain running sums `Σy`, `Σy²` for each group to compute means/variances without storing all traces.

## What gets produced
- `outputs/figs/plot_points_t_order_1.png` — t-values over 1000 points (order 1)
- `outputs/figs/plot_points_t_order_2.png` — t-values (order 2)
- `outputs/figs/plot_points_t_order_3.png` — t-values (order 3)
- `outputs/figs/plot_nrtraces_t_order_1.png` — abs max |t| vs #traces (order 1)
- `outputs/figs/plot_nrtraces_t_order_2.png` — abs max |t| vs #traces (order 2)
- `outputs/figs/plot_nrtraces_t_order_3.png` — abs max |t| vs #traces (order 3)
- `outputs/description.txt` — brief summary and checkpoints

## Run (already configured to use the uploaded file path)
```bash
pip install -r requirements.txt

python -m src.main   --traces_path "/mnt/data/Traces.dat"   --out_dir outputs   --plots_dir outputs/figs   --points_per_trace 1000   --block_size 2000   --schedule 1000 2000 5000 10000 20000 50000 100000
```

## Notes
- Figures use **matplotlib**, one chart per figure, no custom colors.
- No data is re-distributed; only figures and code are included as outputs.
- Code is fully vectorized across time samples; only the streaming over traces is incremental.
