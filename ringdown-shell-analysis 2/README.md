# Ringdown of a black hole with a thin shell of matter

A reanalysis of the public waveforms of Laeuger, Weller, Li & Chen,
*The ringdown of a black hole surrounded by a thin shell of matter*,
[arXiv:2506.00367](https://arxiv.org/abs/2506.00367).

**Question:** can a fit of the ringdown tell a black hole with a shell of matter apart from a black hole
in vacuum? And does this still hold with realistic detector noise?

The full write-up, with method, tables and figures, is in **[ringdown_report.pdf](ringdown_report.pdf)**.

## What the code does

For three shell configurations (heavy and close, light and distant, heavy and distant with an echo):

1. **Fixed-mode fits.** Fits the ringdown with the vacuum (Schwarzschild) quasinormal modes and with
   the shell modes, and compares the mismatch.
2. **Free fit.** Fits damped sinusoids with free frequencies and checks which set of modes they are closest to.
3. **Bias test.** Repeats the free fit on a vacuum-only signal, to check that the method does not create
   shell-like modes by itself.
4. **Advanced LIGO noise.** Adds noise from the Advanced LIGO design sensitivity (for a 60 solar-mass
   black hole) and checks how often the shell modes still fit better than the vacuum ones.
   This test is not in the original paper.

## Main results

With Advanced LIGO noise (100 noise realizations per SNR), the shell modes give the better fit (lower χ²) in at least
90% of the realizations:

| Case | Shell | From SNR |
|---|---|---|
| 1 | heavy, close (δM = M/2, R = 6M) | 30 |
| 2 | light, distant (δM = M/100, R = 30M) | never |
| 3 | heavy, distant (δM = M/2, R = 30M), echo | 300 (89 ± 3% already at SNR 100) |

The report also gives, for each SNR, the χ²/ndof and p-value of both fits and the difference
Δχ² = χ²(vacuum) − χ²(shell), i.e. twice the log-likelihood ratio between the two hypotheses.

## Data

The data are **not included** in this repository: they belong to the authors of the paper and are publicly
available on Zenodo ([DOI 10.5281/zenodo.15586889](https://doi.org/10.5281/zenodo.15586889)).
The program uses six files from the `data` folder of that archive:

- `infall_waveform_coupled_M0.5_deltaM{0.25, 0.005, 0.25}_ell2_vs0.1_rshell{3.0, 15.0, 15.0}_rstart5M.npz` (waveforms)
- `PolarQNMlist_M0.5_deltaM{0.25, 0.005, 0.25}_ell2_vs0.1_rshell{3.0, 15.0, 15.0}.npz` (shell quasinormal modes)

## How to run it

1. Download the archive from Zenodo and unzip it.
2. Install the libraries: `python3 -m pip install -r requirements.txt`
3. Run the program, giving the path of the `data` folder:
   `python3 ringdown_analysis.py path/to/data`

The program saves the plots in `plots/` and writes the LaTeX report in `report/`
(compiled to PDF if LaTeX is installed; the generated `ringdown_report.zip` can be uploaded to Overleaf).

## Author

Darin La Rocca, BSc Physics student, University of Milano-Bicocca
