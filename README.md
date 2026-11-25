# SPECO QLSTM model

Stage 1 of **SPECO** (Shanza Sami, *SPECO: A Novel Multiparametric Methane Forecasting and Mitigation System Using
Quantum Long Short-term Memory and Sonophotoelectrochemical Oxidation*): a hybrid quantum–classical LSTM (QLSTM) that
forecasts atmospheric methane at U.S. hotspots, benchmarked against a classical LSTM trained on the same inputs.

## Model

```
monthly features per cell (CH4, wind speed, temperature, pressure, CH4 lags/rolling stats, season, location)
        │  window of 12 months
        ▼
┌──────────────────────── QLSTM cell, repeated for each month ────────────────────────┐
│ [x_t, h_{t-1}] ──Linear──► 4 angles ──┬─► VQC_forget ─► Linear ─► σ  ─┐               │
│                                       ├─► VQC_input  ─► Linear ─► σ  ─┤  c_t, h_t    │
│                                       ├─► VQC_cell   ─► Linear ─► tanh┤  (hidden = 5) │
│                                       └─► VQC_output ─► Linear ─► σ  ─┘               │
│ each VQC: 4 qubits, AngleEmbedding(RY) + 2 StronglyEntanglingLayers, ⟨Z⟩ on every wire │
└──────────────────────────────────────────────────────────────────────────────────────┘
        │  h_T
        ▼
Linear ─► CH4 forecast for the next 6 months (ppb)
```

- Four parameterised quantum circuits, one per LSTM gate, following Chen, Yoo & Fang (2020),
  [Quantum Long Short-Term Memory](https://arxiv.org/abs/2009.01783). Five hidden units, as in the paper.
- Circuits are simulated with PennyLane (`default.qubit`, backprop) and trained end to end in PyTorch.
- Baseline: a classical `nn.LSTM` with the same hidden size, inputs and output head.
- Evaluation: chronological 80/20 split (every test month comes after every training month), scaler fit on
  training data only, Adam with early stopping on a validation slice of the training period, MAE and RMSE in ppb,
  three seeds, paired t-tests across seeds and across test windows.

## Data

| Variable | Source | Notes |
|---|---|---|
| CH4 (ppb), default | [NOAA GML CarbonTracker-CH4](https://gml.noaa.gov/ccgg/carbontracker-ch4/), release CT-CH4-2025 | Lowest model level (surface layer) mole fraction, 3° lon × 2° lat, 3-hourly means averaged to calendar months, 2003–2020. Read from the open NOAA archive, no account |
| CH4 (ppb), alternative | [CAMS global greenhouse gas reanalysis (EGG4)](https://ads.atmosphere.copernicus.eu/datasets/cams-global-ghg-reanalysis-egg4-monthly), monthly, 0.75° | Column-averaged. Copernicus Atmosphere Data Store; needs an account and the ADS terms accepted |
| Wind speed, air temperature, surface pressure, default | [NCEP/NCAR Reanalysis 1](https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html) monthly means, 2.5°, from NOAA PSL | `ncep.reanalysis.derived/surface/`: `uwnd`/`vwnd`/`air.sig995.mon.mean.nc` (lowest sigma level, σ = 0.995, ~40 m above ground) and `pres.sfc.mon.mean.nc`. Open download, no account. Area-weighted (conservative, cos-latitude) onto the 3° × 2° cells |
| Wind speed (10 m), temperature (2 m), surface pressure, alternative | [ERA5 monthly means](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means), 0.75° | Copernicus Climate Data Store (account + licence). With CT-CH4, 0.75° points block-averaged into the 3° × 2° cells (`--weather era5`) |
| Hotspot selection | EDGAR v8.0 CH4 total emissions, 2022, 0.1° | JRC, summed into the methane grid's cells; top 40 cells in the box lat 24–50°N, lon 125–66°W |

Weather caveats (NCEP R1): wind speed is computed from the monthly-mean u and v components, √(ū² + v̄²). Because
winds change direction within a month, this understates the true monthly mean wind speed (typical values here are
1.5–4 m/s). Temperature is at σ = 0.995, not 2 m. NCEP R1 is an older, coarser reanalysis than ERA5.

Cell caveats: cells are boxes in a lat/lon window, not clipped to U.S. borders. A few of the top-40 cells lie partly
or mostly outside the U.S. (e.g. 25°N 103.5°W is in northern Mexico; the 49°N row straddles the Canadian border).

**This differs from the paper.** The paper's inputs were satellite and airborne retrievals (Sentinel-5P TROPOMI,
EMIT and MethaneAIR via Google Earth Engine) plus EDGAR. Here methane comes from CarbonTracker-CH4, a data-assimilation
model (TM5 transport with surface fluxes optimised against the NOAA and partner in-situ/flask network), not from
satellites. It is gap-free and monthly for 2003–2020, which suits sequence models, but it is model output at
3° × 2°, so a "cell" here is a ~300 km × 220 km box and fine-scale plumes are smoothed out. Its CH4 is the lowest model
layer (near-surface), not the column average that TROPOMI or EGG4 report.

How CT-CH4 is fetched: the release has one ~36 MB global file per day (8 three-hourly steps × 25 levels). The fetch
asks NOAA's server for only the bytes holding the lowest level over CONUS (one HTTP multi-range request per day,
~100 kB), checks the file size in every response, compares the byte-range read with a normal netCDF read for
1 January of each year, then averages all 3-hourly steps of each month. Only the monthly CONUS array is stored
(`data/raw/ctch4_2025_sfc_conus_monthly_2003-2020.nc`, under 1 MB).

### Terms, citations and acknowledgments

- **NOAA CarbonTracker-CH4.** NOAA's usage policy: "CarbonTracker is an open product of NOAA's Global Monitoring
  Laboratory using data from the NOAA GML greenhouse gas observational network and collaborating institutions.
  Results, including figures and tabular material found on the CarbonTracker website may be used for
  non-commercial purposes without restriction." No login or licence click-through. Note the *non-commercial*
  wording: fine for this research code; ask NOAA (carbontracker.team@noaa.gov) before any commercial use. NOAA also
  asks that work relying heavily on CarbonTracker be discussed with them before publication.
  - Acknowledgment (as requested by NOAA): "CarbonTracker CT-CH4-2025 results provided by NOAA GML, Boulder, Colorado,
    USA from the website at https://gml.noaa.gov/ccgg/carbontracker-ch4/"
  - Citation: Oh, Y., Bruhwiler, L., Lan, X., et al. (2025). CarbonTracker CH4 2025. NOAA Global Monitoring
    Laboratory. https://doi.org/10.25925/hxks-v755 (full author list at
    https://gml.noaa.gov/ccgg/carbontracker-ch4/citation.php)
- **NCEP/NCAR Reanalysis 1.** Public NOAA data, no account. Acknowledgment, in PSL's suggested wording: "NCEP/NCAR
  Reanalysis 1 data provided by the NOAA PSL, Boulder, Colorado, USA, from their website at https://psl.noaa.gov"
  (PSL asks for "data provided by the NOAA PSL, Boulder, Colorado, USA, from their website at https://psl.noaa.gov"
  and would appreciate a copy of resulting publications). Dataset reference: Kalnay, E. et al. (1996), The NCEP/NCAR
  40-Year Reanalysis Project, Bull. Amer. Meteor. Soc., 77, 437–471.
- **ERA5** (alternative weather path only). Contains modified Copernicus Climate Change Service information [2003–2020]. Neither the European
  Commission nor ECMWF is responsible for any use that may be made of the Copernicus information or data it contains.
  Hersbach, H. et al. (2023): ERA5 monthly averaged data on single levels from 1940 to present. C3S Climate Data Store,
  https://doi.org/10.24381/cds.f17050d7
- **EDGAR.** EDGAR (Emissions Database for Global Atmospheric Research) v8.0, European Commission, Joint Research
  Centre.
- **CAMS EGG4** (alternative path only). Contains modified Copernicus Atmosphere Monitoring Service information
  [2003–2020]. Neither the European Commission nor ECMWF is responsible for any use that may be made of the
  information it contains.

Downloaded and derived data stay out of git (`data/`).

## Setup

```bash
python3.10 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

The default path (CT-CH4 + NCEP R1) needs no account. ERA5 (`--weather era5`) needs a Copernicus account with a personal access token in `~/.cdsapirc` (`url: https://cds.climate.copernicus.eu/api`
and `key: <token>`) and the ERA5 licence accepted on the dataset page. CarbonTracker-CH4 needs nothing. Only for
`--methane copernicus`: accept the [ADS terms of use](https://ads.atmosphere.copernicus.eu/licences/terms-of-use-ads),
the ADS data protection statement and the EGG4 licence, or downloads are refused with a 403.

## Run

```bash
.venv/bin/python -m speco_qlstm.fetch --methane noaa-ct --weather ncep --start 2003 --end 2020 --cells 40   # writes data/conus_monthly.csv
.venv/bin/python -m speco_qlstm.fetch --methane noaa-ct --weather era5                       # same, ERA5 weather (CDS account)
.venv/bin/python -m speco_qlstm.fetch --methane copernicus                                   # alternative (EGG4, 0.75°)
.venv/bin/python -m speco_qlstm.train --data csv --seeds 0 1 2                               # writes results/
.venv/bin/python -m pytest -q                                                                # tests (synthetic data, no network)
```

The CT-CH4 step takes a few minutes (about 6,600 small requests, 8 in parallel) and resumes month by month if
interrupted. `--data synthetic` runs the whole pipeline on generated data with the same schema. It's for testing only.

## Results

Data: CT-CH4-2025 near-surface CH4 + NCEP/NCAR R1 weather, 40 top-EDGAR cells (3° × 2°), 2003-01..2020-12.
Training months 2003-07..2017-06, test months 2017-07..2020-12; 5,000 train / 840 validation / 1,480 test windows
(12-month input, 6-month horizon). Seeds 0, 1, 2. Other settings are the repo defaults (hidden 5, 4 qubits,
2 entangling layers, lr 0.01, batch 64, early stopping with patience 8). Three pre-specified runs, nothing tuned:

```bash
python -m speco_qlstm.train --data csv --seeds 0 1 2 --epochs 8                                  # first run (renamed *_level_e8)
python -m speco_qlstm.train --data csv --seeds 0 1 2 --target level --tag level_e40              # 40 epochs
python -m speco_qlstm.train --data csv --seeds 0 1 2 --target seasonal-diff --tag seasonal_diff_e40
```

`seasonal-diff` predicts y(t+k) − y(t+k−12), the change from the same month a year earlier, and adds that
earlier value back (it is always inside the input window; the code asserts this). All errors are in ppb on the same
1,480 test windows; persistence and seasonal-naive are computed on those same windows.

Test MAE / RMSE (ppb), per seed and mean:

| Run | Model | Seed 0 | Seed 1 | Seed 2 | **Mean** | QLSTM vs LSTM | p (seeds) MAE / RMSE | p (windows) |
|---|---|---|---|---|---|---|---|---|
| level, 8 epochs | QLSTM | 33.45 / 50.06 | 33.24 / 48.37 | 34.61 / 52.74 | **33.76 / 50.39** | +8.3 % / +5.9 % | 0.12 / 0.36 | 3.7e-39 |
| | LSTM | 32.71 / 50.66 | 30.32 / 46.83 | 30.46 / 45.27 | **31.16 / 47.58** | | | |
| level, 40 epochs | QLSTM | 28.57 / 42.88 | 27.36 / 41.28 | 27.63 / 42.15 | **27.85 / 42.10** | +0.9 % / +1.1 % | 0.70 / 0.61 | 0.054 |
| | LSTM | 27.99 / 42.05 | 28.18 / 42.33 | 26.67 / 40.53 | **27.61 / 41.63** | | | |
| seasonal-diff, 40 epochs | QLSTM | 27.98 / 44.56 | 28.36 / 44.92 | 31.24 / 56.21 | **29.19 / 48.56** | +3.4 % / +8.4 % | 0.62 / 0.41 | 2.8e-5 |
| | LSTM | 29.71 / 44.65 | 27.69 / 44.60 | 27.29 / 45.21 | **28.23 / 44.82** | | | |
| baseline | seasonal naive (same month last year) | | | | **30.19 / 46.10** | | | |
| baseline | persistence (last month repeated) | | | | **57.92 / 86.85** | | | |

p (seeds): paired t-test of the three per-seed values (n = 3, so low power). p (windows): paired t-test of per-window
absolute error averaged over seeds (n = 1,480); windows overlap in time and share cells, so they are not
independent and this p-value overstates the evidence.

Epochs actually run (early stopping), seeds 0/1/2: level-40 LSTM 30/22/32, QLSTM 27/31/38; seasonal-diff LSTM
35/17/19, QLSTM 10/17/40. Wall time: 8-epoch run 8.6 min; level-40 62 min; seasonal-diff-40 33 min (QLSTM
~21–36 s/epoch depending on machine load, LSTM ~0.1 s/epoch; MacBook Air CPU, runs at `nice 5`).

What this says, plainly:
- With 40 epochs both models beat the seasonal-naive forecast, but only modestly: level-target LSTM by 8.5 % MAE /
  9.7 % RMSE, QLSTM by 7.7 % / 8.7 %. After 8 epochs neither did.
- QLSTM and LSTM are statistically indistinguishable at 40 epochs on the level target (QLSTM ~1 % worse, p ≈ 0.6–0.7
  across seeds). The QLSTM has fewer parameters (304 vs 416) and is a few hundred times slower to train in simulation.
- The seasonal-difference target did not help: both models were slightly worse than with the level target, and
  QLSTM seed 2 had one bad run (RMSE 56.2, worse than seasonal naive), which drives its higher mean. On MAE the
  seasonal-diff models still beat seasonal naive (LSTM −6.5 %, QLSTM −3.3 %); on RMSE the QLSTM did not (+5.3 %).

These numbers are not comparable to the paper's: different methane product (model reanalysis, not satellite), grid
(3° × 2°), weather source and period.

Files in `results/`: `metrics_level_e8.json`, `metrics_level_e40.json`, `metrics_seasonal_diff_e40.json`, matching
`train_log_*.txt`, `fetch_summary.txt` (data sanity summary), and per run `loss_curves_<tag>.png`,
`forecast_examples_<tag>.png`, `error_boxplot_<tag>.png` (PNGs are not committed; re-run to regenerate).

## Layout

```
speco_qlstm/data.py    fetch (NOAA CT-CH4, NOAA PSL NCEP R1, Copernicus ADS/CDS), features, chronological split, sequence windows
speco_qlstm/models.py  QLSTM (four VQC gates) and classical LSTM
speco_qlstm/train.py   training, evaluation, t-tests, plots
speco_qlstm/fetch.py   download CLI
tests/                 shape, leakage and smoke tests
```

## License

The code is released under the [MIT License](LICENSE). The data it downloads is not covered by this license: NOAA CarbonTracker-CH4, NCEP/NCAR Reanalysis 1, ERA5 and EDGAR each remain under their providers' terms (see Data above).
