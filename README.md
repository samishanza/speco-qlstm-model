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

Run on 2026-11-25 with the open data path: CT-CH4-2025 near-surface CH4 + NCEP/NCAR R1 weather, 40 top-EDGAR cells
(3° × 2°), 2003-01..2020-12. Command: `python -m speco_qlstm.train --data csv --seeds 0 1 2 --epochs 8`.

**Epochs were reduced from the default 40 to 8** to keep the run short; all other settings are the repo defaults
(hidden 5, 4 qubits, 2 entangling layers, lr 0.01, batch 64, patience 8, 12-month input window, 6-month horizon).
With 8 epochs the models are likely under-trained, so treat these as a first baseline, not final numbers.

- Training months 2003-07..2017-06 (after 6 months consumed by lag features), test months 2017-07..2020-12.
  Windows: 5,000 train, 840 validation, 1,480 test (40 cells × 37).
- Wall time 8.6 min on a MacBook Air CPU: QLSTM ~21 s/epoch (simulated circuits), LSTM ~0.1 s/epoch.
- Parameters: QLSTM 304, LSTM 416.

Test error in ppb (all 6 forecast months pooled):

| Seed | QLSTM MAE | QLSTM RMSE | LSTM MAE | LSTM RMSE |
|---|---|---|---|---|
| 0 | 33.45 | 50.06 | 32.71 | 50.66 |
| 1 | 33.24 | 48.37 | 30.32 | 46.83 |
| 2 | 34.61 | 52.74 | 30.46 | 45.27 |
| **Mean** | **33.76** | **50.39** | **31.16** | **47.58** |

The QLSTM is worse than the classical LSTM: MAE +8.3 %, RMSE +5.9 %.
- Paired t-test across the 3 seeds: MAE p = 0.12, RMSE p = 0.36 (not significant; n = 3 is very small).
- Paired t-test across the 1,480 test windows (absolute error averaged over seeds): t = 13.5, p = 3.7e-39, LSTM
  lower. Windows from the same cell and overlapping months are not independent, so this p-value overstates the
  evidence.

Context from simple baselines on the same test windows (computed with the same scaler and windows, not part of
`train.py`): persistence (last observed month repeated) MAE 57.9 / RMSE 86.9 ppb; seasonal naive (same month one year
earlier) MAE 30.2 / RMSE 46.1 ppb. Neither model beats the seasonal-naive forecast in this run, which is expected for
a rising trend that a year-old value lags only slightly; more epochs, trend-aware features or differencing may change
this, but none of that was tried here.

These numbers are not comparable to the paper's: different methane product (model reanalysis, not satellite), grid
(3° × 2°), weather source and period.

Files: `results/metrics.json`, `results/loss_curves.png`, `results/forecast_examples.png`,
`results/error_boxplot.png` (PNGs are not committed; re-run to regenerate), training log `results/train_log.txt`,
data summary `results/fetch_summary.txt`.

## Layout

```
speco_qlstm/data.py    fetch (NOAA CT-CH4, NOAA PSL NCEP R1, Copernicus ADS/CDS), features, chronological split, sequence windows
speco_qlstm/models.py  QLSTM (four VQC gates) and classical LSTM
speco_qlstm/train.py   training, evaluation, t-tests, plots
speco_qlstm/fetch.py   download CLI
tests/                 shape, leakage and smoke tests
```
