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
| Wind speed (10 m), temperature (2 m), surface pressure | [ERA5 monthly means](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means), 0.75° | Copernicus Climate Data Store. With CT-CH4, block-averaged onto the 3° × 2° cells |
| Hotspot selection | EDGAR v8.0 CH4 total emissions, 2022, 0.1° | JRC, summed into the methane grid's cells; top 40 cells over CONUS (lat 24–50°N, lon 125–66°W) |

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
- **ERA5.** Contains modified Copernicus Climate Change Service information [2003–2020]. Neither the European
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

ERA5 needs a Copernicus account with a personal access token in `~/.cdsapirc` (`url: https://cds.climate.copernicus.eu/api`
and `key: <token>`) and the ERA5 licence accepted on the dataset page. CarbonTracker-CH4 needs nothing. Only for
`--methane copernicus`: accept the [ADS terms of use](https://ads.atmosphere.copernicus.eu/licences/terms-of-use-ads),
the ADS data protection statement and the EGG4 licence, or downloads are refused with a 403.

## Run

```bash
.venv/bin/python -m speco_qlstm.fetch --methane noaa-ct --start 2003 --end 2020 --cells 40   # writes data/conus_monthly.csv
.venv/bin/python -m speco_qlstm.fetch --methane copernicus                                   # alternative (EGG4, 0.75°)
.venv/bin/python -m speco_qlstm.train --data csv --seeds 0 1 2                               # writes results/
.venv/bin/python -m pytest -q                                                                # tests (synthetic data, no network)
```

The CT-CH4 step takes a few minutes (about 6,600 small requests, 8 in parallel) and resumes month by month if
interrupted. `--data synthetic` runs the whole pipeline on generated data with the same schema. It's for testing only.

## Results

Not run on real data yet. Run `python -m speco_qlstm.train` to populate `results/metrics.json`, the loss curves,
example forecasts and the error box plot.

## Layout

```
speco_qlstm/data.py    fetch (NOAA CT-CH4, Copernicus ADS/CDS), features, chronological split, sequence windows
speco_qlstm/models.py  QLSTM (four VQC gates) and classical LSTM
speco_qlstm/train.py   training, evaluation, t-tests, plots
speco_qlstm/fetch.py   download CLI
tests/                 shape, leakage and smoke tests
```
