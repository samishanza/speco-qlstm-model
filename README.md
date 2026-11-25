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
| CH4 (column-averaged, ppb) | [CAMS global greenhouse gas reanalysis (EGG4)](https://ads.atmosphere.copernicus.eu/datasets/cams-global-ghg-reanalysis-egg4-monthly), monthly, 0.75° | Copernicus Atmosphere Data Store, 2003–2020 |
| Wind speed (10 m), temperature (2 m), surface pressure | [ERA5 monthly means](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means), 0.75° | Copernicus Climate Data Store |
| Hotspot selection | EDGAR v8.0 CH4 total emissions, 2022, 0.1° | JRC, aggregated to the 0.75° grid; top 40 cells over CONUS |

The paper's original inputs (Sentinel-5P, EMIT and MethaneAIR via Google Earth Engine, plus EDGAR) are replaced here
by CAMS EGG4 and ERA5 from Copernicus. EGG4 gives a gap-free monthly CH4 record for 2003–2020, which suits sequence
models better than sparse satellite retrievals, but it is a model reanalysis rather than direct observation.

Attribution:
- Contains modified Copernicus Atmosphere Monitoring Service information [2003–2020]. Neither the European
  Commission nor ECMWF is responsible for any use that may be made of the information it contains.
- Contains modified Copernicus Climate Change Service information [2003–2020]. Neither the European Commission nor
  ECMWF is responsible for any use that may be made of the Copernicus information or data it contains.
- EDGAR (Emissions Database for Global Atmospheric Research) v8.0, European Commission, Joint Research Centre.

Downloaded and derived data stay out of git (`data/`).

## Setup

```bash
python3.10 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

You need a Copernicus account with a personal access token in `~/.cdsapirc` (`url: https://cds.climate.copernicus.eu/api`
and `key: <token>`). The same token is used for the Atmosphere Data Store. In the browser, accept the
[ADS terms of use](https://ads.atmosphere.copernicus.eu/licences/terms-of-use-ads), the ADS data protection statement,
and the licence on the EGG4 and ERA5 dataset pages, or downloads are refused with a 403.

## Run

```bash
.venv/bin/python -m speco_qlstm.fetch --start 2003 --end 2020 --cells 40   # writes data/conus_monthly.csv
.venv/bin/python -m speco_qlstm.train --data csv --seeds 0 1 2              # writes results/
.venv/bin/python -m pytest -q                                              # tests (synthetic data)
```

`--data synthetic` runs the whole pipeline on generated data with the same schema. It's for testing only.

## Results

Not run on real data yet. Run `python -m speco_qlstm.train` to populate `results/metrics.json`, the loss curves,
example forecasts and the error box plot.

## Layout

```
speco_qlstm/data.py    fetch (Copernicus ADS/CDS), features, chronological split, sequence windows
speco_qlstm/models.py  QLSTM (four VQC gates) and classical LSTM
speco_qlstm/train.py   training, evaluation, t-tests, plots
speco_qlstm/fetch.py   download CLI
tests/                 shape, leakage and smoke tests
```
