IoT Data Analytics — Batch vs Online Learning with Concept Drift

## Overview
This project compares batch learning and online learning on a synthetic stream with concept drift (SEA dataset).
It trains:
- Online models (River): Hoeffding Tree, Gaussian Naive Bayes, Perceptron (with StandardScaler)
- One batch model (scikit-learn): Logistic Regression

The script writes two CSV files and two figures:
- `online_accuracy.csv`: cumulative online accuracy at each timestep
- `summary_results.csv`: final accuracy and training time for all models
- `figures/accuracy_over_time.png`: accuracy over time for online models
- `figures/summary_bar.png`: bar charts of final accuracy and training time

## Project structure
- `src/data_utils.py`: data stream generator with abrupt drifts (SEA)
- `src/batch_vs_online.py`: main experiment (runs training and saves outputs)
- `src/plots.py`: functions to create the figures
- `figures/`: output images are saved here
- `requirements.txt`: Python dependencies
- `docker-compose.yml`: run the experiment with Docker

## Requirements
- Python 3.10+ (tested with Python 3.11)
- Install dependencies from `requirements.txt`

## Quick start (local)
1) Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
```

2) Install dependencies

```bash
pip install -r requirements.txt
```

3) Run the experiment

```bash
python src/batch_vs_online.py
```

Outputs will appear in the project folder:
- CSV: `online_accuracy.csv`, `summary_results.csv`
- Figures in `figures/`: `accuracy_over_time.png`, `summary_bar.png`

## Run with Docker
If you prefer Docker:

```bash
docker compose up --build
```

This will run `python src/batch_vs_online.py` inside the container and write outputs to your local folder.

## Change parameters
Edit the call at the end of `src/batch_vs_online.py` to set your values:

```python
if __name__ == "__main__":
    run_experiment(
        n_samples=5000,
        drift_points=[1250, 2500, 3750],
        seed=42,
        noise=0.0,
        online_accuracy_csv="online_accuracy.csv",
        summary_csv="summary_results.csv",
    )
```

- `n_samples`: number of samples in the stream
- `drift_points`: indices where abrupt concept drifts occur
- `seed`: random seed for reproducibility
- `noise`: label noise level (SEA)

## Notes
- Figures are generated automatically after the CSV files are saved.
- All outputs are overwritten if the same file names are used.

## Acknowledgements
- River (online machine learning): `river`
- scikit-learn (batch learning): `scikit-learn`


