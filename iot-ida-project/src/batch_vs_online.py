import os
import time
from typing import Dict, List

import numpy as np
import pandas as pd
from river import compose, linear_model, metrics, naive_bayes, preprocessing, tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from data_utils import generate_sea_stream_with_drifts, make_sea_arrays_with_drifts
import plots


def run_experiment(
    n_samples: int = 5000,
    drift_points: List[int] = None,
    seed: int = 42,
    noise: float = 0.0,
    online_accuracy_csv: str = "online_accuracy.csv",
    summary_csv: str = "summary_results.csv",
) -> None:
    """
    Runs Batch vs Online learning on SEA with concept drift and writes required outputs.
    - online_accuracy.csv: timestep-wise cumulative accuracy for online models
    - summary_results.csv: final accuracy and training time for all models (batch + online)
    """
    if drift_points is None:
        drift_points = [1250, 2500, 3750]

    # -----------------------
    # Define ONLINE models
    # -----------------------
    online_models: Dict[str, object] = {
        "HoeffdingTreeClassifier": tree.HoeffdingTreeClassifier(),
        "GaussianNB": naive_bayes.GaussianNB(),
        "Perceptron": compose.Pipeline(
            preprocessing.StandardScaler(), linear_model.Perceptron()
        ),
    }
    online_metrics: Dict[str, metrics.Metric] = {
        name: metrics.Accuracy() for name in online_models.keys()
    }
    online_train_time: Dict[str, float] = {name: 0.0 for name in online_models.keys()}

    # Records for accuracy over time
    acc_records: Dict[str, List[float]] = {"timestep": []}
    for name in online_models.keys():
        acc_records[name] = []

    # -----------------------
    # Stream data with drifts
    # -----------------------
    stream = generate_sea_stream_with_drifts(
        n_samples=n_samples, drift_points=drift_points, seed=seed, noise=noise
    )

    # Prequential evaluation: predict -> evaluate -> learn
    for t, (x, y, _concept) in enumerate(stream, start=1):
        acc_records["timestep"].append(t)

        for name, model in online_models.items():
            y_pred = model.predict_one(x) if hasattr(model, "predict_one") else None
            # Importante: no reasignar, algunos .update() no devuelven self en versiones antiguas
            online_metrics[name].update(y_true=y, y_pred=y_pred)

            start_fit = time.perf_counter()
            model.learn_one(x, y)
            end_fit = time.perf_counter()
            online_train_time[name] += end_fit - start_fit

            acc_records[name].append(online_metrics[name].get())

    # -----------------------
    # Save online accuracy CSV
    # -----------------------
    online_df = pd.DataFrame(acc_records)
    online_df.to_csv(online_accuracy_csv, index=False)

    # -----------------------
    # BATCH model (LogisticRegression)
    # -----------------------
    X, y, _concepts, feature_names = make_sea_arrays_with_drifts(
        n_samples=n_samples, drift_points=drift_points, seed=seed, noise=noise
    )
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)

    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    batch_model = LogisticRegression(max_iter=1000, solver="lbfgs")

    start_batch = time.perf_counter()
    batch_model.fit(X_train, y_train)
    end_batch = time.perf_counter()
    batch_train_time = end_batch - start_batch

    y_pred_batch = batch_model.predict(X_test)
    batch_final_acc = float(accuracy_score(y_test, y_pred_batch))

    # -----------------------
    # Summary results CSV
    # -----------------------
    rows = []
    # Online models: final cumulative accuracy + total training time
    for name in online_models.keys():
        rows.append(
            {
                "model": name,
                "training_time_seconds": online_train_time[name],
                "final_accuracy": online_metrics[name].get(),
            }
        )

    # Batch model
    rows.append(
        {
            "model": "LogisticRegression",
            "training_time_seconds": batch_train_time,
            "final_accuracy": batch_final_acc,
        }
    )

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(summary_csv, index=False)

    # -----------------------
    # Create figures
    # -----------------------
    plots.generate_all_figures(
        accuracy_csv_path=online_accuracy_csv,
        summary_csv_path=summary_csv,
        output_dir="figures",
    )


if __name__ == "__main__":
    run_experiment()

