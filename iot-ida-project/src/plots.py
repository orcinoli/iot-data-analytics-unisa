import os
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_accuracy_over_time(
    csv_path: str = "online_accuracy.csv", output_dir: str = "figures"
):
    _ensure_dir(output_dir)
    df = pd.read_csv(csv_path)
    # Expect columns: timestep, <model1>, <model2>, ...
    model_cols = [c for c in df.columns if c != "timestep"]

    plt.figure(figsize=(10, 6))
    for col in model_cols:
        plt.plot(df["timestep"], df[col], label=col, linewidth=1.6)
    plt.xlabel("Timestep")
    plt.ylabel("Cumulative Accuracy")
    plt.title("Online Accuracy Over Time (SEA with Concept Drift)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    out_path = os.path.join(output_dir, "accuracy_over_time.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_summary(
    csv_path: str = "summary_results.csv", output_dir: str = "figures"
):
    _ensure_dir(output_dir)
    df = pd.read_csv(csv_path)
    models = df["model"].tolist()
    final_acc = df["final_accuracy"].tolist()
    train_time = df["training_time_seconds"].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Final accuracy
    axes[0].bar(models, final_acc, color="#1f77b4")
    axes[0].set_title("Final Accuracy")
    axes[0].set_ylabel("Accuracy")
    axes[0].tick_params(axis="x", labelrotation=20)
    axes[0].set_ylim(0, 1.0)
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.4)

    # Training time
    axes[1].bar(models, train_time, color="#ff7f0e")
    axes[1].set_title("Training Time (s)")
    axes[1].set_ylabel("Seconds")
    axes[1].tick_params(axis="x", labelrotation=20)
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.4)

    out_path = os.path.join(output_dir, "summary_bar.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def generate_all_figures(
    accuracy_csv_path: str = "online_accuracy.csv",
    summary_csv_path: str = "summary_results.csv",
    output_dir: str = "figures",
):
    plot_accuracy_over_time(csv_path=accuracy_csv_path, output_dir=output_dir)
    plot_summary(csv_path=summary_csv_path, output_dir=output_dir)

