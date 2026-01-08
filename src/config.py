from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    data_path: Path = Path("data/raw/Online_Retail.csv")
    churn_window_days: int = 90

    win_30: int = 30
    win_60: int = 60
    win_90: int = 90

    outputs_dir: Path = Path("outputs")
    customer_features_path: Path = Path("outputs/customer_features.csv")
    churn_priority_path: Path = Path("outputs/churn_priority_table.csv")
    metrics_path: Path = Path("outputs/model_metrics.json")

    test_size: float = 0.2
    random_state: int = 42
