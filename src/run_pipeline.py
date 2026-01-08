from __future__ import annotations

from src.config import Config
from src.load_data import load_transactions
from src.churn_label import customer_last_purchase, add_churn_label
from src.features import build_rfm_features, build_window_features, merge_customer_features
from src.modeling import train_and_score, save_metrics
from src.decisioning import build_priority_table


def main() -> None:
    cfg = Config()

    if not cfg.data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {cfg.data_path}\n"
            "Put CSV into data/raw/ and update src/config.py if needed."
        )

    cfg.outputs_dir.mkdir(parents=True, exist_ok=True)

    df = load_transactions(str(cfg.data_path))
    reference_date = df["InvoiceDate"].max()

    cust_last = customer_last_purchase(df)
    churn_df = add_churn_label(cust_last, reference_date, window_days=cfg.churn_window_days)

    rfm = build_rfm_features(df, reference_date)
    w30 = build_window_features(df, reference_date, cfg.win_30)
    w60 = build_window_features(df, reference_date, cfg.win_60)
    w90 = build_window_features(df, reference_date, cfg.win_90)
    feats = merge_customer_features(rfm, w30, w60, w90)

    dataset = feats.merge(churn_df[["CustomerID", "churn"]], on="CustomerID", how="inner")
    dataset.to_csv(cfg.customer_features_path, index=False)

    scores, metrics = train_and_score(
        dataset,
        target_col="churn",
        id_col="CustomerID",
        test_size=cfg.test_size,
        random_state=cfg.random_state,
    )
    save_metrics(metrics, cfg.metrics_path)

    priority = build_priority_table(dataset, scores, id_col="CustomerID")
    priority.to_csv(cfg.churn_priority_path, index=False)

    print("Reference date:", reference_date)
    print("Transactions:", df.shape)
    print("Customers:", dataset.shape[0])
    print("Churn share:", float(dataset["churn"].mean()))
    print("Metrics:", metrics)
    print(f"Saved features: {cfg.customer_features_path}")
    print(f"Saved priority table: {cfg.churn_priority_path}")
    print(f"Saved metrics: {cfg.metrics_path}")


if __name__ == "__main__":
    main()
