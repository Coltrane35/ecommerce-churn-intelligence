from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine


def load_from_sql(
    connection_string: str,
    query: str,
) -> pd.DataFrame:
    """
    Load data from SQL database into a pandas DataFrame.

    Example connection string:
    mysql+pymysql://user:password@localhost:3306/database_name
    """

    engine = create_engine(connection_string)

    with engine.connect() as connection:
        df = pd.read_sql(query, connection)

    return df


def save_predictions_to_sql(
    df: pd.DataFrame,
    connection_string: str,
    table_name: str = "churn_predictions",
) -> None:
    """
    Save churn / CLV / ROI predictions to a SQL table.
    """

    engine = create_engine(connection_string)

    with engine.connect() as connection:
        df.to_sql(
            table_name,
            connection,
            if_exists="replace",
            index=False,
        )