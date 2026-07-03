"""SQL data layer (SQLite, standard-library only).

Represents the "collecte SQL" stage: the generated applicants are written to a
real relational table, and the pipeline *reads them back with a SQL query* —
so the ETL genuinely starts from a database, not a CSV. In production this
module's ``collect`` query is the only thing that changes (point it at
PostgreSQL/MySQL/Snowflake); the rest of the pipeline is untouched.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

TABLE = "applicants"

# The collection query. Filters out obviously invalid rows at the source and
# documents exactly which fields the pipeline consumes.
COLLECT_QUERY = f"""
SELECT
    client_id, age, monthly_income, employment_length, loan_amount,
    loan_term_months, debt_to_income, credit_history_length,
    num_past_defaults, credit_utilization, num_open_accounts,
    num_recent_inquiries, home_ownership, purpose, "default"
FROM {TABLE}
WHERE age BETWEEN 18 AND 100
  AND loan_amount > 0
"""


def create_database(df: pd.DataFrame, db_path: str | Path) -> None:
    """Write the applicants dataframe to a SQLite table (replacing any prior)."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        df.to_sql(TABLE, conn, if_exists="replace", index=False)
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_client ON {TABLE}(client_id)")
        conn.commit()


def collect(db_path: str | Path, query: str = COLLECT_QUERY) -> pd.DataFrame:
    """Collect the applicant dataset from the database via SQL."""
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(query, conn)


def fetch_client(db_path: str | Path, client_id: int) -> dict | None:
    """Fetch a single applicant profile (used by the demo / app)."""
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            f'SELECT * FROM {TABLE} WHERE client_id = ?', (client_id,)
        )
        row = cur.fetchone()
        return dict(row) if row else None
