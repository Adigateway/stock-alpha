"""
db_config.py — database connection.

Copy this to db_config.py in the project root and fill in your own values.
The real file is gitignored and must never be committed.
"""

import pandas as pd
import psycopg2
from sqlalchemy import create_engine

PG_CONFIG = {
    "host":     "localhost",
    "database": "stock_alpha",
    "user":     "stockuser",
    "password": "YOUR_PASSWORD_HERE",
}

# '@' and other specials must be percent-encoded here: @ -> %40
ENGINE_URL = "postgresql+psycopg2://stockuser:YOUR_PASSWORD_HERE@localhost/stock_alpha"


def get_engine():
    return create_engine(ENGINE_URL)


def get_conn():
    return psycopg2.connect(**PG_CONFIG)


def db_read(query: str) -> pd.DataFrame:
    with get_engine().connect() as conn:
        return pd.read_sql(query, conn)


def db_write(df: pd.DataFrame, table: str, if_exists="replace"):
    conn, cur = get_conn(), None
    try:
        cur = conn.cursor()
        if if_exists == "replace":
            cur.execute(f'DROP TABLE IF EXISTS "{table}"')
            cols = []
            for c, dt in zip(df.columns, df.dtypes):
                t = ("DOUBLE PRECISION" if "int" in str(dt) or "float" in str(dt)
                     else "TIMESTAMP" if "datetime" in str(dt) else "TEXT")
                cols.append(f'"{c}" {t}')
            cur.execute(f'CREATE TABLE "{table}" ({", ".join(cols)})')
        rows = [tuple(None if pd.isna(v) else v for v in r)
                for r in df.itertuples(index=False)]
        ph = ", ".join(["%s"] * len(df.columns))
        names = ", ".join(f'"{c}"' for c in df.columns)
        cur.executemany(f'INSERT INTO "{table}" ({names}) VALUES ({ph})', rows)
        conn.commit()
        print(f"  {table} -> {len(df):,} rows written")
    except Exception:
        conn.rollback()
        raise
    finally:
        if cur:
            cur.close()
        conn.close()
