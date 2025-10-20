import sys, sqlite3, pandas as pd

db = sys.argv[1] if len(sys.argv) > 1 else "sql_data.db"
con = sqlite3.connect(db)
# r = con.execute(
#    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name LIMIT 1"
# ).fetchone()
# if not r:
#    raise SystemExit("No tables found.")
# t = r[0]

query = """
WITH hq_data AS (
    SELECT
        MONTH,
        kpi_text,
        SUM(Act) AS act,
        SUM(py) AS py,
        SUM(rf) AS rf
    FROM forecast_data
    WHERE region = 'HQ'
      AND YEAR = 2025
      AND MONTH = 202509
      AND kpi_text IN ('Volume (MM) Kgs', 'Net Revenue', 'Gross Profit (MM)', 'Operating Income')
      AND Act IS NOT NULL
    GROUP BY MONTH, kpi_text
),

annual_summary AS (
    SELECT
        kpi_text AS kpi,
        SUM(act) AS act_ytd,
        SUM(rf) AS rf_ytd,
        SUM(py) AS py_ytd,
        SUM(act) - SUM(rf) AS diff_vs_fcst_ytd,
        CASE WHEN SUM(rf) <> 0 THEN (SUM(act) - SUM(rf)) / SUM(rf) * 100 ELSE NULL END AS pct_vs_fcst_ytd,
        SUM(act) - SUM(py) AS diff_vs_py_ytd,
        CASE WHEN SUM(py) <> 0 THEN (SUM(act) - SUM(py)) / SUM(py) * 100 ELSE NULL END AS pct_vs_py_ytd
    FROM hq_data
    GROUP BY kpi_text
)

SELECT *
FROM annual_summary
ORDER BY CASE
            WHEN kpi = 'Volume (MM) Kgs' THEN 1
            WHEN kpi = 'Net Revenue' THEN 2
            WHEN kpi = 'Gross Profit (MM)' THEN 3
            WHEN kpi = 'Operating Income' THEN 4
            ELSE 5
         END;
"""

df = pd.read_sql_query(query, con)
print(df)

# for i in df.columns:
#    # print(i, df[i].dtype),
#    print(df[i].head(3))
