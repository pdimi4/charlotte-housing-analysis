# FinalFixedDatasetSQL.py

import pandas as pd
import sqlite3
from pathlib import Path

# -------------------------
# 1. File Paths
# -------------------------
DATA_DIR = Path("data")

employment = pd.read_csv(DATA_DIR / "Employment.csv")
graduation = pd.read_csv(DATA_DIR / "High School Graduation Rate.csv")
housing = pd.read_csv(DATA_DIR / "Home Sales Price.csv")
job_density = pd.read_csv(DATA_DIR / "Job Density.csv")
crime = pd.read_csv(DATA_DIR / "CMPD_Homicide.csv")

# -------------------------
# 2. Clean Data
# -------------------------
employment["2023"] = employment["2023"].replace("--", None)
graduation["2023"] = graduation["2023"].replace("--", None)

employment["2023"] = employment["2023"].str.replace("%", "", regex=False)
graduation["2023"] = graduation["2023"].str.replace("%", "", regex=False)

employment["2023"] = pd.to_numeric(employment["2023"], errors="coerce")
graduation["2023"] = pd.to_numeric(graduation["2023"], errors="coerce")

housing["2021"] = housing["2021"].replace(r"[\$,]", "", regex=True)
housing["2023"] = housing["2023"].replace(r"[\$,]", "", regex=True)

housing["2021"] = pd.to_numeric(housing["2021"], errors="coerce")
housing["2023"] = pd.to_numeric(housing["2023"], errors="coerce")

# -------------------------
# 3. Create SQLite Database
# -------------------------
conn = sqlite3.connect(":memory:")

employment.to_sql("Employment", conn, index=False, if_exists="replace")
graduation.to_sql("Graduation", conn, index=False, if_exists="replace")
housing.to_sql("Housing", conn, index=False, if_exists="replace")
job_density.to_sql("JobDensity", conn, index=False, if_exists="replace")
crime.to_sql("Crime", conn, index=False, if_exists="replace")

# -------------------------
# 4. SQL Query
# -------------------------
query = """
WITH crime_agg AS (
    SELECT
        NPA,
        COUNT(*) AS homicide_count
    FROM Crime
    GROUP BY NPA
)

SELECT
    e.NPA,
    e."2023" AS employment_2023,
    g."2023" AS grad_2023,
    j."2022" AS job_density_2022,
    h."2021" AS home_price_2021,
    h."2023" AS home_price_2023,
    COALESCE(c.homicide_count, 0) AS homicide_count
FROM Employment e
INNER JOIN Graduation g ON e.NPA = g.NPA
INNER JOIN JobDensity j ON e.NPA = j.NPA
INNER JOIN Housing h ON e.NPA = h.NPA
LEFT JOIN crime_agg c ON e.NPA = c.NPA
ORDER BY e.NPA
"""

# -------------------------
# 5. Execute SQL
# -------------------------
final_dataset = pd.read_sql(query, conn)

# Convert remaining model features to numeric values
final_dataset["job_density_2022"] = pd.to_numeric(
    final_dataset["job_density_2022"], errors="coerce"
)

final_dataset["homicide_count"] = final_dataset["homicide_count"].astype(int)

# -------------------------
# 6. Save Processed Dataset
# -------------------------
final_dataset.to_csv("final_fixed_dataset.csv", index=False)

# -------------------------
# 7. Output Preview
# -------------------------
print("\nFinal Dataset Preview:")
print(final_dataset.head())

print("\nData Types:")
print(final_dataset.dtypes)

print("\nDataset saved as final_fixed_dataset.csv")

conn.close()
