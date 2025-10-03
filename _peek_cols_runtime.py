import sqlite3, pathlib, pandas as pd
p = (pathlib.Path(__file__).resolve().parent / ".." / "data" / "Energy.db").resolve()
print("Using DB:", p)
con = sqlite3.connect(str(p))
cols = [r[1] for r in con.execute("PRAGMA table_info('Oil_Gas_fields')").fetchall()]
print("PRAGMA columns:", cols)
df = pd.read_sql("SELECT * FROM Oil_Gas_fields LIMIT 1", con)
print("pandas df.columns:", df.columns.tolist())
con.close()
