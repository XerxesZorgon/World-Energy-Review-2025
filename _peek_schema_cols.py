import sqlite3, pathlib
p = (pathlib.Path(__file__).resolve().parent / '..' / 'data' / 'Energy.db').resolve()
con = sqlite3.connect(str(p))
cols = [r[1] for r in con.execute("PRAGMA table_info('Oil_Gas_fields')").fetchall()]
print("DB:", p)
print("Total cols:", len(cols))
print("All columns:\n", cols)

def show(label, pred):
    out = [c for c in cols if pred(c.lower())]
    print(f"\n{label} ({len(out)}):", out)

show("Fuel-like", lambda s: "fuel" in s)
show("Country-like", lambda s: "country" in s)
show("Lat/Lon-like", lambda s: "lat" in s or "lon" in s or "latitude" in s or "longitude" in s)
show("Quantity-like", lambda s: "quant" in s)
show("EJ-like", lambda s: "ej" in s)
show("Quantity+EJ", lambda s: "quant" in s and "ej" in s)
con.close()
