import sqlite3, pathlib
p=(pathlib.Path(__file__).resolve().parent/'..'/'data'/'Energy.db').resolve()
con=sqlite3.connect(str(p))
print('DB:', p)
cols=[r[1] for r in con.execute("PRAGMA table_info('Oil_Gas_fields')").fetchall()]
print('Columns:', cols)
con.close()
