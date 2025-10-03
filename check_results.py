import sqlite3
import pandas as pd

conn = sqlite3.connect('data/Energy.db')
stats = pd.read_sql("""
    SELECT 
        COUNT(*) as total_fields,
        COUNT(Quantity_initial_EJ) as quantities_populated,
        SUM(_imputed_flag_quantity) as quantities_imputed
    FROM Oil_Gas_fields
""", conn)

print('=== FINAL IMPUTATION RESULTS ===')
print(f'Total fields: {stats.iloc[0]["total_fields"]}')
print(f'Quantities populated: {stats.iloc[0]["quantities_populated"]}')
print(f'Quantities imputed: {stats.iloc[0]["quantities_imputed"]}')
print(f'Coverage: {stats.iloc[0]["quantities_populated"]/stats.iloc[0]["total_fields"]*100:.1f}%')

conn.close()
