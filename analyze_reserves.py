import sqlite3
import pandas as pd

conn = sqlite3.connect('data/Energy.db')

# Get fields missing quantities
missing = pd.read_sql("""
    SELECT [Unit ID], [Fuel type] 
    FROM Oil_Gas_fields 
    WHERE Quantity_initial_EJ IS NULL
""", conn)

# Get reserves records
pr_reserves = pd.read_sql("""
    SELECT [Unit ID], [Fuel description], [Units (converted)], [Quantity (converted)]
    FROM Oil_Gas_Production_Reserves 
    WHERE [Production/reserves] = 'reserves' 
    AND [Quantity (converted)] IS NOT NULL
""", conn)

# Find overlap
overlap_ids = missing[missing['Unit ID'].isin(pr_reserves['Unit ID'])]['Unit ID'].head(5)
sample_pr = pr_reserves[pr_reserves['Unit ID'].isin(overlap_ids)]

print("=== FIELDS WITH RESERVES BUT MISSING QUANTITIES ===")
print("Sample reserves records for fields missing quantities:")
print(sample_pr.to_string(index=False))

print(f"\nTotal fields missing quantities: {len(missing)}")
print(f"Fields with reserves data: {len(missing[missing['Unit ID'].isin(pr_reserves['Unit ID'])])}")

# Check for conversion issues
print("\n=== CONVERSION ANALYSIS ===")
for _, row in sample_pr.head(3).iterrows():
    fuel = row['Fuel description']
    unit = row['Units (converted)']
    qty = row['Quantity (converted)']
    print(f"Unit ID: {row['Unit ID']}")
    print(f"  Fuel: '{fuel}' -> Unit: '{unit}' -> Qty: {qty}")
    
    # Try conversion
    try:
        from convert_gas_oil_to_EJ import convert_to_ej
        result = convert_to_ej(qty, unit, fuel_type=fuel)
        print(f"  Conversion result: {result}")
    except Exception as e:
        print(f"  Conversion FAILED: {e}")
    print()

conn.close()
