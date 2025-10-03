import sqlite3
import pandas as pd

conn = sqlite3.connect('data/Energy.db')

# Get fields with missing Quantity_initial_EJ
fields = pd.read_sql("""
    SELECT [Unit ID], [Country/Area], [Fuel type], [Quantity_initial_EJ]
    FROM Oil_Gas_fields
    WHERE Quantity_initial_EJ IS NULL
""", conn)

total_missing = len(fields)
fuel_missing = fields['Fuel type'].isnull().sum() + (fields['Fuel type'].astype(str).str.strip() == '').sum()

# Try to import the fuel type map for unsupported check
try:
    from convert_gas_oil_to_EJ import FUEL_TYPE_MAPPING
    # Get unique fuel types in missing data
    fuel_types = fields['Fuel type'].dropna().unique()
    unsupported_types = [f for f in fuel_types if f not in FUEL_TYPE_MAPPING]
    unsupported = len(fields[fields['Fuel type'].isin(unsupported_types)])
    print(f"Unsupported fuel types found: {unsupported_types}")
except Exception as e:
    unsupported = f'N/A (could not import: {e})'

# Check for matching reserves in PR table
pr_reserves = pd.read_sql("""
    SELECT DISTINCT [Unit ID]
    FROM Oil_Gas_Production_Reserves
    WHERE [Production/reserves] = 'reserves'
    AND [Quantity (converted)] IS NOT NULL
""", conn)

# Check for matching production in PR table
pr_production = pd.read_sql("""
    SELECT DISTINCT [Unit ID]
    FROM Oil_Gas_Production_Reserves
    WHERE [Production/reserves] = 'production'
    AND [Quantity (converted)] IS NOT NULL
""", conn)

fields_with_reserves = fields['Unit ID'].isin(pr_reserves['Unit ID']).sum()
fields_with_production = fields['Unit ID'].isin(pr_production['Unit ID']).sum()
fields_without_reserves = total_missing - fields_with_reserves
fields_without_production = total_missing - fields_with_production

print(f"\n=== MISSING QUANTITY DIAGNOSTIC ===")
print(f"Total missing quantities: {total_missing}")
print(f"\n=== FUEL TYPE ISSUES ===")
print(f"Fuel type missing/blank: {fuel_missing}")
print(f"Fuel type unsupported: {unsupported}")
print(f"\n=== DATA AVAILABILITY ===")
print(f"With matching reserves record: {fields_with_reserves}")
print(f"Without matching reserves record: {fields_without_reserves}")
print(f"With matching production record: {fields_with_production}")
print(f"Without matching production record: {fields_without_production}")

# Show sample of problematic records
print(f"\n=== SAMPLE MISSING RECORDS ===")
sample = fields[['Unit ID', 'Country/Area', 'Fuel type']].head(10)
print(sample.to_string(index=False))

conn.close()