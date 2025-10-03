import sqlite3
import pandas as pd
import numpy as np

conn = sqlite3.connect('data/Energy.db')

print("=== COMPREHENSIVE COVERAGE ANALYSIS ===")

# 1. Overall status
total = pd.read_sql("SELECT COUNT(*) as total FROM Oil_Gas_fields", conn).iloc[0]['total']
populated = pd.read_sql("SELECT COUNT(*) as populated FROM Oil_Gas_fields WHERE Quantity_initial_EJ IS NOT NULL", conn).iloc[0]['populated']
missing = total - populated

print(f"Total fields: {total}")
print(f"Populated: {populated} ({populated/total*100:.1f}%)")
print(f"Missing: {missing} ({missing/total*100:.1f}%)")

# 2. Missing fields breakdown
missing_fields = pd.read_sql("""
    SELECT [Unit ID], [Country/Area], [Fuel type], [Status], [Basin], 
           [Production start year], [FID Year], [Operator]
    FROM Oil_Gas_fields 
    WHERE Quantity_initial_EJ IS NULL
""", conn)

print(f"\n=== MISSING FIELDS BREAKDOWN ===")
print(f"Missing by Country (top 10):")
country_counts = missing_fields['Country/Area'].value_counts().head(10)
for country, count in country_counts.items():
    print(f"  {country}: {count}")

print(f"\nMissing by Fuel Type:")
fuel_counts = missing_fields['Fuel type'].value_counts()
for fuel, count in fuel_counts.items():
    print(f"  {fuel}: {count}")

print(f"\nMissing by Status:")
status_counts = missing_fields['Status'].value_counts()
for status, count in status_counts.items():
    print(f"  {status}: {count}")

# 3. Data availability for missing fields
print(f"\n=== DATA AVAILABILITY FOR MISSING FIELDS ===")

# Check reserves availability
reserves_available = pd.read_sql("""
    SELECT COUNT(DISTINCT f.[Unit ID]) as count
    FROM Oil_Gas_fields f
    JOIN Oil_Gas_Production_Reserves pr ON f.[Unit ID] = pr.[Unit ID]
    WHERE f.Quantity_initial_EJ IS NULL 
    AND pr.[Production/reserves] = 'reserves'
    AND pr.[Quantity (converted)] IS NOT NULL
""", conn).iloc[0]['count']

# Check production availability  
production_available = pd.read_sql("""
    SELECT COUNT(DISTINCT f.[Unit ID]) as count
    FROM Oil_Gas_fields f
    JOIN Oil_Gas_Production_Reserves pr ON f.[Unit ID] = pr.[Unit ID]
    WHERE f.Quantity_initial_EJ IS NULL 
    AND pr.[Production/reserves] = 'production'
    AND pr.[Quantity (converted)] IS NOT NULL
""", conn).iloc[0]['count']

print(f"Missing fields with reserves data: {reserves_available}")
print(f"Missing fields with production data: {production_available}")
print(f"Missing fields with NO reserves data: {missing - reserves_available}")
print(f"Missing fields with NO production data: {missing - production_available}")

# 4. Sample of completely missing data
no_data_fields = pd.read_sql("""
    SELECT f.[Unit ID], f.[Country/Area], f.[Fuel type], f.[Status]
    FROM Oil_Gas_fields f
    LEFT JOIN Oil_Gas_Production_Reserves pr ON f.[Unit ID] = pr.[Unit ID]
    WHERE f.Quantity_initial_EJ IS NULL 
    AND pr.[Unit ID] IS NULL
    LIMIT 10
""", conn)

print(f"\n=== SAMPLE FIELDS WITH NO PR DATA ===")
print(no_data_fields.to_string(index=False))

# 5. Fields that should be imputable but aren't
should_be_imputable = pd.read_sql("""
    SELECT f.[Unit ID], f.[Country/Area], f.[Fuel type], 
           COUNT(pr.[Unit ID]) as pr_records,
           SUM(CASE WHEN pr.[Production/reserves] = 'reserves' THEN 1 ELSE 0 END) as reserves_records,
           SUM(CASE WHEN pr.[Production/reserves] = 'production' THEN 1 ELSE 0 END) as production_records
    FROM Oil_Gas_fields f
    JOIN Oil_Gas_Production_Reserves pr ON f.[Unit ID] = pr.[Unit ID]
    WHERE f.Quantity_initial_EJ IS NULL 
    AND pr.[Quantity (converted)] IS NOT NULL
    GROUP BY f.[Unit ID], f.[Country/Area], f.[Fuel type]
    HAVING COUNT(pr.[Unit ID]) > 0
    LIMIT 10
""", conn)

print(f"\n=== SAMPLE FIELDS THAT SHOULD BE IMPUTABLE ===")
print(should_be_imputable.to_string(index=False))

conn.close()
