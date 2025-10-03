"""
Quick verification script to check that imputation logic is working correctly
"""
import sqlite3
import pandas as pd

def verify_coal_imputation():
    conn = sqlite3.connect('data/Energy.db')
    
    # Get sample of results
    df = pd.read_sql('''
        SELECT "Opening Year", opening_year_final, _imputed_flag_year 
        FROM Coal_open_mines 
        WHERE "Opening Year" IS NOT NULL 
        LIMIT 10
    ''', conn)
    
    print("=== COAL IMPUTATION VERIFICATION ===")
    print("Sample rows with original Opening Year values:")
    print(df)
    print()
    
    # Check if original values are preserved
    preserved = (df["Opening Year"] == df["opening_year_final"]).all()
    print(f"All original Opening Year values preserved: {preserved}")
    
    # Get summary statistics
    total_rows = pd.read_sql('SELECT COUNT(*) as cnt FROM Coal_open_mines', conn).iloc[0,0]
    with_original = pd.read_sql('SELECT COUNT(*) as cnt FROM Coal_open_mines WHERE "Opening Year" IS NOT NULL', conn).iloc[0,0]
    imputed_count = pd.read_sql('SELECT COUNT(*) as cnt FROM Coal_open_mines WHERE _imputed_flag_year = 1', conn).iloc[0,0]
    
    print(f"\nSummary:")
    print(f"Total coal mines: {total_rows}")
    print(f"With original Opening Year: {with_original}")
    print(f"Imputed (missing): {imputed_count}")
    print(f"Expected missing: {total_rows - with_original}")
    print(f"Imputation count matches: {imputed_count == (total_rows - with_original)}")
    
    # Check range of final values
    year_stats = pd.read_sql('SELECT MIN(opening_year_final) as min_year, MAX(opening_year_final) as max_year FROM Coal_open_mines WHERE opening_year_final IS NOT NULL', conn)
    print(f"\nYear range in opening_year_final: {year_stats.iloc[0,0]} to {year_stats.iloc[0,1]}")
    
    conn.close()

if __name__ == "__main__":
    verify_coal_imputation()
