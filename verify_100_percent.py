import sqlite3
import pandas as pd

conn = sqlite3.connect('data/Energy.db')

print("=== 100% COVERAGE VERIFICATION ===")

# Overall stats
total_stats = pd.read_sql("""
    SELECT 
        COUNT(*) as total,
        SUM(CASE WHEN _imputed_flag_quantity = 0 THEN 1 ELSE 0 END) as observed,
        SUM(CASE WHEN _imputed_flag_quantity = 1 THEN 1 ELSE 0 END) as imputed,
        COUNT(Quantity_initial_EJ) as populated
    FROM Oil_Gas_fields
""", conn)

print(f"Total fields: {total_stats.iloc[0]['total']}")
print(f"Populated: {total_stats.iloc[0]['populated']}")
print(f"Observed: {total_stats.iloc[0]['observed']}")
print(f"Imputed: {total_stats.iloc[0]['imputed']}")
print(f"Coverage: {total_stats.iloc[0]['populated']/total_stats.iloc[0]['total']*100:.1f}%")

# Method breakdown
method_stats = pd.read_sql("""
    SELECT 
        _imputed_by_quantity as method,
        COUNT(*) as count,
        ROUND(AVG(_imputation_conf_quantity), 3) as avg_confidence
    FROM Oil_Gas_fields 
    WHERE Quantity_initial_EJ IS NOT NULL 
    GROUP BY _imputed_by_quantity 
    ORDER BY count DESC
""", conn)

print(f"\n=== IMPUTATION METHOD BREAKDOWN ===")
print(method_stats.to_string(index=False))

# Quality distribution
quality_stats = pd.read_sql("""
    SELECT 
        CASE 
            WHEN _imputation_conf_quantity >= 0.8 THEN 'High (0.8-1.0)'
            WHEN _imputation_conf_quantity >= 0.4 THEN 'Medium (0.4-0.8)'
            WHEN _imputation_conf_quantity >= 0.2 THEN 'Low (0.2-0.4)'
            ELSE 'Very Low (0.1-0.2)'
        END as quality_tier,
        COUNT(*) as count
    FROM Oil_Gas_fields 
    WHERE Quantity_initial_EJ IS NOT NULL 
    GROUP BY 
        CASE 
            WHEN _imputation_conf_quantity >= 0.8 THEN 'High (0.8-1.0)'
            WHEN _imputation_conf_quantity >= 0.4 THEN 'Medium (0.4-0.8)'
            WHEN _imputation_conf_quantity >= 0.2 THEN 'Low (0.2-0.4)'
            ELSE 'Very Low (0.1-0.2)'
        END
    ORDER BY count DESC
""", conn)

print(f"\n=== QUALITY DISTRIBUTION ===")
print(quality_stats.to_string(index=False))

conn.close()
