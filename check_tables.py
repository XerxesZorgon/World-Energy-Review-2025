#!/usr/bin/env python3
import sqlite3
import pandas as pd

con = sqlite3.connect('data/Energy.db')

# Check Coal_open_mines table structure and data using quoted column names
print('Coal_open_mines data sample:')
sample = pd.read_sql('SELECT "GEM Mine ID", opening_year_final, reserves_initial_EJ FROM Coal_open_mines LIMIT 10', con)
print(sample)

print('\nCoal_open_mines reserves_initial_EJ stats:')
stats = pd.read_sql('SELECT COUNT(*) as total, COUNT(reserves_initial_EJ) as non_null, SUM(reserves_initial_EJ) as total_reserves FROM Coal_open_mines', con)
print(stats)

print('\nCoal discoveries by year (sample):')
discoveries = pd.read_sql('SELECT opening_year_final as Year, SUM(reserves_initial_EJ) as Discoveries FROM Coal_open_mines WHERE opening_year_final IS NOT NULL AND reserves_initial_EJ IS NOT NULL GROUP BY opening_year_final ORDER BY opening_year_final LIMIT 10', con)
print(discoveries)

con.close()
