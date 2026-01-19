import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')
# Filter rows for years 2004 to 2013 (inclusive)
filtered_df = df[(df['Year'].astype(str).str.startswith('2004')) | 
                  (df['Year'].astype(str).str.startswith('2005')) | 
                  (df['Year'].astype(str).str.startswith('2006')) | 
                  (df['Year'].astype(str).str.startswith('2007')) | 
                  (df['Year'].astype(str).str.startswith('2008')) | 
                  (df['Year'].astype(str).str.startswith('2009')) | 
                  (df['Year'].astype(str).str.startswith('2010')) | 
                  (df['Year'].astype(str).str.startswith('2011')) | 
                  (df['Year'].astype(str).str.startswith('2012')) | 
                  (df['Year'].astype(str).str.startswith('2013'))]

# Convert 'No. of performances' to numeric, replacing non-numeric entries with NaN
performances = pd.to_numeric(filtered_df['No. of performances'], errors='coerce')

# Calculate the mean of valid performance numbers
mean_performances = performances.mean()

print(f"Final Answer: {mean_performances:.1f}")