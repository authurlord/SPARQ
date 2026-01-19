import pandas as pd

df = pd.read_csv('table.csv')
# Convert '2007' and '2011' columns to numeric, coercing errors to NaN
df['2007'] = pd.to_numeric(df['2007'], errors='coerce')
df['2011'] = pd.to_numeric(df['2011'], errors='coerce')

# Calculate the increase from 2007 to 2011
df['increase'] = df['2011'] - df['2007']

# Find the school with the maximum increase
max_increase_school = df.loc[df['increase'].idxmax(), 'School']

print(f"Final Answer: {max_increase_school}")