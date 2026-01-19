import pandas as pd

df = pd.read_csv('table.csv')

# Convert GDP columns to numeric, coercing errors to NaN if needed
df['2011 (imf)'] = pd.to_numeric(df['2011 (imf)'], errors='coerce')
df['2008 (cia factbook)'] = pd.to_numeric(df['2008 (cia factbook)'], errors='coerce')

# Calculate the absolute difference
df['diff'] = abs(df['2011 (imf)'] - df['2008 (cia factbook)'])

# Identify countries with significant deviation (difference > 10000)
significant_deviation = df[df['diff'] > 10000]['nation'].tolist()

print(f"Final Answer: {', '.join(significant_deviation)}")