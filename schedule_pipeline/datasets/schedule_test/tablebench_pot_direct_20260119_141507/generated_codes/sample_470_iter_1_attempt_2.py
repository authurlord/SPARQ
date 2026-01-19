import pandas as pd

df = pd.read_csv('table.csv')

# Convert GDP columns to numeric, coercing errors to NaN (though data appears consistent)
df['2011 (imf)'] = pd.to_numeric(df['2011 (imf)'], errors='coerce')
df['2008 (cia factbook)'] = pd.to_numeric(df['2008 (cia factbook)'], errors='coerce')

# Calculate absolute difference
df['difference'] = abs(df['2011 (imf)'] - df['2008 (cia factbook)'])

# Filter countries with significant deviation (difference > 10,000)
significant_deviations = df[df['difference'] > 10000]['nation'].tolist()

print(f"Final Answer: {', '.join(significant_deviations)}")