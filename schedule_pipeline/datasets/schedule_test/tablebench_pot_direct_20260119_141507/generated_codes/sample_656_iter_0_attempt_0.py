import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'bp comp 1 (˚C)' and '% wt comp 1' to numeric, handling any formatting issues
df['bp comp 1 (˚c)'] = pd.to_numeric(df['bp comp 1 (˚c)'], errors='coerce')
df['% wt comp 1'] = pd.to_numeric(df['% wt comp 1'], errors='coerce')

# Drop rows with NaN due to invalid conversion
df_clean = df.dropna(subset=['bp comp 1 (˚c)', '% wt comp 1'])

# Compute correlation coefficient
correlation = df_clean['bp comp 1 (˚c)'].corr(df_clean['% wt comp 1'])

print(f"Final Answer: {correlation:.3f}")