import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'bp comp 1 (˚c)' and '% wt comp 1' to numeric, handling errors
df['bp comp 1 (˚c)'] = pd.to_numeric(df['bp comp 1 (˚c)'], errors='coerce')
df['% wt comp 1'] = pd.to_numeric(df['% wt comp 1'], errors='coerce')

# Drop rows with NaN due to invalid conversion
df = df.dropna(subset=['bp comp 1 (˚c)', '% wt comp 1'])

# Calculate the correlation coefficient
correlation = df['bp comp 1 (˚c)'].corr(df['% wt comp 1'])
print(f"Final Answer: {correlation:.3f}")