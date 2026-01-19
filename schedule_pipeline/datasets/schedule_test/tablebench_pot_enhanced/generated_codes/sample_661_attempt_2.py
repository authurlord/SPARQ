import pandas as pd

df = pd.read_csv('table.csv')

# Clean and convert 'sunshine (hrs / year)' column by removing spaces and converting to int
df['sunshine (hrs / year)'] = df['sunshine (hrs / year)'].str.replace(' ', '').astype(int)

# Convert 'storms (days / year)' column to int
df['storms (days / year)'] = df['storms (days / year)'].astype(int)

# Calculate correlation coefficient
correlation = df['sunshine (hrs / year)'].corr(df['storms (days / year)'])

print(f"Final Answer: {correlation:.3f}")