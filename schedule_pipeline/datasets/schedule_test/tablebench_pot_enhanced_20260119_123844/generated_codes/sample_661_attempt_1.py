import pandas as pd

df = pd.read_csv('table.csv')

# Clean the 'sunshine (hrs / year)' and 'storms (days / year)' columns by removing spaces
df['sunshine (hrs / year)'] = df['sunshine (hrs / year)'].str.replace(' ', '').astype(int)
df['storms (days / year)'] = df['storms (days / year)'].astype(int)

# Calculate the correlation coefficient
correlation = df['sunshine (hrs / year)'].corr(df['storms (days / year)'])

print(f"Final Answer: {correlation:.3f}")