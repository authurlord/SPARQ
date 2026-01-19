import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Clean and convert the 'Crude birth rate (per 1000)' and 'Crude death rate (per 1000)' columns
df['Crude birth rate (per 1000)'] = df['Crude birth rate (per 1000)'].str.strip().astype(float)
df['Crude death rate (per 1000)'] = df['Crude death rate (per 1000)'].str.strip().astype(float)

# Calculate the difference between birth and death rates
df['difference'] = df['Crude birth rate (per 1000)'] - df['Crude death rate (per 1000)']

# Find the year with the maximum positive difference
max_diff_year = df.loc[df['difference'].idxmax(), 'Unnamed: 0']

print(f"Final Answer: {max_diff_year}")