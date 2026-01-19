import pandas as pd

df = pd.read_csv('table.csv')
# Convert the relevant columns to numeric, coercing errors to NaN if any
df['property taxes'] = pd.to_numeric(df['property taxes'], errors='coerce')
df['investment earnings'] = pd.to_numeric(df['investment earnings'], errors='coerce')

# Calculate the difference between property taxes and investment earnings
df['difference'] = df['property taxes'] - df['investment earnings']

# Find the year with the maximum difference
max_diff_year = df.loc[df['difference'].idxmax(), 'year']
print(f"Final Answer: {max_diff_year}")