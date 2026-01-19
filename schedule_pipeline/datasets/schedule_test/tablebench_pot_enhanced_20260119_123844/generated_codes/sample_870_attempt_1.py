import pandas as pd

df = pd.read_csv('table.csv')
# Calculate the difference between birth rate and death rate
df['difference'] = df['Crude birth rate (per 1000)'] - df['Crude death rate (per 1000)']
# Find the year with the maximum difference
max_diff_year = df.loc[df['difference'].idxmax(), 'Unnamed: 0']
print(f"Final Answer: {max_diff_year}")