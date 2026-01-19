import pandas as pd

df = pd.read_csv('table.csv')
# Calculate the difference between crude birth rate and crude death rate
df['difference'] = df['Crude birth rate (per 1000)'] - df['Crude death rate (per 1000)']
# Find the year with the largest positive difference
max_diff_year = df[df['difference'] == df['difference'].max()]['Unnamed: 0'].values[0]
print(f"Final Answer: {max_diff_year}")