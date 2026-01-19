import pandas as pd

df = pd.read_csv('table.csv')
# Calculate the difference between crude birth rate and crude death rate
df['difference'] = df['Crude birth rate (per 1000)'] - df['Crude death rate (per 1000)']
# Find the year with the largest positive difference
max_diff_row = df[df['difference'] > 0].loc[df['difference'].idxmax()]
year_with_max_margin = max_diff_row['Unnamed: 0']
print(f"Final Answer: {year_with_max_margin}")