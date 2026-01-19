import pandas as pd

df = pd.read_csv('table.csv')

# Replace '–' with 0 for numerical processing
df = df.replace('–', 0)

# Convert the relevant columns to integers
df['First Duma'] = df['First Duma'].str.extract('(\d+)').astype(int)
df['Fourth Duma'] = df['Fourth Duma'].str.extract('(\d+)').astype(int)

# Calculate percentage increase
df['Percentage Increase'] = ((df['Fourth Duma'] - df['First Duma']) / df['First Duma']) * 100

# Exclude the 'TOTAL' row
df = df[df['Party'] != 'TOTAL']

# Find the party with the highest percentage increase
max_increase_party = df.loc[df['Percentage Increase'].idxmax()]['Party']

print(f"Final Answer: {max_increase_party}")