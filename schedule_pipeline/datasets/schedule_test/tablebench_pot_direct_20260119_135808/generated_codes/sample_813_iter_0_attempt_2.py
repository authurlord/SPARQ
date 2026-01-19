import pandas as pd

df = pd.read_csv('table.csv')

# Clean the data: Replace '–' with NaN and convert to numeric
df = df.replace('–', None)
df[['First Duma', 'Second Duma', 'Third Duma', 'Fourth Duma']] = df[['First Duma', 'Second Duma', 'Third Duma', 'Fourth Duma']].apply(pd.to_numeric, errors='coerce')

# Drop the 'TOTAL' row
df = df[df['Party'] != 'TOTAL']

# Calculate percentage increase from First to Fourth Duma
df['percentage_increase'] = ((df['Fourth Duma'] - df['First Duma']) / df['First Duma']) * 100

# Find the party with the highest percentage increase
max_increase_party = df.loc[df['percentage_increase'].idxmax()]['Party']

print(f"Final Answer: {max_increase_party}")