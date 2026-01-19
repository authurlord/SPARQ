import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'First Duma' and 'Fourth Duma' columns to numeric, handling '–' as NaN
df['First Duma'] = pd.to_numeric(df['First Duma'].str.split(' ').str[0], errors='coerce')
df['Fourth Duma'] = pd.to_numeric(df['Fourth Duma'].str.split(' ').str[0], errors='coerce')

# Drop the 'TOTAL' row since it's a summary
df = df[df['Party'] != 'TOTAL']

# Calculate percentage increase
df['Percentage Increase'] = ((df['Fourth Duma'] - df['First Duma']) / df['First Duma']) * 100

# Find the party with the highest percentage increase
max_increase_party = df.loc[df['Percentage Increase'].idxmax()]['Party']

print(f"Final Answer: {max_increase_party}")