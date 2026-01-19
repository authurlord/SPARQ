import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'First Duma' and 'Fourth Duma' to numeric, treating '–' as 0
df['First Duma'] = pd.to_numeric(df['First Duma'].str.replace(r'\(.*\)', '', regex=True).str.strip(), errors='coerce').fillna(0)
df['Fourth Duma'] = pd.to_numeric(df['Fourth Duma'].str.replace(r'\(.*\)', '', regex=True).str.strip(), errors='coerce').fillna(0)

# Calculate percentage increase
df['percentage_increase'] = ((df['Fourth Duma'] - df['First Duma']) / df['First Duma']) * 100

# Exclude the 'TOTAL' row and find the party with the highest percentage increase
max_increase_row = df[df['Party'] != 'TOTAL'].loc[df['percentage_increase'].idxmax()]

print(f"Final Answer: {max_increase_row['Party']}")