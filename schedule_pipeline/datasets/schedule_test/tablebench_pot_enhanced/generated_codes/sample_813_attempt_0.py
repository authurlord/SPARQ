import pandas as pd

df = pd.read_csv('table.csv')

# Replace '–' with 0 for numerical operations
df = df.replace('–', 0)

# Convert seat columns to numeric
seat_columns = ['First Duma', 'Fourth Duma']
for col in seat_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Calculate percentage increase from First Duma to Fourth Duma
df['percentage_increase'] = ((df['Fourth Duma'] - df['First Duma']) / df['First Duma']) * 100

# Drop the 'TOTAL' row as it's a summary
df = df[df['Party'] != 'TOTAL']

# Find the party with the highest percentage increase
max_increase_row = df.loc[df['percentage_increase'].idxmax()]

print(f"Final Answer: {max_increase_row['Party']}")