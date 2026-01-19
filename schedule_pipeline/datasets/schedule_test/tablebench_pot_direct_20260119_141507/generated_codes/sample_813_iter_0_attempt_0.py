import pandas as pd

# Load the dataframe
df = pd.read_csv('table.csv')

# Convert the relevant columns to numeric, handling missing values
df['First Duma'] = pd.to_numeric(df['First Duma'], errors='coerce')
df['Fourth Duma'] = pd.to_numeric(df['Fourth Duma'], errors='coerce')

# Filter out rows where First Duma is missing or zero
valid_rows = df.dropna(subset=['First Duma', 'Fourth Duma'])
valid_rows = valid_rows[valid_rows['First Duma'] != 0]

# Calculate percentage increase
valid_rows['increase'] = ((valid_rows['Fourth Duma'] - valid_rows['First Duma']) / valid_rows['First Duma']) * 100

# Find the party with the highest increase
max_increase_row = valid_rows.loc[valid_rows['increase'].idxmax()]
party_with_max_increase = max_increase_row['Party']

print(f"Final Answer: {party_with_max_increase}")