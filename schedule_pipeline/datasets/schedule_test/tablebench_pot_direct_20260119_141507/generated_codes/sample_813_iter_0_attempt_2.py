import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert relevant columns to numeric, replacing '–' with NaN
df['First Duma'] = df['First Duma'].replace('–', np.nan).astype(float)
df['Fourth Duma'] = df['Fourth Duma'].replace('–', np.nan).astype(float)

# Filter out rows where First Duma is NaN (i.e., missing data)
valid_rows = df.dropna(subset=['First Duma', 'Fourth Duma'])

# Calculate percentage increase
valid_rows['increase'] = ((valid_rows['Fourth Duma'] - valid_rows['First Duma']) / valid_rows['First Duma']) * 100

# Find the party with the highest increase
max_increase_row = valid_rows.loc[valid_rows['increase'].idxmax()]
final_party = max_increase_row['Party']

print(f"Final Answer: {final_party}")