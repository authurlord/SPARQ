import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert the data to numeric, handling missing values (–) as NaN
df = df.replace('--', np.nan)
df['First Duma'] = pd.to_numeric(df['First Duma'], errors='coerce')
df['Fourth Duma'] = pd.to_numeric(df['Fourth Duma'], errors='coerce')

# Filter out rows where First Duma is NaN (no data)
valid_parties = df.dropna(subset=['First Duma', 'Fourth Duma'])

# Calculate percentage increase
valid_parties['increase'] = ((valid_parties['Fourth Duma'] - valid_parties['First Duma']) / valid_parties['First Duma']) * 100

# Find the party with the highest increase
max_increase_party = valid_parties.loc[valid_parties['increase'].idxmax(), 'Party']

print(f"Final Answer: {max_increase_party}")