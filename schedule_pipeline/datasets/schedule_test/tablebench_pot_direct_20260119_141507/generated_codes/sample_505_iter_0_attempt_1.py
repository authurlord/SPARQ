import pandas as pd

df = pd.read_csv('table.csv')

# Calculate Hurricane Severity Index (HSI) for each year
df['hs_index'] = (df['number of hurricanes'] + df['number of major hurricanes']) / df['number of tropical storms']

# Find the year with the highest HSI
max_hsi_year = df.loc[df['hs_index'].idxmax(), 'year']

print(f"Final Answer: {max_hsi_year}")